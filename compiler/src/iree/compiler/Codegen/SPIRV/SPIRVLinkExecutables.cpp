// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-spirv-link-executable"

namespace mlir::iree_compiler {

namespace IREE::HAL {
// Compares two ExecutableTargetAttr according to the alphabetical order of used
// SPIR-V features.
//
// Note that this is a very specific ordering per the needs of this pass--we
// guarantee that input ExectuableTargetAttr only differ w.r.t. their used
// SPIR-V features, and we want a deterministic order when mutating the IR.
bool operator<(const ExecutableTargetAttr &a, const ExecutableTargetAttr &b) {
  auto aFeatures = a.getConfiguration().getAs<ArrayAttr>("iree.spirv.features");
  auto bFeatures = b.getConfiguration().getAs<ArrayAttr>("iree.spirv.features");
  for (unsigned i = 0; i < std::min(aFeatures.size(), bFeatures.size()); ++i) {
    if (aFeatures[i] != bFeatures[i]) {
      return cast<StringAttr>(aFeatures[i]).getValue() <
             cast<StringAttr>(bFeatures[i]).getValue();
    }
  }
  return aFeatures.size() < bFeatures.size();
}
} // namespace IREE::HAL

namespace {

using IREE::HAL::ExecutableTargetAttr;

bool isSPIRVBasedBackend(StringRef backend) {
  return backend.starts_with("vulkan") || backend.starts_with("metal") ||
         backend.starts_with("webgpu");
}

struct SPIRVLinkExecutablesPass final
    : SPIRVLinkExecutablesBase<SPIRVLinkExecutablesPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // Collect all source executable ops.
    SmallVector<IREE::HAL::ExecutableOp, 8> sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1)
      return;

    // Retain only non-external source executables. Linking right now happens as
    // placing spirv.module ops into the same hal.executable.variant ops.
    // External source executables won't have any spirv.modules inside.
    int retainSize = 0;
    for (int i = 0, e = sourceExecutableOps.size(); i < e; ++i) {
      IREE::HAL::ExecutableOp executable = sourceExecutableOps[i];
      if (llvm::none_of(executable.getOps<IREE::HAL::ExecutableVariantOp>(),
                        [](auto op) { return op.getObjects().has_value(); })) {
        sourceExecutableOps[retainSize++] = executable;
      }
    }
    sourceExecutableOps.resize(retainSize);

    // Note that at runtime, for a particular executable, only one variant of it
    // will be loaded. So, all variants of an executable are expected to provide
    // the exact same set of entry points; this way we can guarantee no matter
    // which variant is chosen, we have all entry points to call into. The same
    // entry point in different variants may have different target requirements
    // though.
    //
    // The input to the linking stage are a collection of executables, each may
    // have multiple variants, but only ever provide one entry point. Together
    // with the above restriction, we can link two executables if and only if
    // their variants have the exact same set of target requirements. Under such
    // circumstances, we can make sure for a particular target requirement
    // (loaded as one variant during runtime), we can provide all entry points.

    // Build a map from all variants' target requirements to their wrapping
    // executable ops.
    std::map<SmallVector<ExecutableTargetAttr, 0>,
             SmallVector<IREE::HAL::ExecutableOp>>
        executableBuckets;

    SmallVector<ExecutableTargetAttr, 0> currentTargets;
    for (IREE::HAL::ExecutableOp executable : sourceExecutableOps) {
      // Go through all variants and collect all their target requirements and
      // sort as the unique key.
      currentTargets.clear();
      for (auto variant : executable.getOps<IREE::HAL::ExecutableVariantOp>()) {
        ExecutableTargetAttr target = variant.getTarget();
        if (isSPIRVBasedBackend(target.getBackend())) {
          currentTargets.push_back(target);
        }
      }
      llvm::sort(currentTargets);
      LLVM_DEBUG({
        llvm::dbgs() << "executable op @" << executable.getSymName()
                     << " targets:\n";
        for (ExecutableTargetAttr attr : currentTargets) {
          llvm::dbgs() << "  " << attr << "\n";
        }
      });

      // Put this executable into its proper bucket.
      executableBuckets[std::move(currentTargets)].push_back(executable);
    }

    // Scan through the buckets and drop those with only one executables, given
    // nothing to link for such cases.
    for (auto it = executableBuckets.begin(), ie = executableBuckets.end();
         it != ie;) {
      if (it->second.size() <= 1) {
        it = executableBuckets.erase(it);
      } else {
        ++it;
      }
    }

    // Guess a base module name, if needed, to make the output files readable.
    std::string baseModuleName =
        guessModuleName(moduleOp, "spirv_module") + "_linked_spirv";
    // Go reverse order with index, so when we keep inserting at the beginning,
    // the final IR has ascending order.
    int bucketIndex = executableBuckets.size();

    for (auto [key, bucket] : llvm::reverse(executableBuckets)) {
      --bucketIndex;
      // Build a unique name for this particular executable.
      std::string moduleName =
          executableBuckets.size() == 1
              ? baseModuleName
              : llvm::formatv("{0}_{1}", baseModuleName, bucketIndex);

      LLVM_DEBUG({
        llvm::dbgs() << "executable bucket #" << bucketIndex << " targets:\n";
        for (ExecutableTargetAttr attr : key) {
          llvm::dbgs() << "  " << attr << "\n";
        }
        llvm::dbgs() << "executable bucket #" << bucketIndex
                     << " exectuables:\n";
        for (IREE::HAL::ExecutableOp executable : bucket) {
          llvm::dbgs() << "  " << executable.getSymName() << "\n";
        }
      });

      if (failed(linkOneExecutableBucket(moduleOp, moduleName, key, bucket)))
        return signalPassFailure();
    }
  }

  // Links all executables that are known to be in the same bucket.
  LogicalResult linkOneExecutableBucket(
      mlir::ModuleOp moduleOp, StringRef linkedExecutableName,
      ArrayRef<ExecutableTargetAttr> executableTargetAttrs,
      SmallVectorImpl<IREE::HAL::ExecutableOp> &sourceExecutableOps) const {
    OpBuilder moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

    // Create our new "linked" hal.executable.
    auto linkedExecutableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), linkedExecutableName);
    linkedExecutableOp.setVisibility(
        sourceExecutableOps.front().getVisibility());
    OpBuilder executableBuilder =
        OpBuilder::atBlockBegin(&linkedExecutableOp.getBlock());

    for (auto [index, attr] : llvm::enumerate(executableTargetAttrs)) {
      // Add our hal.executable.variant with an empty module.
      std::string linkedVariantName =
          executableTargetAttrs.size() == 1
              ? attr.getSymbolNameFragment()
              : llvm::formatv("{0}_{1}", attr.getSymbolNameFragment(), index);
      auto linkedTargetOp =
          executableBuilder.create<IREE::HAL::ExecutableVariantOp>(
              moduleOp.getLoc(), linkedVariantName, attr);
      auto targetBuilder = OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
      targetBuilder.create<mlir::ModuleOp>(moduleOp.getLoc());

      auto mergeModuleFn = [](mlir::ModuleOp sourceInnerModule,
                              mlir::ModuleOp linkedInnerModule,
                              DenseMap<StringRef, Operation *> &symbolMap) {
        // spirv.module is isolated from above. It does not define symbols or
        // reference outside symbols too. So we can just simply move it to the
        // linked inner module.
        auto srcModules = sourceInnerModule.getOps<spirv::ModuleOp>();
        assert(std::distance(srcModules.begin(), srcModules.end()) == 1);
        Operation *srcModule = *srcModules.begin();
        Block &targetBlock = *linkedInnerModule->getRegion(0).begin();
        if (!targetBlock.empty()) {
          srcModule->moveAfter(&targetBlock.back());
        } else {
          srcModule->moveBefore(&targetBlock, targetBlock.end());
        }
        return success();
      };

      // Try linking together all executables in moduleOp.
      if (failed(linkExecutablesInto(moduleOp, sourceExecutableOps,
                                     linkedExecutableOp, linkedTargetOp,
                                     mergeModuleFn))) {
        return failure();
      }
    }
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createSPIRVLinkExecutablesPass() {
  return std::make_unique<SPIRVLinkExecutablesPass>();
}

} // namespace mlir::iree_compiler
