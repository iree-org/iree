// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_MATERIALIZERESOURCECACHESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

// TODO(multi-device): rewrite this to shard resources per device.
struct MaterializeResourceCachesPass
    : public IREE::HAL::impl::MaterializeResourceCachesPassBase<
          MaterializeResourceCachesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty())
      return;
    moduleBuilder = OpBuilder(&moduleOp.getBody()->front());

    // Find all relevant ops. If we don't find any we skip the pass as it's
    // likely it's already been run. We could fix the pass to better support
    // partial materialization but there's no use cases for that today.
    auto executableOps = llvm::to_vector<8>(moduleOp.getOps<ExecutableOp>());
    SmallVector<IREE::HAL::DescriptorSetLayoutLookupOp>
        descriptorSetLayoutLookupOps;
    SmallVector<IREE::HAL::PipelineLayoutLookupOp> pipelineLayoutLookupOps;
    SmallVector<IREE::HAL::ExecutableLookupOp> executableLookupOps;
    for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
      for (auto &block : funcOp.getFunctionBody()) {
        block.walk([&](Operation *op) {
          if (auto lookupOp = dyn_cast<DescriptorSetLayoutLookupOp>(op)) {
            descriptorSetLayoutLookupOps.push_back(lookupOp);
          } else if (auto lookupOp = dyn_cast<PipelineLayoutLookupOp>(op)) {
            pipelineLayoutLookupOps.push_back(lookupOp);
          } else if (auto lookupOp = dyn_cast<ExecutableLookupOp>(op)) {
            executableLookupOps.push_back(lookupOp);
          }
        });
      }
    }
    if (descriptorSetLayoutLookupOps.empty() &&
        pipelineLayoutLookupOps.empty() && executableLookupOps.empty()) {
      return;
    }

    // Declare all layouts used by the executables. This will ensure that the
    // initialization order is correct as any pipeline layout needed (and its
    // dependencies) will be created prior to the executable cache below. The
    // other nice thing is that we get ordering similar to the executable
    // variables above.
    for (auto executableOp : executableOps) {
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        for (auto exportOp : variantOp.getExportOps()) {
          definePipelineLayoutOp(exportOp.getLoc(), exportOp.getLayout());
        }
      }
    }

    // Declare executable variables so that we can reference them during lookup
    // replacement.
    for (auto executableOp : executableOps) {
      defineExecutableOp(executableOp);
    }

    // Generate cached resource singletons and replace lookup ops with direct
    // loads from variables.
    for (auto lookupOp : descriptorSetLayoutLookupOps) {
      replaceDescriptorSetLayoutLookupOp(lookupOp);
    }
    for (auto lookupOp : pipelineLayoutLookupOps) {
      replacePipelineLayoutLookupOp(lookupOp);
    }
    for (auto lookupOp : executableLookupOps) {
      replaceExecutableLookupOp(lookupOp);
    }
  }

private:
  IREE::Util::GlobalOp
  defineDescriptorSetLayoutOp(Location loc, ArrayAttr bindingAttrs,
                              IREE::HAL::DescriptorSetLayoutFlags flags) {
    std::pair<Attribute, IREE::HAL::DescriptorSetLayoutFlags> key = {
        bindingAttrs, flags};
    auto existingIt = descriptorSetLayoutCache_.find(key);
    if (existingIt != descriptorSetLayoutCache_.end()) {
      return existingIt->second;
    }

    auto symbolName = (StringRef("_descriptor_set_layout_") +
                       std::to_string(nextUniqueDescriptorSetLayoutId++))
                          .str();

    auto layoutType = DescriptorSetLayoutType::get(loc.getContext());
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, symbolName,
        /*isMutable=*/false, layoutType);
    globalOp.setPrivate();
    descriptorSetLayoutCache_.try_emplace(key, globalOp);

    auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
    OpBuilder blockBuilder =
        OpBuilder::atBlockEnd(initializerOp.addEntryBlock());
    // TODO(multi-device): pass in resolve info to the call and reuse.
    Value device = IREE::HAL::DeviceType::resolveAny(loc, blockBuilder);
    Value layout = blockBuilder.createOrFold<DescriptorSetLayoutCreateOp>(
        loc, layoutType, device, flags, bindingAttrs);
    globalOp.createStoreOp(loc, layout, blockBuilder);
    blockBuilder.create<IREE::Util::ReturnOp>(loc);

    return globalOp;
  }

  IREE::Util::GlobalOp
  definePipelineLayoutOp(Location loc,
                         IREE::HAL::PipelineLayoutAttr layoutAttr) {
    auto existingIt = pipelineLayoutCache_.find(layoutAttr);
    if (existingIt != pipelineLayoutCache_.end()) {
      return existingIt->second;
    }

    // First lookup (or create) all the required descriptor sets. This ensures
    // they end up in the proper initialization order.
    SmallVector<IREE::Util::GlobalOp> setLayoutGlobalOps;
    for (auto setLayoutAttr : layoutAttr.getSetLayouts()) {
      SmallVector<Attribute> bindingAttrs;
      for (auto bindingAttr : setLayoutAttr.getBindings()) {
        bindingAttrs.push_back(bindingAttr);
      }
      setLayoutGlobalOps.push_back(defineDescriptorSetLayoutOp(
          loc, ArrayAttr::get(loc.getContext(), bindingAttrs),
          setLayoutAttr.getFlags().value_or(
              IREE::HAL::DescriptorSetLayoutFlags::None)));
    }

    auto symbolName = (StringRef("_pipeline_layout_") +
                       std::to_string(nextUniquePipelineLayoutId++))
                          .str();

    auto layoutType = PipelineLayoutType::get(loc.getContext());
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, symbolName, /*isMutable=*/false, layoutType);
    globalOp.setPrivate();
    pipelineLayoutCache_.try_emplace(layoutAttr, globalOp);

    auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
    OpBuilder blockBuilder =
        OpBuilder::atBlockEnd(initializerOp.addEntryBlock());
    SmallVector<Value> setLayoutValues;
    for (auto setLayoutGlobalOp : setLayoutGlobalOps) {
      setLayoutValues.push_back(
          setLayoutGlobalOp.createLoadOp(loc, blockBuilder)
              .getLoadedGlobalValue());
    }
    // TODO(multi-device): pass in resolve info to the call and reuse.
    Value device = IREE::HAL::DeviceType::resolveAny(loc, blockBuilder);
    Value layout = blockBuilder.createOrFold<PipelineLayoutCreateOp>(
        loc, layoutType, device,
        blockBuilder.getIndexAttr(layoutAttr.getPushConstants()),
        setLayoutValues);
    globalOp.createStoreOp(loc, layout, blockBuilder);
    blockBuilder.create<IREE::Util::ReturnOp>(loc);

    return globalOp;
  }

  void defineExecutableOp(ExecutableOp executableOp) {
    auto loc = executableOp.getLoc();
    auto symbolName =
        (StringRef("_executable_") + executableOp.getSymName()).str();

    auto executableType = ExecutableType::get(executableOp.getContext());
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, symbolName, /*isMutable=*/false, executableType);
    globalOp.setPrivate();
    executableCache_.try_emplace(executableOp.getSymName(), globalOp);

    auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
    OpBuilder blockBuilder =
        OpBuilder::atBlockEnd(initializerOp.addEntryBlock());
    // TODO(multi-device): pass in resolve info to the call and reuse.
    Value device = IREE::HAL::DeviceType::resolveAny(loc, blockBuilder);

    // Create a switch statement with a case for each variant.
    // Each case should then cache only executables which contain a matching
    // ExecutableVariantOp.
    // Afterwards, canonicalization will take care of de-duping/etc.
    SmallVector<int64_t> caseIndices;
    SmallVector<IREE::HAL::ExecutableVariantOp> caseVariantOps;
    for (auto variantOp :
         executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
      caseIndices.push_back(caseIndices.size());
      caseVariantOps.push_back(variantOp);
    }

    // Select the variant index.
    Value selectedIndex = buildIfElseTree(
        loc, caseVariantOps.size(),
        [&](Location loc, size_t i, OpBuilder &builder) {
          return caseVariantOps[i].buildCondition(device, builder);
        },
        blockBuilder);

    // Allow each variant to define how it is loaded and what pipeline it has.
    auto switchOp = blockBuilder.create<scf::IndexSwitchOp>(
        loc, executableType, selectedIndex, caseIndices, caseIndices.size());
    for (auto [i, variantOp] : llvm::enumerate(caseVariantOps)) {
      auto &caseBlock = switchOp.getCaseRegions()[i].emplaceBlock();
      auto caseBuilder = OpBuilder::atBlockBegin(&caseBlock);

      // Gather each of the pipeline layouts needed for each entry point in
      // the executable.
      SmallVector<Value, 8> pipelineLayoutValues;
      for (auto exportOp : variantOp.getExportOps()) {
        auto pipelineLayoutGlobalOp =
            definePipelineLayoutOp(executableOp.getLoc(), exportOp.getLayout());
        pipelineLayoutValues.push_back(
            pipelineLayoutGlobalOp.createLoadOp(loc, caseBuilder)
                .getLoadedGlobalValue());
      }

      // Inline constant initializer from the variant.
      // We want these to all happen inside of this device switch case; they'll
      // get deduplicated/hoisted if possible in future canonicalization passes.
      SmallVector<Value> constantValues;
      for (auto blockOp :
           llvm::make_early_inc_range(variantOp.getConstantBlockOps())) {
        constantValues.append(
            inlineConstantBlockOp(blockOp, moduleBuilder, caseBuilder, device));
        blockOp.erase();
      }

      Value executable = caseBuilder.createOrFold<ExecutableCreateOp>(
          loc, executableType, device,
          SymbolRefAttr::get(executableOp.getSymNameAttr(),
                             {SymbolRefAttr::get(variantOp.getSymNameAttr())}),
          pipelineLayoutValues, constantValues);

      caseBuilder.create<scf::YieldOp>(loc, executable);
    }

    // Fallback for no available variant.
    auto &defaultBlock = switchOp.getDefaultRegion().emplaceBlock();
    auto defaultBuilder = OpBuilder::atBlockBegin(&defaultBlock);
    Value status = defaultBuilder.create<arith::ConstantIntOp>(
        loc, static_cast<int>(IREE::Util::StatusCode::Unavailable), 32);
    defaultBuilder.create<IREE::Util::StatusCheckOkOp>(
        loc, status,
        "none of the executable binaries in the module are supported by the "
        "runtime");
    auto nullValue =
        defaultBuilder.createOrFold<IREE::Util::NullOp>(loc, executableType);
    defaultBuilder.create<scf::YieldOp>(loc, nullValue);

    auto executableValue = switchOp.getResult(0);
    globalOp.createStoreOp(loc, executableValue, blockBuilder);
    blockBuilder.create<IREE::Util::ReturnOp>(loc);
  }

  // Inlines a constant block as a function in |moduleBuilder| and then inserts
  // a call to it in |callerBuilder|.
  SmallVector<Value> inlineConstantBlockOp(ExecutableConstantBlockOp blockOp,
                                           OpBuilder &moduleBuilder,
                                           OpBuilder &callerBuilder,
                                           Value device) {
    // Create the function with the region contents of the constant block.
    auto funcName = (StringRef("__constant_block_") +
                     std::to_string(nextUniqueConstantBlockId++))
                        .str();
    auto funcOp = moduleBuilder.create<IREE::Util::FuncOp>(
        blockOp.getLoc(), funcName, blockOp.getFunctionType());
    funcOp.setPrivate();
    funcOp.getRegion().takeBody(blockOp.getRegion());

    // Replace the hal.return with a func.return.
    for (auto returnOp :
         llvm::make_early_inc_range(funcOp.getOps<IREE::HAL::ReturnOp>())) {
      OpBuilder(returnOp).create<IREE::Util::ReturnOp>(returnOp.getLoc(),
                                                       returnOp.getOperands());
      returnOp.erase();
    }

    // Create the call passing in the device if needed.
    SmallVector<Value> callOperands;
    if (funcOp.getNumArguments() > 0) {
      callOperands.push_back(device);
    }
    auto callOp = callerBuilder.create<IREE::Util::CallOp>(
        blockOp.getLoc(), funcOp, callOperands);

    return llvm::map_to_vector(callOp.getResults(),
                               [](OpResult result) -> Value { return result; });
  }

  void
  replaceDescriptorSetLayoutLookupOp(DescriptorSetLayoutLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto globalOp = defineDescriptorSetLayoutOp(
        lookupOp.getLoc(), lookupOp.getBindings(), lookupOp.getFlags());
    auto loadedValue = globalOp.createLoadOp(lookupOp.getLoc(), builder)
                           .getLoadedGlobalValue();
    lookupOp.replaceAllUsesWith(loadedValue);
    lookupOp.erase();
  }

  void replacePipelineLayoutLookupOp(PipelineLayoutLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto globalOp =
        definePipelineLayoutOp(lookupOp.getLoc(), lookupOp.getLayout());
    auto loadedValue = globalOp.createLoadOp(lookupOp.getLoc(), builder)
                           .getLoadedGlobalValue();
    lookupOp.replaceAllUsesWith(loadedValue);
    lookupOp.erase();
  }

  void replaceExecutableLookupOp(ExecutableLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto executableIt = executableCache_.find(lookupOp.getExecutable());
    assert(executableIt != executableCache_.end() &&
           "executable must have been cached");
    auto globalOp = executableIt->second;
    auto loadedValue = globalOp.createLoadOp(lookupOp.getLoc(), builder)
                           .getLoadedGlobalValue();
    lookupOp.replaceAllUsesWith(loadedValue);
    lookupOp.erase();
  }

  OpBuilder moduleBuilder{static_cast<MLIRContext *>(nullptr)};
  DenseMap<std::pair<Attribute, IREE::HAL::DescriptorSetLayoutFlags>,
           IREE::Util::GlobalOp>
      descriptorSetLayoutCache_;
  DenseMap<Attribute, IREE::Util::GlobalOp> pipelineLayoutCache_;
  DenseMap<StringRef, IREE::Util::GlobalOp> executableCache_;

  int nextUniqueConstantBlockId = 0;
  int nextUniquePipelineLayoutId = 0;
  int nextUniqueDescriptorSetLayoutId = 0;
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
