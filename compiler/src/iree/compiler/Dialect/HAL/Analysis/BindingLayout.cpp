// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/BindingLayout.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-hal-binding-layout-analysis"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

void PipelineLayout::print(llvm::raw_ostream &os) const {
  os << "PipelineLayout:\n";
  os << "  push constants: " << pushConstantCount << "\n";
  os << "  sets:\n";
  for (auto &setLayout : setLayouts) {
    os << "    set[" << setLayout.ordinal
       << "]: " << stringifyDescriptorSetLayoutFlags(setLayout.flags) << "\n";
    for (auto &binding : setLayout.bindings) {
      os << "      binding[" << binding.ordinal
         << "]: " << stringifyDescriptorType(binding.type) << "\n";
    }
  }
  os << "  resource map:\n";
  for (auto setBinding : llvm::enumerate(resourceMap)) {
    os << "    resource[" << setBinding.index() << "]: set "
       << setBinding.value().first << " binding " << setBinding.value().second
       << "\n";
  }
}

// Finds all dispatches within |rootOp| and groups them by executable export.
static BindingLayoutAnalysis::ExportDispatchMap findAllDispatchSites(
    Operation *rootOp) {
  SymbolTable symbolTable(rootOp);
  BindingLayoutAnalysis::ExportDispatchMap dispatchMap;
  rootOp->walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
      auto exportOp =
          symbolTable.lookupNearestSymbolFrom(dispatchOp, entryPointAttr);
      dispatchMap[exportOp].push_back(dispatchOp);
    });
  });
  return dispatchMap;
}

// Derives an pipeline layout from all of the dispatches to |exportOp|.
static PipelineLayout deriveExportLayout(
    IREE::Stream::ExecutableExportOp exportOp,
    SmallVector<IREE::Stream::CmdDispatchOp> &dispatchOps) {
  auto funcOp = exportOp.lookupFunctionRef();
  assert(funcOp && "export target not found");

  // TODO(#3502): a real derivation based on dispatch sites.
  // We want to get all slowly changing bindings earlier in the sets with the
  // goal of having set 0 be nearly command buffer static so that we only do one
  // binding per command buffer. We may still have some extra sets for one-off
  // bindings (external resources) or even repeat bindings for ones that are
  // both static (based at the same offset across all dispatches) and dynamic
  // (dynamic ssa value offsets) to prevent us from needing to update the offset
  // back and forth.
  //
  // Though we are looking at dispatch sites here we can use the context around
  // them to rank things: for example, looking at the use count of a resource
  // would tell us how often that resource is used within the same command
  // buffer and thus how many duplicate binding updates we could avoid if we
  // made it fixed.
  //
  // Today for implementation expediency we just splat things in order. It's
  // kind of rubbish.
  unsigned operandCount = 0;
  unsigned bindingCount = 0;
  for (auto arg : funcOp.getArgumentTypes()) {
    if (llvm::isa<IREE::Stream::BindingType>(arg)) {
      ++bindingCount;
    } else {
      ++operandCount;
    }
  }

  // Check the usage of each binding at each dispatch site.
  SmallVector<DescriptorFlags> bindingFlags(bindingCount);
  for (auto dispatchOp : dispatchOps) {
    auto resourceAccessesAttrs = dispatchOp.getResourceAccesses().getValue();
    for (unsigned i = 0; i < bindingCount; ++i) {
      auto resourceAccessAttr =
          llvm::cast<IREE::Stream::ResourceAccessBitfieldAttr>(
              resourceAccessesAttrs[i]);
      auto resourceAccess = static_cast<IREE::Stream::ResourceAccessBitfield>(
          resourceAccessAttr.getInt());
      if (!bitEnumContainsAll(resourceAccess,
                              IREE::Stream::ResourceAccessBitfield::Write)) {
        // Read-only.
        bindingFlags[i] =
            bindingFlags[i] | IREE::HAL::DescriptorFlags::ReadOnly;
      }
    }
  }

  PipelineLayout pipelineLayout;
  pipelineLayout.pushConstantCount = operandCount;
  pipelineLayout.resourceMap.resize(bindingCount);

  // Only one set today - this creates a lot of pushes that we can't elide later
  // on once interfaces are materialized.
  DescriptorSetLayout setLayout;
  setLayout.ordinal = 0;
  setLayout.flags = IREE::HAL::DescriptorSetLayoutFlags::None;
  setLayout.bindings.resize(bindingCount);
  for (unsigned i = 0; i < bindingCount; ++i) {
    DescriptorSetLayoutBinding setBinding;
    setBinding.ordinal = i;
    setBinding.type = IREE::HAL::DescriptorType::StorageBuffer;
    setBinding.flags = bindingFlags[i];
    setLayout.bindings[i] = setBinding;
    pipelineLayout.resourceMap[i] =
        std::make_pair(setLayout.ordinal, setBinding.ordinal);
  }
  pipelineLayout.setLayouts.push_back(setLayout);

  LLVM_DEBUG({
    auto executableOp = exportOp->getParentOfType<IREE::Stream::ExecutableOp>();
    llvm::dbgs() << "deriveExportLayout(@" << executableOp.getSymName() << "::@"
                 << exportOp.getSymName() << "):\n";
    pipelineLayout.print(llvm::dbgs());
  });

  return pipelineLayout;
}

static BindingLayoutAnalysis::ExportLayoutMap deriveExportLayouts(
    Operation *rootOp, BindingLayoutAnalysis::ExportDispatchMap dispatchMap) {
  BindingLayoutAnalysis::ExportLayoutMap layoutMap;
  rootOp->walk([&](IREE::Stream::ExecutableExportOp exportOp) {
    auto &dispatchOps = dispatchMap[exportOp];
    layoutMap[exportOp] = deriveExportLayout(exportOp, dispatchOps);
  });
  return layoutMap;
}

BindingLayoutAnalysis::BindingLayoutAnalysis(Operation *rootOp) {
  exportDispatches = findAllDispatchSites(rootOp);
  exportLayouts = deriveExportLayouts(rootOp, exportDispatches);
}

SmallVector<IREE::Stream::CmdDispatchOp>
BindingLayoutAnalysis::getExportDispatches(
    IREE::Stream::ExecutableExportOp exportOp) const {
  auto it = exportDispatches.find(exportOp);
  if (it == exportDispatches.end()) return {};  // no dispatches
  return it->second;
}

const PipelineLayout &BindingLayoutAnalysis::getPipelineLayout(
    IREE::Stream::ExecutableExportOp exportOp) const {
  auto it = exportLayouts.find(exportOp);
  assert(it != exportLayouts.end() && "unanalyzed export");
  return it->second;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
