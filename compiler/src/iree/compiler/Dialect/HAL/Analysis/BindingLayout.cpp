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

namespace mlir::iree_compiler::IREE::HAL {

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

// Assumes an explicit layout as specified on an export.
static PipelineLayout
assumeExportLayout(IREE::HAL::PipelineLayoutAttr layoutAttr) {
  PipelineLayout pipelineLayout;
  pipelineLayout.pushConstantCount = layoutAttr.getPushConstants();

  auto setLayoutAttrs = layoutAttr.getSetLayouts();
  int64_t bindingCount = 0;
  for (auto setLayoutAttr : setLayoutAttrs) {
    bindingCount += setLayoutAttr.getBindings().size();
  }

  pipelineLayout.setLayouts.resize(setLayoutAttrs.size());
  pipelineLayout.resourceMap.resize(bindingCount);
  for (auto setLayoutAttr : setLayoutAttrs) {
    DescriptorSetLayout setLayout;
    setLayout.ordinal = setLayoutAttr.getOrdinal();
    setLayout.flags = setLayoutAttr.getFlags().value_or(
        IREE::HAL::DescriptorSetLayoutFlags::None);
    auto bindingAttrs = setLayoutAttr.getBindings();
    setLayout.bindings.resize(bindingAttrs.size());
    for (auto bindingAttr : bindingAttrs) {
      DescriptorSetLayoutBinding setBinding;
      setBinding.ordinal = bindingAttr.getOrdinal();
      setBinding.type = bindingAttr.getType();
      setBinding.flags =
          bindingAttr.getFlags().value_or(IREE::HAL::DescriptorFlags::None);
      setLayout.bindings[setBinding.ordinal] = setBinding;
      pipelineLayout.resourceMap.emplace_back(setLayout.ordinal,
                                              setBinding.ordinal);
    }
    pipelineLayout.setLayouts[setLayout.ordinal] = setLayout;
  }

  return pipelineLayout;
}

// Derives an pipeline layout from all of the dispatches to |exportOp|.
static PipelineLayout
deriveStreamExportLayout(IREE::Stream::ExecutableExportOp exportOp,
                         ArrayRef<IREE::Stream::CmdDispatchOp> dispatchOps) {
  if (auto layoutAttr = exportOp->getAttrOfType<IREE::HAL::PipelineLayoutAttr>(
          "hal.interface.layout")) {
    auto assumedLayout = assumeExportLayout(layoutAttr);
    LLVM_DEBUG({
      auto executableOp =
          exportOp->getParentOfType<IREE::Stream::ExecutableOp>();
      llvm::dbgs() << "assumeExportLayout(@" << executableOp.getSymName()
                   << "::@" << exportOp.getSymName() << "):\n";
      assumedLayout.print(llvm::dbgs());
    });
    return assumedLayout;
  }

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
    if (isa<IREE::Stream::BindingType>(arg)) {
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
      auto resourceAccessAttr = cast<IREE::Stream::ResourceAccessBitfieldAttr>(
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

BindingLayoutAnalysis::BindingLayoutAnalysis(Operation *rootOp,
                                             SymbolTable &symbolTable) {
  // Finds all exports and dispatches within rootOp and groups them by
  // executable export. We need to complete gathering all of the information
  // before we derive the layouts.
  auto getExportInfo = [&](Operation *exportOp) -> ExportInfo & {
    auto &exportInfo = exportInfos[exportOp];
    if (!exportInfo)
      exportInfo = std::make_unique<ExportInfo>();
    return *exportInfo;
  };
  rootOp->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<IREE::Stream::ExecutableExportOp>(
            [&](auto exportOp) { (void)getExportInfo(exportOp); })
        .Case<IREE::HAL::ExecutableExportOp>([&](auto exportOp) {
          auto &exportInfo = getExportInfo(exportOp);
          exportInfo.pipelineLayout =
              assumeExportLayout(exportOp.getLayoutAttr());
        })
        .Case<IREE::Stream::CmdDispatchOp>([&](auto dispatchOp) {
          dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
            auto exportOp =
                symbolTable.lookupNearestSymbolFrom(dispatchOp, entryPointAttr);
            auto &exportInfo = getExportInfo(exportOp);
            exportInfo.dispatchOps.push_back(dispatchOp);
          });
        })
        .Default([](auto op) {});
  });

  // Derive the layouts for each export op.
  for (auto &it : exportInfos) {
    TypeSwitch<Operation *>(it.first)
        .Case<IREE::Stream::ExecutableExportOp>([&](auto exportOp) {
          it.second->pipelineLayout =
              deriveStreamExportLayout(exportOp, it.second->dispatchOps);
        })
        .Default([&](auto op) {});
  }
}

ArrayRef<IREE::Stream::CmdDispatchOp>
BindingLayoutAnalysis::getExportDispatches(Operation *exportOp) const {
  auto it = exportInfos.find(exportOp);
  if (it == exportInfos.end())
    return {}; // not analyzed
  return it->second.get()->dispatchOps;
}

const PipelineLayout &
BindingLayoutAnalysis::getPipelineLayout(Operation *exportOp) const {
  auto it = exportInfos.find(exportOp);
  assert(it != exportInfos.end() && "unanalyzed export");
  return it->second.get()->pipelineLayout;
}

} // namespace mlir::iree_compiler::IREE::HAL
