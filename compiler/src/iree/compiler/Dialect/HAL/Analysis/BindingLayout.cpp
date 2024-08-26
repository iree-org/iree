// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/BindingLayout.h"

#include "iree/compiler/Dialect/HAL/Analysis/Captures.h"
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
      setBinding.flags = bindingAttr.getFlags();
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
  struct DescriptorInfo {
    DescriptorFlags flags = DescriptorFlags::None;
  };
  SmallVector<DescriptorInfo> descriptorInfos(bindingCount);
  for (auto dispatchOp : dispatchOps) {
    // If any dispatch is performed within a reusable (non-one-shot) execution
    // region we may opt in to indirect references. For those only executed once
    // (though maybe from multiple dispatch sites) we try to bias towards direct
    // references to avoid additional overheads.
    auto parentOp = dispatchOp->getParentOfType<IREE::Stream::CmdExecuteOp>();
    bool isRegionExecutedOnce = parentOp ? parentOp.getOnce() : false;

    auto resourceAccessesAttrs = dispatchOp.getResourceAccesses().getValue();
    for (unsigned i = 0; i < bindingCount; ++i) {
      auto &descriptorInfo = descriptorInfos[i];

      // Opt into indirect descriptors when dynamic values are used from
      // execution regions that may be executed more than once.
      if (!isRegionExecutedOnce) {
        Value resource = dispatchOp.getResources()[i];
        if (auto blockArg = dyn_cast<BlockArgument>(resource)) {
          if (blockArg.getOwner()->getParentOp() == parentOp) {
            resource = parentOp.getResourceOperands()[blockArg.getArgNumber()];
          }
        }
        switch (categorizeValue(resource)) {
        default:
        case ValueOrigin::Unknown:
        case ValueOrigin::MutableGlobal:
          descriptorInfo.flags =
              descriptorInfo.flags | IREE::HAL::DescriptorFlags::Indirect;
          break;
        case ValueOrigin::LocalConstant:
        case ValueOrigin::ImmutableGlobal:
          break;
        }
      }

      // Set binding flags based on the OR of all dispatch site access.
      auto resourceAccess = static_cast<IREE::Stream::ResourceAccessBitfield>(
          cast<IREE::Stream::ResourceAccessBitfieldAttr>(
              resourceAccessesAttrs[i])
              .getInt());
      if (!bitEnumContainsAll(resourceAccess,
                              IREE::Stream::ResourceAccessBitfield::Write)) {
        // Read-only.
        descriptorInfo.flags =
            descriptorInfo.flags | IREE::HAL::DescriptorFlags::ReadOnly;
      }
    }
  }

  PipelineLayout pipelineLayout;
  pipelineLayout.pushConstantCount = operandCount;
  pipelineLayout.resourceMap.resize(bindingCount);

  // TODO(#18154): simplify binding setup.
  DescriptorSetLayout setLayout;
  setLayout.ordinal = 0;
  setLayout.flags = IREE::HAL::DescriptorSetLayoutFlags::None;
  setLayout.bindings.reserve(bindingCount);
  for (unsigned i = 0; i < bindingCount; ++i) {
    const auto &descriptorInfo = descriptorInfos[i];
    if (allEnumBitsSet(descriptorInfo.flags,
                       IREE::HAL::DescriptorFlags::Indirect)) {
      setLayout.flags =
          setLayout.flags | IREE::HAL::DescriptorSetLayoutFlags::Indirect;
    }
    DescriptorSetLayoutBinding setBinding;
    setBinding.ordinal = setLayout.bindings.size();
    setBinding.type = IREE::HAL::DescriptorType::StorageBuffer;
    setBinding.flags = descriptorInfo.flags;
    setLayout.bindings.push_back(setBinding);
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

bool BindingLayoutAnalysis::hasDispatches() const {
  for (auto &it : exportInfos) {
    if (!it.second->dispatchOps.empty()) {
      return true; // found at least one dispatch
    }
  }
  return false;
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
