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

void ExecutableLayout::print(llvm::raw_ostream &os) const {
  os << "ExecutableLayout:\n";
  os << "  push constants: " << pushConstantCount << "\n";
  os << "  sets:\n";
  for (auto &setLayout : setLayouts) {
    os << "    set[" << setLayout.ordinal
       << "]: " << stringifyDescriptorSetLayoutUsageType(setLayout.usage)
       << "\n";
    for (auto &binding : setLayout.bindings) {
      os << "      binding[" << binding.ordinal
         << "]: " << stringifyDescriptorType(binding.type) << " / "
         << stringifyMemoryAccessBitfield(binding.access) << "\n";
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
    auto exportOp = symbolTable.lookupNearestSymbolFrom(
        dispatchOp, dispatchOp.entry_pointAttr());
    dispatchMap[exportOp].push_back(dispatchOp);
  });
  return dispatchMap;
}

// Derives an executable layout from all of the dispatches to |exportOp|.
static ExecutableLayout deriveExportLayout(
    IREE::Stream::ExecutableExportOp exportOp,
    SmallVector<IREE::Stream::CmdDispatchOp> &dispatchOps) {
  auto executableOp = exportOp->getParentOfType<IREE::Stream::ExecutableOp>();
  assert(executableOp && "unnested export");
  auto funcOp = executableOp.getInnerModule().lookupSymbol<mlir::FuncOp>(
      exportOp.function_ref());
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
    if (arg.isa<IREE::Stream::BindingType>()) {
      ++bindingCount;
    } else {
      ++operandCount;
    }
  }

  // In lieu of actual analysis we just check per dispatch-site which bindings
  // need to be dynamic vs those that are at a constant uniform offset.
  // The number of dynamic storage buffer bindings available is limited:
  // https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxDescriptorSetStorageBuffersDynamic&platform=all
  // Earlier optimizations in the stream dialect try to rebase bindings to 0 to
  // make this possible.
  // NOTE: we could check the actual constant values being uniform within a
  // single command buffer (as that's all that really matters) but this is all
  // just a temporary hack so ¯\_(ツ)_/¯.
  llvm::BitVector staticBindings(bindingCount, /*t=*/true);
  for (auto dispatchOp : dispatchOps) {
    auto resourceOffsets = dispatchOp.resource_offsets();
    for (unsigned i = 0; i < bindingCount; ++i) {
      if (!matchPattern(resourceOffsets[i], m_Zero())) staticBindings.reset(i);
    }
  }

  // Compute the access requirements of the binding based on any dispatch (as
  // they should all have the same requirements). If we have no dispatches then
  // we don't need access. Note that as we start to reuse bindings across
  // multiple executables we may want to widen the access.
  //
  // TODO(benvanik): set MayAlias if multiple bindings share the same resource
  // and may have overlapping ranges.
  SmallVector<IREE::HAL::MemoryAccessBitfield> bindingAccess(
      bindingCount, IREE::HAL::MemoryAccessBitfield::None);
  if (!dispatchOps.empty()) {
    auto anyDispatchOp = dispatchOps.front();
    for (unsigned i = 0; i < bindingCount; ++i) {
      auto resourceAccess =
          anyDispatchOp.resource_accesses()[i]
              .cast<IREE::Stream::ResourceAccessBitfieldAttr>()
              .getValue();
      // MLIR's generated bitfield enums could use some ergonomic improvements.
      auto memoryAccess = IREE::HAL::MemoryAccessBitfield::None;
      if (bitEnumContains(resourceAccess,
                          IREE::Stream::ResourceAccessBitfield::Read)) {
        memoryAccess = memoryAccess | IREE::HAL::MemoryAccessBitfield::Read;
      }
      if (bitEnumContains(resourceAccess,
                          IREE::Stream::ResourceAccessBitfield::Write)) {
        memoryAccess = memoryAccess | IREE::HAL::MemoryAccessBitfield::Write;
      }
      bindingAccess[i] = bindingAccess[i] | memoryAccess;
    }
  }

  ExecutableLayout executableLayout;
  executableLayout.pushConstantCount = operandCount;
  executableLayout.resourceMap.resize(bindingCount);

  // Only one set today - this creates a lot of pushes that we can't elide later
  // on once interfaces are materialized.
  DescriptorSetLayout setLayout;
  setLayout.ordinal = 0;
  setLayout.usage = IREE::HAL::DescriptorSetLayoutUsageType::PushOnly;
  setLayout.bindings.resize(bindingCount);
  for (unsigned i = 0; i < bindingCount; ++i) {
    DescriptorSetLayoutBinding setBinding;
    setBinding.ordinal = i;
    setBinding.type = staticBindings.test(i)
                          ? IREE::HAL::DescriptorType::StorageBuffer
                          : IREE::HAL::DescriptorType::StorageBufferDynamic;
    setBinding.access = bindingAccess[i];
    setLayout.bindings[i] = setBinding;
    executableLayout.resourceMap[i] =
        std::make_pair(setLayout.ordinal, setBinding.ordinal);
  }
  executableLayout.setLayouts.push_back(setLayout);

  LLVM_DEBUG({
    llvm::dbgs() << "deriveExportLayout(@" << executableOp.sym_name() << "::@"
                 << exportOp.sym_name() << "):\n";
    executableLayout.print(llvm::dbgs());
  });

  return executableLayout;
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

const SmallVector<IREE::Stream::CmdDispatchOp>
    &BindingLayoutAnalysis::getExportDispatches(
        IREE::Stream::ExecutableExportOp exportOp) const {
  auto it = exportDispatches.find(exportOp);
  assert(it != exportDispatches.end() && "unanalyzed export");
  return it->second;
}

const ExecutableLayout &BindingLayoutAnalysis::getExecutableLayout(
    IREE::Stream::ExecutableExportOp exportOp) const {
  auto it = exportLayouts.find(exportOp);
  assert(it != exportLayouts.end() && "unanalyzed export");
  return it->second;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
