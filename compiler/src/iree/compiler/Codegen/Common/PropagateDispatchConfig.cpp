// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "iree-codegen-propagate-dispatch-config"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PROPAGATEDISPATCHCONFIGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class PropagateDispatchConfigPass final
    : public impl::PropagateDispatchConfigPassBase<
          PropagateDispatchConfigPass> {
public:
  using Base::Base;
  void runOnOperation() override;
};

void PropagateDispatchConfigPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModule = variantOp.getInnerModule();
  if (!innerModule) {
    return;
  }

  // Collect all dispatch_config ops.
  SmallVector<IREE::Codegen::DispatchConfigOp> configOps;
  innerModule->walk(
      [&](IREE::Codegen::DispatchConfigOp op) { configOps.push_back(op); });
  if (configOps.empty()) {
    return;
  }

  // Build a map from export name to export op.
  DenseMap<StringRef, IREE::HAL::ExecutableExportOp> exportMap;
  for (auto exportOp : variantOp.getExportOps()) {
    exportMap[exportOp.getSymName()] = exportOp;
  }

  for (IREE::Codegen::DispatchConfigOp configOp : configOps) {
    StringRef funcRef = configOp.getFunctionRef();

    if (!exportMap.contains(funcRef)) {
      // No export for this function (e.g. a helper that is not an entry
      // point).  Erase the dispatch_config and move on.
      configOp.erase();
      continue;
    }
    IREE::HAL::ExecutableExportOp exportOp = exportMap[funcRef];

    // Replace the export count region body with the dispatch_config body.
    // Map dispatch_config block args to export block args (offset by 1
    // for !hal.device at position 0).
    Region &countRegion = exportOp.getWorkgroupCount();
    OpBuilder builder(&getContext());

    if (!countRegion.empty()) {
      Block &configBlock = configOp.getBody().front();
      Block *exportBlock = exportOp.getWorkgroupCountBody();
      unsigned configArity = configBlock.getNumArguments();
      unsigned exportArity = exportBlock->getNumArguments();
      // Export count region has !hal.device as block arg 0, then workloads.
      if (configArity + 1 > exportArity) {
        configOp.emitError("workload arity mismatch: dispatch_config has ")
            << configArity << " args but export count region has "
            << exportArity << " (expected >= " << configArity + 1
            << " = config args + !hal.device)";
        return signalPassFailure();
      }
      exportBlock->clear();
      IRMapping mapping;
      for (unsigned i = 0; i < configArity; ++i) {
        mapping.map(configBlock.getArgument(i),
                    exportBlock->getArgument(i + 1));
      }
      builder.setInsertionPointToEnd(exportBlock);
      for (Operation &op : configBlock.without_terminator()) {
        builder.clone(op, mapping);
      }
      // Replace iree_codegen.yield with hal.return.
      auto yieldOp = cast<IREE::Codegen::YieldOp>(configBlock.getTerminator());
      auto returnValues =
          llvm::map_to_vector(yieldOp.getOperands(), [&](Value v) {
            return mapping.lookupOrDefault(v);
          });
      IREE::HAL::ReturnOp::create(builder, yieldOp.getLoc(), returnValues);
    }

    // Set workgroup_size and subgroup_size on the export.
    auto wgSize = configOp.getWorkgroupSize();
    if (!wgSize) {
      configOp.emitError("missing workgroup_size attribute");
      return signalPassFailure();
    }
    SmallVector<int64_t, 3> wgSizePadded(wgSize->begin(), wgSize->end());
    while (wgSizePadded.size() < 3) {
      wgSizePadded.push_back(1);
    }
    exportOp.setWorkgroupSizeAttr(builder.getIndexArrayAttr(wgSizePadded));
    if (auto subgroupSize = configOp.getSubgroupSize()) {
      exportOp.setSubgroupSizeAttr(builder.getIndexAttr(subgroupSize.value()));
    }

    configOp.erase();
  }
}

} // namespace
} // namespace mlir::iree_compiler
