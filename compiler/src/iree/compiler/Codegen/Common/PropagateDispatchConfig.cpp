// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/DebugLog.h"

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

  // Collect all dispatch_config ops. These are direct children of the module
  // (like func.func), so no walk needed.
  SmallVector<IREE::Codegen::DispatchConfigOp> configOps =
      llvm::to_vector(innerModule.getOps<IREE::Codegen::DispatchConfigOp>());
  if (configOps.empty()) {
    return;
  }

  SymbolTable symbolTable(variantOp);
  for (IREE::Codegen::DispatchConfigOp configOp : configOps) {
    StringRef funcRef = configOp.getFunctionRef();
    auto exportOp = symbolTable.lookup<IREE::HAL::ExecutableExportOp>(funcRef);
    if (!exportOp) {
      // No export for this function, so erase the dispatch_config and move on.
      configOp.erase();
      continue;
    }

    // Move the dispatch_config region body into the export count region,
    // replacing iree_codegen.yield with hal.return.
    Region &countRegion = exportOp.getWorkgroupCount();
    OpBuilder builder(&getContext());

    if (!countRegion.empty()) {
      Block &configBlock = configOp.getBody().front();
      Block *exportBlock = exportOp.getWorkgroupCountBody();
      TypeRange configArgTypes = configBlock.getArgumentTypes();
      TypeRange exportArgTypes = exportBlock->getArgumentTypes();
      if (configArgTypes != exportArgTypes) {
        configOp.emitError("block argument mismatch: dispatch_config has (")
            << configArgTypes << ") but export count region has ("
            << exportArgTypes << ")";
        return signalPassFailure();
      }
      countRegion.takeBody(configOp.getBody());
      // Replace iree_codegen.yield with hal.return.
      Block &block = countRegion.front();
      auto yieldOp = cast<IREE::Codegen::YieldOp>(block.getTerminator());
      builder.setInsertionPoint(yieldOp);
      IREE::HAL::ReturnOp::create(builder, yieldOp.getLoc(),
                                  yieldOp.getOperands());
      yieldOp.erase();
    }

    // Set workgroup_size and subgroup_size on the export.
    std::optional<ArrayRef<int64_t>> wgSize = configOp.getWorkgroupSize();
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
