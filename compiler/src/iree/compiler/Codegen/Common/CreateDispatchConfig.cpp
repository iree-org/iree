// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CREATEDISPATCHCONFIGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class CreateDispatchConfigPass final
    : public impl::CreateDispatchConfigPassBase<CreateDispatchConfigPass> {
public:
  using Base::Base;
  void runOnOperation() override;
};

void CreateDispatchConfigPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModule = variantOp.getInnerModule();
  if (!innerModule) {
    return;
  }

  // Build a map from symbol name to func op for placement.
  SymbolTable symbolTable(innerModule);

  OpBuilder builder(&getContext());
  for (auto exportOp : variantOp.getExportOps()) {
    // Find the corresponding function.
    auto funcOp =
        symbolTable.lookup<FunctionOpInterface>(exportOp.getSymNameAttr());
    if (!funcOp) {
      continue;
    }

    Location loc = funcOp.getLoc();
    Block *exportBlock = exportOp.getWorkgroupCountBody();
    if (!exportBlock || exportBlock->getNumArguments() == 0) {
      // No count region — create a dispatch_config with a stub {1,1,1} body.
      builder.setInsertionPointAfter(funcOp);
      auto configOp = IREE::Codegen::DispatchConfigOp::create(builder, loc,
                                                              FlatSymbolRefAttr::get(funcOp.getNameAttr()));
      Block *block = builder.createBlock(&configOp.getBody());
      builder.setInsertionPointToStart(block);
      auto c1 = arith::ConstantIndexOp::create(builder, loc, 1);
      IREE::Codegen::YieldOp::create(builder, loc, ValueRange{c1, c1, c1});
      continue;
    }

    // Export count region block args: (!hal.device, index, index, ...).
    // Validate first arg is !hal.device.
    if (!isa<IREE::HAL::DeviceType>(exportBlock->getArgument(0).getType())) {
      exportOp.emitError("expected first count region arg to be !hal.device");
      return signalPassFailure();
    }

    // Create dispatch_config right after the function, clone the export region
    // and drop the first block argument (!hal.device).
    builder.setInsertionPointAfter(funcOp);
    auto configOp = IREE::Codegen::DispatchConfigOp::create(
        builder, loc, FlatSymbolRefAttr::get(funcOp.getNameAttr()));
    builder.cloneRegionBefore(exportOp.getWorkgroupCount(), configOp.getBody(),
                              configOp.getBody().end());
    Block *configBlock = &configOp.getBody().front();
    configBlock->eraseArgument(0);
    auto returnOp = cast<IREE::HAL::ReturnOp>(configBlock->getTerminator());
    builder.setInsertionPoint(returnOp);
    IREE::Codegen::YieldOp::create(builder, returnOp.getLoc(),
                                   returnOp.getOperands());
    returnOp.erase();
  }
}

} // namespace
} // namespace mlir::iree_compiler
