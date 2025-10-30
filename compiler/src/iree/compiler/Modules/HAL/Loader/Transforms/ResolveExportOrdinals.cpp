// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderDialect.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL::Loader {

#define GEN_PASS_DEF_RESOLVEEXPORTORDINALSPASS
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h.inc"

namespace {

class ResolveExportOrdinalsPass final
    : public impl::ResolveExportOrdinalsPassBase<ResolveExportOrdinalsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<IREE::HAL::Loader::HALLoaderDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      funcOp.walk([&](IREE::HAL::Loader::ExecutableExportOrdinalOp ordinalOp) {
        auto exportOp =
            symbolTable.lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
                ordinalOp, ordinalOp.getEntryPointAttr());
        OpBuilder builder(ordinalOp);
        Value value = arith::ConstantIndexOp::create(
            builder, ordinalOp.getLoc(), exportOp.getOrdinalAttr().getInt());
        ordinalOp.replaceAllUsesWith(value);
        ordinalOp.erase();
      });
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL::Loader
