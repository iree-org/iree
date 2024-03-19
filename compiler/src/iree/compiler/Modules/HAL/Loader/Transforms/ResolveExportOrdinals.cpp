// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderDialect.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/PassDetail.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL::Loader {

class ResolveExportOrdinalsPass
    : public ResolveExportOrdinalsBase<ResolveExportOrdinalsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<IREE::HAL::Loader::HALLoaderDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      funcOp.walk([&](IREE::HAL::Loader::ExecutableExportOrdinalOp ordinalOp) {
        auto exportOp =
            symbolTable.lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
                ordinalOp, ordinalOp.getEntryPointAttr());
        Value value = OpBuilder(ordinalOp).create<arith::ConstantIndexOp>(
            ordinalOp.getLoc(), exportOp.getOrdinalAttr().getInt());
        ordinalOp.replaceAllUsesWith(value);
        ordinalOp.erase();
      });
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createResolveExportOrdinalsPass() {
  return std::make_unique<ResolveExportOrdinalsPass>();
}

static PassRegistration<ResolveExportOrdinalsPass> pass;

} // namespace mlir::iree_compiler::IREE::HAL::Loader
