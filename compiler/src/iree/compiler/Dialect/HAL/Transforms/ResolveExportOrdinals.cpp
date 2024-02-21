// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_RESOLVEEXPORTORDINALSPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-resolve-export-ordinals
//===----------------------------------------------------------------------===//

struct ResolveExportOrdinalsPass
    : public IREE::HAL::impl::ResolveExportOrdinalsPassBase<
          ResolveExportOrdinalsPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      funcOp.walk([&](IREE::HAL::ExecutableExportOrdinalOp ordinalOp) {
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

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
