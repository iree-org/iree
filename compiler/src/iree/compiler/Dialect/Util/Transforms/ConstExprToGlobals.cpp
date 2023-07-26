// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::Util {

namespace {
class ConstExprToGlobalsPass
    : public ConstExprToGlobalsBase<ConstExprToGlobalsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<UtilDialect>();
  }
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable moduleSymbols(moduleOp);

    // Collect all constexpr ops in topological order.
    OpBuilder moduleBuilder = OpBuilder::atBlockEnd(moduleOp.getBody());
    IRMapping globalMapping;
    moduleOp.walk<WalkOrder::PreOrder>([&](ConstExprOp constExprOp) {
      // One global per result.
      llvm::SmallVector<GlobalOp> globalOps;
      OpBuilder opBuilder(&getContext());
      opBuilder.setInsertionPoint(constExprOp);
      for (auto result : constExprOp.getResults()) {
        // Create the global.
        GlobalOp globalOp = moduleBuilder.create<GlobalOp>(
            constExprOp.getLoc(), "constexpr", false, result.getType());
        moduleSymbols.insert(globalOp);
        SymbolTable::setSymbolVisibility(globalOp,
                                         SymbolTable::Visibility::Private);
        globalOps.push_back(globalOp);

        // Redirect the original result to a GlobalLoad.
        auto load =
            opBuilder.create<GlobalLoadOp>(constExprOp.getLoc(), globalOp);
        result.replaceAllUsesWith(load);
      }

      // Since we proceeded topologically, any inputs to the constexpr (which
      // must themselves have been constexpr) will have been transformed into
      // a GlobalLoad. So, we just zip across the op/block args and clone the
      // producer into the block, replacing all of the block arguments with
      // local loads.
      Block *bodyBlock = constExprOp.getBody();
      OpBuilder bodyBuilder(&getContext());
      bodyBuilder.setInsertionPointToStart(bodyBlock);
      for (auto it :
           llvm::zip(constExprOp.getOperands(), bodyBlock->getArguments())) {
        Value input = std::get<0>(it);
        Value blockArg = std::get<1>(it);
        Operation *inputOp = input.getDefiningOp();
        assert((inputOp && llvm::isa<GlobalLoadOp>(inputOp)) &&
               "all constexpr inputs should have been replace with GlobalLoad");
        Operation *newInputOp = inputOp->clone();
        bodyBuilder.insert(newInputOp);
        bodyBuilder.setInsertionPointAfter(newInputOp);
        blockArg.replaceAllUsesWith(newInputOp->getResult(0));
      }
      bodyBlock->eraseArguments(0, bodyBlock->getNumArguments());

      // Now just move the body to the initializer and rewriter the terminator
      // to a GlobalStore for each operand.
      auto initializerOp =
          moduleBuilder.create<InitializerOp>(constExprOp.getLoc());
      initializerOp.getBody().takeBody(constExprOp.getBodyRegion());

      Operation *terminator = bodyBlock->getTerminator();
      bodyBuilder.setInsertionPoint(terminator);
      for (auto it : llvm::zip(terminator->getOperands(), globalOps)) {
        auto storeValue = std::get<0>(it);
        auto intoGlobal = std::get<1>(it);
        bodyBuilder.create<GlobalStoreOp>(intoGlobal.getLoc(), storeValue,
                                          intoGlobal);
      }
      bodyBuilder.create<InitializerReturnOp>(constExprOp.getLoc());
      terminator->erase();

      constExprOp->erase();
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConstExprToGlobalsPass() {
  return std::make_unique<ConstExprToGlobalsPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
