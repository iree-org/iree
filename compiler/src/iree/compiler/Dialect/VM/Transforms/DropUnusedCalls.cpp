// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VM {

namespace {

/// Removes vm.call ops to functions that are marked as having no side-effects
/// if the results are unused.
template <typename T>
struct EraseUnusedCallOp : public OpRewritePattern<T> {
  DenseSet<StringRef> &noSideEffectsSymbols;
  EraseUnusedCallOp(MLIRContext *context,
                    DenseSet<StringRef> &noSideEffectsSymbols,
                    PatternBenefit benefit = 1)
      : OpRewritePattern<T>(context, benefit),
        noSideEffectsSymbols(noSideEffectsSymbols) {}
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // First check if the call is unused - this ensures we only do the symbol
    // lookup if we are actually going to use it.
    for (auto result : op.getResults()) {
      if (!result.use_empty()) {
        return failure();
      }
    }

    // Check that
    bool hasNoSideEffects = noSideEffectsSymbols.contains(op.getCallee());
    if (!hasNoSideEffects) {
      // Op has side-effects (or may have them); can't remove.
      return failure();
    }

    // Erase op as it is unused.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

class DropUnusedCallsPass
    : public PassWrapper<DropUnusedCallsPass,
                         OperationPass<IREE::VM::ModuleOp>> {
public:
  StringRef getArgument() const override { return "iree-vm-drop-unused-calls"; }

  StringRef getDescription() const override {
    return "Drops vm.call ops that have no side effects and are unused.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Find all top-level symbols that have no side effects.
    DenseSet<StringRef> noSideEffectsSymbols;
    for (auto symbolOp : moduleOp.getOps<SymbolOpInterface>()) {
      if (symbolOp->getAttr("nosideeffects")) {
        noSideEffectsSymbols.insert(symbolOp.getName());
      } else if (auto importOp =
                     dyn_cast<ImportInterface>(symbolOp.getOperation())) {
        if (!importOp.hasSideEffects()) {
          noSideEffectsSymbols.insert(symbolOp.getName());
        }
      }
    }

    // Remove all unused calls.
    // Note that we want to remove entire chains of unused calls and run this
    // as a pattern application.
    RewritePatternSet patterns(&getContext());
    patterns.insert<EraseUnusedCallOp<IREE::VM::CallOp>,
                    EraseUnusedCallOp<IREE::VM::CallVariadicOp>>(
        &getContext(), noSideEffectsSymbols);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>> createDropUnusedCallsPass() {
  return std::make_unique<DropUnusedCallsPass>();
}

static PassRegistration<DropUnusedCallsPass> pass;

} // namespace mlir::iree_compiler::IREE::VM
