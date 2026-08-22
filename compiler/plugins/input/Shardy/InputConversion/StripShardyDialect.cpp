// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Shardy/InputConversion/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::iree_compiler::shardy {

namespace {

// Pattern to strip sdy.sharding attribute from operations
struct StripShardyAttributesPattern : public RewritePattern {
  StripShardyAttributesPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    bool modified = false;
    // Remove sdy.sharding attributes
    if (op->hasAttr("sdy.sharding")) {
      op->removeAttr("sdy.sharding");
      modified = true;
    }
    return modified ? success() : failure();
  }
};

struct StripShardyDialectPass
    : public PassWrapper<StripShardyDialectPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripShardyDialectPass)

  StringRef getArgument() const override { return "iree-shardy-strip-dialect"; }
  StringRef getDescription() const override {
    return "Strip Shardy dialect ops and attributes for single-device "
           "execution";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<StripShardyAttributesPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }

    // Collect sdy ops to remove (can't erase during walk - iterator invalidation)
    SmallVector<Operation *> sdyOpsToErase;
    module.walk([&](Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "sdy") {
        if (op->getNumResults() == 1 && op->getNumOperands() >= 1) {
          sdyOpsToErase.push_back(op);
        } else if (op->getNumResults() > 0 || op->getNumOperands() > 0) {
          // Warn about unexpected patterns we can't handle
          op->emitWarning() << "Unexpected Shardy op pattern (results="
                            << op->getNumResults()
                            << ", operands=" << op->getNumOperands()
                            << ") - may not be fully stripped";
        }
      }
    });

    // Erase in reverse order to handle nested ops correctly
    for (Operation *op : llvm::reverse(sdyOpsToErase)) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createStripShardyDialectPass() {
  return std::make_unique<StripShardyDialectPass>();
}

} // namespace mlir::iree_compiler::shardy
