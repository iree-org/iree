// Copyright 2022 Scott Reid
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TFL/PassDetail.h"
#include "iree_tf_compiler/TFL/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

// Clone block, convert yield from TFL to TOSA
static void inlineWhileCase(Region &srcRegion, Region &dstRegion,
                            PatternRewriter &rewriter) {
  rewriter.cloneRegionBefore(srcRegion, &dstRegion.back());
  rewriter.eraseBlock(&dstRegion.back());

  Block *headBlock = &dstRegion.front();

  auto yield = cast<mlir::TFL::YieldOp>(headBlock->getTerminator());  
  rewriter.setInsertionPoint(yield);
  rewriter.create<mlir::tosa::YieldOp>(yield.getLoc(), yield.operands());
  rewriter.eraseOp(yield);
}

namespace { // anonymous

class WhileOpConverter : public OpRewritePattern<mlir::TFL::WhileOp> {
public:
  using OpRewritePattern<mlir::TFL::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::TFL::WhileOp op,
                                PatternRewriter &rewriter) const final {
    auto newWhile = rewriter.create<mlir::tosa::WhileOp>(
        op.getLoc(), op.getResultTypes(), op.input());
    rewriter.createBlock(&newWhile.cond());
    rewriter.createBlock(&newWhile.body());

    inlineWhileCase(op.cond(), newWhile.cond(), rewriter);
    inlineWhileCase(op.body(), newWhile.body(), rewriter);

    rewriter.replaceOp(op, newWhile.getResults());

    return success();
  }
};

} // anonymous namespace

namespace {
struct ConvertTFLConditionalsPass 
    : public ConvertTFLConditionalsBase<ConvertTFLConditionalsPass> {
  public:
    void runOnOperation() override {
      RewritePatternSet patterns(&getContext());
      ConversionTarget target(getContext());
      target.addIllegalOp<mlir::TFL::WhileOp>(); // mlir::TFL::IfOp, mlir::TFL::YieldOp,
      target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

      MLIRContext *context = &getContext();

      auto op = getOperation();
      patterns.add<WhileOpConverter>(context);
      if (failed(applyPartialConversion(op, target, std::move(patterns))))
        signalPassFailure();
    }
  };

} // anon namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertTFLConditionalsPass() {
  return std::make_unique<ConvertTFLConditionalsPass>();
}

} // namespace TFL
} // namespace iree_integrations
} // namespace mlir

