// this implementation is inspired by: TosaToSCF.cpp

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
  rewriter.create<mlir::tosa::YieldOp>(yield.getLoc(), yield.inputs());
  rewriter.eraseOp(yield);
}

namespace { // anonymous

class WhileOpConverter : public OpRewritePattern<mlir::TFL::WhileOp> {
public:
  using OpRewritePattern<mlir::TFL::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::TFL::WhileOp op,
                                PatternRewriter &rewriter) const final {
    auto newWhile = rewriter.create<mlir::tosa::WhileOp>(
        op.getLoc(), op.getResultTypes(), op.inputs());
    rewriter.createBlock(&newWhile.getCond());
    rewriter.createBlock(&newWhile.getBody());

    inlineWhileCase(op.cond(), newWhile.getCond(), rewriter);
    inlineWhileCase(op.body(), newWhile.getBody(), rewriter);

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
      // ConversionTarget target(getContext());
      // target.addIllegalOp<TFL::WhileOp>(); // TFL::IfOp, TFL::YieldOp,
      // target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

      MLIRContext *context = &getContext();

      auto *op = getOperation();
      patterns.add<WhileOpConverter>(context);
      if (failed(applyPartialConversion(op, target, std::move(patterns))))
        signalPassFailure();
    }
  };

} // anon namespace

std::unique_ptr<OperationPass<>> createConvertTFLConditionalsPass() {
  return std::make_unique<ConvertTFLConditionalsPass>();
}

} // namespace TFL
} // namespace iree_integrations
} // namespace mlir

