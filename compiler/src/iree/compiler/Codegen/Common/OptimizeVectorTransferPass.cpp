// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-codegen-optimize-vector-transfer"

namespace mlir {
namespace iree_compiler {
namespace {

static void debugPrint(func::FuncOp funcOp, const char *message) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << message << " ---//\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

// Pattern to canonialize tranpose where only one dimension is not unit
// dimension. In this case the transpose is a no-op and should be simplified
// before getting to the conversion to llvm/spirv.
// TODO(thomasraoux): This should be moved in
// `populateCastAwayVectorLeadingOneDimPatterns` but might need more discussion
// on the semantic of transpose in this case.
class TransposeUnitDimToShapeCast
    : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    unsigned numNonUnitSrcDim =
        llvm::count_if(op.getSourceVectorType().getShape(),
                       [](int64_t dim) { return dim != 1; });
    if (numNonUnitSrcDim > 1)
      return failure();
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        op, op.getResultVectorType(), op.getVector());
    return success();
  }
};

static void loopInvariantCodeMotion(func::FuncOp funcOp) {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  funcOp.walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}

struct OptimizeVectorTransferPass
    : public OptimizeVectorTransferBase<OptimizeVectorTransferPass> {
  OptimizeVectorTransferPass(bool flatten, bool dropUnitDims)
      : flatten(flatten), dropUnitDims(dropUnitDims) {}
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    debugPrint(funcOp, "before optimize vector transfer");
    // Generate vector.shape_cast for dropping leading one dimensions in vector
    // ops. This increases the chance that we can forward more transfer writes
    // to transfer reads.
    {
      RewritePatternSet patterns(&getContext());
      mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, &getContext());
      patterns.add<TransposeUnitDimToShapeCast>(&getContext());
      mlir::vector::
          populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
              patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after dropping leading unit dims");

    // Workaround, run loop invariant code motion before hoist redudant vector
    // transfer to workaround a bug upstream.
    // TODO(thomasraoux): Remove it once the fix is merged.
    loopInvariantCodeMotion(funcOp);
    linalg::hoistRedundantVectorTransfers(funcOp);
    IRRewriter rewriter(funcOp->getContext());
    vector::transferOpflowOpt(rewriter, funcOp);

    debugPrint(funcOp, "after folding redundant vector transfers");

    // Move bitcast inwards from loop region boundaries to increase chances to
    // cancel them.
    {
      RewritePatternSet patterns(&getContext());
      vector::populateBubbleVectorBitCastOpPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after bubbling vector bitcasts");

    // TODO(#14191): SPIR-V can't handle the vector.shape_cast created for
    // dropping unit dims so this option is disabled in SPIR-V pipeline.
    // This option should go away after all backend issues have been resolved.
    if (dropUnitDims) {
      RewritePatternSet patterns(&getContext());
      mlir::vector::populateVectorTransferDropUnitDimsPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }

      debugPrint(funcOp, "after dropping vector transfer unit dims");
    }

    // Second stage of patterns to flatten transfer ops.
    if (flatten) {
      RewritePatternSet patterns(&getContext());
      mlir::vector::populateFlattenVectorTransferPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    debugPrint(funcOp, "after flattening vector transfers");
    // Delete potential dead alloc and associated ops after store to load
    // forwarding.
    memref::eraseDeadAllocAndStores(rewriter, funcOp);
    debugPrint(funcOp, "after erasing unused allocs and stores");
  }

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    // `flatten` may have been set to `true` in the constructor already.
    // The |= is so we preserve that rather than overwrite it with the default
    // value `false` of `optionFlatten`.
    flatten |= optionFlatten;
    return success();
  }

private:
  bool flatten;
  bool dropUnitDims;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createOptimizeVectorTransferPass(bool flatten, bool dropUnitDims) {
  return std::make_unique<OptimizeVectorTransferPass>(flatten, dropUnitDims);
}

} // namespace iree_compiler
} // namespace mlir
