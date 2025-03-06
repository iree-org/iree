// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_OPTIMIZEVECTORTRANSFERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Pattern to canonialize transpose where only one dimension is not unit
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

static void loopInvariantCodeMotion(mlir::FunctionOpInterface funcOp) {
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  funcOp.walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}

struct OptimizeVectorTransferPass final
    : impl::OptimizeVectorTransferPassBase<OptimizeVectorTransferPass> {
  using impl::OptimizeVectorTransferPassBase<
      OptimizeVectorTransferPass>::OptimizeVectorTransferPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    LDBG("before optimize vector transfer\n" << funcOp);
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
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LDBG("after dropping leading unit dims\n" << funcOp);

    if (redundantHoisting) {
      // Workaround, run loop invariant code motion before hoist redundant
      // vector transfer to workaround a bug upstream.
      loopInvariantCodeMotion(funcOp);
      linalg::hoistRedundantVectorTransfers(cast<func::FuncOp>(funcOp),
                                            /*verifyNonZeroTrip=*/true);
    }
    IRRewriter rewriter(funcOp->getContext());
    vector::transferOpflowOpt(rewriter, funcOp);

    LDBG("after folding redundant vector transfers\n" << funcOp);

    // Move bitcast inwards from loop region boundaries to increase chances to
    // cancel them.
    {
      RewritePatternSet patterns(&getContext());
      vector::populateBubbleVectorBitCastOpPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LDBG("after bubbling vector bitcasts\n" << funcOp);

    // Second stage of patterns to flatten transfer ops.
    if (flatten) {
      RewritePatternSet patterns(&getContext());
      mlir::vector::populateFlattenVectorTransferPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    LDBG("after flattening vector transfers\n" << funcOp);
    // Delete potential dead alloc and associated ops after store to load
    // forwarding.
    memref::eraseDeadAllocAndStores(rewriter, funcOp);
    LDBG("after erasing unused allocs and stores\n" << funcOp);
  }
};

} // namespace
} // namespace mlir::iree_compiler
