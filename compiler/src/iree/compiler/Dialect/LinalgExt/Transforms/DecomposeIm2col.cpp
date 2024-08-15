// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEIM2COLPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

/// Pattern to decompose the tiled im2col op.
struct DecomposeIm2col : public OpRewritePattern<Im2colOp> {
  using OpRewritePattern<Im2colOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Im2colOp im2colOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Value>> decomposedIm2col =
        im2colOp.decomposeOperation(rewriter);
    if (failed(decomposedIm2col)) {
      return failure();
    }
    rewriter.replaceOp(im2colOp, decomposedIm2col.value().front());

    // Unroll the loop nest created by the im2col op decomposition.
    auto outerLoop =
        decomposedIm2col.value().front().getDefiningOp<scf::ForOp>();
    SmallVector<scf::ForOp> loopNest({outerLoop});
    while (auto innerLoop =
               outerLoop.getYieldedValues()[0].getDefiningOp<scf::ForOp>()) {
      loopNest.push_back(innerLoop);
      outerLoop = innerLoop;
    }
    for (auto loop : llvm::reverse(loopNest)) {
      IntegerAttr ub;
      if (!matchPattern(loop.getUpperBound(), m_Constant(&ub))) {
        loop.emitOpError("upper bound should be a constant");
        return failure();
      }
      if (ub.getInt() == 1) {
        continue;
      }
      if (failed(mlir::loopUnrollByFactor(loop, ub.getInt()))) {
        loop.emitOpError("failed unrolling by factor 1");
        return failure();
      }
    }
    return success();
  }
};

} // namespace

namespace {
struct DecomposeIm2colPass final
    : impl::DecomposeIm2colPassBase<DecomposeIm2colPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void DecomposeIm2colPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<DecomposeIm2col>(context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::IREE::LinalgExt
