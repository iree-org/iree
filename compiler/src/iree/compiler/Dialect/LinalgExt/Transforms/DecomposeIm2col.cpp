// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEIM2COLPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

static LogicalResult decomposeIm2col(Im2colOp im2colOp, RewriterBase &rewriter,
                                     bool unroll) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(im2colOp);
  FailureOr<SmallVector<Value>> decomposedIm2col =
      im2colOp.decomposeOperation(rewriter);
  if (failed(decomposedIm2col)) {
    return failure();
  }
  rewriter.replaceOp(im2colOp, decomposedIm2col.value().front());
  if (!unroll) {
    return success();
  }

  // Unroll the loop nest created by the im2col op decomposition.
  auto outerLoop = decomposedIm2col.value().front().getDefiningOp<scf::ForOp>();
  assert(outerLoop &&
         "expected im2col op decomposition to produce scf.for loop nest.");
  SmallVector<scf::ForOp> loopNest({outerLoop});
  while (auto innerLoop =
             outerLoop.getYieldedValues()[0].getDefiningOp<scf::ForOp>()) {
    loopNest.push_back(innerLoop);
    outerLoop = innerLoop;
  }
  for (auto loop : llvm::reverse(loopNest)) {
    std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
    if (!ub.has_value() || ub.value() == 1) {
      continue;
    }
    rewriter.setInsertionPoint(loop);
    if (failed(mlir::loopUnrollByFactor(loop, ub.value()))) {
      loop.emitOpError("failed to unroll loop");
      return failure();
    }
  }
  return success();
}

namespace {
struct DecomposeIm2colPass final
    : impl::DecomposeIm2colPassBase<DecomposeIm2colPass> {
  using impl::DecomposeIm2colPassBase<
      DecomposeIm2colPass>::DecomposeIm2colPassBase;

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
  auto funcOp = getOperation();

  SmallVector<Im2colOp> candidates;
  funcOp->walk([&](Im2colOp op) { candidates.push_back(op); });
  IRRewriter rewriter(context);
  for (auto im2colOp : candidates) {
    if (failed(decomposeIm2col(im2colOp, rewriter, unroll))) {
      return signalPassFailure();
    }
  }

  RewritePatternSet patterns(context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::IREE::LinalgExt
