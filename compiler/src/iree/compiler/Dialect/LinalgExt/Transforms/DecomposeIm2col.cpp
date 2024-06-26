// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {
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
    return success();
  }
};

} // namespace

namespace {
struct DecomposeIm2colPass : public DecomposeIm2colBase<DecomposeIm2colPass> {
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

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDecomposeIm2colPass() {
  return std::make_unique<DecomposeIm2colPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
