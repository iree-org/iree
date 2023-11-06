// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- ExpandVectors.cpp -----------------------------------===//
// Expands vectors in matrix/vector operations (vecmat, matvec, batch_matvec)
// into matrices in order to enable tiling.
//===---------------------------------------------------------------------===//

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

struct ExpandVectors
    : public OpInterfaceRewritePattern<linalg::ContractionOpInterface> {
  using OpInterfaceRewritePattern<
      linalg::ContractionOpInterface>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::ContractionOpInterface op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
    if (!linalgOp.hasTensorSemantics()) {
      return failure();
    }

    Value lhs = linalgOp.getDpsInputs()[0];
    Value rhs = linalgOp.getDpsInputs()[1];

    Value vectorIn;
    Value matrixIn;
    if (op.isVecmat() || op.isBatchVecmat()) {
      vectorIn = lhs;
      matrixIn = rhs;
    } else if (op.isMatvec() || op.isBatchMatvec()) {
      vectorIn = rhs;
      matrixIn = lhs;
    } else {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "unsupported contraction op");
    }

    auto vectorOut = linalgOp.getDpsInits()[0];
    auto vectorInTy = dyn_cast<RankedTensorType>(vectorIn.getType());
    auto matrixInTy = dyn_cast<RankedTensorType>(matrixIn.getType());
    auto vectorOutTy = dyn_cast<RankedTensorType>(vectorOut.getType());

    if (!vectorInTy || !matrixInTy || !vectorOutTy) {
      return failure();
    }

    SmallVector<int64_t> expandedInDims, expandedOutDims;
    bool isBatchMatmul = matrixInTy.getRank() == 3;
    int64_t b = isBatchMatmul ? vectorInTy.getDimSize(0) : 1;
    int64_t m =
        (matrixIn == lhs) ? matrixInTy.getDimSize(matrixInTy.getRank() - 2) : 1;
    int64_t n =
        (matrixIn == rhs) ? matrixInTy.getDimSize(matrixInTy.getRank() - 1) : 1;
    int64_t k = vectorInTy.getDimSize(vectorInTy.getRank() - 1);
    SmallVector<ReassociationIndices> ri;
    if (op.isVecmat()) {
      // Expand (K * KxN -> N) into (1xK * KxN -> 1xN)
      expandedInDims = {1, k};
      expandedOutDims = {1, n};
      ri = {{0, 1}};
    } else if (op.isMatvec()) {
      // Expand (MxK * K -> M) into (MxK * Kx1 -> Mx1)
      expandedInDims = {k, 1};
      expandedOutDims = {m, 1};
      ri = {{0, 1}};
    } else if (op.isBatchVecmat()) {
      // Expand (BxK * BxKxN -> BxN) into (Bx1xK * BxKxN -> Bx1xN)
      expandedInDims = {b, 1, k};
      expandedOutDims = {b, 1, n};
      ri = {{0, 1}, {2}};
    } else if (op.isBatchMatvec()) {
      // Expand (BxMxK * BxK -> BxM) into (BxMxK * BxKx1 -> BxMx1)
      expandedInDims = {b, k, 1};
      expandedOutDims = {b, m, 1};
      ri = {{0}, {1, 2}};
    }

    auto newVectorInTy =
        RankedTensorType::get(expandedInDims, vectorInTy.getElementType());
    auto newVectorOutTy =
        RankedTensorType::get(expandedOutDims, vectorOutTy.getElementType());
    Location loc = linalgOp.getLoc();
    Value expandedIn;
    std::optional<CastOpInterface> castOp = getDefiningCastOp(vectorIn);
    if (castOp) {
      Value castIn = vectorIn.getDefiningOp()->getOperand(0);
      Type castSrcElemType = castOp.value()->getOperand(0).getType();
      if (auto castTensorType = dyn_cast<RankedTensorType>(castSrcElemType)) {
        castSrcElemType = castTensorType.getElementType();
      }
      auto newVectorCastInTy =
          RankedTensorType::get(expandedInDims, castSrcElemType);
      expandedIn =
          rewriter
              .create<tensor::ExpandShapeOp>(loc, newVectorCastInTy, castIn, ri)
              .getResult();
      expandedIn =
          rewriter
              .create(loc, castOp.value()->getName().getIdentifier(),
                      expandedIn, newVectorInTy, castOp.value()->getAttrs())
              ->getResult(0);
    } else {
      expandedIn =
          rewriter
              .create<tensor::ExpandShapeOp>(loc, newVectorInTy, vectorIn, ri)
              .getResult();
    }
    Value expandedOut =
        rewriter
            .create<tensor::ExpandShapeOp>(loc, newVectorOutTy, vectorOut, ri)
            .getResult();

    Value matmul;
    if (vectorIn == lhs) {
      lhs = expandedIn;
    } else {
      rhs = expandedIn;
    }
    if (isBatchMatmul) {
      matmul = rewriter
                   .create<linalg::BatchMatmulOp>(
                       loc, newVectorOutTy, ValueRange{lhs, rhs}, expandedOut)
                   .getResult(0);
    } else {
      matmul = rewriter
                   .create<linalg::MatmulOp>(loc, newVectorOutTy,
                                             ValueRange{lhs, rhs}, expandedOut)
                   .getResult(0);
    }
    Value result =
        rewriter.create<tensor::CollapseShapeOp>(loc, vectorOutTy, matmul, ri)
            .getResult();
    rewriter.replaceOp(linalgOp, result);
    return success();
  }
};

struct ExpandVectorsPass : public ExpandVectorsBase<ExpandVectorsPass> {
  void runOnOperation() override;
};

} // namespace
void ExpandVectorsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  {
    RewritePatternSet patterns(context);
    patterns.insert<ExpandVectors>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> createExpandVectorsPass() {
  return std::make_unique<ExpandVectorsPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
