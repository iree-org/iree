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
    if (!linalgOp || !linalgOp.hasTensorSemantics())
      return failure();

    if (op.isRowMajorMatmul() || op.isColumnMajorMatmul() ||
        op.isRowMajorBatchMatmul()) {
      return rewriter.notifyMatchFailure(linalgOp, "op is already a matmul");
    }
    Value lhs = linalgOp.getDpsInputs()[0];
    Value rhs = linalgOp.getDpsInputs()[1];

    Value vectorIn;
    if (op.isVecmat()) {
      vectorIn = lhs;
    } else if (op.isMatvec() || op.isBatchMatvec()) {
      vectorIn = rhs;
    } else {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "unsupported contraction op");
    }

    auto vectorOut = linalgOp.getDpsInits()[0];
    auto vectorInTy = llvm::dyn_cast<RankedTensorType>(vectorIn.getType());
    auto vectorOutTy = llvm::dyn_cast<RankedTensorType>(vectorOut.getType());

    if (!vectorInTy || !vectorOutTy) {
      return failure();
    }

    Type inEType = vectorInTy.getElementType();
    Type outEType = vectorOutTy.getElementType();

    SmallVector<int64_t> expandedInDims, expandedOutDims;
    bool isBatched = false;
    SmallVector<ReassociationIndices> ri = {{0, 1}};

    // Expand (N * NxM = M) into (1xN * NxM = 1xM)
    if (op.isVecmat()) {
      expandedInDims = {1, vectorInTy.getDimSize(0)};
      expandedOutDims = {1, vectorOutTy.getDimSize(0)};
      // Expand (NxM * M = N) into (NxM * Mx1 = Mx1)
    } else if (op.isMatvec()) {
      expandedInDims = {vectorInTy.getDimSize(0), 1};
      expandedOutDims = {vectorOutTy.getDimSize(0), 1};
      // Expand (BxNxM * BxM = BxN) into (BxNxM * BxMx1 = BxMx1)
    } else {
      expandedInDims = {vectorInTy.getDimSize(0), vectorInTy.getDimSize(1), 1};
      expandedOutDims = {vectorOutTy.getDimSize(0), vectorOutTy.getDimSize(1),
                         1};
      ri = {{0}, {1, 2}};
      isBatched = true;
    }

    auto newVectorInTy = RankedTensorType::get(expandedInDims, inEType);
    auto newVectorOutTy = RankedTensorType::get(expandedOutDims, outEType);
    Location loc = linalgOp.getLoc();
    Value expandedIn =
        rewriter.create<tensor::ExpandShapeOp>(loc, newVectorInTy, vectorIn, ri)
            .getResult();
    Value expandedOut =
        rewriter
            .create<tensor::ExpandShapeOp>(loc, newVectorOutTy, vectorOut, ri)
            .getResult();

    Value matmul;
    if (op.isVecmat()) {
      lhs = expandedIn;
    } else {
      rhs = expandedIn;
    }
    if (isBatched) {
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
