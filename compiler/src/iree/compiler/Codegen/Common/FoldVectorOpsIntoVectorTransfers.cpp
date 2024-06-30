// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fold-into-vector-transfers"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

namespace {

struct FoldVectorOpsIntoVectorTransfersPass
    : public FoldVectorOpsIntoVectorTransfersBase<
          FoldVectorOpsIntoVectorTransfersPass> {
  void runOnOperation() override;
};

class FoldTransposeIntoTransferRead
    : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceRead = op.getVector().getDefiningOp<vector::TransferReadOp>();
    if (!sourceRead) {
      return rewriter.notifyMatchFailure(
          op, "transpose producer is not a transfer read");
    }

    if (!sourceRead->hasOneUse()) {
      return rewriter.notifyMatchFailure(op, "multi-use producer read");
    }

    if (sourceRead.getMask()) {
      return rewriter.notifyMatchFailure(op, "unimplemented: masked read");
    }

    AffineMap permutation = sourceRead.getPermutationMap();
    permutation = AffineMap::get(
        permutation.getNumDims(), permutation.getNumSymbols(),
        applyPermutation(permutation.getResults(), op.getPermutation()),
        rewriter.getContext());

    SmallVector<bool> inBounds = sourceRead.getInBoundsValues();
    applyPermutationToVector(inBounds, op.getPermutation());

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, op.getResultVectorType(), sourceRead.getSource(),
        sourceRead.getIndices(), AffineMapAttr::get(permutation),
        sourceRead.getPadding(), sourceRead.getMask(),
        rewriter.getBoolArrayAttr(inBounds));
    return success();
  }
};

class FoldBroadcastIntoTransferRead
    : public OpRewritePattern<vector::BroadcastOp> {
public:
  using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceRead = op.getSource().getDefiningOp<vector::TransferReadOp>();
    if (!sourceRead) {
      return rewriter.notifyMatchFailure(
          op, "transpose producer is not a transfer read");
    }

    if (!sourceRead->hasOneUse()) {
      return rewriter.notifyMatchFailure(op, "multi-use producer read");
    }

    if (sourceRead.getMask()) {
      return rewriter.notifyMatchFailure(op, "unimplemented: masked read");
    }

    // Because the source is a transfer read we can always do this case.
    auto sourceVectorType = op.getSourceType().cast<VectorType>();
    auto resultVectorType = op.getResultVectorType();

    int64_t numDroppedDims =
        resultVectorType.getRank() - sourceVectorType.getRank();

    AffineMap permutation = sourceRead.getPermutationMap();
    SmallVector<AffineExpr> exprList(numDroppedDims,
                                     rewriter.getAffineConstantExpr(0));
    exprList.append(permutation.getResults().begin(),
                    permutation.getResults().end());

    permutation =
        AffineMap::get(permutation.getNumDims(), permutation.getNumSymbols(),
                       exprList, rewriter.getContext());

    SmallVector<bool> inBounds(numDroppedDims, true);
    inBounds.append(sourceRead.getInBoundsValues());

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, resultVectorType, sourceRead.getSource(), sourceRead.getIndices(),
        AffineMapAttr::get(permutation), sourceRead.getPadding(),
        sourceRead.getMask(), rewriter.getBoolArrayAttr(inBounds));
    return success();
  }
};

void FoldVectorOpsIntoVectorTransfersPass::runOnOperation() {
  auto funcOp = getOperation();
  LDBG("before folding transpose/broadcast into vector transfer\n" << funcOp);
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldTransposeIntoTransferRead>(&getContext());
    patterns.add<FoldBroadcastIntoTransferRead>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
  LDBG("after folding transpose/broadcast into vector transfer\n" << funcOp);
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFoldVectorOpsIntoVectorTransfersPass() {
  return std::make_unique<FoldVectorOpsIntoVectorTransfersPass>();
}

} // namespace mlir::iree_compiler
