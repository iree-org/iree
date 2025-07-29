// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CPUPROPAGATEDATALAYOUTPASS
#include "iree/compiler/Codegen/Common/CPU/Passes.h.inc"

namespace {

// Sinks down tensor.collapse_shape across linalg.unpack op, if the collapsing
// dims are two unit dims where one is outer dimension and the other is inner
// dimension. It implies that we swap two operations by adjusting the packing
// metadata in linalg.unpack op.
// Note that the pattern only supports the case where the destination tensor of
// linalg.unpack op is a tensor.empty op. The constraint can be removed by
// introducing tensor.expand_shape op on the destination tensor. However, it is
// not common in practice, so it is not supported now.
struct SinkDownCollapsingUnitDimsAcrossUnpack final
    : public OpRewritePattern<linalg::UnPackOp> {
  using OpRewritePattern<linalg::UnPackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    if (!isIdentityPermutation(op.getOuterDimsPerm())) {
      return rewriter.notifyMatchFailure(
          op, "expected identity (or unset) outer permutation");
    }
    if (op.getSourceRank() != op.getDestRank() + 1) {
      return rewriter.notifyMatchFailure(
          op, "expected unpacking exactly one dimension");
    }
    auto emptyOp = op.getDest().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp) {
      return rewriter.notifyMatchFailure(
          op, "expected destination to be a tensor.empty op");
    }
    auto collapseOp = op.getSource().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseOp) {
      return rewriter.notifyMatchFailure(
          op, "expected the source to be a tensor.collpase_shape op");
    }

    int64_t srcRank = collapseOp.getSrcType().getRank();
    if (srcRank != 4 && srcRank != 5) {
      return rewriter.notifyMatchFailure(
          op, "expected the rank of collapseOp's source is either 4 or 5");
    }
    bool hasBatch = collapseOp.getSrcType().getRank() == 5;
    SmallVector<ReassociationIndices, 4> ri =
        collapseOp.getReassociationIndices();
    if (hasBatch && ri[0].size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected batch dimension to be not collapsed");
    }
    if (hasBatch + 2 != ri.size()) {
      return rewriter.notifyMatchFailure(
          op, "expected 2 reassociation indices for linalg.mmt4d source (3 for "
              "linalg.batch_mmt4d)");
    }
    ReassociationIndices outerRi = ri[hasBatch + 0];
    ReassociationIndices innerRi = ri[hasBatch + 1];
    if (outerRi.size() != 2 || innerRi.size() != 2) {
      return rewriter.notifyMatchFailure(
          op,
          "expected collapsing one outer dimension and one inner dimension");
    }

    RankedTensorType mmt4dSrcType = collapseOp.getSrcType();
    bool missUnitDimM = mmt4dSrcType.getDimSize(outerRi[0]) == 1 &&
                        mmt4dSrcType.getDimSize(innerRi[0]) == 1;
    bool missUnitDimN = mmt4dSrcType.getDimSize(outerRi[1]) == 1 &&
                        mmt4dSrcType.getDimSize(innerRi[1]) == 1;
    if (!missUnitDimM && !missUnitDimN) {
      return rewriter.notifyMatchFailure(
          op, "expected collapsing either M dimensions or N dimensions");
    }

    SmallVector<int64_t> innerDimPos(op.getInnerDimsPos());
    if (hasBatch && innerDimPos[0] == 0) {
      return rewriter.notifyMatchFailure(
          op, "expected unpacking either M or N dimension");
    }

    SmallVector<OpFoldResult> innerTiles(op.getMixedTiles());
    SmallVector<OpFoldResult> destShape = emptyOp.getMixedSizes();
    if (missUnitDimM) {
      for (auto &pos : innerDimPos) {
        pos++;
      }
      innerDimPos.insert(innerDimPos.begin(), hasBatch + 0);
      innerTiles.insert(innerTiles.begin(), rewriter.getIndexAttr(1));
      destShape.insert(destShape.begin() + hasBatch, rewriter.getIndexAttr(1));
    } else {
      innerDimPos.insert(innerDimPos.end(), hasBatch + 1);
      innerTiles.insert(innerTiles.end(), rewriter.getIndexAttr(1));
      destShape.insert(destShape.end(), rewriter.getIndexAttr(1));
    }

    Location loc = op.getLoc();
    auto newDestOp = rewriter.create<tensor::EmptyOp>(
        loc, destShape, emptyOp.getType().getElementType());
    auto newUnpackOp = rewriter.create<linalg::UnPackOp>(
        loc, collapseOp.getSrc(), newDestOp, innerDimPos, innerTiles);
    SmallVector<ReassociationIndices> newRi;
    if (hasBatch) {
      newRi.push_back({0});
    }
    newRi.push_back({hasBatch, hasBatch + 1});
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        op, newUnpackOp.getResult(), newRi);

    return success();
  }
};

struct CPUPropagateDataLayoutPass final
    : public impl::CPUPropagateDataLayoutPassBase<CPUPropagateDataLayoutPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void CPUPropagateDataLayoutPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  FunctionOpInterface funcOp = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<SinkDownCollapsingUnitDimsAcrossUnpack>(ctx);
  populateReshapeToInterfaceTensorPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns, /*foldSingleUseOnly=*/1);
  linalg::populateFoldReshapeOpsByExpansionPatterns(
      patterns, [](OpOperand *fusedOperand) -> bool {
        Operation *producer = fusedOperand->get().getDefiningOp();
        auto consumerGenercOp =
            dyn_cast_if_present<linalg::GenericOp>(fusedOperand->getOwner());
        if (!isa<tensor::CollapseShapeOp>(producer) || !consumerGenercOp) {
          return false;
        }
        return true;
      });
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
