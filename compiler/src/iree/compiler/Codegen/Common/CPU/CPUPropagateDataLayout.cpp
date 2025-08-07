// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "llvm/ADT/STLExtras.h"
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

/// Sinks down tensor.collapse_shape across linalg.unpack op, if the collapsing
/// dims are two unit dims where one is outer dimension and the other is inner
/// dimension. It implies that we swap two operations by adjusting the packing
/// metadata in linalg.unpack op.
/// Note that the pattern only supports the case where the destination tensor of
/// linalg.unpack op is a tensor.empty op. The constraint can be removed by
/// introducing tensor.expand_shape op on the destination tensor. However, it is
/// not common in practice, so it is not supported now.
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
          op, "expected the source to be a tensor.collapse_shape op");
    }

    SmallVector<ReassociationIndices, 4> ri =
        collapseOp.getReassociationIndices();
    ReassociationIndices outerRi, innerRi;
    for (ArrayRef<int64_t> indices : ri) {
      if (indices.size() == 1) {
        continue;
      }
      if (indices.size() > 2) {
        return rewriter.notifyMatchFailure(
            op, "expected re-association map to have two dimensions");
      }
      if (outerRi.empty()) {
        outerRi.assign(indices.begin(), indices.end());
        continue;
      }
      if (innerRi.empty()) {
        innerRi.assign(indices.begin(), indices.end());
        continue;
      }
      return rewriter.notifyMatchFailure(
          op, "expected only two re-association maps to have two dimensions");
    }
    if (outerRi.empty() || innerRi.empty()) {
      return rewriter.notifyMatchFailure(
          op, "expected only two re-association maps to have two dimensions");
    }

    RankedTensorType srcType = collapseOp.getSrcType();
    if (innerRi.back() != srcType.getRank() - 1) {
      return rewriter.notifyMatchFailure(
          op, "expected that the two innermost dimensions are collapsed");
    }
    SmallVector<int64_t> innerDimPos(op.getInnerDimsPos());
    if (!llvm::is_contained(outerRi, innerDimPos[0])) {
      return rewriter.notifyMatchFailure(
          op, "expected the packed dimension is collapsed");
    }

    bool missLeadingUnitDim = srcType.getDimSize(outerRi[0]) == 1 &&
                              srcType.getDimSize(innerRi[0]) == 1;
    bool missTrailingUnitDim = srcType.getDimSize(outerRi[1]) == 1 &&
                               srcType.getDimSize(innerRi[1]) == 1;
    if (!missLeadingUnitDim && !missTrailingUnitDim) {
      return rewriter.notifyMatchFailure(op,
                                         "expected collapsing either leading "
                                         "unit dims or trailing outer dims");
    }

    // We either add unit dims right before or after the packed dimensions.
    // E.g., AxBxNxCxDxn becomes AxBx1xNxCxDx1xn if `missLeadingUnitDim` is
    // true. It becomes AxBxNx1xCxDxnx1 if `missingTrailingUnitDim` is true.
    // If both are true, the former is prioritized because it does not matter in
    // practice.
    SmallVector<OpFoldResult> innerTiles(op.getMixedTiles());
    SmallVector<OpFoldResult> destShape = emptyOp.getMixedSizes();
    if (missLeadingUnitDim) {
      // The unit dim is inserted before the packed dimension, so we advance one
      // for innerDimPos[0].
      innerDimPos[0]++;
      innerDimPos.insert(innerDimPos.begin(), outerRi[0]);
      innerTiles.insert(innerTiles.begin(), rewriter.getIndexAttr(1));
      destShape.insert(destShape.begin() + outerRi[0],
                       rewriter.getIndexAttr(1));
    } else {
      innerDimPos.insert(innerDimPos.end(), outerRi[1]);
      innerTiles.insert(innerTiles.end(), rewriter.getIndexAttr(1));
      destShape.insert(destShape.end(), rewriter.getIndexAttr(1));
    }

    Location loc = op.getLoc();
    auto newDestOp = rewriter.create<tensor::EmptyOp>(
        loc, destShape, emptyOp.getType().getElementType());
    auto newUnpackOp = rewriter.create<linalg::UnPackOp>(
        loc, collapseOp.getSrc(), newDestOp, innerDimPos, innerTiles);
    SmallVector<ReassociationIndices> newRi;
    for (int64_t i = 0, e = op.getDestRank(); i < e; ++i) {
      if (i == outerRi[0]) {
        newRi.push_back(outerRi);
        ++i;
      } else {
        newRi.push_back({i});
      }
    }
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
        auto consumerGenericOp =
            dyn_cast_if_present<linalg::GenericOp>(fusedOperand->getOwner());
        if (!isa<tensor::CollapseShapeOp>(producer) || !consumerGenericOp) {
          return false;
        }
        return true;
      });
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
