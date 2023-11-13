// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO gather to torch_index_select.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h"
#include "stablehlo-iree/Conversion/Preprocessing/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_GATHERTOTORCHINDEXSELECT
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h.inc"

namespace {
struct GatherIsTorchIndexSelectPattern final
    : OpRewritePattern<mlir::stablehlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GatherOp gather,
                                PatternRewriter &rewriter) const override {
    TypedValue<TensorType> startIndices = gather.getStartIndices();
    auto startIndicesTy = cast<ShapedType>(startIndices.getType());
    if (!startIndicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(gather, "unranked start_indices");
    }

    TypedValue<TensorType> operand = gather.getOperand();
    auto operandTy = cast<ShapedType>(operand.getType());
    if (!operandTy.hasRank()) {
      return rewriter.notifyMatchFailure(gather, "unranked operand");
    }

    int64_t indexVectorDim = std::max<int64_t>(0, startIndicesTy.getRank() - 1);

    // We can use torch_index_select if the last dimension represents the
    // gather indices.
    auto dimensionNumbers = gather.getDimensionNumbers();
    if (dimensionNumbers.getIndexVectorDim() != indexVectorDim) {
      return rewriter.notifyMatchFailure(
          gather, "index_vector_dim not last dimension of start_indices");
    }

    // Index select only works across a single dimension.
    if (!startIndicesTy.getShape().empty() &&
        startIndicesTy.getShape().back() != 1) {
      return rewriter.notifyMatchFailure(
          gather, "start_indices index vector dimension not 1");
    }

    // Only support the default case for start_index_map.
    if (dimensionNumbers.getStartIndexMap().size() != 1 ||
        dimensionNumbers.getStartIndexMap()[0] != 0) {
      return rewriter.notifyMatchFailure(gather, "start_index_map != [0]");
    }

    auto resultTy =
        llvm::dyn_cast<RankedTensorType>(gather.getResult().getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(gather, "unranked result");
    }

    // Offset dimensions should be the defaults.
    if (static_cast<int64_t>(dimensionNumbers.getOffsetDims().size()) !=
        resultTy.getRank() - indexVectorDim) {
      return rewriter.notifyMatchFailure(
          gather, "offset_dims.size not operand rank minus index_vector_dim");
    }

    for (auto [idx, dim] : llvm::enumerate(dimensionNumbers.getOffsetDims())) {
      if (static_cast<int64_t>(idx + indexVectorDim) != dim) {
        return rewriter.notifyMatchFailure(
            gather, "offset_dims != [index_vector_dim, result.rank)");
      }
    }

    for (auto [idx, value] :
         llvm::enumerate(gather.getSliceSizes().getValues<APInt>())) {
      // First shape value must be 1.
      if (idx == 0) {
        if (value.getSExtValue() != 1) {
          return rewriter.notifyMatchFailure(gather, "slice_size[0] != 1");
        }
        continue;
      }

      // The gather needs to index the entire slice for each other dimension.
      if (value.getSExtValue() != operandTy.getDimSize(idx)) {
        return rewriter.notifyMatchFailure(
            gather, "slice_size doesn't match operand dimension");
      }
    }

    auto indexSelectShape = llvm::to_vector(startIndicesTy.getShape());

    for (auto dim : operandTy.getShape().drop_front()) {
      indexSelectShape.push_back(dim);
    }

    if (dimensionNumbers.getCollapsedSliceDims().size() != 1 ||
        dimensionNumbers.getCollapsedSliceDims()[0] != 0) {
      return rewriter.notifyMatchFailure(gather, "collapsed_slice_dims != [0]");
    }

    auto torchIndexSelect =
        rewriter.create<mlir::stablehlo::TorchIndexSelectOp>(
            gather.getLoc(),
            RankedTensorType::get(indexSelectShape, operandTy.getElementType()),
            operand, gather.getStartIndices(), rewriter.getI64IntegerAttr(0),
            rewriter.getI64IntegerAttr(0));

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(
        gather, gather.getType(), torchIndexSelect);

    return success();
  }
};

struct GatherToTorchIndexSelect final
    : impl::GatherToTorchIndexSelectBase<GatherToTorchIndexSelect> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populatePreprocessingGatherToTorchIndexSelectPatterns(ctx, &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populatePreprocessingGatherToTorchIndexSelectPatterns(
    mlir::MLIRContext *context, RewritePatternSet *patterns) {
  patterns->add<GatherIsTorchIndexSelectPattern>(context);
}

} // namespace mlir::iree_compiler::stablehlo
