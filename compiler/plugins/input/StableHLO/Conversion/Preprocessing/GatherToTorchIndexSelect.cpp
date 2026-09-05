// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO gather to torch_index_select.

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_GATHERTOTORCHINDEXSELECT
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h.inc"

namespace {
struct GatherIsTorchIndexSelectPattern final
    : OpRewritePattern<mlir::stablehlo::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::GatherOp gather,
                                PatternRewriter &rewriter) const override {
    TypedValue<RankedTensorType> startIndices = gather.getStartIndices();
    auto startIndicesTy = cast<ShapedType>(startIndices.getType());
    if (!startIndicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(gather, "unranked start_indices");
    }

    TypedValue<RankedTensorType> operand = gather.getOperand();
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

    auto resultTy = dyn_cast<RankedTensorType>(gather.getResult().getType());
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

    for (auto [idx, value] : llvm::enumerate(gather.getSliceSizes())) {
      // First shape value must be 1.
      if (idx == 0) {
        if (value != 1) {
          return rewriter.notifyMatchFailure(gather, "slice_size[0] != 1");
        }
        continue;
      }

      // The gather needs to index the entire slice for each other dimension.
      if (value != operandTy.getDimSize(idx)) {
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

    // `gather` clamps every start index into `[0, operand_dim - slice_size]`
    // (https://openxla.org/stablehlo/spec#gather); `slice_sizes[0]` is 1 here,
    // so the bound is `operand.dim(0) - 1`. `torch_index_select` promises no
    // such thing -- it lowers to a bare `tensor.extract` -- so the clamp has to
    // be materialized here, or an out-of-range index reads unrelated memory.
    auto indexElemTy = dyn_cast<IntegerType>(startIndicesTy.getElementType());
    if (!indexElemTy) {
      return rewriter.notifyMatchFailure(gather, "non-integer start_indices");
    }

    ImplicitLocOpBuilder b(gather.getLoc(), rewriter);
    auto indexScalarTy = RankedTensorType::get({}, indexElemTy);
    unsigned indexWidth = indexElemTy.getWidth();

    Value upperBound;
    if (!operandTy.isDynamicDim(0)) {
      // `slice_sizes[0]` is 1 and the verifier bounds it by the dimension, so
      // a well-formed gather always has an element here. Bail rather than let a
      // negative bound through the unsigned arithmetic below.
      int64_t maxIndex = operandTy.getDimSize(0) - 1;
      if (maxIndex < 0) {
        return rewriter.notifyMatchFailure(gather, "operand dimension 0 empty");
      }
      // An index type too narrow to name element `maxIndex` can never reach
      // past the end, so saturate the bound instead of truncating it into a
      // nonsensical (possibly negative) value.
      uint64_t indexTypeMax =
          (indexElemTy.isUnsigned() ? APInt::getMaxValue(indexWidth)
                                    : APInt::getSignedMaxValue(indexWidth))
              .getLimitedValue();
      APInt bound(indexWidth, std::min<uint64_t>(maxIndex, indexTypeMax));
      upperBound = mlir::stablehlo::ConstantOp::create(
          b, DenseElementsAttr::get(indexScalarTy,
                                    b.getIntegerAttr(indexElemTy, bound)));
    } else if (indexWidth >= 32) {
      // `get_dimension_size` yields an i32, so only index types wide enough to
      // hold one can carry the bound without wrapping.
      Value dimSize = mlir::stablehlo::GetDimensionSizeOp::create(
          b, RankedTensorType::get({}, b.getI32Type()), operand,
          b.getI64IntegerAttr(0));
      if (dimSize.getType() != indexScalarTy) {
        dimSize = mlir::stablehlo::ConvertOp::create(b, indexScalarTy, dimSize);
      }
      Value one = mlir::stablehlo::ConstantOp::create(
          b, DenseElementsAttr::get(
                 indexScalarTy,
                 b.getIntegerAttr(indexElemTy, APInt(indexWidth, 1))));
      upperBound =
          mlir::stablehlo::SubtractOp::create(b, indexScalarTy, dimSize, one);
    } else {
      return rewriter.notifyMatchFailure(
          gather, "start_indices too narrow to bound a dynamic dimension");
    }

    Value lowerBound = mlir::stablehlo::ConstantOp::create(
        b, DenseElementsAttr::get(indexScalarTy, b.getZeroAttr(indexElemTy)));
    Value clampedIndices = mlir::stablehlo::ClampOp::create(
        b, startIndicesTy, lowerBound, startIndices, upperBound);

    auto torchIndexSelect = mlir::stablehlo::TorchIndexSelectOp::create(
        b, RankedTensorType::get(indexSelectShape, operandTy.getElementType()),
        operand, clampedIndices, b.getI64IntegerAttr(0),
        b.getI64IntegerAttr(0));

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
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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
