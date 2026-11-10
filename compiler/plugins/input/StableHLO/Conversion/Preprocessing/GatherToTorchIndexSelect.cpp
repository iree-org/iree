// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO gather to torch_index_select.

#include <limits>

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_GATHERTOTORCHINDEXSELECT
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h.inc"

namespace {
// Returns whether a clamped index of `indexType` may still fall outside a
// 32-bit target index type: unsigned and 64-bit indices are only safe when
// every valid index, below the static `indexedSize`, fits int32.
static bool mayClampedIndexExceedI32(IntegerType indexType,
                                     int64_t indexedSize) {
  bool isWideOrUnsigned =
      indexType.isUnsignedInteger() || indexType.getWidth() >= 64;
  bool fitsI32 = ShapedType::isStatic(indexedSize) && indexedSize > 0 &&
                 indexedSize - 1 <= std::numeric_limits<int32_t>::max();
  return isWideOrUnsigned && !fitsI32;
}

// Returns `indices` clamped to [0, size - 1] of the first dimension of
// `operand`. Gather clamps indices while torch_index_select requires them in
// bounds. The indices are widened to 64 bits first, so the dimension size fits
// even for narrow indices.
static Value clampToIndexedRange(PatternRewriter &rewriter, Location loc,
                                 Value operand, Value indices) {
  auto operandTy = cast<RankedTensorType>(operand.getType());
  auto indicesTy = cast<RankedTensorType>(indices.getType());
  auto indexType = cast<IntegerType>(indicesTy.getElementType());
  auto wideIndexType =
      IntegerType::get(rewriter.getContext(), 64, indexType.getSignedness());
  auto scalarType = RankedTensorType::get({}, wideIndexType);
  Value zero = mlir::stablehlo::ConstantOp::create(
      rewriter, loc, DenseIntElementsAttr::get(scalarType, APInt(64, 0)));
  Value upperBound;
  if (operandTy.isDynamicDim(0)) {
    Value size = tensor::DimOp::create(rewriter, loc, operand, 0);
    size =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), size);
    size = tensor::FromElementsOp::create(
        rewriter, loc, RankedTensorType::get({}, rewriter.getI64Type()), size);
    size = mlir::stablehlo::ConvertOp::create(rewriter, loc, scalarType, size);
    Value one = mlir::stablehlo::ConstantOp::create(
        rewriter, loc, DenseIntElementsAttr::get(scalarType, APInt(64, 1)));
    upperBound = mlir::stablehlo::SubtractOp::create(rewriter, loc, size, one);
  } else {
    upperBound = mlir::stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseIntElementsAttr::get(scalarType,
                                  APInt(64, operandTy.getDimSize(0) - 1)));
  }
  if (indexType != wideIndexType) {
    indices = mlir::stablehlo::ConvertOp::create(
        rewriter, loc,
        RankedTensorType::get(indicesTy.getShape(), wideIndexType), indices);
  }
  return mlir::stablehlo::ClampOp::create(rewriter, loc, zero, indices,
                                          upperBound);
}

// Returns whether a gather slice is one element of the first operand
// dimension and the entire extent of every other dimension.
static bool selectsWholeRows(ArrayRef<int64_t> sliceSizes,
                             ShapedType operandTy) {
  return !sliceSizes.empty() && sliceSizes.front() == 1 &&
         llvm::equal(sliceSizes.drop_front(),
                     operandTy.getShape().drop_front());
}

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

    auto indexType = cast<IntegerType>(startIndicesTy.getElementType());
    if (mayClampedIndexExceedI32(indexType, operandTy.getDimSize(0))) {
      return rewriter.notifyMatchFailure(
          gather, "clamped indices may exceed signed 32-bit range");
    }

    // The final reshape requires a static result; use generic gather otherwise.
    if (!resultTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(gather, "dynamic result shape");
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

    if (!selectsWholeRows(gather.getSliceSizes(), operandTy)) {
      return rewriter.notifyMatchFailure(
          gather, "slice sizes are not 1 followed by the operand dimensions");
    }

    auto indexSelectShape = llvm::to_vector(startIndicesTy.getShape());

    for (auto dim : operandTy.getShape().drop_front()) {
      indexSelectShape.push_back(dim);
    }

    if (dimensionNumbers.getCollapsedSliceDims().size() != 1 ||
        dimensionNumbers.getCollapsedSliceDims()[0] != 0) {
      return rewriter.notifyMatchFailure(gather, "collapsed_slice_dims != [0]");
    }

    Value indices =
        clampToIndexedRange(rewriter, gather.getLoc(), operand, startIndices);

    auto torchIndexSelect = mlir::stablehlo::TorchIndexSelectOp::create(
        rewriter, gather.getLoc(),
        RankedTensorType::get(indexSelectShape, operandTy.getElementType()),
        operand, indices, rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(0));

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(
        gather, gather.getType(), torchIndexSelect);

    return success();
  }
};

struct GatherToTorchIndexSelect final
    : impl::GatherToTorchIndexSelectBase<GatherToTorchIndexSelect> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }

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
