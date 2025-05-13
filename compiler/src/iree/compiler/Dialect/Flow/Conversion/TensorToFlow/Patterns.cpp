// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

/// Converts linalg.fill ops into flow.tensor.splat ops.
///
/// This is expected to improve performance because we can use DMA
/// functionalities for the fill, instead of dispatching kernels.
struct ConvertLinalgFillPattern final
    : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (fillOp->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      // Don't convert linalg.fill ops that were fused together with other ops.
      return failure();
    }

    SmallVector<Value> dynamicDims = tensor::createDynamicDimValues(
        rewriter, fillOp.getLoc(), fillOp.output());
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorSplatOp>(
        fillOp, fillOp.output().getType(), fillOp.value(), dynamicDims);
    return success();
  }
};

/// Convert tensor.insert_slice ops into flow.tensor.update ops where possible.
struct ConvertTensorInsertSlicePattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    return convertInsertSliceOpToFlowUpdateOp(rewriter, insertOp);
  }
};

/// Convert tensor.insert ops into flow.tensor.store ops where possible.
struct ConvertTensorInsertPattern : public OpRewritePattern<tensor::InsertOp> {
  using OpRewritePattern<tensor::InsertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorStoreOp>(
        insertOp, insertOp.getScalar(), insertOp.getDest(),
        insertOp.getIndices());
    return success();
  }
};

/// Convert tensor.extract_slice ops into flow.tensor.slice ops where possible.
struct ConvertTensorExtractSlicePattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    return convertExtractSliceOpToFlowSliceOp(rewriter, sliceOp);
  }
};

struct ConvertTensorExtractPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorLoadOp>(
        op, op.getResult().getType(), op.getTensor(), op.getIndices());
    return success();
  }
};

struct ConvertTensorBitcastPattern
    : public OpRewritePattern<tensor::BitcastOp> {
  using OpRewritePattern<tensor::BitcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::BitcastOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    auto dynamicDims = IREE::Util::buildDynamicDimsForValue(
        op.getLoc(), op.getOperand(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorBitCastOp>(
        op, op.getResult().getType(), op.getOperand(), dynamicDims,
        dynamicDims);
    return success();
  }
};

struct ConvertTensorCastPattern : public OpRewritePattern<tensor::CastOp> {
  using OpRewritePattern<tensor::CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }

    auto loc = op.getLoc();
    Value input = op.getOperand();
    ShapedType inputType = llvm::dyn_cast<ShapedType>(input.getType());
    ShapedType resultType =
        llvm::dyn_cast_if_present<ShapedType>(op.getResult().getType());
    if (!inputType || !resultType || !inputType.hasRank() ||
        !resultType.hasRank()) {
      return rewriter.notifyMatchFailure(op, "not ranked shaped types");
    }
    // This should not happen, except in the context of type conversion.
    if (inputType.getRank() != resultType.getRank()) {
      return rewriter.notifyMatchFailure(op, "mismatched rank");
    }

    // Resolve dims to the most specific value.
    int rank = inputType.getRank();
    SmallVector<Value> dimSizes(rank);
    auto resolveDimSize = [&](int position) -> Value {
      if (!dimSizes[position]) {
        // Find the most specific.
        if (!inputType.isDynamicDim(position) ||
            !resultType.isDynamicDim(position)) {
          // Static dim.
          int64_t dimSize = !inputType.isDynamicDim(position)
                                ? inputType.getDimSize(position)
                                : resultType.getDimSize(position);
          dimSizes[position] =
              rewriter.create<arith::ConstantIndexOp>(loc, dimSize);
        } else {
          // Dynamic dim.
          dimSizes[position] =
              rewriter.create<tensor::DimOp>(loc, input, position);
        }
      }

      return dimSizes[position];
    };

    SmallVector<Value> sourceDynamicDims;
    SmallVector<Value> targetDynamicDims;
    for (int i = 0; i < rank; i++) {
      if (inputType.isDynamicDim(i)) {
        sourceDynamicDims.push_back(resolveDimSize(i));
      }
      if (resultType.isDynamicDim(i)) {
        targetDynamicDims.push_back(resolveDimSize(i));
      }
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        op, resultType, input, sourceDynamicDims, targetDynamicDims);

    return success();
  }
};

struct ConvertTensorConcatPattern : public OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern<tensor::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    if (concatOp->getParentOfType<IREE::Flow::DispatchRegionOp>() ||
        concatOp->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    if (concatOp.getDim() != 0) {
      return rewriter.notifyMatchFailure(
          concatOp, "only outer-dim concat lowering supported");
    }
    assert(cast<RankedTensorType>(concatOp.getInputs().front().getType())
                   .getRank() != 0 &&
           "concat cannot be of zero-rank tensors");

    Location loc = concatOp.getLoc();
    SmallVector<SmallVector<OpFoldResult>> inputShapes;
    inputShapes.reserve(concatOp.getInputs().size());
    // Note the output shape is computed directly without using
    // `reifyResultShapes` since we need the `inputShapes` anyway and using the
    // method would create duplicate `tensor.dim` operations.
    SmallVector<OpFoldResult> outputShape;
    AffineExpr addExpr =
        rewriter.getAffineSymbolExpr(0) + rewriter.getAffineSymbolExpr(1);
    SmallVector<OpFoldResult> concatOffsets;
    concatOffsets.reserve(concatOp.getInputs().size());
    for (auto [index, input] : llvm::enumerate(concatOp.getInputs())) {
      SmallVector<OpFoldResult> inputShape =
          tensor::getMixedSizes(rewriter, input.getLoc(), input);
      if (index == 0) {
        outputShape = inputShape;
        concatOffsets.push_back(rewriter.getIndexAttr(0));
      } else {
        concatOffsets.push_back(outputShape[0]);
        outputShape[0] = affine::makeComposedFoldedAffineApply(
            rewriter, loc, addExpr, {outputShape[0], inputShape[0]});

        // Any dims outside of concatenation axis (only `0` supported currently)
        // should be equal. Fill in any dynamic dims in `outputShape` known from
        // other inputs.
        // Ex. concat([?,?], [?,12]) -> [?,12]
        for (auto [dimIdx, outDim] :
             llvm::drop_begin(llvm::enumerate(outputShape))) {
          OpFoldResult inDim = inputShape[dimIdx];
          bool outDimIsDynamic = isa<Value>(outDim);
          bool inDimIsDynamic = isa<Value>(inDim);
          if (outDimIsDynamic && !inDimIsDynamic) {
            outputShape[dimIdx] = inDim;
          }
        }
      }
      inputShapes.emplace_back(std::move(inputShape));
    }

    Value replacement = rewriter.create<tensor::EmptyOp>(
        loc, outputShape, concatOp.getType().getElementType());

    SmallVector<int64_t> resultStaticDims;
    SmallVector<Value> resultDynamicDims;
    dispatchIndexOpFoldResults(outputShape, resultDynamicDims,
                               resultStaticDims);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Generate the `flow.tensor.update` operations for the concat.
    for (auto [index, input] : llvm::enumerate(concatOp.getInputs())) {
      SmallVector<int64_t> inputStaticShape;
      SmallVector<Value> inputDynamicShape;
      dispatchIndexOpFoldResults(inputShapes[index], inputDynamicShape,
                                 inputStaticShape);
      SmallVector<Value> offsets(inputStaticShape.size(), zero);
      offsets[0] =
          getValueOrCreateConstantIndexOp(rewriter, loc, concatOffsets[index]);
      replacement = rewriter.create<IREE::Flow::TensorUpdateOp>(
          loc, replacement.getType(), replacement, resultDynamicDims, offsets,
          input, inputDynamicShape);
    }
    rewriter.replaceOp(concatOp, replacement);
    return success();
  }
};

struct ConvertTensorFromElementsPattern
    : public OpRewritePattern<tensor::FromElementsOp> {
  using OpRewritePattern<tensor::FromElementsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: This pattern was mainly added to iron out some kinks specific to
    // detensoring (see: https://github.com/iree-org/iree/issues/1159). Do we
    // need to expand this check for other uses?
    if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    auto tensorType = op.getType();
    if (!tensorType.hasRank()) {
      return rewriter.notifyMatchFailure(op,
                                         "unranked result type not supported");
    }

    if (op.getNumOperands() == 1) {
      rewriter.replaceOpWithNewOp<IREE::Flow::TensorSplatOp>(
          op, tensorType, op.getOperand(0), ValueRange());
      return success();
    }

    const int64_t rank = tensorType.getRank();
    Value result = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), tensorType.getShape(), tensorType.getElementType());
    SmallVector<Value> ivs(rank);
    for (int i = 0, s = op.getNumOperands(); i < s; ++i) {
      int64_t index = i;
      for (int j = rank - 1; j >= 0; --j) {
        int64_t iv = index % tensorType.getDimSize(j);
        index = index / tensorType.getDimSize(j);
        ivs[j] = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), iv);
      }

      result = rewriter.create<Flow::TensorStoreOp>(
          op.getLoc(), op.getOperand(i), result, ivs);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Returns a sizes array with the dynamic dims.
static SmallVector<Value> getDynamicTensorSizes(OpBuilder &builder,
                                                Location loc,
                                                RankedTensorType type,
                                                Value tensor) {
  SmallVector<Value> sizes;
  for (const auto [idx, size] : enumerate(type.getShape())) {
    if (type.isDynamicDim(idx)) {
      Value dim = builder.create<tensor::DimOp>(loc, tensor, idx);
      sizes.push_back(dim);
    }
  }
  return sizes;
}

/// Convert tensor.reshape ops into flow.tensor.reshape ops where possible.
struct ConvertTensorDialectReshapeOpPattern
    : public OpRewritePattern<tensor::ReshapeOp> {
  using OpRewritePattern<tensor::ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    auto loc = op.getLoc();
    Value input = op.getSource();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto shapeOperandType = dyn_cast<ShapedType>(op.getShape().getType());
    auto resultType = dyn_cast<ShapedType>(op.getResult().getType());

    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "not ranked shaped types");
    }

    SmallVector<Value> srcSizes;
    srcSizes = getDynamicTensorSizes(rewriter, loc, inputType, input);

    // flow.reshape only takes dynamic dims for the result, source dims
    // (ignore static dimensions)
    SmallVector<Value> destSizes;
    for (int i = 0; i < shapeOperandType.getShape()[0]; i++) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value element = rewriter.create<tensor::ExtractOp>(loc, op.getShape(),
                                                         ValueRange({idx}));
      if (ShapedType::isDynamic(resultType.getShape()[i])) {
        auto elementTy = element.getType();
        if (isa<IntegerType>(elementTy)) {
          element = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), element);
        }
        destSizes.push_back(element);
      }
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        op, resultType, input, srcSizes, destSizes);
    return success();
  }
};

/// Converts linalg.tensor_reshape operations into flow.tensor.reshape
/// operations.
template <typename TensorReshapeOp>
struct ConvertTensorReshapePattern : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    if (reshapeOp
            ->template getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<SmallVector<OpFoldResult>> outputShape;
    ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
        cast<ReifyRankedShapedTypeOpInterface>(reshapeOp.getOperation());
    if (failed(reifyShapedTypeInterface.reifyResultShapes(rewriter,
                                                          outputShape))) {
      return failure();
    }
    SmallVector<Value> outputDynamicShapes;
    for (auto [resultShape, outputShp] : llvm::zip_equal(
             reshapeOp.getResultType().getShape(), outputShape[0])) {
      if (!ShapedType::isDynamic(resultShape))
        continue;
      outputDynamicShapes.push_back(getValueOrCreateConstantIndexOp(
          rewriter, reshapeOp.getLoc(), outputShp));
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        reshapeOp, reshapeOp.getResultType(), reshapeOp.getSrc(),
        outputDynamicShapes);
    return success();
  }
};

} // namespace

void populateTensorToFlowConversionPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns) {
  patterns.insert<ConvertLinalgFillPattern, ConvertTensorBitcastPattern,
                  ConvertTensorCastPattern, ConvertTensorConcatPattern,
                  ConvertTensorExtractPattern, ConvertTensorExtractSlicePattern,
                  ConvertTensorInsertSlicePattern, ConvertTensorInsertPattern,
                  ConvertTensorFromElementsPattern,
                  ConvertTensorDialectReshapeOpPattern,
                  ConvertTensorReshapePattern<tensor::CollapseShapeOp>,
                  ConvertTensorReshapePattern<tensor::ExpandShapeOp>>(context);
}

} // namespace mlir::iree_compiler::IREE::Flow
