// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

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
    if (insertOp->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
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
    if (op->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
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
    if (op->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
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
    if (op->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
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

struct ConvertTensorFromElementsPattern
    : public OpRewritePattern<tensor::FromElementsOp> {
  using OpRewritePattern<tensor::FromElementsOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: This pattern was mainly added to iron out some kinks specific to
    // detensoring (see: https://github.com/openxla/iree/issues/1159). Do we
    // need to expand this check for other uses?
    if (op->getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    auto tensorType = op.getType();
    if (!tensorType.hasRank()) {
      return failure();
    }

    // Check that all the dimensions are 1.
    if (!llvm::all_of(tensorType.getShape(),
                      [](int64_t dim) { return dim == 1; })) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::TensorSplatOp>(
        op, tensorType, op.getOperand(0), ValueRange());
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
    if (reshapeOp->template getParentOfType<Flow::DispatchWorkgroupsOp>()) {
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
      if (resultShape != ShapedType::kDynamic)
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
    rewriter.replaceOpWithNewOp<TensorSplatOp>(
        fillOp, fillOp.output().getType(), fillOp.value(), dynamicDims);
    return success();
  }
};

} // namespace

void populateTensorToFlowConversionPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns) {
  patterns
      .insert<ConvertLinalgFillPattern, ConvertTensorBitcastPattern,
              ConvertTensorCastPattern, ConvertTensorExtractPattern,
              ConvertTensorExtractSlicePattern, ConvertTensorInsertSlicePattern,
              ConvertTensorInsertPattern, ConvertTensorFromElementsPattern,
              ConvertTensorReshapePattern<tensor::CollapseShapeOp>,
              ConvertTensorReshapePattern<tensor::ExpandShapeOp>>(context);
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
