// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements StableHLO-to-StableHLO IREE-specific preprocessing.

#include <numeric>
#include <random>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h"
#include "stablehlo-iree/Conversion/Preprocessing/Rewriters.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_STABLEHLOTOSTABLEHLOPREPROCESSING
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h.inc"

namespace {

bool isIota(ArrayRef<int64_t> array) {
  for (auto [idx, value] : llvm::enumerate(array)) {
    if (static_cast<int64_t>(idx) != value)
      return false;
  }
  return true;
}

DenseIntElementsAttr make1DElementsAttr(OpBuilder &b,
                                        ArrayRef<int64_t> integers) {
  auto type = RankedTensorType::get({static_cast<int64_t>(integers.size())},
                                    b.getIntegerType(64));
  return DenseIntElementsAttr::get(type, integers);
}

Value getF32Const(ImplicitLocOpBuilder b, ArrayRef<int64_t> shapes,
                  ArrayRef<float> values) {
  RankedTensorType ty = RankedTensorType::get(shapes, b.getF32Type());
  return b
      .create<mlir::stablehlo::ConstantOp>(DenseFPElementsAttr::get(ty, values))
      .getResult();
}

// Guarantee that the input dimensions are ordered batch, spatial_dims, feature
// dim.
struct ReorderConvOpInputDimensions final
    : OpRewritePattern<mlir::stablehlo::ConvolutionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = cast<ShapedType>(op.getLhs().getType());
    auto lhsShape = lhsType.getShape();
    if (!lhsType.hasRank()) {
      return failure();
    }

    auto dimensionNumbers = op.getDimensionNumbers();
    auto spatialDims = dimensionNumbers.getInputSpatialDimensions();

    // Compute the permutation required to create a standard order.
    llvm::SmallVector<int64_t> permutations;
    permutations.push_back(dimensionNumbers.getInputBatchDimension());
    llvm::append_range(permutations, spatialDims);
    permutations.push_back(dimensionNumbers.getInputFeatureDimension());

    // If the permutation is iota then no reordering is required.
    if (isIota(permutations)) {
      return failure();
    }

    llvm::SmallVector<int64_t> transposeShape;
    for (int64_t p : permutations) {
      transposeShape.push_back(lhsShape[p]);
    }

    auto transposed = rewriter.create<mlir::stablehlo::TransposeOp>(
        op.getLoc(),
        RankedTensorType::get(transposeShape, lhsType.getElementType()),
        op.getLhs(), rewriter.getI64TensorAttr(permutations));

    llvm::SmallVector<int64_t> newSpatialDimensions(spatialDims.size());
    std::iota(newSpatialDimensions.begin(), newSpatialDimensions.end(), 1);

    auto newDimensionNumbers = mlir::stablehlo::ConvDimensionNumbersAttr::get(
        op.getContext(),
        /*input_batch_dimension=*/0,
        /*input_feature_dimension=*/newSpatialDimensions.size() + 1,
        /*input_spatial_dimensions=*/newSpatialDimensions,
        dimensionNumbers.getKernelInputFeatureDimension(),
        dimensionNumbers.getKernelOutputFeatureDimension(),
        dimensionNumbers.getKernelSpatialDimensions(),
        dimensionNumbers.getOutputBatchDimension(),
        dimensionNumbers.getOutputFeatureDimension(),
        dimensionNumbers.getOutputSpatialDimensions());

    SmallVector<Value, 2> operands = {transposed, op.getRhs()};
    auto newConv = rewriter.create<mlir::stablehlo::ConvolutionOp>(
        op.getLoc(), op.getType(), operands, op->getAttrs());
    newConv.setDimensionNumbersAttr(newDimensionNumbers);
    rewriter.replaceOp(op, newConv.getResult());

    return success();
  }
};

struct ReorderConvOpKernelDimensions final
    : OpRewritePattern<mlir::stablehlo::ConvolutionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::stablehlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto kernel = op.getRhs();
    auto kernelType = cast<ShapedType>(kernel.getType());
    if (!kernelType.hasRank())
      return failure();
    auto kernelShape = kernelType.getShape();

    auto dimensionNumbers = op.getDimensionNumbers();

    auto spatialDims = dimensionNumbers.getKernelSpatialDimensions();

    auto inputFeatureDimension =
        dimensionNumbers.getKernelInputFeatureDimension();
    auto outputFeatureDimension =
        dimensionNumbers.getKernelOutputFeatureDimension();

    // Compute the permutation for the transpose.
    llvm::SmallVector<int64_t> permutation(spatialDims.begin(),
                                           spatialDims.end());
    permutation.push_back(inputFeatureDimension);
    permutation.push_back(outputFeatureDimension);

    // If the permutation is iota, then no transpose is required.
    if (isIota(permutation))
      return failure();

    llvm::SmallVector<int64_t> transposeShape;
    for (int64_t perm : permutation) {
      transposeShape.push_back(kernelShape[perm]);
    }

    llvm::SmallVector<int64_t> newSpatialDimensions(spatialDims.size());
    std::iota(newSpatialDimensions.begin(), newSpatialDimensions.end(), 0);

    auto transposeKernel = rewriter.create<mlir::stablehlo::TransposeOp>(
        op.getLoc(),
        RankedTensorType::get(transposeShape, kernelType.getElementType()),
        kernel, rewriter.getI64TensorAttr(permutation));

    auto newDimensionNumbers = mlir::stablehlo::ConvDimensionNumbersAttr::get(
        op.getContext(), dimensionNumbers.getInputBatchDimension(),
        dimensionNumbers.getInputFeatureDimension(),
        dimensionNumbers.getInputSpatialDimensions(),
        /*kernel_input_feature_dimension=*/
        newSpatialDimensions.size(),
        /*kernel_output_feature_dimension=*/
        newSpatialDimensions.size() + 1, newSpatialDimensions,
        dimensionNumbers.getOutputBatchDimension(),
        dimensionNumbers.getOutputFeatureDimension(),
        dimensionNumbers.getOutputSpatialDimensions());

    SmallVector<Value, 2> operands = {op.getLhs(), transposeKernel};
    mlir::stablehlo::ConvolutionOp newConv =
        rewriter.create<mlir::stablehlo::ConvolutionOp>(
            op.getLoc(), op.getType(), operands, op->getAttrs());
    newConv.setDimensionNumbersAttr(newDimensionNumbers);

    rewriter.replaceOp(op, {newConv.getResult()});
    return success();
  }
};

// Guarantee that the output dimensions are ordered batch, spatial_dims, feature
// dim.
struct ReorderConvOpOutputDimensions final
    : OpRewritePattern<mlir::stablehlo::ConvolutionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<ShapedType>(op.getType());
    auto resultShape = resultType.getShape();
    if (!resultType.hasRank()) {
      return failure();
    }

    auto dimensionNumbers = op.getDimensionNumbers();
    auto spatialDims = dimensionNumbers.getOutputSpatialDimensions();

    // Compute the permutation to transpose to an ordered output.
    llvm::SmallVector<int64_t> permutation;
    permutation.push_back(dimensionNumbers.getOutputBatchDimension());
    permutation.append(spatialDims.begin(), spatialDims.end());
    permutation.push_back(dimensionNumbers.getOutputFeatureDimension());

    // If the permutation is iota then no reordering is required.
    if (isIota(permutation)) {
      return failure();
    }

    // Compute what the new conv shape should be.
    llvm::SmallVector<int64_t> convShape;
    for (auto p : permutation) {
      convShape.push_back(resultShape[p]);
    }

    // Compute the inverse transpose to unordered and ordered output.
    llvm::SmallVector<int64_t> invertPermutation(permutation.size());
    for (auto it : llvm::enumerate(permutation)) {
      invertPermutation[it.value()] = it.index();
    }

    llvm::SmallVector<int64_t> newSpatialDimensions(spatialDims.size());
    std::iota(newSpatialDimensions.begin(), newSpatialDimensions.end(), 1);

    auto newDimensionNumbers = mlir::stablehlo::ConvDimensionNumbersAttr::get(
        op.getContext(), dimensionNumbers.getInputBatchDimension(),
        dimensionNumbers.getInputFeatureDimension(),
        dimensionNumbers.getInputSpatialDimensions(),
        dimensionNumbers.getKernelInputFeatureDimension(),
        dimensionNumbers.getKernelOutputFeatureDimension(),
        dimensionNumbers.getKernelSpatialDimensions(),
        /*output_batch_dimension=*/0,
        /*output_feature_dimension=*/newSpatialDimensions.size() + 1,
        /*output_spatial_dimensions=*/newSpatialDimensions);

    SmallVector<Value, 2> operands = {op.getLhs(), op.getRhs()};
    auto newConv = rewriter.create<mlir::stablehlo::ConvolutionOp>(
        op.getLoc(),
        RankedTensorType::get(convShape, resultType.getElementType()), operands,
        op->getAttrs());
    newConv.setDimensionNumbersAttr(newDimensionNumbers);

    auto transposed = rewriter.create<mlir::stablehlo::TransposeOp>(
        op.getLoc(), resultType, newConv,
        rewriter.getI64TensorAttr(invertPermutation));

    rewriter.replaceOp(op, transposed.getResult());
    return success();
  }
};

bool isConsecutive(ArrayRef<int64_t> array) {
  for (size_t i = 1, e = array.size(); i < e; ++i) {
    if (array[i] - array[i - 1] != 1)
      return false;
  }
  return true;
}

/// Rewrites stablehlo.dot_general so lhs contraction dimensions are innermost
/// and rhs contraction dimensions are dims right after batch dimension. The
/// pattern inserts transposes so the dot_general always has the form:
/// {batch_dims, parallel_dims, contraction_dims}.
///   {batch_dims, contraction_dims, parallel_dims}
/// After that, batch_dims, contraction_dims, parallel_dims are
/// in consecutive order and not spliting the domain. This pattern inserts
/// reshapes to collapse consecutive reduction and parallel dims to always
/// generate a rank-3 dot_general op.
struct TransposeReshapeGenericDotGeneral final
    : OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  Value TransposeIfNonConsecutive(OpBuilder &b, Location loc, Value src,
                                  ArrayRef<int64_t> targetOrder) const {
    if (isConsecutive(targetOrder))
      return src;

    auto type = cast<RankedTensorType>(src.getType());
    SmallVector<int64_t> transposeShape;
    for (int64_t i : targetOrder) {
      transposeShape.push_back(type.getDimSize(i));
    }
    return b.create<mlir::stablehlo::TransposeOp>(
        loc, RankedTensorType::get(transposeShape, type.getElementType()), src,
        b.getI64TensorAttr(targetOrder));
  }

  Value ReshapeIfNonStandard(OpBuilder &b, Location loc, Value src,
                             size_t dimsBorder0, size_t dimsBorder1) const {
    auto type = cast<RankedTensorType>(src.getType());
    ArrayRef<int64_t> shape = type.getShape();
    if (dimsBorder0 <= 1 && dimsBorder1 - dimsBorder0 <= 1 &&
        shape.size() - dimsBorder1 <= 1)
      return src;

    SmallVector<int64_t> result_shape = {
        std::accumulate(shape.begin(), shape.begin() + dimsBorder0, 1,
                        std::multiplies<int64_t>()),
        std::accumulate(shape.begin() + dimsBorder0,
                        shape.begin() + dimsBorder1, 1,
                        std::multiplies<int64_t>()),
        std::accumulate(shape.begin() + dimsBorder1, shape.end(), 1,
                        std::multiplies<int64_t>())};
    return b.create<mlir::stablehlo::ReshapeOp>(
        loc, RankedTensorType::get(result_shape, type.getElementType()), src);
  }

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsShapeType = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsShapeType = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!lhsShapeType || !rhsShapeType || !resultType)
      return failure();

    // TODO(jpienaar): This pattern is not safe for dynamic shapes and seems to
    // be (now) redundant with later pass that does handle them. To decouple
    // fixing and verifying redundant, this just limits to static shapes and
    // then will remove this in follow up.
    if (!lhsShapeType.hasStaticShape() || !rhsShapeType.hasStaticShape())
      return failure();

    SmallVector<int64_t> lhsTargetOrder, rhsTargetOrder;
    mlir::stablehlo::DotDimensionNumbersAttr dimNumbers =
        op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dimNumbers.getRhsContractingDimensions();

    // No contraction dims means this can be represented as a mul.
    if (lhsContractingDims.empty() || rhsContractingDims.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "can be represented as stablehlo.mul");
    }

    // No batching dimensions means this can be represented a dot.
    if (lhsBatchingDims.empty() || rhsBatchingDims.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "can be represented as stablehlo.dot");
    }

    SmallVector<bool> isLhsParallel(lhsShapeType.getRank(), true);
    for (int64_t i : lhsBatchingDims) {
      lhsTargetOrder.push_back(i);
      isLhsParallel[i] = false;
    }
    for (int64_t i : lhsContractingDims) {
      isLhsParallel[i] = false;
    }
    for (int64_t i = 0, e = lhsShapeType.getRank(); i < e; ++i) {
      if (isLhsParallel[i]) {
        lhsTargetOrder.push_back(i);
      }
    }
    for (int64_t i : lhsContractingDims) {
      lhsTargetOrder.push_back(i);
    }

    SmallVector<bool> isRhsParallel(rhsShapeType.getRank(), true);

    for (int64_t i : rhsBatchingDims) {
      rhsTargetOrder.push_back(i);
      isRhsParallel[i] = false;
    }
    for (int64_t i : rhsContractingDims) {
      rhsTargetOrder.push_back(i);
      isRhsParallel[i] = false;
    }
    for (int64_t i = 0, e = rhsShapeType.getRank(); i < e; ++i) {
      if (isRhsParallel[i]) {
        rhsTargetOrder.push_back(i);
      }
    }

    Value lhs = TransposeIfNonConsecutive(rewriter, op.getLoc(), op.getLhs(),
                                          lhsTargetOrder);
    Value rhs = TransposeIfNonConsecutive(rewriter, op.getLoc(), op.getRhs(),
                                          rhsTargetOrder);

    // The dimensions of this will always be transposed into {batch_dims,
    // parallel_dims, contraction_dims}, and the
    // following logic is based on this assumption.
    // TODO(#7443): If we consider transpose performance, the above assumptions
    // may not be true.
    int64_t numLhsContractionDims = lhsContractingDims.size();
    int64_t lhsContractionBase = lhsShapeType.getRank() - numLhsContractionDims;
    int64_t rhsContractionBase = rhsBatchingDims.size();
    int64_t numRhsContractionDims =
        rhsContractionBase + rhsContractingDims.size();

    lhs = ReshapeIfNonStandard(rewriter, op.getLoc(), lhs,
                               lhsBatchingDims.size(), lhsContractionBase);
    rhs = ReshapeIfNonStandard(rewriter, op.getLoc(), rhs,
                               rhsBatchingDims.size(), numRhsContractionDims);

    if (lhs == op.getLhs() && rhs == op.getRhs())
      return rewriter.notifyMatchFailure(op, "already in canonical form");

    auto dimensionNumbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatchingDimensions=*/0,
        /*rhsBatchingDimensions=*/0,
        /*lhsContractingDimensions=*/
        cast<ShapedType>(lhs.getType()).getRank() - 1,
        /*rhsContractingDimensions=*/1);
    auto lhsNewType = cast<RankedTensorType>(lhs.getType());
    auto rhsNewType = cast<RankedTensorType>(rhs.getType());

    // batching、lhs parallel、rhs parallel this order is a conversion
    SmallVector<int64_t, 3> newShape = {lhsNewType.getShape()[0]};

    if (lhsNewType.getRank() > 2)
      newShape.push_back(lhsNewType.getDimSize(1));

    if (rhsNewType.getRank() > 2)
      newShape.push_back(rhsNewType.getDimSize(2));

    TensorType newResultType =
        RankedTensorType::get(newShape, resultType.getElementType());

    auto newOp = rewriter.create<mlir::stablehlo::DotGeneralOp>(
        op.getLoc(), newResultType, lhs, rhs, dimensionNumbers,
        op.getPrecisionConfigAttr());

    // Copy over unknown attributes as we currently rely on it to let user tune
    // lowering parameters.
    ArrayRef<StringRef> odsAttrs = op.getAttributeNames();
    for (NamedAttribute kv : op->getAttrs()) {
      if (!llvm::is_contained(odsAttrs, kv.getName().getValue())) {
        newOp->setAttr(kv.getName(), kv.getValue());
      }
    }

    Value result = newOp.getResult();
    if (op.getType() != newResultType) {
      result = rewriter.create<mlir::stablehlo::ReshapeOp>(
          op.getLoc(), op.getType(), newOp.getResult());
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ScatterInt64Indices final
    : OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getScatterIndices();
    auto indicesTy = indices.getType();
    auto indicesETy = indicesTy.getElementType();
    if (indicesETy.isInteger(32)) {
      return rewriter.notifyMatchFailure(op, "already has i32 index type");
    }

    if (!indicesTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "cannot validate legal size");
    }

    uint64_t maxSize = std::numeric_limits<int32_t>::max();
    if (indicesETy.getIntOrFloatBitWidth() > 32) {
      for (int i = 0, s = indicesTy.getRank(); i < s; ++i) {
        if (indicesTy.getDimSize(i) > maxSize) {
          return rewriter.notifyMatchFailure(op, "index may exceed i32 max");
        }
      }
    }

    indices = rewriter.create<mlir::stablehlo::ConvertOp>(
        op.getLoc(), indicesTy.clone(rewriter.getI32Type()), indices);

    auto newScatter = rewriter.create<mlir::stablehlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices,
        op.getUpdates(), op.getScatterDimensionNumbers(),
        op.getIndicesAreSorted(), op.getUniqueIndices());

    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());

    return success();
  }
};

// If the indices tensor has an implicit index vector dim we expand and make it
// an explicit dim.
struct ScatterImplicitIndex final
    : OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto dimNumbers = op.getScatterDimensionNumbers();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    Value indices = op.getScatterIndices();
    auto indicesTy = llvm::cast<ShapedType>(indices.getType());

    // Check indices vector has an implicit dim.
    if (indexVectorDim != indicesTy.getRank()) {
      return rewriter.notifyMatchFailure(op, "no implicit index dim");
    }

    // Materialize the implicit indices dim.
    SmallVector<ReassociationExprs> reassociationMap;
    reassociationMap.resize(indicesTy.getRank());
    SmallVector<int64_t> newShape;
    for (int i = 0, s = indicesTy.getRank(); i < s; ++i) {
      reassociationMap[i].push_back(rewriter.getAffineDimExpr(i));
      newShape.push_back(indicesTy.getDimSize(i));
    }
    if (!reassociationMap.empty()) {
      reassociationMap.back().push_back(
          rewriter.getAffineDimExpr(indicesTy.getRank()));
    }
    newShape.push_back(1);
    indicesTy = RankedTensorType::get(newShape, indicesTy.getElementType());
    indices = rewriter.create<tensor::ExpandShapeOp>(op.getLoc(), indicesTy,
                                                     indices, reassociationMap);

    auto newScatter = rewriter.create<mlir::stablehlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices,
        op.getUpdates(), dimNumbers, op.getIndicesAreSorted(),
        op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

struct ScatterImplicitBatch final
    : OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  static Value addUnitBatchDim(Location loc, Value value,
                               PatternRewriter &rewriter) {
    auto valueTy = cast<ShapedType>(value.getType());
    if (!valueTy.hasRank())
      return nullptr;

    // Materialize the implicit indices dim.
    SmallVector<ReassociationExprs> reassociationMap(valueTy.getRank());
    if (!reassociationMap.empty()) {
      reassociationMap.front().push_back(rewriter.getAffineDimExpr(0));
    }

    SmallVector<int64_t> newShape = {1};
    for (int i = 0, s = valueTy.getRank(); i < s; i++) {
      reassociationMap[i].push_back(rewriter.getAffineDimExpr(i + 1));
      newShape.push_back(valueTy.getDimSize(i));
    }

    valueTy = RankedTensorType::get(newShape, valueTy.getElementType());
    return rewriter.create<tensor::ExpandShapeOp>(loc, valueTy, value,
                                                  reassociationMap);
  }

  LogicalResult matchAndRewrite(mlir::stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto dimNumbers = op.getScatterDimensionNumbers();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    auto indices = llvm::cast<Value>(op.getScatterIndices());
    auto indicesTy = llvm::dyn_cast<RankedTensorType>(indices.getType());

    // Check whether indices has no batch dimension.
    if (!indicesTy)
      return failure();
    if (indicesTy.getRank() != 1 || indexVectorDim != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "no implicit batch dimension to add.");
    }

    indices = addUnitBatchDim(op.getLoc(), indices, rewriter);
    if (!indices) {
      return rewriter.notifyMatchFailure(
          op, "Unable to add implicit batch dim to indices.");
    }

    llvm::SmallVector<int64_t> newUpdateWindowDims;
    for (auto dim : dimNumbers.getUpdateWindowDims()) {
      // Batch dimension is inserted at the start so window dimensions are shift
      // forwards.
      newUpdateWindowDims.push_back(dim + 1);
    }

    llvm::SmallVector<Value> updates;
    for (Value update : op.getUpdates()) {
      update = addUnitBatchDim(op.getLoc(), update, rewriter);
      if (!update) {
        return rewriter.notifyMatchFailure(
            op, "Unable to add implicit batch dim to update.");
      }
      updates.push_back(update);
    }

    auto newDimNumbers = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
        op.getContext(), newUpdateWindowDims,
        dimNumbers.getInsertedWindowDims(),
        dimNumbers.getScatterDimsToOperandDims(),
        dimNumbers.getIndexVectorDim() + 1);

    auto newScatter = rewriter.create<mlir::stablehlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices, updates,
        newDimNumbers, op.getIndicesAreSorted(), op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

struct ScatterCollapseBatch final
    : OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  static Value collapseBatchDims(Location loc, Value value, int64_t batchCount,
                                 PatternRewriter &rewriter) {
    auto valueTy = dyn_cast<ShapedType>(value.getType());
    if (!valueTy)
      return nullptr;

    SmallVector<ReassociationExprs> reassociationMap(1);
    reassociationMap.reserve(valueTy.getRank() - batchCount + 1);
    int64_t batchSize = 1;
    for (int i = 0, s = batchCount; i < s; i++) {
      reassociationMap.front().push_back(rewriter.getAffineDimExpr(i));
      bool isDynamic =
          valueTy.isDynamicDim(i) || ShapedType::isDynamic(batchSize);
      batchSize =
          isDynamic ? ShapedType::kDynamic : valueTy.getDimSize(i) * batchSize;
    }

    SmallVector<int64_t> newShape = {batchSize};
    for (int i = batchCount, s = valueTy.getRank(); i < s; i++) {
      reassociationMap.push_back({rewriter.getAffineDimExpr(i)});
      newShape.push_back(valueTy.getDimSize(i));
    }

    valueTy = RankedTensorType::get(newShape, valueTy.getElementType());
    return rewriter.create<tensor::CollapseShapeOp>(loc, valueTy, value,
                                                    reassociationMap);
  }

  LogicalResult matchAndRewrite(mlir::stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto dimNumbers = op.getScatterDimensionNumbers();
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    auto indices = llvm::cast<Value>(op.getScatterIndices());
    auto indicesTy = llvm::cast<ShapedType>(indices.getType());
    auto updatedWindowDims = dimNumbers.getUpdateWindowDims();

    if (!indicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "indices has unknown rank");
    }

    // Check for an explicit index dimension.
    if (indexVectorDim != indicesTy.getRank() - 1) {
      return rewriter.notifyMatchFailure(op, "no explicit indices dimension");
    }

    // Check that there are multiple batch dimensions.
    if (indicesTy.getRank() < 3) {
      return rewriter.notifyMatchFailure(op, "no multiple batch dimensions");
    }

    const int64_t batchCount = indicesTy.getRank() - 1;
    for (auto it : llvm::enumerate(updatedWindowDims)) {
      if (it.index() != it.value() - batchCount) {
        return rewriter.notifyMatchFailure(
            op, "update windows should be at the end.");
      }
    }

    indices = collapseBatchDims(op.getLoc(), indices, batchCount, rewriter);
    if (!indices) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot collapse indices batch dims");
    }

    llvm::SmallVector<Value> updates;
    for (Value update : op.getUpdates()) {
      update = collapseBatchDims(op.getLoc(), update, batchCount, rewriter);
      if (!update) {
        return rewriter.notifyMatchFailure(op,
                                           "cannot collapse update batch dims");
      }
      updates.push_back(update);
    }

    llvm::SmallVector<int64_t> newUpdatedWindowDims;
    for (auto dim : updatedWindowDims) {
      newUpdatedWindowDims.push_back(dim - batchCount + 1);
    }

    auto newDimNumbers = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
        op.getContext(), newUpdatedWindowDims,
        dimNumbers.getInsertedWindowDims(),
        dimNumbers.getScatterDimsToOperandDims(),
        /*indexVectorDim=*/1);

    auto newScatter = rewriter.create<mlir::stablehlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices, updates,
        newDimNumbers, op.getIndicesAreSorted(), op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

// Ensure the batch dimensions of both the indices and updates are the first
// dimensions. If they are not, transpose them to the start.
struct ScatterBatchFirst final : OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto dimNumbers = op.getScatterDimensionNumbers();

    // If the index vector dim is not implicitly or explicitly at the end
    // we need to transpose the batch dimensions to the start.
    Value indices = op.getScatterIndices();
    auto indicesTy = llvm::cast<ShapedType>(indices.getType());
    auto indexVectorDim = dimNumbers.getIndexVectorDim();
    if (indexVectorDim < indicesTy.getRank() - 1) {
      llvm::SmallVector<int64_t> perm;
      perm.reserve(indicesTy.getRank());
      for (int i = 0, s = indicesTy.getRank(); i < s; ++i) {
        if (i != indexVectorDim)
          perm.push_back(i);
      }

      if (perm.size() < indicesTy.getRank())
        perm.push_back(indexVectorDim);

      llvm::SmallVector<int64_t> newShape;
      for (int i = 0, s = perm.size(); i < s; ++i) {
        newShape.push_back(indicesTy.getDimSize(perm[i]));
      }

      indices = builder.create<mlir::stablehlo::TransposeOp>(
          indicesTy.clone(newShape), indices, builder.getI64TensorAttr(perm));
      indicesTy = llvm::cast<RankedTensorType>(indices.getType());
      indexVectorDim = indicesTy.getRank() - 1;
    }

    // Compute the permutation require to transpose the batch dimensions to
    // the beginning.
    auto updates = op.getUpdates();
    auto updates0 = updates.front();
    auto updates0Ty = llvm::cast<ShapedType>(updates0.getType());
    auto updatedWindowDims = dimNumbers.getUpdateWindowDims();

    // Determine which dimensions are batch dimensions.
    llvm::SmallVector<bool> isBatch(updates0Ty.getRank(), true);
    for (int i = 0, s = updatedWindowDims.size(); i < s; ++i)
      isBatch[updatedWindowDims[i]] = false;

    // Permute batch dimensions to the start of the update tensor.
    llvm::SmallVector<int64_t> updatePerm;
    updatePerm.reserve(updates0Ty.getRank());
    for (int i = 0, s = isBatch.size(); i < s; ++i)
      if (isBatch[i])
        updatePerm.push_back(i);
    updatePerm.append(updatedWindowDims.begin(), updatedWindowDims.end());

    llvm::SmallVector<int64_t> newUpdatedWindowDims;
    int64_t batchCount = updates0Ty.getRank() - updatedWindowDims.size();
    for (int i = batchCount, s = updates0Ty.getRank(); i < s; i++)
      newUpdatedWindowDims.push_back(i);

    bool indicesChanged = indices != op.getScatterIndices();
    bool updatesChanged =
        llvm::any_of(llvm::enumerate(updatePerm),
                     [](auto it) { return it.index() != it.value(); });
    llvm::SmallVector<Value> newUpdates(updates.begin(), updates.end());
    if (updatesChanged) {
      for (Value &update : newUpdates) {
        auto updateTy = llvm::cast<ShapedType>(update.getType());
        llvm::SmallVector<int64_t> newShape;
        newShape.reserve(updateTy.getRank());
        for (int i = 0, s = updatePerm.size(); i < s; i++)
          newShape.push_back(updateTy.getDimSize(updatePerm[i]));
        update = builder.create<mlir::stablehlo::TransposeOp>(
            updateTy.clone(newShape), update,
            builder.getI64TensorAttr(updatePerm));
      }
    }

    if (!indicesChanged && !updatesChanged)
      return rewriter.notifyMatchFailure(
          op, "batch dimensions are already leading");

    auto newDimNumbers = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
        op.getContext(), newUpdatedWindowDims,
        dimNumbers.getInsertedWindowDims(),
        dimNumbers.getScatterDimsToOperandDims(),
        /*indexVectorDim=*/indexVectorDim);

    auto newScatter = rewriter.create<mlir::stablehlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(), indices, newUpdates,
        newDimNumbers, op.getIndicesAreSorted(), op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

// stablehlo.scatter can materialize a unit dimension at both indexed dimensions
// or at unary dimensions in the destination matrix. linalg_ext.scatter only
// allows unit dimensions at indexed dimensions. This pattern inserts all
// unary dimensions that are not index dimensions to be compatible with
// linalg_ext.scatter.
//
// If converts an stablehlo.scatter as below:
//  %result = "stablehlo.scatter"(...) ({
//    indices_are_sorted = true,
//    scatter_dimension_numbers = #stablehlo.scatter<
//            update_window_dims = [1],
//            inserted_window_dims = [0, 2],
//            scatter_dims_to_operand_dims = [0],
//            index_vector_dim = 1>,
//    unique_indices = true} :
//        (tensor<5x4x1xi32>, tensor<1x1xi32>, tensor<1x4xi32>)
//
// To:
//  %result = "stablehlo.scatter"(...) ({
//    indices_are_sorted = true,
//    scatter_dimension_numbers = #stablehlo.scatter<
//            update_window_dims = [1, 2],
//            inserted_window_dims = [0],
//            scatter_dims_to_operand_dims = [0],
//            index_vector_dim = 1>,
//     unique_indices = true} :
//        (tensor<5x4x1xi32>, tensor<1x1xi32>, tensor<1x4x1xi32>)
//  return %0 : tensor<5x4x1xi32>
struct ScatterMaterializeInsertedDim final
    : OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getScatterIndices();
    auto operand = op.getInputs().front();
    auto indicesTy = llvm::cast<ShapedType>(indices.getType());
    auto operandTy = llvm::cast<ShapedType>(operand.getType());

    if (!operandTy.hasRank() || !indicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "operand/indices have no rank");
    }

    auto dimNumbers = op.getScatterDimensionNumbers();
    auto updateDims = dimNumbers.getUpdateWindowDims();

    if (indicesTy.getRank() != 2 || dimNumbers.getIndexVectorDim() != 1) {
      return rewriter.notifyMatchFailure(
          op, "indices is not of shape [batch, indices]");
    }

    if (!updateDims.empty() && updateDims.front() == 0) {
      return rewriter.notifyMatchFailure(
          op, "updates is not of shape [batch, ...]");
    }

    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    llvm::SmallVector<bool> isIndexDim(operandTy.getRank(), false);
    for (auto val : scatterDimsToOperandDims) {
      isIndexDim[val] = true;
    }

    int64_t firstNonIndex = 0;
    for (int64_t s = scatterDimsToOperandDims.size(); firstNonIndex < s;
         ++firstNonIndex) {
      if (!isIndexDim[firstNonIndex])
        break;
    }

    llvm::SmallVector<bool> isInsertDims(operandTy.getRank(), false);
    for (auto val : dimNumbers.getInsertedWindowDims()) {
      isInsertDims[val] = true;
    }

    int64_t frontInsertedDims = 0;
    for (; frontInsertedDims < firstNonIndex; ++frontInsertedDims) {
      if (!isInsertDims[frontInsertedDims]) {
        break;
      }
    }

    llvm::ArrayRef<bool> toInsertDims =
        llvm::ArrayRef<bool>(isInsertDims).drop_front(frontInsertedDims);
    if (!llvm::any_of(toInsertDims, [](auto d) { return d; })) {
      return rewriter.notifyMatchFailure(op, "no dimensions to insert");
    }

    // Create a reassociation map that starts with the batch dims.
    SmallVector<ReassociationExprs> reassociationMap;
    reassociationMap.push_back({rewriter.getAffineDimExpr(0)});

    for (auto it : llvm::enumerate(llvm::ArrayRef<bool>(toInsertDims))) {
      if (!it.value())
        reassociationMap.push_back({});
      reassociationMap.back().push_back(
          rewriter.getAffineDimExpr(it.index() + 1));
    }

    llvm::SmallVector<Value> expandedUpdates;
    for (auto update : op.getUpdates()) {
      auto updatesTy = llvm::cast<ShapedType>(update.getType());

      llvm::SmallVector<int64_t> newShape;
      for (int i = 0, s = reassociationMap.size(); i < s; ++i) {
        newShape.push_back(updatesTy.getDimSize(i));
        for (int j = 1, s = reassociationMap[i].size(); j < s; ++j) {
          newShape.push_back(1);
        }
      }

      Value expandUpdate = rewriter.create<tensor::ExpandShapeOp>(
          op.getLoc(),
          RankedTensorType::get(newShape, updatesTy.getElementType()), update,
          reassociationMap);
      expandedUpdates.push_back(expandUpdate);
    }

    llvm::SmallVector<int64_t> newUpdatedWindowDims(toInsertDims.size());
    llvm::SmallVector<int64_t> newInsertedWindowDims(frontInsertedDims);
    std::iota(newUpdatedWindowDims.begin(), newUpdatedWindowDims.end(), 1);
    std::iota(newInsertedWindowDims.begin(), newInsertedWindowDims.end(), 0);

    auto newDimNumbers = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
        op.getContext(), newUpdatedWindowDims, newInsertedWindowDims,
        dimNumbers.getScatterDimsToOperandDims(),
        /*indexVectorDim=*/1);

    auto newScatter = rewriter.create<mlir::stablehlo::ScatterOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs(),
        op.getScatterIndices(), expandedUpdates, newDimNumbers,
        op.getIndicesAreSorted(), op.getUniqueIndices());
    Region &region = newScatter.getUpdateComputation();
    rewriter.cloneRegionBefore(op.getUpdateComputation(), region, region.end());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
  }
};

// Traverse upward past common operations to see if the value came from a
// boolean tensor.
bool isFromBool(Value val) {
  while (true) {
    Operation *op = val.getDefiningOp();
    if (!op)
      return false;

    if (auto convertOp = dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
      auto inTy = llvm::cast<ShapedType>(convertOp.getOperand().getType());
      if (inTy.getElementType().isInteger(1)) {
        return true;
      }
      val = convertOp.getOperand();
      continue;
    }

    if (isa<mlir::stablehlo::DynamicBroadcastInDimOp>(op) ||
        isa<mlir::stablehlo::BroadcastInDimOp>(op) ||
        isa<mlir::stablehlo::BroadcastOp>(op)) {
      val = op->getOperand(0);
      continue;
    }

    return false;
  }
}

// Mul of non-finite values (e.g. NaN, inf) and 0.0 produce 0.0 in StableHLO.
// For linalg we need to convert these to select operations.
struct MulCastOfBool final : OpRewritePattern<mlir::stablehlo::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ShapedType>(op.getType());
    if (!isa<FloatType>(resultTy.getElementType()))
      return failure();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    bool lhsIsBool = isFromBool(lhs);
    bool rhsIsBool = isFromBool(rhs);

    if (lhsIsBool == rhsIsBool)
      return failure();
    if (rhsIsBool)
      std::swap(lhs, rhs);

    Type eType = resultTy.getElementType();
    auto lhsTy = cast<ShapedType>(lhs.getType());
    Value lhsBool = rewriter.create<mlir::stablehlo::ConvertOp>(
        op.getLoc(), lhsTy.clone(rewriter.getIntegerType(1)), lhs);
    Value zero = rewriter.create<mlir::stablehlo::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(RankedTensorType::get({}, eType),
                                            rewriter.getZeroAttr(eType)));

    auto lhsShape = rewriter.create<shape::ShapeOfOp>(
        op.getLoc(),
        RankedTensorType::get({lhsTy.getRank()}, rewriter.getIndexType()), lhs);

    int64_t resultRank = resultTy.getRank();
    auto broadcast = [&](Value value) -> Value {
      auto valueTy = cast<ShapedType>(value.getType());
      auto newTy =
          RankedTensorType::get(resultTy.getShape(), valueTy.getElementType());
      if (valueTy == newTy)
        return value;
      auto dimensions = llvm::to_vector(
          llvm::seq<int64_t>(resultRank - valueTy.getRank(), resultRank));
      return rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
          op.getLoc(), newTy, value, lhsShape,
          rewriter.getI64TensorAttr(dimensions));
    };

    zero = broadcast(zero);

    rewriter.replaceOpWithNewOp<mlir::stablehlo::SelectOp>(op, resultTy,
                                                           lhsBool, rhs, zero);
    return success();
  }
};

// Generates Gaussian noise with uniform random generator based on Box-Muller
// transform.
struct ExpandRngNormal final : OpRewritePattern<mlir::stablehlo::RngOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::RngOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRngDistribution() != mlir::stablehlo::RngDistribution::NORMAL)
      return failure();

    auto resTy = dyn_cast<RankedTensorType>(op.getType());
    // We can support static shapes, but it's easier to implement Box-Muller
    // transform if we know the number of elements.
    if (!resTy || !resTy.hasStaticShape())
      return failure();

    // The algorithm requires even numbers and will generate pairs.
    auto numElems = resTy.getNumElements();
    if (numElems & 1)
      numElems++;
    auto halfNumElems = numElems / 2;

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Explicitly set the seed to 0, so we have stateless generator. This is not
    // a hard limit. Random generator is still a new topic, and we start with
    // stateless random generator.
    std::mt19937 rng{0};
    std::uniform_real_distribution<> runif(0.0, 1.0);
    SmallVector<float> sqrtValues(halfNumElems), cosValues(halfNumElems),
        sinValues(halfNumElems);
    for (auto i : llvm::seq<unsigned>(0, numElems / 2)) {
      static constexpr float kEpsilon = std::numeric_limits<float>::epsilon();
      static constexpr float kTwoPi = static_cast<float>(2.0 * M_PI);
      float u1, u2;
      do {
        u1 = runif(rng);
        u2 = runif(rng);
      } while (u1 <= kEpsilon);
      sqrtValues[i] = -2.0 * log(u1);
      cosValues[i] = cos(kTwoPi * u2);
      sinValues[i] = sin(kTwoPi * u2);
    }

    // mag = sigma * sqrt(-2.0 * log(u1));
    Value mag = getF32Const(b, /*shapes=*/{halfNumElems}, sqrtValues);
    Value sigma = b.create<mlir::stablehlo::BroadcastOp>(
        mag.getType(), op.getB(), make1DElementsAttr(b, halfNumElems));
    mag = b.create<mlir::stablehlo::MulOp>(
        sigma, b.create<mlir::stablehlo::SqrtOp>(mag));

    // z0 = mag * cos(two_pi * u2) + mu;
    // z1 = mag * sin(two_pi * u2) + mu;
    Value mu = b.create<mlir::stablehlo::BroadcastOp>(
        mag.getType(), op.getA(), make1DElementsAttr(b, halfNumElems));
    Value z0 = getF32Const(b, /*shapes=*/{halfNumElems}, cosValues);
    z0 = b.create<mlir::stablehlo::MulOp>(mag, z0);
    z0 = b.create<mlir::stablehlo::AddOp>(z0, mu);
    Value z1 = getF32Const(b, /*shapes=*/{halfNumElems}, sinValues);
    z1 = b.create<mlir::stablehlo::MulOp>(mag, z1);
    z1 = b.create<mlir::stablehlo::AddOp>(z1, mu);

    Value res = b.create<mlir::stablehlo::ConcatenateOp>(
        ValueRange{z0, z1}, b.getI64IntegerAttr(0));
    if (numElems != resTy.getNumElements()) {
      OpFoldResult zero = b.getIndexAttr(0);
      OpFoldResult one = b.getIndexAttr(1);
      OpFoldResult size = b.getIndexAttr(resTy.getNumElements());
      res = b.create<tensor::ExtractSliceOp>(res, zero, size, one);
    }
    if (resTy.getRank() != 1) {
      res = b.create<mlir::stablehlo::ReshapeOp>(resTy, res);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

// clang-format off
//
// Reorder BroadcastInDimOp and N-ary elementwise op.
//
// Rewrites the following pattern (take binary elementwise op as example)
//
// %bcastx = "stablehlo.broadcast_in_dim"(%x) {broadcast_dimensions = %[[BCAST_DIMS]]} : (%[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
// %bcasty = "stablehlo.broadcast_in_dim"(%y) {broadcast_dimensions = %[[BCAST_DIMS]]} : (%[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
// %result = "BinaryElementwiseOpT"(%bcastx, %bcasty) : (%[[SHAPE_AFTER_BCAST]], %[[SHAPE_AFTER_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
//
// into
//
// %z = "BinaryElementwiseOpT"(%x, %y) : (%[[SHAPE_BEFORE_BCAST]], %[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_BEFORE_BCAST]]
// %result = "stablehlo.broadcast_in_dim"(%z) {broadcast_dimensions = %[[BCAST_DIMS]]} : (%[[SHAPE_BEFORE_BCAST]]) -> %[[SHAPE_AFTER_BCAST]]
//
// clang-format on
template <typename ElementwiseOpT>
struct ReorderBroadcastInDimOpAndElementwiseOp final
    : OpRewritePattern<ElementwiseOpT> {
  using OpRewritePattern<ElementwiseOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOpT op,
                                PatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    assert(operation->getNumOperands() >= 1 && operation->getNumResults() == 1);

    // Verify if all operands are from BroadcastInDimOp and its
    // broadcast_dimensions is the same.
    llvm::SmallVector<mlir::stablehlo::BroadcastInDimOp, 2> bcastOps;
    for (auto operand : operation->getOperands()) {
      if (auto bcastOp =
              operand.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        bcastOps.push_back(bcastOp);
      } else {
        return failure();
      }
    }

    if (llvm::any_of(bcastOps,
                     [&bcastOps](mlir::stablehlo::BroadcastInDimOp bcastOp) {
                       return bcastOp.getBroadcastDimensions() !=
                              bcastOps[0].getBroadcastDimensions();
                     })) {
      return failure();
    }

    // Verify if all operands of BroadcastInDimOp are of same type and have
    // static shape.
    auto bcastOperandType =
        dyn_cast<ShapedType>(bcastOps[0].getOperand().getType());
    llvm::SmallVector<Value, 2> bcastOperands;
    for (auto bcastOp : bcastOps) {
      auto bcastOperand = bcastOp.getOperand();
      auto type = dyn_cast<ShapedType>(bcastOperand.getType());
      if (!type || !type.hasStaticShape() || type != bcastOperandType) {
        return failure();
      }
      bcastOperands.push_back(bcastOperand);
    }

    // Some elementwise ops, mlir::stablehlo::RealOp for example, do not have
    // SameOperandsAndResultType trait, so resultType might be different
    // from bcastOperandType.
    auto elementType = getElementTypeOrSelf(op.getResult());
    auto resultShape = bcastOperandType.getShape();
    auto resultType = RankedTensorType::get(resultShape, elementType);

    Value result =
        rewriter.create<ElementwiseOpT>(op.getLoc(), resultType, bcastOperands);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
        op, op.getType(), result, bcastOps[0].getBroadcastDimensions());

    llvm::SetVector<Operation *> opsToErase;
    for (auto bcastOp : bcastOps) {
      if (bcastOp.getOperation()->use_empty()) {
        opsToErase.insert(bcastOp.getOperation());
      }
    }

    for (auto opToErase : opsToErase) {
      rewriter.eraseOp(opToErase);
    }
    return success();
  }
};

// Identifies cases where a dense operation has inputs that come from widening
// operations. For instance, a dot product widening from FP16 to FP32 is better
// to have the casting operation fused into the dot operation. This decreases
// the loading required during a dense computation.
template <class Op>
struct FuseWidenOperands final : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      auto convertOp =
          dyn_cast_or_null<mlir::stablehlo::ConvertOp>(operand.getDefiningOp());
      if (convertOp) {
        auto inputType = getElementTypeOrSelf(convertOp.getOperand().getType());
        auto castedType = getElementTypeOrSelf(convertOp.getResult().getType());
        if (!isa<FloatType, IntegerType>(inputType) ||
            !isa<FloatType, IntegerType>(castedType)) {
          return rewriter.notifyMatchFailure(
              op, "non-integer or floating point type");
          ;
        }

        if (inputType.getIntOrFloatBitWidth() <
            castedType.getIntOrFloatBitWidth()) {
          operands.push_back(convertOp.getOperand());
          continue;
        }
      }
      operands.push_back(operand);
    }

    if (llvm::all_of(
            llvm::zip_equal(operands, op->getOperands()),
            [](auto pair) { return std::get<0>(pair) == std::get<1>(pair); }))
      return failure();

    rewriter.replaceOpWithNewOp<Op>(op, op->getResultTypes(), operands,
                                    op->getAttrs());
    return success();
  }
};

struct DotToMul final : OpRewritePattern<mlir::stablehlo::DotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getType());

    if (!lhsTy || !rhsTy) {
      return rewriter.notifyMatchFailure(op, "lhs and rhs must be ranked");
    }

    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "lhs and rhs must be rank-2");
    }

    if (lhsTy.getDimSize(1) != 1)
      return failure();

    // Dynamically compute the shape of the result of the DotOp by querying
    // the 0-th dimensions, of the left, and the 1st dimension of the right.
    // Concatenating them togething to make the final shape.
    Value batchSize = rewriter.create<mlir::stablehlo::GetDimensionSizeOp>(
        op.getLoc(), lhs, rewriter.getI64IntegerAttr(0));
    Value batchSize1 = rewriter.create<mlir::stablehlo::ReshapeOp>(
        op.getLoc(), RankedTensorType::get({1}, rewriter.getI32Type()),
        batchSize);

    Value featureSize = rewriter.create<mlir::stablehlo::GetDimensionSizeOp>(
        op.getLoc(), rhs, rewriter.getI64IntegerAttr(1));
    Value featureSize1 = rewriter.create<mlir::stablehlo::ReshapeOp>(
        op.getLoc(), RankedTensorType::get({1}, rewriter.getI32Type()),
        featureSize);

    Value outSize = rewriter.create<mlir::stablehlo::ConcatenateOp>(
        op.getLoc(), RankedTensorType::get({2}, rewriter.getI32Type()),
        ValueRange{batchSize1, featureSize1}, rewriter.getI64IntegerAttr(0));

    lhs = rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
        op.getLoc(), resultTy.clone(lhsTy.getElementType()), lhs, outSize,
        rewriter.getI64TensorAttr({0, 1}));

    rhs = rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
        op.getLoc(), resultTy.clone(rhsTy.getElementType()), rhs, outSize,
        rewriter.getI64TensorAttr({0, 1}));

    auto computeETy = lhsTy.getElementType();
    if (computeETy.getIntOrFloatBitWidth() < rhsTy.getElementTypeBitWidth())
      computeETy = rhsTy.getElementType();
    if (computeETy.getIntOrFloatBitWidth() < resultTy.getElementTypeBitWidth())
      computeETy = resultTy.getElementType();

    auto computeTy = resultTy.clone(computeETy);

    rhs = rewriter.create<mlir::stablehlo::ConvertOp>(op.getLoc(), computeTy,
                                                      rhs);
    lhs = rewriter.create<mlir::stablehlo::ConvertOp>(op.getLoc(), computeTy,
                                                      lhs);

    auto result = rewriter.create<mlir::stablehlo::MulOp>(
        op.getLoc(), resultTy.clone(computeETy), lhs, rhs);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvertOp>(op, resultTy,
                                                            result);
    return success();
  }
};

// Rewrite RngBitGenerator with f32 return type to instead generate the same
// number of i32 outputs, then BitcastConvert to return f32.
struct RngBitcastFloat final
    : OpRewritePattern<mlir::stablehlo::RngBitGeneratorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::RngBitGeneratorOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto resultTy = dyn_cast<RankedTensorType>(op.getType(1));
    auto stateTy = dyn_cast<RankedTensorType>(op.getType(0));

    if (!isa<FloatType>(resultTy.getElementType())) {
      return failure();
    }

    llvm::SmallVector<Type> castedTypes;
    castedTypes.push_back(stateTy);
    castedTypes.push_back(resultTy.clone(rewriter.getI32Type()));

    TypeRange castedTypeRange = TypeRange{castedTypes};

    auto resultOp = rewriter.create<mlir::stablehlo::RngBitGeneratorOp>(
        op.getLoc(), castedTypeRange, op.getRngAlgorithm(), op.getOperand());

    auto casted = rewriter.create<mlir::stablehlo::BitcastConvertOp>(
        resultOp.getLoc(), resultTy, resultOp.getResult(1));

    llvm::SmallVector<Value> results;
    results.push_back(resultOp.getResult(0));
    results.push_back(casted);

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ZeroConcat final : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type || !type.hasStaticShape())
      return failure();

    uint64_t axis = op.getDimension();
    OperandRange origInputs = op.getInputs();
    SmallVector<Value> nonzeroInputs;
    for (auto input : origInputs) {
      auto type = dyn_cast<RankedTensorType>(input.getType());
      ArrayRef<int64_t> shape = type.getShape();
      if (axis > shape.size())
        return failure();

      if (shape[axis] != 0)
        nonzeroInputs.push_back(input);
    }

    if (nonzeroInputs.size() == origInputs.size())
      return failure();

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConcatenateOp>(
        op, nonzeroInputs, /*dimension=*/axis);
    return success();
  }
};

// Similar to DotIsMul, this finds the case where a dot general
// can be represented using a mul operation. This includes possibly making
// an implicit cast explicit prior the mul.
struct DotGeneralIsMul final : OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = cast<Value>(op.getLhs());
    auto rhs = cast<Value>(op.getRhs());
    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getType());
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (!lhsTy || !rhsTy || !resultTy)
      return failure();

    auto dNums = op.getDotDimensionNumbers();
    auto batchDimsL = dNums.getLhsBatchingDimensions();
    auto batchDimsR = dNums.getRhsBatchingDimensions();
    auto contractDimsL = dNums.getLhsContractingDimensions();
    auto contractDimsR = dNums.getRhsContractingDimensions();

    llvm::SmallVector<bool> isLhsParallelDim(lhsTy.getRank(), true);
    llvm::SmallVector<bool> isRhsParallelDim(rhsTy.getRank(), true);

    for (auto dim : batchDimsL)
      isLhsParallelDim[dim] = false;
    for (auto dim : batchDimsR)
      isRhsParallelDim[dim] = false;
    for (auto dim : contractDimsL)
      isLhsParallelDim[dim] = false;
    for (auto dim : contractDimsR)
      isRhsParallelDim[dim] = false;

    for (auto dim : contractDimsL) {
      if (lhsTy.getDimSize(dim) != 1) {
        return rewriter.notifyMatchFailure(op, "Non unit contract dimensions");
      }
    }

    // Generate the permutation matrix to order BatchDims, ParallelDims,
    // ContractDims.
    llvm::SmallVector<int64_t> permLhs;
    llvm::SmallVector<int64_t> permRhs;
    permLhs.append(batchDimsL.begin(), batchDimsL.end());
    permRhs.append(batchDimsR.begin(), batchDimsR.end());

    for (auto [idx, value] : llvm::enumerate(isLhsParallelDim)) {
      if (value)
        permLhs.push_back(idx);
    }

    for (auto [idx, value] : llvm::enumerate(isRhsParallelDim)) {
      if (value)
        permRhs.push_back(idx);
    }

    llvm::append_range(permLhs, contractDimsL);
    llvm::append_range(permRhs, contractDimsR);

    // Determine the transpose shape based on the generate permutations.
    llvm::SmallVector<int64_t> lhsTransposeShape;
    llvm::SmallVector<int64_t> rhsTransposeShape;
    for (auto dim : permLhs)
      lhsTransposeShape.push_back(lhsTy.getDimSize(dim));
    for (auto dim : permRhs)
      rhsTransposeShape.push_back(rhsTy.getDimSize(dim));

    // Transpose the left hand side and the right hand side.
    lhs = builder.create<mlir::stablehlo::TransposeOp>(
        RankedTensorType::get(lhsTransposeShape, lhsTy.getElementType()), lhs,
        builder.getI64TensorAttr(permLhs));
    lhsTy = llvm::cast<RankedTensorType>(lhs.getType());

    rhs = builder.create<mlir::stablehlo::TransposeOp>(
        RankedTensorType::get(rhsTransposeShape, rhsTy.getElementType()), rhs,
        builder.getI64TensorAttr(permRhs));
    rhsTy = llvm::cast<RankedTensorType>(rhs.getType());

    auto dimI32Ty = RankedTensorType::get({1}, builder.getI32Type());

    // Drop all of the non-concat dimensions from the lhs.
    llvm::SmallVector<Value> lhsReshapeDims;
    for (int i = 0, s = lhsTy.getRank() - contractDimsL.size(); i < s; i++) {
      Value dim = builder.create<mlir::stablehlo::GetDimensionSizeOp>(lhs, i);
      lhsReshapeDims.push_back(
          builder.create<mlir::stablehlo::ReshapeOp>(dimI32Ty, dim));
    }
    Value lhsDynShape = builder.create<mlir::stablehlo::ConcatenateOp>(
        RankedTensorType::get({static_cast<int64_t>(lhsReshapeDims.size())},
                              builder.getI32Type()),
        lhsReshapeDims, 0);
    lhsTy =
        RankedTensorType::get(lhsTy.getShape().drop_back(contractDimsL.size()),
                              lhsTy.getElementType());
    lhs = builder.create<mlir::stablehlo::DynamicReshapeOp>(lhsTy, lhs,
                                                            lhsDynShape);

    // Drop all of the non concat dimensions from the rhs.
    llvm::SmallVector<Value> rhsReshapeDims;
    for (int i = 0, s = rhsTy.getRank() - contractDimsR.size(); i < s; i++) {
      Value dim = builder.create<mlir::stablehlo::GetDimensionSizeOp>(rhs, i);
      rhsReshapeDims.push_back(
          builder.create<mlir::stablehlo::ReshapeOp>(dimI32Ty, dim));
    }
    Value rhsDynShape = builder.create<mlir::stablehlo::ConcatenateOp>(
        RankedTensorType::get({static_cast<int64_t>(rhsReshapeDims.size())},
                              builder.getI32Type()),
        rhsReshapeDims, 0);
    rhsTy =
        RankedTensorType::get(rhsTy.getShape().drop_back(contractDimsR.size()),
                              rhsTy.getElementType());
    rhs = builder.create<mlir::stablehlo::DynamicReshapeOp>(rhsTy, rhs,
                                                            rhsDynShape);

    // Compute the size of the output shape with dynamic shape support using the
    // lhs and rhs dimensions.
    llvm::SmallVector<Value> outputDims;
    outputDims.append(lhsReshapeDims);
    outputDims.append(rhsReshapeDims.begin() + batchDimsR.size(),
                      rhsReshapeDims.end());
    Value outputShape = builder.create<mlir::stablehlo::ConcatenateOp>(
        RankedTensorType::get({resultTy.getRank()}, builder.getI32Type()),
        outputDims, 0);

    // Broadcast the left hand side to match the expect output shape.
    llvm::SmallVector<int64_t> lhsDimMapping(lhsTy.getRank());
    std::iota(lhsDimMapping.begin(), lhsDimMapping.end(), 0);
    auto lhsBroadcastTy =
        RankedTensorType::get(resultTy.getShape(), lhsTy.getElementType());
    lhs = builder.createOrFold<mlir::stablehlo::DynamicBroadcastInDimOp>(
        lhsBroadcastTy, lhs, outputShape,
        rewriter.getI64TensorAttr(lhsDimMapping));

    // Broadcast the right hand side to match the expected output shape.
    llvm::SmallVector<int64_t> rhsDimMapping(rhsTy.getRank());
    std::iota(rhsDimMapping.begin(), rhsDimMapping.begin() + batchDimsR.size(),
              0);
    std::iota(rhsDimMapping.begin() + batchDimsR.size(), rhsDimMapping.end(),
              lhsTy.getRank());
    auto rhsBroadcastTy =
        RankedTensorType::get(resultTy.getShape(), rhsTy.getElementType());
    rhs = builder.createOrFold<mlir::stablehlo::DynamicBroadcastInDimOp>(
        rhsBroadcastTy, rhs, outputShape,
        rewriter.getI64TensorAttr(rhsDimMapping));

    lhs = builder.createOrFold<mlir::stablehlo::ConvertOp>(resultTy, lhs);
    rhs = builder.createOrFold<mlir::stablehlo::ConvertOp>(resultTy, rhs);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::MulOp>(op, resultTy, lhs, rhs);
    return success();
  }
};

struct CustomCallIsTopK final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != "TopK") {
      return rewriter.notifyMatchFailure(op, "not a TopK custom call");
    }

    if (op.getNumOperands() != 1 ||
        !(op.getNumResults() == 1 || op.getNumResults() == 2)) {
      return rewriter.notifyMatchFailure(
          op, "incorrect number of operands / results");
    }

    ArrayAttr computations = op.getCalledComputations();
    if (computations.size() != 1) {
      return rewriter.notifyMatchFailure(op,
                                         "incorrect number of computations");
    }

    SymbolRefAttr computation = dyn_cast<SymbolRefAttr>(computations[0]);
    if (!computation) {
      return rewriter.notifyMatchFailure(op, "not a ref attr");
    }

    auto operand = op.getOperand(0);
    auto operandTy = cast<ShapedType>(operand.getType());
    if (!operandTy.hasRank() || operandTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "rank-2 input not found");
    }

    ShapedType topVTy;
    ShapedType topITy;
    if (op.getNumResults() == 1) {
      if (auto tupleTy = dyn_cast<TupleType>(op.getType(0))) {
        if (tupleTy.size() != 2) {
          return rewriter.notifyMatchFailure(
              op, "tuple return does not tuple two values");
        }
        topVTy = dyn_cast<ShapedType>(tupleTy.getType(0));
        topITy = dyn_cast<ShapedType>(tupleTy.getType(1));
      }
    }

    if (op.getNumResults() == 2) {
      topVTy = dyn_cast<ShapedType>(op.getType(0));
      topITy = dyn_cast<ShapedType>(op.getType(1));
    }

    if (!topVTy || !topITy) {
      return rewriter.notifyMatchFailure(op, "unknown return type behavior");
    }

    int64_t k = topVTy.getDimSize(1);
    if (ShapedType::isDynamic(k)) {
      return rewriter.notifyMatchFailure(op, "dynamic top-k k value");
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto funcOp = dyn_cast<func::FuncOp>(moduleOp.lookupSymbol(computation));

    Block &block = funcOp.getRegion().front();
    auto stablehloCompareOp =
        dyn_cast<mlir::stablehlo::CompareOp>(block.front());
    if (!stablehloCompareOp) {
      return rewriter.notifyMatchFailure(op, "not stablehlo compare op");
    }

    auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
    if (!returnOp) {
      return rewriter.notifyMatchFailure(op, "could not find ReturnOp");
    }

    if (returnOp.getNumOperands() != 1 ||
        returnOp.getOperand(0) != stablehloCompareOp.getResult()) {
      return rewriter.notifyMatchFailure(op, "ReturnOp operand not compare op");
    }

    auto direction = stablehloCompareOp.getComparisonDirection();
    bool getTop = direction == mlir::stablehlo::ComparisonDirection::GT ||
                  direction == mlir::stablehlo::ComparisonDirection::GE;

    if (!getTop) {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported comparison direction");
    }

    auto newTopK = rewriter.create<chlo::TopKOp>(
        op.getLoc(), TypeRange{topVTy, topITy}, operand, k);

    if (op.getNumResults() == 2) {
      rewriter.replaceOp(op, newTopK.getResults());
      return success();
    }

    if (auto tupleTy = dyn_cast<TupleType>(op.getType(0))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::TupleOp>(
          op, op.getType(0), newTopK.getResults());
      return success();
    }

    return failure();
  }
};

// Recursive helper function that identifies an Iota followed by a set of
// broadcasts where the last dimension of the iota is preserved throughout.
bool isIotaOrIotaBroadcast(PatternRewriter &rewriter, Value input) {
  if (auto iotaOp =
          dyn_cast_or_null<mlir::stablehlo::IotaOp>(input.getDefiningOp())) {
    int64_t iotaDim = iotaOp.getIotaDimension();
    auto iotaLastDim = cast<ShapedType>(iotaOp.getType()).getRank() - 1;
    if (iotaDim == iotaLastDim) {
      return true;
    }

    (void)rewriter.notifyMatchFailure(iotaOp, "Iota must be on last dimension");
    return false;
  }

  if (auto broadcastOp = dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
          input.getDefiningOp())) {
    auto broadcastLastDim =
        cast<ShapedType>(broadcastOp.getType()).getRank() - 1;
    SmallVector<int64_t> broadcastDimensions = llvm::to_vector(
        broadcastOp.getBroadcastDimensions().getValues<int64_t>());
    if (broadcastDimensions.back() != broadcastLastDim) {
      (void)rewriter.notifyMatchFailure(
          broadcastOp, "Last dimension must be maintained in broadcast");
      return false;
    }
    return isIotaOrIotaBroadcast(rewriter, broadcastOp.getOperand());
  }

  return false;
}

struct IotaSortSliceIsTopK final : OpRewritePattern<mlir::stablehlo::SortOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SortOp op,
                                PatternRewriter &rewriter) const override {
    auto opOperands = op.getOperands();
    auto opResults = op.getResults();
    Value topKInput;
    if (opOperands.size() != 2 || opResults.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "Slice that maps to TopK must have exactly two inputs/outputs");
    }

    Value inputIota;
    // Check that one of the inputs is iota, assume that the other one is the
    // input.
    for (Value operand : opOperands) {
      if (isIotaOrIotaBroadcast(rewriter, operand)) {
        inputIota = operand;
      } else {
        topKInput = operand;
      }
    }

    if (!inputIota) {
      return rewriter.notifyMatchFailure(op, "Sort isn't called from Iota.");
    }

    Block &block = op.getRegion().front();
    auto stablehloCompareOp =
        dyn_cast<mlir::stablehlo::CompareOp>(block.front());
    if (!stablehloCompareOp) {
      return rewriter.notifyMatchFailure(op, "not stablehlo compare op");
    }

    auto direction = stablehloCompareOp.getComparisonDirection();
    bool getTop = direction == mlir::stablehlo::ComparisonDirection::GT ||
                  direction == mlir::stablehlo::ComparisonDirection::GE;

    if (!getTop) {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported comparison direction");
    }

    Value topV, topI;
    int64_t k;
    // Check that the output of the sort op gets fed into a slice.
    for (auto [idx, result] : llvm::enumerate(opResults)) {
      if (result.getUsers().empty())
        return rewriter.notifyMatchFailure(
            op, "Sort isn't calling into a slice op.");
      auto sliceOp =
          dyn_cast<mlir::stablehlo::SliceOp>(*result.getUsers().begin());
      if (!sliceOp) {
        return rewriter.notifyMatchFailure(
            op, "Sort isn't calling into a slice op.");
      }

      for (auto stride : sliceOp.getStrides().getValues<int64_t>()) {
        if (stride != 1) {
          return rewriter.notifyMatchFailure(
              op, "All slice strides must be 1 in order to match to TopK.");
        }
      }

      // Treat the first slice as inputs, the second as indices.
      if (idx == 0) {
        topV = sliceOp.getResult();
        SmallVector<int64_t> limitIndices =
            llvm::to_vector(sliceOp.getLimitIndices().getValues<int64_t>());
        k = limitIndices.back();
      } else {
        topI = sliceOp.getResult();
      }
    }

    auto topK = rewriter.create<chlo::TopKOp>(
        op.getLoc(), TypeRange{topV.getType(), topI.getType()}, topKInput, k);
    topV.replaceAllUsesWith(topK.getResults()[0]);
    topI.replaceAllUsesWith(topK.getResults()[1]);
    return success();
  }
};

struct ApproxTopK final : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != "ApproxTopK")
      return rewriter.notifyMatchFailure(op, "not ApproxTopK operation.");

    auto computationName =
        dyn_cast<SymbolRefAttr>(op.getCalledComputationsAttr()[0]);
    Operation *funcOp;
    for (auto parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          parent, computationName);
      if (funcOp)
        break;
    }
    if (!funcOp)
      return rewriter.notifyMatchFailure(op, "computation function not found.");

    int64_t k = cast<ShapedType>(op.getType(0)).getShape().back();
    auto input = op.getOperand(0);
    auto iota = op.getOperand(1);

    if (auto iotaOp =
            dyn_cast_or_null<mlir::stablehlo::IotaOp>(iota.getDefiningOp())) {
      int64_t iotaDim = iotaOp.getIotaDimension();
      auto iotaLastDim = cast<ShapedType>(iotaOp.getType()).getRank() - 1;
      if (iotaDim != iotaLastDim) {
        return rewriter.notifyMatchFailure(op, "Iota of last dim not found.");
      }
    }

    Block &block = funcOp->getRegion(0).front();
    auto stablehloCompareOp =
        dyn_cast<mlir::stablehlo::CompareOp>(block.front());
    if (!stablehloCompareOp) {
      return rewriter.notifyMatchFailure(op, "not stablehlo compare op");
    }

    auto returnOp = block.getTerminator();
    auto freturnOp = dyn_cast<func::ReturnOp>(returnOp);
    auto sreturnOp = dyn_cast<mlir::stablehlo::ReturnOp>(returnOp);
    if (!freturnOp && !sreturnOp) {
      return rewriter.notifyMatchFailure(op, "could not find ReturnOp");
    }

    if (returnOp->getNumOperands() != 1 ||
        returnOp->getOperand(0) != stablehloCompareOp.getResult()) {
      return rewriter.notifyMatchFailure(op, "ReturnOp operand not compare op");
    }

    auto direction = stablehloCompareOp.getComparisonDirection();
    bool getTop = direction == mlir::stablehlo::ComparisonDirection::GT ||
                  direction == mlir::stablehlo::ComparisonDirection::GE;
    if (getTop) {
      auto topK =
          rewriter.create<chlo::TopKOp>(op.getLoc(), op.getResultTypes(), input,
                                        rewriter.getI64IntegerAttr(k));
      rewriter.replaceOp(op, topK);
      return success();
    }

    bool getBottom = direction == mlir::stablehlo::ComparisonDirection::LT ||
                     direction == mlir::stablehlo::ComparisonDirection::LE;
    if (getBottom) {
      input = rewriter.create<mlir::stablehlo::NegOp>(op.getLoc(), input);
      auto topK =
          rewriter.create<chlo::TopKOp>(op.getLoc(), op.getResultTypes(), input,
                                        rewriter.getI64IntegerAttr(k));
      rewriter.replaceOp(op, topK);
      return success();
    }

    return failure();
  }
};

struct StableHLOToStableHLOPreprocessing final
    : impl::StableHLOToStableHLOPreprocessingBase<
          StableHLOToStableHLOPreprocessing> {
  using StableHLOToStableHLOPreprocessingBase::
      StableHLOToStableHLOPreprocessingBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, mlir::stablehlo::StablehloDialect,
                    chlo::ChloDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget conversionTarget(*context);
    RewritePatternSet conversionPatterns(context);
    conversionTarget
        .addLegalDialect<shape::ShapeDialect, chlo::ChloDialect,
                         mlir::stablehlo::StablehloDialect, math::MathDialect,
                         mlir::func::FuncDialect, mlir::arith::ArithDialect,
                         mlir::tensor::TensorDialect>();
    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    // General StableHLO canonicalization patterns. Run these with a high
    // benefit to enable more rewrites and avoid needless expansions that could
    // be more difficult to fold away. Note that we need to manually add these
    // because StableHLO does not provide constant folders or canonicalization
    // patterns for its ops.
    populateCanonicalizationPatterns(context, &patterns, PatternBenefit{1024});

    // TODO: Remove once we have a general contraction to matmul pass.
    populatePreprocessingEinsumToDotGeneralPatterns(context, &patterns);
    populatePreprocessingUnfuseBatchNormPatterns(context, &patterns);
    populatePreprocessingComplexPatterns(context, &patterns);
    populatePreprocessingGatherToTorchIndexSelectPatterns(context, &patterns);
    patterns.insert<ExpandRngNormal, MulCastOfBool>(context);

    // rng float conversion pattern
    patterns.insert<RngBitcastFloat>(context);

    // scatter canonicalization patterns
    patterns.insert<ScatterInt64Indices, ScatterImplicitIndex,
                    ScatterImplicitBatch, ScatterMaterializeInsertedDim,
                    ScatterCollapseBatch, ScatterBatchFirst>(context);

    // dot_general canonicalization patterns.
    populatePreprocessingDotGeneralToDotPatterns(context, &patterns);

    // TODO(jpienaar): This may be redundant with lower_general_dot. Remove if
    // so.
    patterns.insert<TransposeReshapeGenericDotGeneral>(context,
                                                       /*benefit=*/200);
    patterns.insert<DotGeneralIsMul>(context, /*benefit=*/300);

    // Fusion operations.
    // TODO: Reconsider performing this optimization in preprocessing.
    patterns.insert<FuseWidenOperands<mlir::stablehlo::DotOp>,
                    FuseWidenOperands<mlir::stablehlo::DotGeneralOp>,
                    FuseWidenOperands<mlir::stablehlo::ConvolutionOp>>(
        context,
        /*benefit=*/400);

    // Identify known custom calls and convert them to equivalent StableHLO.
    patterns.insert<CustomCallIsTopK>(context);

    // Identify an iota->sort->slice pattern that maps to TopK.
    patterns.insert<IotaSortSliceIsTopK, ApproxTopK>(context);

    // Additional canonicalizers that simplify to computationally
    // less-complex operations.
    patterns.insert<DotToMul, ZeroConcat>(context);

    // Unary elementwise op.
    patterns.insert<
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::AbsOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::CeilOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::ConvertOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::ClzOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::CosineOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::ExpOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::Expm1Op>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::FloorOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::ImagOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::IsFiniteOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::LogOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::Log1pOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::LogisticOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::NotOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::NegOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<
            mlir::stablehlo::PopulationCountOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::RealOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::RoundOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::RsqrtOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::SignOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::SineOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::SqrtOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::TanhOp>>(
        context);
    // Binary elementwise op.
    patterns.insert<
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::AddOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::Atan2Op>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::ComplexOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::DivOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::MaxOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::MinOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::MulOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::PowOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::RemOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::ShiftLeftOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<
            mlir::stablehlo::ShiftRightArithmeticOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<
            mlir::stablehlo::ShiftRightLogicalOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::SubtractOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::AndOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::OrOp>,
        ReorderBroadcastInDimOpAndElementwiseOp<mlir::stablehlo::XorOp>>(
        context);
    if (orderConvFeatures) {
      patterns.insert<ReorderConvOpInputDimensions>(context);
      patterns.insert<ReorderConvOpKernelDimensions>(context);
      patterns.insert<ReorderConvOpOutputDimensions>(context);
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
