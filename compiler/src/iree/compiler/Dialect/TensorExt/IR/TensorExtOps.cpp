// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtAttrs.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Dominance.h"

namespace mlir::iree_compiler::IREE::TensorExt {

//===----------------------------------------------------------------------===//
// Op utilities used within the IREETensorExt dialect
//===----------------------------------------------------------------------===//

static LogicalResult produceSliceErrorMsg(SliceVerificationResult result,
                                          Operation *op,
                                          RankedTensorType expectedType) {
  switch (result) {
  case SliceVerificationResult::Success:
    return success();
  case SliceVerificationResult::RankTooLarge:
    return op->emitError("expected rank to be smaller or equal to ")
           << "the other rank. ";
  case SliceVerificationResult::SizeMismatch:
    return op->emitError("expected type to be ")
           << expectedType << " or a rank-reduced version. (size mismatch) ";
  case SliceVerificationResult::ElemTypeMismatch:
    return op->emitError("expected element type to be ")
           << expectedType.getElementType();
  default:
    llvm_unreachable("unexpected slicing op verification result");
  }
}

// Verifies that |dynamicDims| contains the appropriate number of dims for all
// of the dynamic dimensions in |values|.
static LogicalResult verifyOpDynamicDims(Operation *op, ValueRange values,
                                         ValueRange dynamicDims) {
  unsigned requiredCount = 0;
  for (auto value : values) {
    if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
      requiredCount += shapedType.getNumDynamicDims();
    } else if (auto tensorType = dyn_cast<IREE::TensorExt::DispatchTensorType>(
                   value.getType())) {
      requiredCount += tensorType.getNumDynamicDims();
    }
  }
  if (dynamicDims.size() != requiredCount) {
    return op->emitOpError()
           << "value set has " << requiredCount
           << " dynamic dimensions but only " << dynamicDims.size()
           << " dimension values are attached";
  }
  return success();
}

static Attribute getEncodingOrLayout(ShapedType type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.getEncoding();
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return memrefType.getLayout();
  }
  return {};
}

static bool hasEncodingOrNonIdentityLayout(ShapedType type) {
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return memrefType.getLayout() && !memrefType.getLayout().isIdentity();
  }
  return getEncodingOrLayout(type) != nullptr;
}

// Gets the dropped dimensions for `iree_tensor_ext.dispatch.tensor.load/store`.
static llvm::SmallBitVector
getDroppedDimsImpl(RankedTensorType slicedObjectType,
                   ArrayRef<OpFoldResult> mixedSizes) {
  ArrayRef<int64_t> resultShape = slicedObjectType.getShape();
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  size_t maxDroppedDims = mixedSizes.size() - resultShape.size();
  if (maxDroppedDims == 0) {
    return droppedDims;
  }
  unsigned shapePos = 0;
  int numSet = 0;
  for (const auto &size : llvm::enumerate(mixedSizes)) {
    std::optional<int64_t> sizeVal = getConstantIntValue(size.value());
    // If the size is not 1, or if the current matched dimension of the result
    // is the same static shape as the size value (which is 1), then the
    // dimension is preserved.
    if (!sizeVal || sizeVal.value() != 1 ||
        (shapePos < resultShape.size() && resultShape[shapePos] == 1)) {
      shapePos++;
      continue;
    }
    droppedDims.set(size.index());
    numSet++;
    if (numSet == maxDroppedDims) {
      break;
    }
  }
  return droppedDims;
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.bitcast
//===----------------------------------------------------------------------===//

LogicalResult BitCastOp::verify() {
  // The element types don't need to match, we can just check the requisite
  // number of dynamic dims.
  if (failed(verifyOpDynamicDims(getOperation(), {getSource()},
                                 getSourceDims())) ||
      failed(verifyOpDynamicDims(getOperation(), {getResult()},
                                 {getResultDims()}))) {
    return failure();
  }

  return success();
}

Value BitCastOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getSource());
}

::std::optional<unsigned>
BitCastOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // source
}

SmallVector<int64_t> BitCastOp::getTiedResultOperandIndices() {
  return {0}; // source
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.dispatch.tensor.load
//===----------------------------------------------------------------------===//

LogicalResult DispatchTensorLoadOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getSource()},
                                 getSourceDims()))) {
    return failure();
  }
  return success();
}

/// Extracts static and dynamic values from list of `OpFoldResult`.
static void processMixedOperands(ArrayRef<OpFoldResult> valueOrAttrs,
                                 SmallVectorImpl<Value> &dynamicValues,
                                 SmallVectorImpl<int64_t> &staticValues,
                                 int64_t dynamicIndexValue) {
  for (OpFoldResult valueOrAttr : valueOrAttrs) {
    if (auto value = dyn_cast<Value>(valueOrAttr)) {
      dynamicValues.push_back(value);
      staticValues.push_back(dynamicIndexValue);
    } else {
      auto operandValue =
          cast<IntegerAttr>(dyn_cast<Attribute>(valueOrAttr)).getInt();
      staticValues.push_back(operandValue);
    }
  }
}

/// Implements default offset, sizes and strides, for
/// `iree_tensor_ext.dispatch.tensor.load/store` ops. When no offsets, sizes and
/// strides are specified, the offsets are all zeros, sizes are same as the
/// dispatch tensor and strides are all 1.
static void getDefaultOffsetSizeAndStrides(
    OpBuilder &builder, IREE::TensorExt::DispatchTensorType dispatchTensorType,
    ValueRange dynamicDims, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) {
  auto zeroAttr = builder.getI64IntegerAttr(0);
  auto oneAttr = builder.getI64IntegerAttr(1);
  int64_t dispatchTensorRank = dispatchTensorType.getRank();
  offsets.assign(dispatchTensorRank, zeroAttr);
  strides.assign(dispatchTensorRank, oneAttr);
  sizes.resize(dispatchTensorRank);
  unsigned pos = 0;
  for (auto dim : llvm::enumerate(dispatchTensorType.getShape())) {
    if (ShapedType::isDynamic(dim.value())) {
      assert(pos < dynamicDims.size() && "missing dynamic dims specifications");
      sizes[dim.index()] = dynamicDims[pos++];
      continue;
    }
    sizes[dim.index()] = builder.getI64IntegerAttr(dim.value());
  }
  return;
}

RankedTensorType DispatchTensorLoadOp::inferResultType(
    IREE::TensorExt::DispatchTensorType sourceType,
    ArrayRef<OpFoldResult> mixedSizes) {
  auto shape =
      llvm::map_to_vector(mixedSizes, [&](OpFoldResult valueOrAttr) -> int64_t {
        if (auto attr = dyn_cast<Attribute>(valueOrAttr)) {
          return cast<IntegerAttr>(attr).getInt();
        }
        return ShapedType::kDynamic;
      });
  return RankedTensorType::get(shape, sourceType.getBoundElementType());
}

llvm::SmallBitVector DispatchTensorLoadOp::getDroppedDims() {
  return getDroppedDimsImpl(getType(), getMixedSizes());
}

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 RankedTensorType returnType, Value source,
                                 ValueRange sourceDynamicDims,
                                 ArrayRef<NamedAttribute> attributes) {
  SmallVector<OpFoldResult> offsets, strides, sizes;
  getDefaultOffsetSizeAndStrides(
      builder, cast<IREE::TensorExt::DispatchTensorType>(source.getType()),
      sourceDynamicDims, offsets, sizes, strides);
  build(builder, state, returnType, source, sourceDynamicDims, offsets, sizes,
        strides, attributes);
}

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 RankedTensorType returnType, Value source,
                                 ValueRange sourceDynamicDims,
                                 ArrayRef<OpFoldResult> mixedOffsets,
                                 ArrayRef<OpFoldResult> mixedSizes,
                                 ArrayRef<OpFoldResult> mixedStrides,
                                 ArrayRef<NamedAttribute> attributes) {
  SmallVector<Value> offsets;
  SmallVector<Value> sizes;
  SmallVector<Value> strides;
  SmallVector<int64_t> staticOffsets;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> staticStrides;

  processMixedOperands(mixedOffsets, offsets, staticOffsets,
                       ShapedType::kDynamic);
  processMixedOperands(mixedSizes, sizes, staticSizes, ShapedType::kDynamic);
  processMixedOperands(mixedStrides, strides, staticStrides,
                       ShapedType::kDynamic);

  build(builder, state, returnType, source, sourceDynamicDims, offsets, sizes,
        strides, staticOffsets, staticSizes, staticStrides);
  state.addAttributes(attributes);
}

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 Value source, ValueRange sourceDynamicDims,
                                 ArrayRef<OpFoldResult> mixedOffsets,
                                 ArrayRef<OpFoldResult> mixedSizes,
                                 ArrayRef<OpFoldResult> mixedStrides,
                                 ArrayRef<NamedAttribute> attributes) {
  auto returnType = inferResultType(
      cast<IREE::TensorExt::DispatchTensorType>(source.getType()), mixedSizes);
  build(builder, state, returnType, source, sourceDynamicDims, mixedOffsets,
        mixedSizes, mixedStrides);
}

LogicalResult DispatchTensorLoadOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  auto mixedSizes = getMixedSizes();
  SmallVector<OpFoldResult> shape;
  if (!mixedSizes.empty()) {
    // Slicing out a tile; return the size sliced.
    shape.reserve(mixedSizes.size());
    auto droppedDims = getDroppedDims();
    for (auto mixedSize : llvm::enumerate(mixedSizes)) {
      if (droppedDims.test(mixedSize.index())) {
        continue;
      }
      shape.push_back(mixedSize.value());
    }
  } else {
    // Result size matches the source size (no slicing).
    unsigned dynamicIdx = 0;
    for (int64_t dim : getType().getShape()) {
      if (ShapedType::isDynamic(dim)) {
        shape.push_back(getSourceDims()[dynamicIdx++]);
      } else {
        shape.push_back(b.getIndexAttr(dim));
      }
    }
  }
  reifiedReturnShapes.push_back(shape);
  return success();
}

Value DispatchTensorLoadOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getSource());
}

::std::optional<unsigned>
DispatchTensorLoadOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // source
}

SmallVector<int64_t> DispatchTensorLoadOp::getTiedResultOperandIndices() {
  return {0}; // source
}

bool DispatchTensorLoadOp::isLoadOfWholeSource() {
  return getSourceType().doesSliceSpanWholeTensor(
      getSourceDims(), getMixedOffsets(), getMixedSizes(), getMixedStrides());
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.dispatch.tensor.store
//===----------------------------------------------------------------------===//

LogicalResult DispatchTensorStoreOp::verify() {
  if (failed(verifyOpDynamicDims(getOperation(), {getTarget()},
                                 getTargetDims()))) {
    return failure();
  }

  // We only verify that the source tensor type is consistent with the type
  // inferred from the slice sizes.
  RankedTensorType sourceTensorType = getValue().getType();
  auto inferredType = RankedTensorType::get(getStaticSizes(),
                                            sourceTensorType.getElementType());
  SliceVerificationResult result =
      isRankReducedType(inferredType, sourceTensorType);
  return produceSliceErrorMsg(result, *this, inferredType);
}

void DispatchTensorStoreOp::build(OpBuilder &builder, OperationState &state,
                                  Value value, Value target,
                                  ValueRange targetDynamicDims,
                                  ArrayRef<NamedAttribute> attributes) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  getDefaultOffsetSizeAndStrides(
      builder, cast<IREE::TensorExt::DispatchTensorType>(target.getType()),
      targetDynamicDims, offsets, sizes, strides);
  build(builder, state, value, target, targetDynamicDims, offsets, sizes,
        strides, attributes);
}

void DispatchTensorStoreOp::build(OpBuilder &builder, OperationState &state,
                                  Value value, Value target,
                                  ValueRange targetDynamicDims,
                                  ArrayRef<OpFoldResult> mixedOffsets,
                                  ArrayRef<OpFoldResult> mixedSizes,
                                  ArrayRef<OpFoldResult> mixedStrides,
                                  ArrayRef<NamedAttribute> attributes) {
  SmallVector<Value> offsets, sizes, strides;
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  processMixedOperands(mixedOffsets, offsets, staticOffsets,
                       ShapedType::kDynamic);
  processMixedOperands(mixedSizes, sizes, staticSizes, ShapedType::kDynamic);
  processMixedOperands(mixedStrides, strides, staticStrides,
                       ShapedType::kDynamic);

  build(builder, state, ArrayRef<Type>(), value, target, targetDynamicDims,
        offsets, sizes, strides, staticOffsets, staticSizes, staticStrides);
  state.addAttributes(attributes);
}

llvm::SmallBitVector DispatchTensorStoreOp::getDroppedDims() {
  return getDroppedDimsImpl(cast<RankedTensorType>(getValue().getType()),
                            getMixedSizes());
}

bool DispatchTensorStoreOp::isStoreToWholeTarget() {
  return getTargetType().doesSliceSpanWholeTensor(
      getTargetDims(), getMixedOffsets(), getMixedSizes(), getMixedStrides());
}

//===----------------------------------------------------------------------===//
// dispatch.workgroup_count_splitk_modifier
//===----------------------------------------------------------------------===//

void DispatchWorkgroupCountSplitReductionModifierOp::build(
    OpBuilder &b, OperationState &state, ValueRange workgroups,
    ValueRange workload) {
  assert(workgroups.size() == 3);
  return build(b, state, workgroups[0], workgroups[1], workgroups[2], workload);
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.dispatch.workload.ordinal
//===----------------------------------------------------------------------===//

void DispatchWorkloadOrdinalOp::inferResultDivisibility(
    ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
    IREE::Util::SetIntDivisibilityFn setResultDivisibility) {
  if (argDivs[0].isUninitialized()) {
    setResultDivisibility(getResult(),
                          IREE::Util::ConstantIntDivisibility(1, 1));
    return;
  }
  setResultDivisibility(getResult(), argDivs[0].getValue());
}

void DispatchWorkloadOrdinalOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> argRanges, SetIntRangeFn setResultRange) {
  assert(!argRanges.empty() && "expected range of input to be set");
  setResultRange(getResult(), argRanges[0]);
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.compute_barrier.start
//===----------------------------------------------------------------------===//

LogicalResult ComputeBarrierStartOp::verify() {
  ComputeBarrierStartOp op = *this;
  if (failed(verifyOpDynamicDims(op, {op.getValue()}, op.getValueDims()))) {
    return failure();
  }

  if (op.getValue().getType() != op.getResult().getType()) {
    return op.emitOpError("value and result types must match");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.compute_barrier.end
//===----------------------------------------------------------------------===//

LogicalResult ComputeBarrierEndOp::verify() {
  ComputeBarrierEndOp op = *this;
  if (failed(verifyOpDynamicDims(op, {op.getValue()}, op.getValueDims()))) {
    return failure();
  }

  if (op.getValue().getType() != op.getResult().getType()) {
    return op.emitOpError("value and result types must match");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.cast_to_ragged_shape
//===----------------------------------------------------------------------===//

LogicalResult CastToRaggedShapeOp::verify() {
  // Check that the ragged row dimensions `raggedRowDim` < rank(`result`) - 1.
  int64_t raggedDim = getRaggedDim().getSExtValue();
  ShapedType sourceType = getSourceType();
  if (raggedDim >= sourceType.getRank()) {
    return emitOpError("expected `ragged_dim` to be less than ")
           << sourceType.getRank() << ", i.e the rank of source";
  }

  auto columnLengthsType = cast<ShapedType>(getColumnLengths().getType());
  if (columnLengthsType.getRank() != 1) {
    return emitOpError("expected `column_lengths` to be of rank 1");
  }

  ShapedType resultType = getResultType();
  if (resultType.getRank() != sourceType.getRank() + 1) {
    return emitOpError("expected result rank to be ")
           << sourceType.getRank() + 1
           << ", i.e. one more than the source rank, but got "
           << resultType.getRank();
  }
  RaggedShapeAttr raggedTensorAttr = getResultSparseEncoding();
  if (!raggedTensorAttr) {
    return emitOpError("expected result type to have an encoding of type "
                       "`RaggedShapeAttr`");
  }
  if (raggedTensorAttr.getRaggedRow() != raggedDim) {
    return emitOpError("mismatch in specified `ragged_dim` value of ")
           << raggedDim << " and `raggedRow` value in the sparse encoding "
           << raggedTensorAttr.getRaggedRow();
  }

  if (Value numRaggedRows = getNumRaggedRows()) {
    // If `num_ragged_rows` is dynamic, check that columnLengths shape is
    // dynamic as well.
    if (!columnLengthsType.isDynamicDim(0)) {
      return emitOpError("invalid to have static dimensions for "
                         "`column_lengths` when `num_ragged_rows` is dynamic");
    }

    // Check that the result has the number of rows dynamic as well.
    if (!resultType.isDynamicDim(raggedDim)) {
      return emitOpError("invalid to have static value for dimension ")
             << raggedDim
             << " of result when `num_ragged_rows` is specified as dynamic "
                "value";
    }
  } else {
    // if `num_ragged_rows` is static, check that the `column_lengths` shape is
    // static as well. This effectively where the number of rows is statically
    // recorded.
    if (columnLengthsType.isDynamicDim(0) ||
        columnLengthsType.getDimSize(0) <= 1) {
      return emitOpError(
          "expected shape of `column_lengths` to be static and "
          "greater than 1 when `num_ragged_rows` is unspecified, i.e. "
          "number of ragged rows is statically known");
    }

    // Check that the size of `ragged_dim` dimension in result type is
    // consistent with the shape of column_lengths`.
    if (resultType.isDynamicDim(raggedDim) ||
        resultType.getDimSize(raggedDim) !=
            columnLengthsType.getDimSize(0) - 1) {
      return emitOpError("expected shape of dimension ")
             << raggedDim
             << " of result, i.e. the `ragged_dim`, to be static and equal to "
             << columnLengthsType.getDimSize(0) - 1
             << " when `num_ragged_rows` is unspecified, i.e number of ragged "
                "rows is "
                "statically known";
    }
  }

  // The dimension for the "ragged columns" should be dynamic.
  if (!resultType.isDynamicDim(raggedDim + 1)) {
    return emitOpError("expected dimension ")
           << (raggedDim + 1)
           << " of result, i.e. `ragged_dim` + 1, to be dynamic";
  }

  bool foundRaggedDim = false;
  int expectedNumSourceDynamicDims = 0;
  for (auto [index, sourceShape] : llvm::enumerate(sourceType.getShape())) {
    if (index == raggedDim) {
      foundRaggedDim = true;
      if (ShapedType::isDynamic(sourceShape)) {
        expectedNumSourceDynamicDims++;
      }
      continue;
    }
    int resultIndex = index + foundRaggedDim;
    int64_t resultShape = resultType.getDimSize(resultIndex);
    if (ShapedType::isDynamic(sourceShape)) {
      if (ShapedType::isStatic(resultShape)) {
        return emitOpError("expected dimension ")
               << resultIndex
               << " of result to be dynamic since the corresponding dimension "
               << index << " in the source is dynamic";
      }
      expectedNumSourceDynamicDims++;
      continue;
    }

    if (ShapedType::isDynamic(resultShape) || resultShape != sourceShape) {
      return emitOpError("expected shape of dimension ")
             << resultIndex << " of result to match the shape of dimension "
             << index << " of source, but got "
             << (ShapedType::isDynamic(resultShape)
                     ? std::string("?")
                     : std::to_string(resultShape))
             << " and " << sourceShape << " respectively";
    }
  }

  if (expectedNumSourceDynamicDims != getSourceDynamicDims().size()) {
    return emitOpError("mismatch in number of dynamic dimensions specified for "
                       "source, expected ")
           << expectedNumSourceDynamicDims << " values, got "
           << getSourceDynamicDims().size();
  }

  return success();
}

OpFoldResult CastToRaggedShapeOp::getNumRaggedRowsAsOfr() {
  if (Value v = getNumRaggedRows()) {
    return v;
  }
  ArrayRef<int64_t> resultShape = getResultType().getShape();
  int64_t raggedDim = getRaggedDimAttr().getInt();
  assert(ShapedType::isStatic(resultShape[raggedDim]) &&
         "expected number of ragged rows to be static");
  return IntegerAttr::get(IndexType::get(getContext()), resultShape[raggedDim]);
}

// Cast `v` to index type if it's not already index type.
static Value castToIndexTypeOrSelf(RewriterBase &rewriter, Location loc,
                                   Value v) {
  if (v.getType() != rewriter.getIndexType()) {
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                      v);
  }
  return v;
}

// Starting at `v`, compute a backward slice dominated by the given `sparseOp`
// and clone the operations in the slice, replacing any occurrence of
// `dim(%sparseOp, givenDim)` with the value returned by
// `getDimReplacementValue`. Return the cloned values corresponding to
// `queriedValues`.
static FailureOr<SmallVector<Value>> cloneAndReplaceDimInBackwardSlice(
    RewriterBase &rewriter, Location loc, DominanceInfo &dominanceInfo, Value v,
    IREE::TensorExt::SparseCastOpInterface sparseOp, int64_t givenDim,
    std::function<Value(RewriterBase &, Location Loc)> getDimReplacementValue,
    ArrayRef<Value> queriedValues) {
  BackwardSliceOptions sliceOptions;
  sliceOptions.inclusive = true;
  sliceOptions.filter = [&](Operation *op) -> bool {
    return dominanceInfo.properlyDominates(sparseOp, op);
  };
  llvm::SetVector<Operation *> slice;
  if (failed(mlir::getBackwardSlice(v, &slice, sliceOptions))) {
    return sparseOp->emitOpError("failed to compute backward slice for value");
  }

  IRMapping mapping;
  for (auto op : slice) {
    IntegerAttr attr;
    Value dimSource;
    if (!matchDimOp(op, dimSource, attr) ||
        dimSource != sparseOp->getResult(0)) {
      rewriter.clone(*op, mapping);
      continue;
    }
    int64_t foundDim = attr.getInt();
    if (foundDim != givenDim) {
      return sparseOp->emitOpError("invalid dim expression found in slice");
    }
    Value dimReplacementVal = getDimReplacementValue(rewriter, loc);
    mapping.map(op->getResult(0), dimReplacementVal);
  }
  SmallVector<Value> clonedQueriedValues;
  clonedQueriedValues.reserve(queriedValues.size());
  for (Value qv : queriedValues) {
    Value cv = mapping.lookupOrDefault(qv);
    clonedQueriedValues.push_back(cv);
  }
  return clonedQueriedValues;
}

FailureOr<SmallVector<Range>>
CastToRaggedShapeOp::getEstimatedLoopRange(RewriterBase &rewriter,
                                           ArrayRef<int64_t> sparseDims,
                                           ArrayRef<Range> givenRange) {
  SmallVector<int64_t> expectedSparseDims =
      getResultSparseEncoding().getSparseDimensions();
  assert(expectedSparseDims.size() == 2 &&
         "invalid specification of op with more than two sparse dimensions");
  if (expectedSparseDims != sparseDims) {
    return emitOpError(
        "cannot estimate the loop range for given sparse dimensions");
  }
  assert(givenRange.size() == sparseDims.size() &&
         "expected ranges for all dims");

  SmallVector<Range> estimatedRange = llvm::to_vector(givenRange);

  DominanceInfo dominanceInfo(getOperation()->getParentOp());
  Location loc = getLoc();

  // For the outer sparse dimension, the estimated range is obtained by
  // replacing `dim(%sparseOp, outerSparseDim)` with `%num_ragged_rows` in the
  // backward slice of the upper bound.
  Value outerUb =
      getValueOrCreateConstantIndexOp(rewriter, loc, givenRange[0].size);
  // Replace the `memref.dim`/`tensor.dim` operation in the backward slice of
  // the upper bound with the number of ragged rows.
  FailureOr<SmallVector<Value>> outerDimReplacementResult =
      cloneAndReplaceDimInBackwardSlice(
          rewriter, loc, dominanceInfo, outerUb, *this, expectedSparseDims[0],
          [&](RewriterBase &rewriter, Location loc) {
            OpFoldResult numRaggedRows = getNumRaggedRowsAsOfr();
            Value numRaggedRowsVal =
                getValueOrCreateConstantIndexOp(rewriter, loc, numRaggedRows);
            return numRaggedRowsVal;
          },
          {outerUb});
  if (failed(outerDimReplacementResult)) {
    return emitOpError("failed to replace outer dim in backward slice");
  }
  estimatedRange[0].size = outerDimReplacementResult.value()[0];

  // For the inner sparse dimension, the estimated range is obtained by
  // replacing the `dim(%sparseOp, innerSparseDim)` with
  // `dim(%source, outerSparseDim) / %num_ragged_rows`.

  Value innerUb =
      getValueOrCreateConstantIndexOp(rewriter, loc, givenRange[1].size);
  FailureOr<SmallVector<Value>> innerDimReplacementResult =
      cloneAndReplaceDimInBackwardSlice(
          rewriter, loc, dominanceInfo, innerUb, *this, expectedSparseDims[1],
          [&](RewriterBase &rewriter, Location loc) {
            OpFoldResult sourceDim;
            if (isa<MemRefType>(getSource().getType())) {
              sourceDim = memref::DimOp::create(rewriter, loc, getSource(),
                                                expectedSparseDims[0])
                              .getResult();
            } else {
              sourceDim = tensor::DimOp::create(rewriter, loc, getSource(),
                                                expectedSparseDims[0])
                              .getResult();
            }
            OpFoldResult numRaggedRows = getNumRaggedRowsAsOfr();
            AffineExpr s0, s1;
            bindSymbols(rewriter.getContext(), s0, s1);
            AffineMap divMap =
                AffineMap::get(0, 2, s0.ceilDiv(s1), rewriter.getContext());
            OpFoldResult estimatedNumColumns =
                affine::makeComposedFoldedAffineApply(
                    rewriter, loc, divMap,
                    ArrayRef<OpFoldResult>{sourceDim, numRaggedRows});
            Value estimatedNumColumnsVal = getValueOrCreateConstantIndexOp(
                rewriter, loc, estimatedNumColumns);
            return estimatedNumColumnsVal;
          },
          {innerUb});
  if (failed(innerDimReplacementResult)) {
    return emitOpError("failed to replace inner dim in backward slice");
  }
  estimatedRange[1].size = innerDimReplacementResult.value()[0];
  return estimatedRange;
}

FailureOr<SmallVector<Value>>
CastToRaggedShapeOp::lowerLoopRange(RewriterBase &rewriter,
                                    ArrayRef<int64_t> sparseDims,
                                    ArrayRef<Range> givenRange) {
  SmallVector<int64_t> expectedSparseDims =
      getResultSparseEncoding().getSparseDimensions();
  assert(expectedSparseDims.size() == 2 &&
         "invalid specification of op with more than two sparse dimensions");

  // Check that all requested sparse dims are valid.
  for (int64_t sparseDim : sparseDims) {
    if (!llvm::is_contained(expectedSparseDims, sparseDim)) {
      return rewriter.notifyMatchFailure(
          *this, "cannot lower the loop range for given sparse dimensions");
    }
  }
  assert(givenRange.size() == sparseDims.size() &&
         "expected ranges for all dims");

  // Generate the loop for sparse dimensions.
  Location loc = getLoc();

  // The upper bounds of the range is expected to be the
  // `memref.dim`/`tensor.dim` operation of the result of this operation.
  DominanceInfo dominanceInfo(getOperation()->getParentOp());

  SmallVector<Value> ivs;

  // Find indices of outer and inner sparse dims in the sparseDims array.
  auto outerIt = llvm::find(sparseDims, expectedSparseDims[0]);
  auto innerIt = llvm::find(sparseDims, expectedSparseDims[1]);

  // Handle outer sparse dimension (row) if present.
  Value outerIv;
  if (outerIt != sparseDims.end()) {
    int64_t outerIdx = std::distance(sparseDims.begin(), outerIt);
    Value outerLb = getValueOrCreateConstantIndexOp(
        rewriter, loc, givenRange[outerIdx].offset);
    Value outerUb = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                    givenRange[outerIdx].size);
    Value outerStep = getValueOrCreateConstantIndexOp(
        rewriter, loc, givenRange[outerIdx].stride);

    // Replace the `memref.dim`/`tensor.dim` operation in the backward slice of
    // the upper bound with the number of ragged rows.
    FailureOr<SmallVector<Value>> outerDimReplacementResult =
        cloneAndReplaceDimInBackwardSlice(
            rewriter, loc, dominanceInfo, outerUb, *this, expectedSparseDims[0],
            [&](RewriterBase &rewriter, Location loc) {
              OpFoldResult numRaggedRows = getNumRaggedRowsAsOfr();
              Value numRaggedRowsVal =
                  getValueOrCreateConstantIndexOp(rewriter, loc, numRaggedRows);
              return numRaggedRowsVal;
            },
            {outerUb});
    if (failed(outerDimReplacementResult)) {
      return emitOpError("failed to replace outer dim in backward slice");
    }

    auto outerFor =
        scf::ForOp::create(rewriter, loc, outerLb,
                           outerDimReplacementResult.value()[0], outerStep);
    outerIv = outerFor.getInductionVar();
    Block *outerForBody = outerFor.getBody();
    rewriter.setInsertionPointToStart(outerForBody);
    ivs.push_back(outerIv);
  }

  // Handle inner sparse dimension (column) if present.
  if (innerIt != sparseDims.end()) {
    int64_t innerIdx = std::distance(sparseDims.begin(), innerIt);
    // If we don't have an outer IV from this call, we cannot create the inner
    // loop since it depends on the outer IV for computing column bounds.
    if (!outerIv) {
      return emitOpError(
          "cannot lower inner sparse dimension without outer dimension");
    }

    Value innerLb = getValueOrCreateConstantIndexOp(
        rewriter, loc, givenRange[innerIdx].offset);
    Value innerUb = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                    givenRange[innerIdx].size);
    Value innerStep = getValueOrCreateConstantIndexOp(
        rewriter, loc, givenRange[innerIdx].stride);

    // Replace the `memref.dim`/`tensor.dim` operation in the backward slice of
    // the inner dim with the column length range (column_lengths[outerIv+1] -
    // column_lengths[outerIv]). This ensures the loop variable iterates in
    // relative index space (0, 1, 2, ...) which is correct for the dense output
    // tensor. The sparse access resolution pass will adjust load indices to add
    // column_lengths[outerIv] to get the linearized source index.
    Value columnLengths = getColumnLengths();
    FailureOr<SmallVector<Value>> innerDimReplacementResult =
        cloneAndReplaceDimInBackwardSlice(
            rewriter, loc, dominanceInfo, innerUb, *this, expectedSparseDims[1],
            [&](RewriterBase &rewriter, Location loc) {
              Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
              Value plusOne =
                  arith::AddIOp::create(rewriter, loc, outerIv, one);
              Value columnEnd, columnStart;
              if (isa<MemRefType>(columnLengths.getType())) {
                columnEnd = memref::LoadOp::create(rewriter, loc, columnLengths,
                                                   plusOne);
                columnStart = memref::LoadOp::create(rewriter, loc,
                                                     columnLengths, outerIv);
              } else {
                columnEnd = tensor::ExtractOp::create(rewriter, loc,
                                                      columnLengths, plusOne);
                columnStart = tensor::ExtractOp::create(rewriter, loc,
                                                        columnLengths, outerIv);
              }
              columnEnd = castToIndexTypeOrSelf(rewriter, loc, columnEnd);
              columnStart = castToIndexTypeOrSelf(rewriter, loc, columnStart);
              // Compute column_lengths[outerIv+1] - column_lengths[outerIv]
              AffineExpr s0, s1;
              bindSymbols(rewriter.getContext(), s0, s1);
              AffineMap subMap =
                  AffineMap::get(0, 2, {s0 - s1}, rewriter.getContext());
              OpFoldResult columnRangeOfr =
                  affine::makeComposedFoldedAffineApply(
                      rewriter, loc, subMap,
                      ArrayRef<OpFoldResult>{columnEnd, columnStart});
              return getValueOrCreateConstantIndexOp(rewriter, loc,
                                                     columnRangeOfr);
            },
            {innerLb, innerUb});
    if (failed(innerDimReplacementResult)) {
      return emitOpError("failed to replace inner dims in backward slice");
    }

    Value clonedLb = innerDimReplacementResult.value()[0];
    Value clonedUb = innerDimReplacementResult.value()[1];
    // Use the cloned lower bound directly - no need to max with
    // column_lengths[outerIv] since the loop now iterates in relative index
    // space.
    auto innerFor =
        scf::ForOp::create(rewriter, loc, clonedLb, clonedUb, innerStep);
    Block *innerForBody = innerFor.getBody();
    rewriter.setInsertionPointToStart(innerForBody);
    Value innerIv = innerFor.getInductionVar();
    ivs.push_back(innerIv);
  }

  return ivs;
}

FailureOr<SmallVector<Range>> CastToRaggedShapeOp::resolveRange(
    RewriterBase &rewriter, ArrayRef<Range> givenRanges,
    const llvm::BitVector &inBounds, std::optional<Value> paddingValue) {
  // NOTE: paddingValue parameter is accepted for interface compatibility.
  // Future work: Generate conditional code/guards when paddingValue is provided
  // to handle out-of-bounds accesses by returning the padding value.
  (void)paddingValue; // Unused for now

  ShapedType resultType = getResultType();

  if (givenRanges.size() != resultType.getRank()) {
    return emitOpError("expected givenRanges to have ")
           << resultType.getRank() << " entries, but got "
           << givenRanges.size();
  }

  if (inBounds.size() != givenRanges.size()) {
    return emitOpError("expected inBounds to have ")
           << givenRanges.size() << " bits, but got " << inBounds.size();
  }

  // Check if all accesses are in-bounds. Out-of-bounds handling is not yet
  // supported.
  if (!inBounds.all()) {
    return emitOpError("out-of-bounds accesses are not yet supported");
  }

  SmallVector<int64_t> sparseDims =
      getResultSparseEncoding().getSparseDimensions();
  assert(sparseDims.size() == 2 &&
         "invalid specification of op with more than two sparse dimensions");

  // Verify constraints on sparse dimension ranges.
  // For the outer sparse dimension: size must be 1, stride must be 1.
  OpFoldResult ofrsToCheck[] = {givenRanges[sparseDims[0]].size,
                                givenRanges[sparseDims[0]].stride};
  if (!llvm::all_of(ofrsToCheck, isOneInteger)) {
    return emitOpError("expected outer sparse dimension size and stride to be "
                       "1 in givenRanges");
  }

  Location loc = getLoc();
  SmallVector<Range> resolvedRange;

  // Process all dimensions except the sparse dimensions.
  int64_t resultRank = resultType.getRank();

  // The outer sparse dimension index.
  int64_t outerSparseDim = sparseDims[0];
  int64_t innerSparseDim = sparseDims[1];

  // Create a set of sparse dimensions for easy lookup.
  llvm::SmallDenseSet<int64_t> sparseDimsSet(sparseDims.begin(),
                                             sparseDims.end());

  // Build the resolved range for the source.
  for (int64_t resultDim = 0; resultDim < resultRank; ++resultDim) {
    if (!sparseDimsSet.contains(resultDim)) {
      // Non-sparse dimensions are preserved.
      resolvedRange.push_back(givenRanges[resultDim]);
      continue;
    }
    if (resultDim == innerSparseDim) {
      // Skip the inner sparse dimension - it's already handled.
      continue;
    }
    // This is the outer sparse dimension - compute the range for the
    // corresponding source dimension.
    Value columnLengths = getColumnLengths();

    // Extract the column_lengths[offset_0] value.
    Value offset0 = getValueOrCreateConstantIndexOp(
        rewriter, loc, givenRanges[outerSparseDim].offset);
    Value columnLengthsCurrent;
    if (isa<MemRefType>(columnLengths.getType())) {
      columnLengthsCurrent =
          memref::LoadOp::create(rewriter, loc, columnLengths, offset0);
    } else {
      columnLengthsCurrent =
          tensor::ExtractOp::create(rewriter, loc, columnLengths, offset0);
    }
    columnLengthsCurrent =
        castToIndexTypeOrSelf(rewriter, loc, columnLengthsCurrent);

    // Since we verified all accesses are in-bounds, we can use the requested
    // size directly without clamping.
    OpFoldResult sizeVal = givenRanges[innerSparseDim].size;
    OpFoldResult strideVal = givenRanges[innerSparseDim].stride;

    // offset = column_lengths[offset_0] + offset_1
    // The linearized offset is the base offset from column_lengths plus the
    // inner sparse dimension offset.
    AffineExpr d0_offset, d1_offset;
    bindDims(rewriter.getContext(), d0_offset, d1_offset);
    AffineMap addMap =
        AffineMap::get(2, 0, {d0_offset + d1_offset}, rewriter.getContext());
    OpFoldResult linearizedOffset = affine::makeComposedFoldedAffineApply(
        rewriter, loc, addMap,
        ArrayRef<OpFoldResult>{columnLengthsCurrent,
                               givenRanges[innerSparseDim].offset});

    resolvedRange.push_back(Range{linearizedOffset, sizeVal, strideVal});
  }

  return resolvedRange;
}

llvm::BitVector CastToRaggedShapeOp::getDistributionInfoForSparseDimensions() {
  SmallVector<int64_t> sparseDims =
      getResultSparseEncoding().getSparseDimensions();

  // For ragged tensors with 2 sparse dimensions:
  // - First sparse dimension (raggedRow) CAN be distributed.
  // - Second sparse dimension (raggedRow + 1) CANNOT be distributed
  //   because its bounds depend on the outer dimension.
  llvm::BitVector result(sparseDims.size(), false);
  if (!sparseDims.empty()) {
    result.set(0); // Only first sparse dimension is distributable.
  }
  return result;
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.linearize_ragged_dims
//===----------------------------------------------------------------------===//

LogicalResult LinearizeRaggedDimsOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  ArrayRef<int64_t> resultShape =
      cast<ShapedType>(getResult().getType()).getShape();
  ValueRange dynamicDims = getResultDynamicDims();
  SmallVector<OpFoldResult> mixedResultShapes =
      getMixedValues(resultShape, dynamicDims, builder);
  reifiedReturnShapes.emplace_back(std::move(mixedResultShapes));
  return success();
}

LogicalResult LinearizeRaggedDimsOp::verify() {
  ShapedType sourceType = cast<ShapedType>(getSource().getType());
  ShapedType resultType = cast<ShapedType>(getResult().getType());

  // 1. Verify source has ragged tensor encoding.
  auto raggedAttr =
      dyn_cast_if_present<RaggedShapeAttr>(getEncodingOrLayout(sourceType));

  if (!raggedAttr) {
    return emitOpError("expected source type to have a ragged tensor encoding");
  }

  // Verify result does not have ragged encoding.
  if (hasEncodingOrNonIdentityLayout(resultType)) {
    return emitOpError(
        "expected result type to have no encoding or identity layout");
  }

  // Get the sparse dimensions from the encoding.
  SmallVector<int64_t> sparseDims = raggedAttr.getSparseDimensions();

  if (sparseDims.empty()) {
    return emitOpError("ragged tensor encoding has no sparse dimensions");
  }

  // Verify sparse dimensions are contiguous
  for (size_t i = 1; i < sparseDims.size(); ++i) {
    if (sparseDims[i] != sparseDims[i - 1] + 1) {
      return emitOpError("sparse dimensions must be contiguous, but got ")
             << sparseDims[i - 1] << " and " << sparseDims[i];
    }
  }

  int64_t sourceRank = sourceType.getRank();
  int64_t resultRank = resultType.getRank();

  // 2. Verify result rank: sparse dimensions are linearized into one dimension
  // Result rank = source rank - number of sparse dimensions + 1
  int64_t expectedResultRank = sourceRank - sparseDims.size() + 1;
  if (resultRank != expectedResultRank) {
    return emitOpError("expected result rank to be ")
           << expectedResultRank
           << " (source rank - number of sparse dimensions + 1), but got "
           << resultRank;
  }

  // 3. Build a set of sparse dimensions and a map from non-sparse source
  // dimensions to result dimensions
  llvm::SmallDenseSet<int64_t> sparseDimSet(sparseDims.begin(),
                                            sparseDims.end());

  // Build mapping: non-sparse source dims -> result dims
  int64_t resultDimIndex = 0;
  for (int64_t sourceDim = 0; sourceDim < sourceRank; ++sourceDim) {
    if (sparseDimSet.contains(sourceDim)) {
      if (sourceDim == sparseDims[0]) {
        resultDimIndex++;
      }
      continue;
    }

    if (resultDimIndex >= resultRank) {
      return emitOpError("result rank too small to accommodate all non-sparse "
                         "source dimensions");
    }

    int64_t sourceSize = sourceType.getDimSize(sourceDim);
    int64_t resultSize = resultType.getDimSize(resultDimIndex);
    bool sourceDynamic = ShapedType::isDynamic(sourceSize);
    bool resultDynamic = ShapedType::isDynamic(resultSize);
    if (!sourceDynamic && !resultDynamic && sourceSize != resultSize) {
      return emitOpError("expected source dimension ")
             << sourceDim << " with size " << sourceSize
             << " to be preserved in result dimension " << resultDimIndex
             << " but got size " << resultSize;
    }

    resultDimIndex++;
  }

  // 4. Verify the correct number of dynamic dimensions are provided
  int64_t numResultDynamicDims = getResultDynamicDims().size();
  int64_t numExpectedDynamicDims = 0;
  for (int64_t i = 0; i < resultRank; ++i) {
    if (resultType.isDynamicDim(i)) {
      numExpectedDynamicDims++;
    }
  }

  if (numResultDynamicDims != numExpectedDynamicDims) {
    return emitOpError("expected ")
           << numExpectedDynamicDims << " dynamic dimension values, but got "
           << numResultDynamicDims;
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::TensorExt

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.cpp.inc" // IWYU pragma: keep
