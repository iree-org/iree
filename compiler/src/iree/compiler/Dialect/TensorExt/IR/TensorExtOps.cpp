// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

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
    if (auto shapedType = llvm::dyn_cast<ShapedType>(value.getType())) {
      requiredCount += shapedType.getNumDynamicDims();
    } else if (auto tensorType =
                   llvm::dyn_cast<IREE::TensorExt::DispatchTensorType>(
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
          llvm::cast<IntegerAttr>(dyn_cast<Attribute>(valueOrAttr)).getInt();
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
          return llvm::cast<IntegerAttr>(attr).getInt();
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
      builder,
      llvm::cast<IREE::TensorExt::DispatchTensorType>(source.getType()),
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
      llvm::cast<IREE::TensorExt::DispatchTensorType>(source.getType()),
      mixedSizes);
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
      builder,
      llvm::cast<IREE::TensorExt::DispatchTensorType>(target.getType()),
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
  return getDroppedDimsImpl(llvm::cast<RankedTensorType>(getValue().getType()),
                            getMixedSizes());
}

bool DispatchTensorStoreOp::isStoreToWholeTarget() {
  return getTargetType().doesSliceSpanWholeTensor(
      getTargetDims(), getMixedOffsets(), getMixedSizes(), getMixedStrides());
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

} // namespace mlir::iree_compiler::IREE::TensorExt

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.cpp.inc" // IWYU pragma: keep
