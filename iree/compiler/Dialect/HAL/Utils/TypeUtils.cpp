// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

Value align(Location loc, Value value, int64_t alignment, OpBuilder &builder) {
  // (value + (alignment - 1)) & ~(alignment - 1)
  return builder.createOrFold<AndOp>(
      loc,
      builder.createOrFold<AddIOp>(
          loc, value,
          builder.createOrFold<ConstantIndexOp>(loc, alignment - 1)),
      builder.createOrFold<ConstantIndexOp>(loc, ~(alignment - 1)));
}

int32_t getRoundedElementByteWidth(Type type) {
  return (type.getIntOrFloatBitWidth() + 8 - 1) / 8;
}

// Returns an array of i32 values representing the shape of the |shapedType|.
static SmallVector<Value, 4> getStaticShapeDims(Location loc,
                                                ShapedType shapedType,
                                                OpBuilder &builder) {
  SmallVector<Value, 4> shape;
  if (shapedType.getRank() >= 1) {
    for (auto dim : shapedType.getShape()) {
      shape.push_back(builder.createOrFold<mlir::ConstantIndexOp>(loc, dim));
    }
  }
  return shape;
}

// Returns an array of index values representing the shape of the |shapedValue|.
static llvm::Optional<SmallVector<Value, 4>> getShapeDims(Location loc,
                                                          Value shapedValue,
                                                          OpBuilder &builder) {
  return Shape::buildOrFindDimsForValue(loc, shapedValue, builder);
}

Value getValueSize(Location loc, Value value, OpBuilder &builder) {
  // Function arguments are special as we always have to query.
  auto definingOp = value.getDefiningOp();
  if (!definingOp) {
    return builder.createOrFold<IREE::HAL::BufferLengthOp>(
        loc, builder.getIndexType(), value);
  }

  if (auto awareOp =
          dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(definingOp)) {
    return awareOp.getResultSizeFromValue(value);
  }

  auto type = value.getType();
  if (auto inferType = type.dyn_cast<IREE::Util::InferTypeSizeInterface>()) {
    return inferType.inferSizeFromValue(loc, value, builder);
  }

  auto elementType = IREE::HAL::getElementTypeValue(
      value.getType().cast<ShapedType>().getElementType());
  if (!elementType) return {};

  // TODO(#6762): get encoding type from value.
  auto encodingType = IREE::HAL::getEncodingTypeValue({});
  if (!encodingType) return {};

  auto shape = IREE::HAL::getShapeDims(loc, value, builder);
  if (!shape) return {};

  auto allocatorValue = builder.createOrFold<IREE::HAL::BufferAllocatorOp>(
      loc, IREE::HAL::AllocatorType::get(builder.getContext()), value);
  return builder.createOrFold<IREE::HAL::AllocatorComputeSizeOp>(
      loc, allocatorValue, *shape, elementType.getValue(),
      encodingType.getValue());
}

// static
bool TensorRewriteAdaptor::isValidNewType(Type newType) {
  return newType.isa<IREE::HAL::BufferType>() ||
         newType.isa<IREE::HAL::BufferViewType>();
}

// static
LogicalResult TensorRewriteAdaptor::verifyConstructionInvariants(
    Location loc, Value oldValue, Value newValue,
    ConversionPatternRewriter &rewriter) {
  if (!oldValue || !newValue) {
    return emitError(loc) << "TensorRewriteAdaptor values are null";
  }
  if (!oldValue.getType().isa<TensorType>()) {
    return emitError(loc) << "TensorRewriteAdaptor oldValue is not a Tensor";
  }
  if (!TensorRewriteAdaptor::isValidNewType(newValue.getType())) {
    return emitError(loc) << "TensorRewriteAdaptor newValue is invalid type "
                          << newValue.getType();
  }
  return success();
}

// static
TensorRewriteAdaptor TensorRewriteAdaptor::get(
    Location loc, Value oldValue, Value newValue,
    ConversionPatternRewriter &rewriter) {
  assert(succeeded(TensorRewriteAdaptor::verifyConstructionInvariants(
      loc, oldValue, newValue, rewriter)));
  return TensorRewriteAdaptor(loc, oldValue, newValue, rewriter);
}

// static
llvm::Optional<TensorRewriteAdaptor> TensorRewriteAdaptor::getChecked(
    Location loc, Value oldValue, Value newValue,
    ConversionPatternRewriter &rewriter) {
  if (failed(TensorRewriteAdaptor::verifyConstructionInvariants(
          loc, oldValue, newValue, rewriter))) {
    return llvm::None;
  }
  return TensorRewriteAdaptor(loc, oldValue, newValue, rewriter);
}

Value TensorRewriteAdaptor::getAllocator() {
  return rewriter_.createOrFold<IREE::HAL::BufferAllocatorOp>(
      loc_, AllocatorType::get(rewriter_.getContext()), getBuffer());
}

bool TensorRewriteAdaptor::isBufferView() {
  return newValue_.getType().isa<IREE::HAL::BufferViewType>();
}

Value TensorRewriteAdaptor::getBuffer() {
  if (isBufferView()) {
    return rewriter_.createOrFold<IREE::HAL::BufferViewBufferOp>(
        loc_, IREE::HAL::BufferType::get(rewriter_.getContext()), newValue_);
  } else {
    return newValue_;
  }
}

Value TensorRewriteAdaptor::getBufferView() {
  if (isBufferView()) {
    return newValue_;
  } else if (auto bufferViewBufferOp =
                 llvm::dyn_cast_or_null<IREE::HAL::BufferViewBufferOp>(
                     newValue_.getDefiningOp())) {
    return bufferViewBufferOp.buffer_view();
  } else {
    auto shapeDims = getShapeDims();
    if (!shapeDims) return {};
    return rewriter_.createOrFold<IREE::HAL::BufferViewCreateOp>(
        loc_, newValue_, getElementType(), getEncodingType(), *shapeDims);
  }
}

TensorType TensorRewriteAdaptor::getTensorType() {
  return oldValue_.getType().cast<TensorType>();
}

int32_t TensorRewriteAdaptor::getElementType() {
  return IREE::HAL::getElementTypeValue(getTensorType().getElementType())
      .getValueOr(0);
}

IntegerAttr TensorRewriteAdaptor::getElementTypeAttr() {
  auto type = getTensorType().getElementType();
  auto elementType = getElementTypeValue(type);
  if (!elementType) return {};
  return IntegerAttr::get(IntegerType::get(type.getContext(), 32),
                          elementType.getValue());
}

int32_t TensorRewriteAdaptor::getEncodingType() {
  return (int32_t)getEncodingTypeAttr().getValue().getZExtValue();
}

IntegerAttr TensorRewriteAdaptor::getEncodingTypeAttr() {
  // TODO(#6762): get encoding attribute from the tensor type.
  auto encodingType = getEncodingTypeValue({});
  if (!encodingType) return {};
  return IntegerAttr::get(IntegerType::get(loc_.getContext(), 32),
                          encodingType.getValue());
}

llvm::Optional<SmallVector<Value, 4>> TensorRewriteAdaptor::getShapeDims() {
  return IREE::HAL::getShapeDims(loc_, oldValue_, rewriter_);
}

llvm::Optional<SmallVector<Value, 4>> TensorRewriteAdaptor::getShapeDims(
    ConversionPatternRewriter &rewriter) {
  return IREE::HAL::getShapeDims(loc_, oldValue_, rewriter);
}

Value TensorRewriteAdaptor::getByteLength() {
  auto shapeDims = getShapeDims();
  if (!shapeDims) return {};
  return rewriter_.createOrFold<IREE::HAL::AllocatorComputeSizeOp>(
      loc_, getAllocator(), *shapeDims, getElementType(), getEncodingType());
}

Value TensorRewriteAdaptor::computeOffset(ValueRange indices) {
  auto shapeDims = getShapeDims();
  if (!shapeDims) return {};
  return rewriter_.createOrFold<IREE::HAL::AllocatorComputeOffsetOp>(
      loc_, getAllocator(), *shapeDims, getElementType(), getEncodingType(),
      indices);
}

llvm::Optional<TensorRewriteAdaptor::Range> TensorRewriteAdaptor::computeRange(
    ValueRange indices, ValueRange lengths) {
  auto shapeDims = getShapeDims();
  if (!shapeDims) return llvm::None;
  auto range = rewriter_.create<IREE::HAL::AllocatorComputeRangeOp>(
      loc_, getAllocator(), *shapeDims, getElementType(), getEncodingType(),
      indices, lengths);
  return Range{range.offset(), range.length()};
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
