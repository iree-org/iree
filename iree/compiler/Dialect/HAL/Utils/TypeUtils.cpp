// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

int32_t getRoundedElementByteWidth(Type type) {
  return (type.getIntOrFloatBitWidth() + 8 - 1) / 8;
}

SmallVector<Value, 4> getStaticShapeDims(Location loc, ShapedType shapedType,
                                         PatternRewriter &rewriter) {
  SmallVector<Value, 4> shape;
  if (shapedType.getRank() >= 1) {
    for (auto dim : shapedType.getShape()) {
      shape.push_back(rewriter.createOrFold<mlir::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(dim))));
    }
  }
  return shape;
}

SmallVector<Value, 4> getShapeDims(Value shapedValue,
                                   PatternRewriter &rewriter) {
  // TODO(benvanik): dynamic shape support.
  return getStaticShapeDims(shapedValue.getLoc(),
                            shapedValue.getType().cast<ShapedType>(), rewriter);
}

Value TensorRewriteAdaptor::getAllocator() {
  return rewriter.createOrFold<IREE::HAL::BufferAllocatorOp>(loc, getBuffer());
}

bool TensorRewriteAdaptor::isBufferView() {
  return newValue.getType().isa<IREE::HAL::BufferViewType>();
}

Value TensorRewriteAdaptor::getBuffer() {
  if (isBufferView()) {
    return rewriter.createOrFold<IREE::HAL::BufferViewBufferOp>(loc, newValue);
  } else {
    return newValue;
  }
}

Value TensorRewriteAdaptor::getBufferView() {
  if (isBufferView()) {
    return newValue;
  } else {
    return rewriter.createOrFold<IREE::HAL::BufferViewCreateOp>(
        loc, newValue, getShapeDims(), getElementType());
  }
}

TensorType TensorRewriteAdaptor::getTensorType() {
  return oldValue.getType().cast<TensorType>();
}

int32_t TensorRewriteAdaptor::getElementType() {
  return IREE::HAL::getElementTypeValue(getTensorType().getElementType())
      .getValueOr(0);
}

IntegerAttr TensorRewriteAdaptor::getElementTypeAttr() {
  return IREE::HAL::getElementTypeAttr(getTensorType().getElementType());
}

SmallVector<Value, 4> TensorRewriteAdaptor::getShapeDims() {
  // TODO(benvanik): replace with actual ranked shape tracking to newValue.
  return IREE::HAL::getShapeDims(oldValue, rewriter);
}

Value TensorRewriteAdaptor::getByteLength() {
  if (isBufferView()) {
    return rewriter.createOrFold<IREE::HAL::BufferViewByteLengthOp>(
        loc, getBufferView());
  } else {
    return rewriter.createOrFold<IREE::HAL::AllocatorComputeSizeOp>(
        loc, getAllocator(), getShapeDims(), getElementType());
  }
}

Value TensorRewriteAdaptor::computeOffset(ValueRange indices) {
  if (isBufferView()) {
    return rewriter.createOrFold<IREE::HAL::BufferViewComputeOffsetOp>(
        loc, getBufferView(), indices);
  } else {
    return rewriter.createOrFold<IREE::HAL::AllocatorComputeOffsetOp>(
        loc, getAllocator(), getShapeDims(), getElementType(), indices);
  }
}

TensorRewriteAdaptor::Range TensorRewriteAdaptor::computeRange(
    ValueRange indices, ValueRange lengths) {
  if (isBufferView()) {
    auto range = rewriter.create<IREE::HAL::BufferViewComputeRangeOp>(
        loc, getBufferView(), indices, lengths);
    return Range{range.offset(), range.length()};
  } else {
    auto range = rewriter.create<IREE::HAL::AllocatorComputeRangeOp>(
        loc, getAllocator(), getShapeDims(), getElementType(), indices,
        lengths);
    return Range{range.offset(), range.length()};
  }
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
