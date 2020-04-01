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
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
      shape.push_back(rewriter.createOrFold<mlir::ConstantIndexOp>(loc, dim));
    }
  }
  return shape;
}

llvm::Optional<SmallVector<Value, 4>> getShapeDims(
    Location loc, Value shapedValue, ConversionPatternRewriter &rewriter) {
  ShapedType shapedType = shapedValue.getType().cast<ShapedType>();
  if (shapedType.hasStaticShape()) {
    return getStaticShapeDims(loc, shapedType, rewriter);
  } else {
    // Dynamic shape lookup.
    Value rsValue = Shape::buildOrFindRankedShapeForValue(
        loc, shapedValue, rewriter.getIndexType(), rewriter);
    if (!rsValue) {
      return llvm::None;
    }
    SmallVector<Value, 4> dims;
    // Note that in the following, we require that the dims resolve
    // to discrete SSA values, which in a stream, will be block args.
    if (failed(Shape::getRankedDimsFromRankedShape(loc, rsValue, false, dims,
                                                   rewriter))) {
      return llvm::None;
    }
    for (auto &dim : dims) {
      dim = rewriter.getRemappedValue(dim);
    }
    return dims;
  }
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
  auto shapeDims = getShapeDims();
  if (!shapeDims) return {};

  if (isBufferView()) {
    return newValue;
  } else {
    return rewriter.createOrFold<IREE::HAL::BufferViewCreateOp>(
        loc, newValue, *shapeDims, getElementType());
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

llvm::Optional<SmallVector<Value, 4>> TensorRewriteAdaptor::getShapeDims() {
  return IREE::HAL::getShapeDims(loc, oldValue, rewriter);
}

Value TensorRewriteAdaptor::getByteLength() {
  if (isBufferView()) {
    return rewriter.createOrFold<IREE::HAL::BufferViewByteLengthOp>(
        loc, getBufferView());
  } else {
    auto shapeDims = getShapeDims();
    if (!shapeDims) return {};
    return rewriter.createOrFold<IREE::HAL::AllocatorComputeSizeOp>(
        loc, getAllocator(), *shapeDims, getElementType());
  }
}

Value TensorRewriteAdaptor::computeOffset(ValueRange indices) {
  if (isBufferView()) {
    return rewriter.createOrFold<IREE::HAL::BufferViewComputeOffsetOp>(
        loc, getBufferView(), indices);
  } else {
    auto shapeDims = getShapeDims();
    if (!shapeDims) return {};
    return rewriter.createOrFold<IREE::HAL::AllocatorComputeOffsetOp>(
        loc, getAllocator(), *shapeDims, getElementType(), indices);
  }
}

llvm::Optional<TensorRewriteAdaptor::Range> TensorRewriteAdaptor::computeRange(
    ValueRange indices, ValueRange lengths) {
  if (isBufferView()) {
    auto range = rewriter.create<IREE::HAL::BufferViewComputeRangeOp>(
        loc, getBufferView(), indices, lengths);
    return Range{range.offset(), range.length()};
  } else {
    auto shapeDims = getShapeDims();
    if (!shapeDims) return llvm::None;
    auto range = rewriter.create<IREE::HAL::AllocatorComputeRangeOp>(
        loc, getAllocator(), *shapeDims, getElementType(), indices, lengths);
    return Range{range.offset(), range.length()};
  }
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
