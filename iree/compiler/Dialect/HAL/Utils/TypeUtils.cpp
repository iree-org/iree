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
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

int32_t getRoundedElementByteWidth(Type type) {
  return (type.getIntOrFloatBitWidth() + 8 - 1) / 8;
}

SmallVector<Value, 4> getStaticShapeDims(Location loc, ShapedType shapedType,
                                         OpBuilder &builder) {
  SmallVector<Value, 4> shape;
  if (shapedType.getRank() >= 1) {
    for (auto dim : shapedType.getShape()) {
      shape.push_back(builder.createOrFold<mlir::ConstantIndexOp>(loc, dim));
    }
  }
  return shape;
}

llvm::Optional<SmallVector<Value, 4>> getShapeDims(Location loc,
                                                   Value shapedValue,
                                                   OpBuilder &builder) {
  ShapedType shapedType = shapedValue.getType().cast<ShapedType>();
  if (shapedType.hasStaticShape()) {
    return getStaticShapeDims(loc, shapedType, builder);
  } else {
    // Dynamic shape lookup.
    Value rsValue = Shape::buildOrFindRankedShapeForValue(
        loc, shapedValue, builder.getIndexType(), builder);
    if (!rsValue) {
      return llvm::None;
    }
    SmallVector<Value, 4> dims;
    // Note that in the following, we require that the dims resolve
    // to discrete SSA values, which in a stream, will be block args.
    if (failed(Shape::getRankedDimsFromRankedShape(
            loc, rsValue, /*createIntermediateOps=*/true, dims, builder))) {
      return llvm::None;
    }
    return dims;
  }
}

// static
bool TensorRewriteAdaptor::isValidNewType(Type newType) {
  return newType.isa<IREE::HAL::BufferViewType>() ||
         newType.isa<IREE::HAL::BufferType>();
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
  return rewriter_.createOrFold<IREE::HAL::BufferAllocatorOp>(loc_,
                                                              getBuffer());
}

bool TensorRewriteAdaptor::isBufferView() {
  return newValue_.getType().isa<IREE::HAL::BufferViewType>();
}

Value TensorRewriteAdaptor::getBuffer() {
  if (isBufferView()) {
    return rewriter_.createOrFold<IREE::HAL::BufferViewBufferOp>(loc_,
                                                                 newValue_);
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
        loc_, newValue_, *shapeDims, getElementType());
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
  return IREE::HAL::getElementTypeAttr(getTensorType().getElementType());
}

llvm::Optional<SmallVector<Value, 4>> TensorRewriteAdaptor::getShapeDims() {
  return IREE::HAL::getShapeDims(loc_, oldValue_, rewriter_);
}
llvm::Optional<SmallVector<Value, 4>> TensorRewriteAdaptor::getShapeDims(
    OpBuilder &builder) {
  return IREE::HAL::getShapeDims(loc_, oldValue_, builder);
}

Value TensorRewriteAdaptor::getByteLength() {
  if (isBufferView()) {
    return rewriter_.createOrFold<IREE::HAL::BufferViewByteLengthOp>(
        loc_, getBufferView());
  } else {
    auto shapeDims = getShapeDims();
    if (!shapeDims) return {};
    return rewriter_.createOrFold<IREE::HAL::AllocatorComputeSizeOp>(
        loc_, getAllocator(), *shapeDims, getElementType());
  }
}

Value TensorRewriteAdaptor::computeOffset(ValueRange indices) {
  if (isBufferView()) {
    return rewriter_.createOrFold<IREE::HAL::BufferViewComputeOffsetOp>(
        loc_, getBufferView(), indices);
  } else {
    auto shapeDims = getShapeDims();
    if (!shapeDims) return {};
    return rewriter_.createOrFold<IREE::HAL::AllocatorComputeOffsetOp>(
        loc_, getAllocator(), *shapeDims, getElementType(), indices);
  }
}

llvm::Optional<TensorRewriteAdaptor::Range> TensorRewriteAdaptor::computeRange(
    ValueRange indices, ValueRange lengths) {
  if (isBufferView()) {
    auto range = rewriter_.create<IREE::HAL::BufferViewComputeRangeOp>(
        loc_, getBufferView(), indices, lengths);
    return Range{range.offset(), range.length()};
  } else {
    auto shapeDims = getShapeDims();
    if (!shapeDims) return llvm::None;
    auto range = rewriter_.create<IREE::HAL::AllocatorComputeRangeOp>(
        loc_, getAllocator(), *shapeDims, getElementType(), indices, lengths);
    return Range{range.offset(), range.length()};
  }
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
