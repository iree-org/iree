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

#ifndef IREE_COMPILER_DIALECT_HAL_UTILS_TYPEUTILS_H_
#define IREE_COMPILER_DIALECT_HAL_UTILS_TYPEUTILS_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Returns the number of bytes an element of the given type occupies
// post-conversion. For example, the size of i1 would be '1 byte'.
int32_t getRoundedElementByteWidth(Type type);

// Returns an array of i32 values representing the shape of the |shapedType|.
SmallVector<Value, 4> getStaticShapeDims(Location loc, ShapedType shapedType,
                                         OpBuilder &builder);

// Returns an array of i32 values representing the shape of the |shapedValue|.
llvm::Optional<SmallVector<Value, 4>> getShapeDims(Location loc,
                                                   Value shapedValue,
                                                   OpBuilder &builder);

// An adaptor used for tensor->buffer rewrites.
// This abstracts the source and destination types to allow for implicit
// conversion between buffers and buffer views. Always prefer using this when
// mapping between the types to ensure that the conversion framework can
// flexibly choose which type to use based on target ops.
class TensorRewriteAdaptor {
 public:
  // Returns whether the given type can adapted from a Tensor.
  static bool isValidNewType(Type newType);

  // Emits an error and returns failure if invariants are not satisfied.
  static LogicalResult verifyConstructionInvariants(
      Location loc, Value oldValue, Value newValue,
      ConversionPatternRewriter &rewriter);

  // Create an adaptor between the given values.
  // Aborts if the values cannot be adapted.
  static TensorRewriteAdaptor get(Location loc, Value oldValue, Value newValue,
                                  ConversionPatternRewriter &rewriter);

  // Create an adaptor between the given values.
  // If the values cannot be adapted, emits an error and returns empty.
  static llvm::Optional<TensorRewriteAdaptor> getChecked(
      Location loc, Value oldValue, Value newValue,
      ConversionPatternRewriter &rewriter);

  // Gets the allocator this buffer was allocated with.
  Value getAllocator();

  // Returns true if the new value is a buffer view type.
  bool isBufferView();

  // Returns a hal.buffer type for the value.
  Value getBuffer();

  // Returns a hal.buffer_view type for the value.
  Value getBufferView();

  // Returns the original tensor type of the value.
  TensorType getTensorType();

  // Returns the element type of the tensor as the int32 packed value.
  int32_t getElementType();
  IntegerAttr getElementTypeAttr();

  // Returns the I32 shape dimensions of the tensor.
  llvm::Optional<SmallVector<Value, 4>> getShapeDims();
  llvm::Optional<SmallVector<Value, 4>> getShapeDims(OpBuilder &builder);

  // Performs the equivalent of a hal.buffer_view.byte_length.
  Value getByteLength();

  // Performs the equivalent of a hal.buffer_view.compute_offset.
  Value computeOffset(ValueRange indices);

  struct Range {
    Value offset;
    Value length;
  };

  // Performs the equivalent of a hal.buffer_view.compute_range.
  llvm::Optional<Range> computeRange(ValueRange indices, ValueRange lengths);

 private:
  TensorRewriteAdaptor(Location loc, Value oldValue, Value newValue,
                       ConversionPatternRewriter &rewriter)
      : loc_(loc),
        oldValue_(oldValue),
        newValue_(newValue),
        rewriter_(rewriter) {}

  Location loc_;
  Value oldValue_;
  Value newValue_;
  ConversionPatternRewriter &rewriter_;
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_UTILS_TYPEUTILS_H_
