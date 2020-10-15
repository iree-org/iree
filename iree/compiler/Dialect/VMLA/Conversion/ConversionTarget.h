// Copyright 2020 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_VMLA_CONVERSION_CONVERSIONTARGET_H_
#define IREE_COMPILER_DIALECT_VMLA_CONVERSION_CONVERSIONTARGET_H_

#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

enum class VMLAOpSemantics {
  kDefault = 0,
  // Forces integers to be treated as unsigned integers.
  kForceUnsigned,
};

// A conversion target for the VMLA dialect that ensures that tensor types are
// fully removed. Conversions targeting the VMLA dialect should always use this.
class VMLAConversionTarget : public ConversionTarget {
 public:
  VMLAConversionTarget(MLIRContext *context, TypeConverter &typeConverter);

  // Attempts to rewrite an op that may use tensor values into an op using VMLA
  // buffers. See VMLAOpConversion for more information.
  static LogicalResult applyDefaultBufferRewrite(
      Operation *srcOp, ArrayRef<Value> operands, VMLAOpSemantics semantics,
      StringRef dstOpName, TypeConverter &typeConverter,
      ConversionPatternRewriter &rewriter);

  // Returns the shape of the |originalValue| tensor as an SSA ranked shape.
  static Value getTensorShape(Location loc, Value originalValue,
                              TypeConverter &typeConverter,
                              ConversionPatternRewriter &rewriter);

  // Returns the offset, in bytes, of an index within a linearized dense buffer.
  static Value getBufferOffset(Location loc, Value tensorValue,
                               Value indicesValue, TypeConverter &typeConverter,
                               ConversionPatternRewriter &rewriter);
  static Value getBufferOffset(Location loc, Value tensorValue,
                               ValueRange indices, TypeConverter &typeConverter,
                               ConversionPatternRewriter &rewriter);

  // Returns the length, in bytes, of a linearized dense buffer.
  static Value getBufferLength(Location loc, Value tensorValue,
                               TypeConverter &typeConverter,
                               ConversionPatternRewriter &rewriter);

  // Allocates a VMLA buffer for an output operand of an op.
  // Returns a buffer allocated with the appropriate size for storing the value.
  // Callers must replace uses of |originalValue| with the returned value.
  static Value allocateOutputBuffer(Location loc, Value originalValue,
                                    TypeConverter &typeConverter,
                                    ConversionPatternRewriter &rewriter);

 private:
  bool isDynamicallyLegal(Operation *op) const override;

  TypeConverter &typeConverter;
};

// VMLA tensor-to-buffer conversion utility.
// This can be used by dialects to model custom op conversion from a dialect
// that uses the MLIR tensor type to the IREE VMLA buffer type. At this point
// during conversion the source values will be TensorType and the target values
// will be IREE::VMLA::BufferTypes. Any static information available about the
// tensor (such as static dimensions, element type, layout, etc) are extracted
// here and lowered as expanded values.
template <typename SRC, typename DST,
          VMLAOpSemantics semantics = VMLAOpSemantics::kDefault>
class VMLAOpConversion : public OpConversionPattern<SRC> {
 public:
  using OpConversionPattern<SRC>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SRC srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    return VMLAConversionTarget::applyDefaultBufferRewrite(
        srcOp, operands, semantics, DST::getOperationName(),
        *this->getTypeConverter(), rewriter);
  }
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VMLA_CONVERSION_CONVERSIONTARGET_H_
