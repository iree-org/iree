// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONTARGET_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// A conversion target for the HAL dialect that ensures that tensor types are
// fully removed. Conversions targeting the HAL dialect should always use this.
class HALConversionTarget : public ConversionTarget {
 public:
  HALConversionTarget(MLIRContext *context, TypeConverter &typeConverter);

  // Attempts to rewrite an op that may use tensor values into an op using HAL
  // buffers. See HALOpConversion for more information.
  static LogicalResult applyDefaultBufferRewrite(
      Operation *srcOp, ValueRange operands, StringRef dstOpName,
      TypeConverter &typeConverter, ConversionPatternRewriter &rewriter);
};

// HAL tensor-to-buffer conversion utility.
// This can be used by dialects to model custom op conversion from a dialect
// that uses the MLIR tensor type to the IREE HAL buffer type. At this point
// during conversion the source values will be TensorType and the target values
// will be IREE::HAL::BufferTypes. Any static information available about the
// tensor (such as static dimensions, element type, layout, etc) are extracted
// here and lowered as expanded values.
//
// The ABI is currently very basic and will change with the introduction of more
// dynamic shape logic.
//
// Source:
//   my.tensor_op(%arg0 : tensor<2x4xf32>)
// Target:
//   %arg0_view = hal.buffer_view.create %arg0, ...
//   my.buffer_op(%arg0_view : !hal.buffer_view)
template <typename SRC, typename DST>
class HALOpConversion : public OpConversionPattern<SRC> {
 public:
  HALOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern<SRC>(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      SRC srcOp, typename SRC::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    return HALConversionTarget::applyDefaultBufferRewrite(
        srcOp, adaptor.getOperands(), DST::getOperationName(), typeConverter,
        rewriter);
  }

 protected:
  TypeConverter &typeConverter;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERSIONTARGET_H_
