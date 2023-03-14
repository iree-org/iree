// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"

#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

HALConversionTarget::HALConversionTarget(MLIRContext *context,
                                         TypeConverter &typeConverter)
    : ConversionTarget(*context) {
  // The HAL dialect allows hal ops as input as we may be running on partially
  // processed files or may have already lowered some constructs (like constant
  // pools).
  addLegalDialect("hal");

  // We don't care about the contents of a HAL executable: it may have any kind
  // of dialect and type usage.
  addLegalOp<IREE::HAL::ExecutableOp>();
  markOpRecursivelyLegal<IREE::HAL::ExecutableOp>();

  // Setup the fallback handler such that all ops without explicitly
  // registered patterns will be checked to ensure that they don't use any
  // illegal types.
  markUnknownOpDynamicallyLegal([&](Operation *op) {
    // Short-circuit test that bails on the first illegal type.
    const auto isTypeIllegal = [&](Type type) {
      return !typeConverter.isLegal(type);
    };
    return !(llvm::any_of(op->getOperandTypes(), isTypeIllegal) ||
             llvm::any_of(op->getResultTypes(), isTypeIllegal));
  });
}

// static
LogicalResult HALConversionTarget::applyDefaultBufferRewrite(
    Operation *srcOp, ValueRange operands, StringRef dstOpName,
    TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  OperationState state{srcOp->getLoc(), dstOpName};
  state.addAttributes(srcOp->getAttrs());

  for (auto [srcOperand, dstOperand] :
       llvm::zip_equal(srcOp->getOperands(), operands)) {
    // Check that any type that should have been mapped to buffer view was.
    // This is just to catch conflicts in type conversions that may sneak in
    // during development.
    assert(!HALTypeConverter::shouldConvertToBufferView(srcOperand.getType()) ||
           dstOperand.getType().isa<IREE::HAL::BufferViewType>() &&
               "expect that tensors have been mapped to buffer views");
    state.addOperands({dstOperand});
  }
  for (auto resultType : srcOp->getResultTypes()) {
    if (HALTypeConverter::shouldConvertToBufferView(resultType)) {
      state.addTypes(IREE::HAL::BufferViewType::get(rewriter.getContext()));
    } else {
      // Normal pass-through result.
      if (failed(typeConverter.convertType(resultType, state.types))) {
        return failure();
      }
    }
  }

  auto *dstOp = rewriter.create(state);
  rewriter.replaceOp(srcOp, dstOp->getResults());
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
