// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/HAL/Inline/IR/HALInlineOps.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace Inline {

//===----------------------------------------------------------------------===//
// hal_inline.buffer.allocate
//===----------------------------------------------------------------------===//

void BufferAllocateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer");
  setNameFn(getStorage(), "storage");
}

Value BufferAllocateOp::getOperandSize(unsigned idx) { return {}; }

Value BufferAllocateOp::getResultSize(unsigned idx) {
  return getAllocationSize();
}

//===----------------------------------------------------------------------===//
// hal_inline.buffer.allocate.initialized
//===----------------------------------------------------------------------===//

void BufferAllocateInitializedOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer");
  setNameFn(getStorage(), "storage");
}

Value BufferAllocateInitializedOp::getOperandSize(unsigned idx) { return {}; }

Value BufferAllocateInitializedOp::getResultSize(unsigned idx) {
  return getLength();
}

//===----------------------------------------------------------------------===//
// hal_inline.buffer.wrap
//===----------------------------------------------------------------------===//

void BufferWrapOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "mapped");
}

Value BufferWrapOp::getOperandSize(unsigned idx) { return {}; }

Value BufferWrapOp::getResultSize(unsigned idx) { return getLength(); }

//===----------------------------------------------------------------------===//
// hal_inline.buffer.subspan
//===----------------------------------------------------------------------===//

void BufferSubspanOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer");
}

Value BufferSubspanOp::getOperandSize(unsigned idx) { return getLength(); }

Value BufferSubspanOp::getResultSize(unsigned idx) { return getLength(); }

//===----------------------------------------------------------------------===//
// hal_inline.buffer.byte_length
//===----------------------------------------------------------------------===//

void BufferLengthOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "length");
}

OpFoldResult BufferLengthOp::fold(ArrayRef<Attribute> operands) {
  Operation *op = this->getOperation();
  return IREE::Util::SizeAwareTypeInterface::findSizeValue(
      getBuffer(), op->getBlock(), Block::iterator(op));
}

//===----------------------------------------------------------------------===//
// hal_inline.buffer.storage
//===----------------------------------------------------------------------===//

void BufferStorageOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "storage");
}

OpFoldResult BufferStorageOp::fold(ArrayRef<Attribute> operands) {
  auto *definingOp = getBuffer().getDefiningOp();
  if (!definingOp) return {};
  if (auto sourceOp =
          dyn_cast_or_null<IREE::HAL::Inline::BufferAllocateOp>(definingOp)) {
    return sourceOp.getStorage();
  } else if (auto sourceOp = dyn_cast_or_null<
                 IREE::HAL::Inline::BufferAllocateInitializedOp>(definingOp)) {
    return sourceOp.getStorage();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// hal_inline.buffer_view.create
//===----------------------------------------------------------------------===//

void BufferViewCreateOp::build(OpBuilder &builder, OperationState &state,
                               Value buffer, int32_t elementType,
                               int32_t encodingType, ValueRange shape) {
  build(builder, state, buffer,
        builder.createOrFold<arith::ConstantIntOp>(state.location, elementType,
                                                   32),
        builder.createOrFold<arith::ConstantIntOp>(state.location, encodingType,
                                                   32),
        shape);
}

void BufferViewCreateOp::build(OpBuilder &builder, OperationState &state,
                               Value buffer, Value elementType,
                               Value encodingType, ValueRange shape) {
  state.addOperands({buffer, elementType, encodingType});
  state.addOperands(shape);
  state.addTypes({BufferViewType::get(builder.getContext())});
}

void BufferViewCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "view");
}

//===----------------------------------------------------------------------===//
// hal_inline.buffer_view.buffer
//===----------------------------------------------------------------------===//

void BufferViewBufferOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer");
}

namespace {

/// Skips a hal.buffer_view.buffer accessor when the buffer view was created in
/// the same scope and we know the origin buffer.
struct SkipBufferViewBufferOp : public OpRewritePattern<BufferViewBufferOp> {
  using OpRewritePattern<BufferViewBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferViewBufferOp op,
                                PatternRewriter &rewriter) const override {
    if (auto createOp = dyn_cast_or_null<BufferViewCreateOp>(
            op.getBufferView().getDefiningOp())) {
      rewriter.replaceOp(op, createOp.getBuffer());
      return success();
    }
    return failure();
  }
};

}  // namespace

void BufferViewBufferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.insert<SkipBufferViewBufferOp>(context);
}

//===----------------------------------------------------------------------===//
// hal_inline.device.query
//===----------------------------------------------------------------------===//

LogicalResult DeviceQueryOp::verify() {
  DeviceQueryOp op = *this;
  if (op.getDefaultValue().has_value()) {
    if (auto typedDefaultValue = op.getDefaultValue()->dyn_cast<TypedAttr>()) {
      if (typedDefaultValue.getType() != op.getValue().getType()) {
        return op.emitOpError()
               << "type mismatch between result and default value";
      }
    }
  }
  return success();
}

}  // namespace Inline
}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Modules/HAL/Inline/IR/HALInlineOps.cpp.inc"
