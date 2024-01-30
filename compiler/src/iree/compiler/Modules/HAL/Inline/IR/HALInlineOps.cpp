// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::iree_compiler::IREE::HAL::Inline {

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

OpFoldResult BufferLengthOp::fold(FoldAdaptor operands) {
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

OpFoldResult BufferStorageOp::fold(FoldAdaptor operands) {
  auto *definingOp = getBuffer().getDefiningOp();
  if (!definingOp)
    return {};
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
                               Value sourceBuffer, Value sourceOffset,
                               Value sourceLength, int32_t elementType,
                               int32_t encodingType, ValueRange shape) {
  build(builder, state, sourceBuffer, sourceOffset, sourceLength,
        builder.createOrFold<arith::ConstantIntOp>(state.location, elementType,
                                                   32),
        builder.createOrFold<arith::ConstantIntOp>(state.location, encodingType,
                                                   32),
        shape);
}

void BufferViewCreateOp::build(OpBuilder &builder, OperationState &state,
                               Value sourceBuffer, Value sourceOffset,
                               Value sourceLength, Value elementType,
                               Value encodingType, ValueRange shape) {
  state.addOperands(
      {sourceBuffer, sourceOffset, sourceLength, elementType, encodingType});
  state.addOperands(shape);
  state.addTypes({BufferViewType::get(builder.getContext())});
}

void BufferViewCreateOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "view");
}

namespace {

/// Folds hal_inline.buffer_view.subspans into buffer view creation subspans.
struct FoldBufferViewCreateSubspan
    : public OpRewritePattern<BufferViewCreateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BufferViewCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto newSourceBuffer = op.getSourceBuffer();
    auto newSourceOffset = llvm::cast<Value>(op.getSourceOffset());
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.getSourceBuffer().getDefiningOp())) {
      newSourceBuffer = subspanOp.getSourceBuffer();
      newSourceOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(),
          op.getSourceOffset());
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate)
      return failure();
    rewriter.modifyOpInPlace(op, [&]() {
      op.getSourceBufferMutable().assign(newSourceBuffer);
      op.getSourceOffsetMutable().assign(newSourceOffset);
    });
    return success();
  }
};

} // namespace

void BufferViewCreateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.insert<FoldBufferViewCreateSubspan>(context);
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
      rewriter.replaceOp(op, createOp.getSourceBuffer());
      return success();
    }
    return failure();
  }
};

} // namespace

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
    if (auto typedDefaultValue =
            llvm::dyn_cast<TypedAttr>(*op.getDefaultValue())) {
      if (typedDefaultValue.getType() != op.getValue().getType()) {
        return op.emitOpError()
               << "type mismatch between result and default value";
      }
    }
  }
  return success();
}

} // namespace mlir::iree_compiler::IREE::HAL::Inline

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.cpp.inc"
