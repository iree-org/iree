// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

class BufferLoadOpConversion
    : public OpConversionPattern<IREE::HAL::BufferLoadOp> {
public:
  BufferLoadOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                         TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::BufferLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    auto originalType = op.getResult().getType();
    auto targetType = typeConverter->convertType(originalType);
    auto targetBitwidth = IREE::Util::getTypeBitWidth(targetType);
    int32_t validByteWidth =
        IREE::Util::getRoundedElementByteWidth(originalType);

    if (originalType.isF16() || originalType.isBF16()) {
      return rewriter.notifyMatchFailure(
          op, "half-width floats not supported on the host (yet)");
    }

    auto sourceOffset = castToImportType(adaptor.getSourceOffset(),
                                         rewriter.getI64Type(), rewriter);

    // 32-bit values are loaded directly, 64-bit are combined from 32 | 32.
    Value value;
    if (validByteWidth <= 4) {
      auto byteWidth = arith::ConstantIntOp::create(rewriter, op.getLoc(),
                                                    validByteWidth, 32);
      auto callOp = IREE::VM::CallOp::create(
          rewriter, op.getLoc(), SymbolRefAttr::get(importOp),
          importType.getResults(),
          ArrayRef<Value>{adaptor.getSourceBuffer(), sourceOffset, byteWidth});
      copyImportAttrs(importOp, callOp);
      value = callOp.getResult(0);
    } else {
      auto halfByteWidth =
          arith::ConstantIntOp::create(rewriter, op.getLoc(), 4, 32);

      // value = (i64(hi) << 32) | i64(lo)
      auto hiOffset = rewriter.createOrFold<arith::AddIOp>(
          op.getLoc(), sourceOffset,
          IREE::VM::ConstI64Op::create(rewriter, op.getLoc(), 4));
      auto hiCallOp = IREE::VM::CallOp::create(
          rewriter, op.getLoc(), SymbolRefAttr::get(importOp),
          importType.getResults(),
          ArrayRef<Value>{adaptor.getSourceBuffer(), hiOffset, halfByteWidth});
      auto hi = arith::ShLIOp::create(
          rewriter, op.getLoc(),
          arith::ExtUIOp::create(rewriter, op.getLoc(),
                                 rewriter.getIntegerType(targetBitwidth),
                                 hiCallOp.getResult(0)),
          arith::ConstantIntOp::create(rewriter, op.getLoc(), 32, 32));

      auto loCallOp = IREE::VM::CallOp::create(
          rewriter, op.getLoc(), SymbolRefAttr::get(importOp),
          importType.getResults(),
          ArrayRef<Value>{adaptor.getSourceBuffer(), sourceOffset,
                          halfByteWidth});
      auto lo = arith::ExtUIOp::create(rewriter, op.getLoc(),
                                       rewriter.getIntegerType(targetBitwidth),
                                       loCallOp.getResult(0));

      value = arith::OrIOp::create(rewriter, op.getLoc(), lo, hi);
    }

    // i32 -> f32, etc
    if (isa<FloatType>(targetType)) {
      value =
          arith::BitcastOp::create(rewriter, op.getLoc(), targetType, value);
    }

    rewriter.replaceOp(op, {value});
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

class BufferStoreOpConversion
    : public OpConversionPattern<IREE::HAL::BufferStoreOp> {
public:
  BufferStoreOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                          TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::BufferStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    auto elementType = op.getValue().getType();
    int32_t validByteWidth =
        IREE::Util::getRoundedElementByteWidth(elementType);

    if (elementType.isF16() || elementType.isBF16()) {
      return rewriter.notifyMatchFailure(
          op, "half-width floats not supported on the host (yet)");
    }

    auto targetOffset = castToImportType(adaptor.getTargetOffset(),
                                         rewriter.getI64Type(), rewriter);

    // f32 -> i32, etc
    auto value = adaptor.getValue();
    if (isa<FloatType>(elementType)) {
      value = rewriter.createOrFold<arith::BitcastOp>(
          op.getLoc(),
          rewriter.getIntegerType(value.getType().getIntOrFloatBitWidth()),
          value);
    }

    // 32-bit values are stored directly, 64-bit are split into 32 | 32.
    if (validByteWidth <= 4) {
      auto byteWidth = arith::ConstantIntOp::create(rewriter, op.getLoc(),
                                                    validByteWidth, 32);
      auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
          op, SymbolRefAttr::get(importOp), importType.getResults(),
          ArrayRef<Value>{value, adaptor.getTargetBuffer(), targetOffset,
                          byteWidth});
      copyImportAttrs(importOp, callOp);
    } else {
      auto halfByteWidth =
          arith::ConstantIntOp::create(rewriter, op.getLoc(), 4, 32);

      auto lo = rewriter.createOrFold<arith::TruncIOp>(
          op.getLoc(), rewriter.getI32Type(), value);
      auto loOffset = targetOffset;
      auto loCallOp = IREE::VM::CallOp::create(
          rewriter, op.getLoc(), SymbolRefAttr::get(importOp),
          importType.getResults(),
          ArrayRef<Value>{lo, adaptor.getTargetBuffer(), loOffset,
                          halfByteWidth});
      copyImportAttrs(importOp, loCallOp);

      auto hi = rewriter.createOrFold<arith::TruncIOp>(
          op.getLoc(), rewriter.getI32Type(),
          rewriter.createOrFold<arith::ShRUIOp>(
              op.getLoc(), value,
              arith::ConstantIntOp::create(rewriter, op.getLoc(), 32, 64)));
      auto hiOffset = rewriter.createOrFold<arith::AddIOp>(
          op.getLoc(), targetOffset,
          IREE::VM::ConstI64Op::create(rewriter, op.getLoc(), 4));
      auto hiCallOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
          op, SymbolRefAttr::get(importOp), importType.getResults(),
          ArrayRef<Value>{hi, adaptor.getTargetBuffer(), hiOffset,
                          halfByteWidth});
      copyImportAttrs(importOp, hiCallOp);
    }

    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

struct MemoryTypeOpConversion
    : public OpConversionPattern<IREE::HAL::MemoryTypeOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::HAL::MemoryTypeOp op,
                  IREE::HAL::MemoryTypeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(
        op, op.getTypeAttr().getInt());
    return success();
  }
};

struct BufferUsageOpConversion
    : public OpConversionPattern<IREE::HAL::BufferUsageOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::HAL::BufferUsageOp op,
                  IREE::HAL::BufferUsageOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(
        op, op.getUsageAttr().getInt());
    return success();
  }
};

void populateHALBufferToVMPatterns(MLIRContext *context,
                                   SymbolTable &importSymbols,
                                   TypeConverter &typeConverter,
                                   RewritePatternSet &patterns) {
  patterns.insert<MemoryTypeOpConversion>(context);
  patterns.insert<BufferUsageOpConversion>(context);
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferAssertOp>>(
      context, importSymbols, typeConverter, "hal.buffer.assert");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferAllocationPreserveOp>>(
      context, importSymbols, typeConverter, "hal.buffer.allocation.preserve");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferAllocationDiscardOp>>(
      context, importSymbols, typeConverter, "hal.buffer.allocation.discard");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::BufferAllocationIsTerminalOp>>(
          context, importSymbols, typeConverter,
          "hal.buffer.allocation.is_terminal");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferSubspanOp>>(
      context, importSymbols, typeConverter, "hal.buffer.subspan");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferLengthOp>>(
      context, importSymbols, typeConverter, "hal.buffer.length");
  patterns.insert<BufferLoadOpConversion>(context, importSymbols, typeConverter,
                                          "hal.buffer.load");
  patterns.insert<BufferStoreOpConversion>(context, importSymbols,
                                           typeConverter, "hal.buffer.store");
}

} // namespace mlir::iree_compiler
