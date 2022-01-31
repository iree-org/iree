// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

class BufferLoadOpConversion
    : public OpConversionPattern<IREE::HAL::BufferLoadOp> {
 public:
  BufferLoadOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                         TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::BufferLoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getType();

    auto originalType = op.result().getType();
    auto targetType = typeConverter->convertType(op.result().getType());
    int32_t validByteWidth =
        IREE::Util::getRoundedElementByteWidth(originalType);

    if (originalType.isF16() || originalType.isBF16()) {
      return rewriter.notifyMatchFailure(
          op, "half-width floats not supported on the host (yet)");
    }

    // 32-bit values are loaded directly, 64-bit are combined from 32 | 32.
    Value value;
    if (validByteWidth <= 4) {
      auto byteWidth = rewriter.create<arith::ConstantIntOp>(
          op.getLoc(), validByteWidth, 32);
      auto callOp = rewriter.create<IREE::VM::CallOp>(
          op.getLoc(), SymbolRefAttr::get(importOp), importType.getResults(),
          ArrayRef<Value>{adaptor.source_buffer(), adaptor.source_offset(),
                          byteWidth});
      copyImportAttrs(importOp, callOp);
      value = callOp.getResult(0);
    } else {
      auto halfByteWidth =
          rewriter.create<arith::ConstantIntOp>(op.getLoc(), 4, 32);

      // value = (i64(hi) << 32) | i64(lo)
      auto hiOffset = rewriter.createOrFold<arith::AddIOp>(
          op.getLoc(), adaptor.source_offset(),
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 4));
      auto hiCallOp = rewriter.create<IREE::VM::CallOp>(
          op.getLoc(), SymbolRefAttr::get(importOp), importType.getResults(),
          ArrayRef<Value>{adaptor.source_buffer(), hiOffset, halfByteWidth});
      auto hi = rewriter.create<arith::ShLIOp>(
          op.getLoc(),
          rewriter.create<arith::ExtUIOp>(
              op.getLoc(),
              rewriter.getIntegerType(targetType.getIntOrFloatBitWidth()),
              hiCallOp.getResult(0)),
          rewriter.create<arith::ConstantIntOp>(op.getLoc(), 32, 32));

      auto loCallOp = rewriter.create<IREE::VM::CallOp>(
          op.getLoc(), SymbolRefAttr::get(importOp), importType.getResults(),
          ArrayRef<Value>{adaptor.source_buffer(), adaptor.source_offset(),
                          halfByteWidth});
      auto lo = rewriter.create<arith::ExtUIOp>(
          op.getLoc(),
          rewriter.getIntegerType(targetType.getIntOrFloatBitWidth()),
          loCallOp.getResult(0));

      value = rewriter.create<arith::OrIOp>(op.getLoc(), lo, hi);
    }

    // i32 -> f32, etc
    if (targetType.isa<FloatType>()) {
      value = rewriter.create<arith::BitcastOp>(op.getLoc(), targetType, value);
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

  LogicalResult matchAndRewrite(
      IREE::HAL::BufferStoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getType();

    auto elementType = op.value().getType();
    int32_t validByteWidth =
        IREE::Util::getRoundedElementByteWidth(elementType);

    if (elementType.isF16() || elementType.isBF16()) {
      return rewriter.notifyMatchFailure(
          op, "half-width floats not supported on the host (yet)");
    }

    // f32 -> i32, etc
    auto value = adaptor.value();
    if (elementType.isa<FloatType>()) {
      value = rewriter.createOrFold<arith::BitcastOp>(
          op.getLoc(),
          rewriter.getIntegerType(value.getType().getIntOrFloatBitWidth()),
          value);
    }

    // 32-bit values are stored directly, 64-bit are split into 32 | 32.
    if (validByteWidth <= 4) {
      auto byteWidth = rewriter.create<arith::ConstantIntOp>(
          op.getLoc(), validByteWidth, 32);
      auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
          op, SymbolRefAttr::get(importOp), importType.getResults(),
          ArrayRef<Value>{value, adaptor.target_buffer(),
                          adaptor.target_offset(), byteWidth});
      copyImportAttrs(importOp, callOp);
    } else {
      auto halfByteWidth =
          rewriter.create<arith::ConstantIntOp>(op.getLoc(), 4, 32);

      auto lo = rewriter.createOrFold<arith::TruncIOp>(
          op.getLoc(), rewriter.getI32Type(), value);
      auto loOffset = adaptor.target_offset();
      auto loCallOp = rewriter.create<IREE::VM::CallOp>(
          op.getLoc(), SymbolRefAttr::get(importOp), importType.getResults(),
          ArrayRef<Value>{lo, adaptor.target_buffer(), loOffset,
                          halfByteWidth});
      copyImportAttrs(importOp, loCallOp);

      auto hi = rewriter.createOrFold<arith::TruncIOp>(
          op.getLoc(), rewriter.getI32Type(),
          rewriter.createOrFold<arith::ShRUIOp>(
              op.getLoc(), value,
              rewriter.create<arith::ConstantIntOp>(op.getLoc(), 32, 64)));
      auto hiOffset = rewriter.createOrFold<arith::AddIOp>(
          op.getLoc(), adaptor.target_offset(),
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 4));
      auto hiCallOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
          op, SymbolRefAttr::get(importOp), importType.getResults(),
          ArrayRef<Value>{hi, adaptor.target_buffer(), hiOffset,
                          halfByteWidth});
      copyImportAttrs(importOp, hiCallOp);
    }

    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

void populateHALBufferToVMPatterns(MLIRContext *context,
                                   SymbolTable &importSymbols,
                                   TypeConverter &typeConverter,
                                   RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferAssertOp>>(
      context, importSymbols, typeConverter, "hal.buffer.assert");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferSubspanOp>>(
      context, importSymbols, typeConverter, "hal.buffer.subspan");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferLengthOp>>(
      context, importSymbols, typeConverter, "hal.buffer.length");
  patterns.insert<BufferLoadOpConversion>(context, importSymbols, typeConverter,
                                          "hal.buffer.load");
  patterns.insert<BufferStoreOpConversion>(context, importSymbols,
                                           typeConverter, "hal.buffer.store");
}

}  // namespace iree_compiler
}  // namespace mlir
