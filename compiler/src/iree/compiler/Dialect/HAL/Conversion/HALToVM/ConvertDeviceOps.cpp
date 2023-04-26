// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Rewrites a hal.device.query of an i1/i16/i32 value to a hal.device.query of
// i64 with a truncation on the result.
class DeviceQueryIntCastOpConversion
    : public OpConversionPattern<IREE::HAL::DeviceQueryOp> {
 public:
  DeviceQueryIntCastOpConversion(MLIRContext *context,
                                 TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, context) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::DeviceQueryOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // We only deal with in-dialect conversions to i32 in this pattern.
    auto targetType = op.getValue().getType();
    if (targetType.isInteger(64)) return failure();
    if (!targetType.isIntOrIndex()) return failure();

    // Query as I64.
    // Note that due to type conversion we need to handle the default logic
    // ourselves instead of allowing the i64 do the same. We could let it handle
    // things but then we are generating more IR that may prevent other
    // canonicalizations (a select of i1 to i1 is easier to handle).
    auto queryOp = rewriter.create<IREE::HAL::DeviceQueryOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getI64Type(),
        adaptor.getDevice(), op.getCategoryAttr(), op.getKeyAttr(),
        TypedAttr{});
    auto ok = queryOp.getOk().cast<Value>();
    auto value = queryOp.getValue();

    // Truncate or extend based on the target type.
    if (targetType.isIndex()) {
      // i64 -> index cast.
      value = rewriter.createOrFold<arith::IndexCastOp>(op.getLoc(), targetType,
                                                        value);
    } else if (targetType.isa<IntegerType>()) {
      // i64 -> {integer} cast.
      if (targetType.getIntOrFloatBitWidth() <
          value.getType().getIntOrFloatBitWidth()) {
        // i64 -> narrowing cast.
        value = rewriter.createOrFold<arith::TruncIOp>(op.getLoc(), targetType,
                                                       value);
      } else {
        // i64 -> widening cast.
        value = rewriter.createOrFold<arith::ExtUIOp>(op.getLoc(), targetType,
                                                      value);
      }
    }

    if (op.getDefaultValue().has_value()) {
      // Select the default value based on the converted type as that's the type
      // of the attribute we have is in. 'ok' result is set to true as we've
      // already handled the error case.
      value = rewriter.createOrFold<arith::SelectOp>(
          op.getLoc(), ok, value,
          rewriter.createOrFold<arith::ConstantOp>(op.getLoc(),
                                                   op.getDefaultValueAttr()));
      ok = rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1);
    }

    rewriter.replaceOp(op, {ok, value});
    return success();
  }
};

class DeviceQueryI64OpConversion
    : public OpConversionPattern<IREE::HAL::DeviceQueryOp> {
 public:
  DeviceQueryI64OpConversion(MLIRContext *context, SymbolTable &importSymbols,
                             TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::DeviceQueryOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!op.getValue().getType().isInteger(64)) return failure();
    auto results =
        rewriteToCall(op, adaptor, importOp, *getTypeConverter(), rewriter);
    if (!results.has_value()) return failure();
    auto ok = results->front();
    auto value = results->back();
    if (op.getDefaultValue().has_value()) {
      value = rewriter.createOrFold<arith::SelectOp>(
          op.getLoc(), ok, value,
          rewriter.createOrFold<IREE::VM::ConstI64Op>(
              op.getLoc(), op.getDefaultValueAttr()));
      ok = rewriter.createOrFold<IREE::VM::ConstI64Op>(op.getLoc(), 1);
    }
    rewriter.replaceOp(op, {ok, value});
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

void populateHALDeviceToVMPatterns(MLIRContext *context,
                                   SymbolTable &importSymbols,
                                   TypeConverter &typeConverter,
                                   RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceAllocatorOp>>(
      context, importSymbols, typeConverter, "hal.device.allocator");

  patterns.insert<DeviceQueryIntCastOpConversion>(context, typeConverter);
  patterns.insert<DeviceQueryI64OpConversion>(
      context, importSymbols, typeConverter, "hal.device.query.i64");

  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueAllocaOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.alloca");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueDeallocaOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.dealloca");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueExecuteOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.execute");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueFlushOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.flush");
}

}  // namespace iree_compiler
}  // namespace mlir
