// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Rewrites a hal.device.query of an i1 value to a hal.device.query of i32 with
// a truncation on the result.
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
    auto targetType = op.value().getType();
    if (targetType.isInteger(32)) return failure();
    if (!targetType.isIntOrIndex()) return failure();

    // Query as I32.
    // Note that due to type conversion we need to handle the default logic
    // ourselves instead of allowing the i32 do the same. We could let it handle
    // things but then we are generating more IR that may prevent other
    // canonicalizations (a select of i1 to i1 is easier to handle).
    auto queryOp = rewriter.create<IREE::HAL::DeviceQueryOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getI32Type(),
        adaptor.device(), op.categoryAttr(), op.keyAttr(), Attribute{});
    auto ok = queryOp.ok();
    auto value = queryOp.value();

    // Truncate or extend based on the target type.
    if (targetType.isIndex()) {
      // i32 -> index cast.
      value = rewriter.createOrFold<arith::IndexCastOp>(op.getLoc(), targetType,
                                                        value);
    } else if (targetType.isa<IntegerType>()) {
      // i32 -> {integer} cast.
      if (targetType.getIntOrFloatBitWidth() <
          value.getType().getIntOrFloatBitWidth()) {
        // i32 -> narrowing cast.
        value = rewriter.createOrFold<arith::TruncIOp>(op.getLoc(), targetType,
                                                       value);
      } else {
        // i32 -> widening cast.
        value = rewriter.createOrFold<arith::ExtUIOp>(op.getLoc(), targetType,
                                                      value);
      }
    }

    if (op.default_value().hasValue()) {
      // Select the default value based on the converted type as that's the type
      // of the attribute we have is in. 'ok' result is set to true as we've
      // already handled the error case.
      value = rewriter.createOrFold<SelectOp>(
          op.getLoc(), ok, value,
          rewriter.createOrFold<arith::ConstantOp>(op.getLoc(),
                                                   op.default_valueAttr()));
      ok = rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1);
    }

    rewriter.replaceOp(op, {ok, value});
    return success();
  }
};

class DeviceQueryI32OpConversion
    : public OpConversionPattern<IREE::HAL::DeviceQueryOp> {
 public:
  DeviceQueryI32OpConversion(MLIRContext *context, SymbolTable &importSymbols,
                             TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::DeviceQueryOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!op.value().getType().isInteger(32)) return failure();
    auto results =
        rewriteToCall(op, adaptor, importOp, *getTypeConverter(), rewriter);
    if (!results.hasValue()) return failure();
    auto ok = results->front();
    auto value = results->back();
    if (op.default_value().hasValue()) {
      value = rewriter.createOrFold<SelectOp>(
          op.getLoc(), ok, value,
          rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(),
                                                      op.default_valueAttr()));
      ok = rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1);
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
  patterns.insert<DeviceQueryI32OpConversion>(
      context, importSymbols, typeConverter, "hal.device.query.i32");
}

}  // namespace iree_compiler
}  // namespace mlir
