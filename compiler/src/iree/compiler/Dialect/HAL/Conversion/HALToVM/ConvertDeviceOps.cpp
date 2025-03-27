// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/Patterns.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Rewrites a hal.device.query of an i1/i16/i32 value to a hal.device.query of
// i64 with a truncation on the result.
class DeviceQueryCastOpConversion
    : public OpConversionPattern<IREE::HAL::DeviceQueryOp> {
public:
  DeviceQueryCastOpConversion(MLIRContext *context,
                              TypeConverter &typeConverter)
      : OpConversionPattern(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(IREE::HAL::DeviceQueryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto targetType = op.getValue().getType();
    if (targetType.isInteger(64))
      return failure(); // handled natively
    if (!targetType.isIntOrIndex())
      return rewriter.notifyMatchFailure(op, "unsupported result type");

    // Query as i64.
    // Note that due to type conversion we need to handle the default logic
    // ourselves instead of allowing the i64 do the same. We could let it handle
    // things but then we are generating more IR that may prevent other
    // canonicalizations (a select of i1 to i1 is easier to handle).
    auto queryOp = rewriter.create<IREE::HAL::DeviceQueryOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getI64Type(),
        adaptor.getDevice(), op.getCategoryAttr(), op.getKeyAttr(),
        TypedAttr{});
    auto ok = llvm::cast<Value>(queryOp.getOk());
    auto value = queryOp.getValue();

    // Truncate or extend based on the target type.
    if (targetType.isIndex()) {
      // i64 -> index cast.
      value =
          rewriter.create<arith::IndexCastOp>(op.getLoc(), targetType, value);
    } else if (targetType.isInteger(1)) {
      // i64 -> i1 cast.
      value = rewriter.create<IREE::VM::CmpNZI64Op>(
          op.getLoc(), rewriter.getI32Type(), value);
    } else {
      // i64 -> {integer} cast.
      if (targetType.getIntOrFloatBitWidth() <
          value.getType().getIntOrFloatBitWidth()) {
        // i64 -> narrowing cast.
        value =
            rewriter.create<arith::TruncIOp>(op.getLoc(), targetType, value);
      } else {
        // i64 -> widening cast.
        value = rewriter.create<arith::ExtUIOp>(op.getLoc(), targetType, value);
      }
    }

    if (op.getDefaultValue().has_value()) {
      // Select the default value based on the converted type as that's the type
      // of the attribute we have is in. 'ok' result is set to true as we've
      // already handled the error case.
      value = rewriter.create<arith::SelectOp>(
          op.getLoc(), ok, value,
          rewriter.create<arith::ConstantOp>(op.getLoc(),
                                             op.getDefaultValueAttr()));
      ok = rewriter.create<IREE::VM::ConstI32Op>(op.getLoc(), 1);
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

  LogicalResult
  matchAndRewrite(IREE::HAL::DeviceQueryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getValue().getType().isInteger(64))
      return failure();
    auto results =
        rewriteToCall(op, adaptor, importOp, *getTypeConverter(), rewriter);
    if (!results.has_value())
      return failure();
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

class DeviceQueueFillOpConversion
    : public OpConversionPattern<IREE::HAL::DeviceQueueFillOp> {
public:
  DeviceQueueFillOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                              TypeConverter &typeConverter,
                              StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::DeviceQueueFillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();
    auto i64Type = rewriter.getI64Type();
    auto patternLength = rewriter.create<IREE::VM::ConstI32Op>(
        op.getLoc(),
        llvm::divideCeil(op.getPattern().getType().getIntOrFloatBitWidth(), 8));
    std::array<Value, 10> callOperands = {
        adaptor.getDevice(),
        castToImportType(adaptor.getQueueAffinity(), i64Type, rewriter),
        adaptor.getWaitFence(),
        adaptor.getSignalFence(),
        adaptor.getTargetBuffer(),
        castToImportType(adaptor.getTargetOffset(), i64Type, rewriter),
        castToImportType(adaptor.getLength(), i64Type, rewriter),
        castToImportType(adaptor.getPattern(), i64Type, rewriter),
        patternLength,
        getFlagsI64(op.getLoc(), adaptor.getFlagsAttr(), rewriter),
    };
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(),
        callOperands);
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

class DeviceQueueExecuteIndirectOpConversion
    : public OpConversionPattern<IREE::HAL::DeviceQueueExecuteIndirectOp> {
public:
  DeviceQueueExecuteIndirectOpConversion(MLIRContext *context,
                                         SymbolTable &importSymbols,
                                         TypeConverter &typeConverter,
                                         StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::DeviceQueueExecuteIndirectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();
    auto i64Type = rewriter.getI64Type();

    Value queueAffinity =
        castToImportType(adaptor.getQueueAffinity(), i64Type, rewriter);
    SmallVector<Value, 8> callOperands = {
        adaptor.getDevice(),
        queueAffinity,
        adaptor.getWaitFence(),
        adaptor.getSignalFence(),
        adaptor.getCommandBuffer(),
        getFlagsI64(op.getLoc(), adaptor.getFlagsAttr(), rewriter),
    };
    SmallVector<int16_t, 5> segmentSizes = {
        /*device=*/-1,
        /*queue_affinity=*/-1,
        /*wait_fence=*/-1,
        /*signal_fence=*/-1,
        /*command_buffer=*/-1,
        /*flags=*/-1,
        /*bindings=*/
        static_cast<int16_t>(adaptor.getBindingBuffers().size()),
    };
    for (auto [bindingBuffer, bindingOffset, bindingLength] : llvm::zip_equal(
             adaptor.getBindingBuffers(), adaptor.getBindingOffsets(),
             adaptor.getBindingLengths())) {
      callOperands.push_back(bindingBuffer);
      callOperands.push_back(
          castToImportType(bindingOffset, i64Type, rewriter));
      callOperands.push_back(
          castToImportType(bindingLength, i64Type, rewriter));
    }

    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(), segmentSizes,
        importType.getInputs(), callOperands);
    copyImportAttrs(importOp, callOp);
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

  patterns.insert<DeviceQueryCastOpConversion>(context, typeConverter);
  patterns.insert<DeviceQueryI64OpConversion>(
      context, importSymbols, typeConverter, "hal.device.query.i64");

  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueAllocaOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.alloca");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueDeallocaOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.dealloca");
  patterns.insert<DeviceQueueFillOpConversion>(
      context, importSymbols, typeConverter, "hal.device.queue.fill");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueUpdateOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.update");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueCopyOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.copy");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueReadOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.read");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueWriteOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.write");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueBarrierOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.barrier");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueExecuteOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.execute");
  patterns.insert<DeviceQueueExecuteIndirectOpConversion>(
      context, importSymbols, typeConverter,
      "hal.device.queue.execute.indirect");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceQueueFlushOp>>(
      context, importSymbols, typeConverter, "hal.device.queue.flush");
}

} // namespace mlir::iree_compiler
