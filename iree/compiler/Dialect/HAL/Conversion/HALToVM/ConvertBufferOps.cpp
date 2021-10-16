// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
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
      IREE::HAL::BufferLoadOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::BufferLoadOp::Adaptor adaptor(operands);
    auto importType = importOp.getType();
    auto sizeConst = rewriter.createOrFold<mlir::arith::ConstantOp>(
        op.getLoc(),
        rewriter.getI32IntegerAttr(
            IREE::HAL::getRoundedElementByteWidth(op.getResult().getType())));
    auto callOp = rewriter.create<IREE::VM::CallOp>(
        op.getLoc(), SymbolRefAttr::get(importOp), importType.getResults(),
        ArrayRef<Value>{adaptor.source_buffer(), adaptor.source_offset(),
                        sizeConst});
    copyImportAttrs(importOp, callOp);
    // If the original result was a floating point type, we want to bitcast
    // from importType (i32) to a matching bit depth floating point type (f32).
    auto originalResultType = op.getResult().getType();
    auto newResultType = typeConverter->convertType(originalResultType);
    auto callResult = callOp.getResult(0);
    if (newResultType == callResult.getType()) {
      rewriter.replaceOp(op, {callResult});
    } else {
      rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, newResultType,
                                                    callResult);
    }

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
      IREE::HAL::BufferStoreOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::BufferStoreOp::Adaptor adaptor(operands);
    auto importType = importOp.getType();
    auto sizeConst = rewriter.createOrFold<mlir::arith::ConstantOp>(
        op.getLoc(),
        rewriter.getI32IntegerAttr(
            IREE::HAL::getRoundedElementByteWidth(op.value().getType())));
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(),
        ArrayRef<Value>{adaptor.value(), adaptor.target_buffer(),
                        adaptor.target_offset(), sizeConst});
    copyImportAttrs(importOp, callOp);
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

void populateHALBufferToVMPatterns(MLIRContext *context,
                                   SymbolTable &importSymbols,
                                   TypeConverter &typeConverter,
                                   OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferAllocatorOp>>(
      context, importSymbols, typeConverter, "hal.buffer.allocator");
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
