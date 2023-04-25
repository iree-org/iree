// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class AllocatorAllocateInitializedOpConversion
    : public OpConversionPattern<IREE::HAL::AllocatorAllocateInitializedOp> {
 public:
  AllocatorAllocateInitializedOpConversion(TypeConverter &typeConverter,
                                           MLIRContext *context,
                                           SymbolTable &importSymbols)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(
        "hal.allocator.allocate.initialized");
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::AllocatorAllocateInitializedOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, importOp.getName(),
        ArrayRef<Type>{
            getTypeConverter()->convertType(op.getType()),
        },
        ArrayRef<Value>{
            adaptor.getAllocator(),
            rewriter.createOrFold<IREE::VM::ConstI32Op>(
                op.getLoc(), op.getMemoryTypesAttr().getInt()),
            rewriter.createOrFold<IREE::VM::ConstI32Op>(
                op.getLoc(), op.getBufferUsageAttr().getInt()),
            adaptor.getSource(),
            castToImportType(adaptor.getOffset(), rewriter.getI64Type(),
                             rewriter),
            castToImportType(adaptor.getLength(), rewriter.getI64Type(),
                             rewriter),
        });
    copyImportAttrs(importOp, callOp);
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

class AllocatorTryMapOpConversion
    : public OpConversionPattern<IREE::HAL::AllocatorTryMapOp> {
 public:
  AllocatorTryMapOpConversion(TypeConverter &typeConverter,
                              MLIRContext *context, SymbolTable &importSymbols)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(
        "hal.allocator.map.byte_buffer");
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::AllocatorTryMapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto callOp = rewriter.create<IREE::VM::CallOp>(
        op.getLoc(), importOp.getName(),
        ArrayRef<Type>{
            getTypeConverter()->convertType(op.getResult().getType()),
        },
        ArrayRef<Value>{
            adaptor.getAllocator(),
            rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), /*try=*/1),
            rewriter.createOrFold<IREE::VM::ConstI32Op>(
                op.getLoc(), op.getMemoryTypesAttr().getInt()),
            rewriter.createOrFold<IREE::VM::ConstI32Op>(
                op.getLoc(), op.getBufferUsageAttr().getInt()),
            adaptor.getSource(),
            castToImportType(adaptor.getOffset(), rewriter.getI64Type(),
                             rewriter),
            castToImportType(adaptor.getLength(), rewriter.getI64Type(),
                             rewriter),
        });
    copyImportAttrs(importOp, callOp);
    auto result = callOp.getResults().front();
    auto didMap = rewriter.create<IREE::VM::CmpNZRefOp>(
        op.getLoc(), rewriter.getI32Type(), result);
    rewriter.replaceOp(op, {didMap, result});
    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

}  // namespace

void populateHALAllocatorToVMPatterns(MLIRContext *context,
                                      SymbolTable &importSymbols,
                                      TypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::AllocatorAllocateOp>>(
      context, importSymbols, typeConverter, "hal.allocator.allocate");
  patterns.insert<AllocatorAllocateInitializedOpConversion>(
      typeConverter, context, importSymbols);
  patterns.insert<AllocatorTryMapOpConversion>(typeConverter, context,
                                               importSymbols);
}

}  // namespace iree_compiler
}  // namespace mlir
