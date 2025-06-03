// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/Patterns.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

class AllocatorSelectOpConversion
    : public OpConversionPattern<IREE::HAL::AllocatorSelectOp> {
public:
  AllocatorSelectOpConversion(TypeConverter &typeConverter,
                              MLIRContext *context, SymbolTable &importSymbols)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>("hal.allocator.select");
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::AllocatorSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();
    auto i32Type = rewriter.getI32Type();

    SmallVector<Value> callOperands = {
        castToImportType(adaptor.getMemoryTypes(), i32Type, rewriter),
        castToImportType(adaptor.getBufferUsage(), i32Type, rewriter),
        getFlagsI64(op.getLoc(), {}, rewriter),
    };
    SmallVector<int16_t> segmentSizes = {
        /*memory_types=*/-1,
        /*buffer_usage=*/-1,
        /*flags=*/-1,
        /*from=*/
        static_cast<int16_t>(adaptor.getDevices().size()),
    };
    for (auto [device, queueAffinity] :
         llvm::zip_equal(adaptor.getDevices(), adaptor.getQueueAffinities())) {
      callOperands.push_back(device);
      callOperands.push_back(queueAffinity);
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

class AllocatorAllocateOpConversion
    : public OpConversionPattern<IREE::HAL::AllocatorAllocateOp> {
public:
  AllocatorAllocateOpConversion(TypeConverter &typeConverter,
                                MLIRContext *context,
                                SymbolTable &importSymbols)
      : OpConversionPattern(typeConverter, context) {
    importOp =
        importSymbols.lookup<IREE::VM::ImportOp>("hal.allocator.allocate");
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::AllocatorAllocateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, importOp.getName(),
        ArrayRef<Type>{
            getTypeConverter()->convertType(op.getType()),
        },
        ArrayRef<Value>{
            adaptor.getAllocator(),
            castToImportType(adaptor.getQueueAffinity(), rewriter.getI64Type(),
                             rewriter),
            castToImportType(adaptor.getMemoryTypes(), rewriter.getI32Type(),
                             rewriter),
            castToImportType(adaptor.getBufferUsage(), rewriter.getI32Type(),
                             rewriter),
            castToImportType(adaptor.getResultSize(), rewriter.getI64Type(),
                             rewriter),
        });
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

class AllocatorImportOpConversion
    : public OpConversionPattern<IREE::HAL::AllocatorImportOp> {
public:
  AllocatorImportOpConversion(TypeConverter &typeConverter,
                              MLIRContext *context, SymbolTable &importSymbols)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>("hal.allocator.import");
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(IREE::HAL::AllocatorImportOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto callOp = rewriter.create<IREE::VM::CallOp>(
        op.getLoc(), importOp.getName(),
        ArrayRef<Type>{
            getTypeConverter()->convertType(op.getResult().getType()),
        },
        ArrayRef<Value>{
            adaptor.getAllocator(),
            rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), /*try=*/1),
            castToImportType(adaptor.getQueueAffinity(), rewriter.getI64Type(),
                             rewriter),
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
    auto didImport = rewriter.create<IREE::VM::CmpNZRefOp>(
        op.getLoc(), rewriter.getI32Type(), result);
    rewriter.replaceOp(op, {didImport, result});
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

} // namespace

void populateHALAllocatorToVMPatterns(MLIRContext *context,
                                      SymbolTable &importSymbols,
                                      TypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  patterns.insert<AllocatorSelectOpConversion>(typeConverter, context,
                                               importSymbols);
  patterns.insert<AllocatorAllocateOpConversion>(typeConverter, context,
                                                 importSymbols);
  patterns.insert<AllocatorImportOpConversion>(typeConverter, context,
                                               importSymbols);
  patterns.insert<VMImportOpConversion<IREE::HAL::MemoryTypeOp>>(
      context, importSymbols, typeConverter, "hal.memory_type");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferUsageOp>>(
      context, importSymbols, typeConverter, "hal.buffer_usage");
}

} // namespace mlir::iree_compiler
