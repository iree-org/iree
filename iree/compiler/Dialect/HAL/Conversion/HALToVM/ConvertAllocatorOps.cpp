// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class AllocatorMapOpConversion
    : public OpConversionPattern<IREE::HAL::AllocatorMapOp> {
 public:
  AllocatorMapOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                           SymbolTable &importSymbols)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(
        "hal.allocator.wrap.byte_buffer");
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::AllocatorMapOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::AllocatorMapOp::Adaptor opAdaptor(operands);
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        op, importOp.getName(),
        ArrayRef<Type>{getTypeConverter()->convertType(op.getType())},
        ArrayRef<Value>{opAdaptor.allocator(),
                        rewriter.createOrFold<IREE::VM::ConstI32Op>(
                            op.getLoc(), op.memory_typesAttr()),
                        rewriter.createOrFold<IREE::VM::ConstI32Op>(
                            op.getLoc(), op.buffer_usageAttr()),
                        opAdaptor.source(), opAdaptor.offset(),
                        opAdaptor.length()});
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
      IREE::HAL::AllocatorTryMapOp op, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::AllocatorTryMapOp::Adaptor operands(rawOperands);
    auto callOp = rewriter.create<IREE::VM::CallOp>(
        op.getLoc(), importOp.getName(),
        ArrayRef<Type>{getTypeConverter()->convertType(op.result().getType())},
        ArrayRef<Value>{
            operands.allocator(),
            rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), /*try=*/1),
            rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(),
                                                        op.memory_typesAttr()),
            rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(),
                                                        op.buffer_usageAttr()),
            operands.source(),
            operands.offset(),
            operands.length(),
        });
    copyImportAttrs(importOp, callOp);
    auto result = callOp.results().front();
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
                                      OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::AllocatorAllocateOp>>(
      context, importSymbols, typeConverter, "hal.allocator.allocate");
  patterns.insert<AllocatorMapOpConversion>(typeConverter, context,
                                            importSymbols);
  patterns.insert<AllocatorTryMapOpConversion>(typeConverter, context,
                                               importSymbols);
}

}  // namespace iree_compiler
}  // namespace mlir
