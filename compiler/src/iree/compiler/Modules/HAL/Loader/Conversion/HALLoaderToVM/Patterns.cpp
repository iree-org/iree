// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Loader/Conversion/HALLoaderToVM/Patterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/Patterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

// Casts |value| to i32 if it is not already.
static Value castToI32(Value value, OpBuilder &builder) {
  if (value.getType().isInteger(32))
    return value;
  return builder.createOrFold<IREE::VM::TruncI64I32Op>(
      value.getLoc(), builder.getI32Type(), value);
}

struct ExecutableLoadOpConversion
    : public OpConversionPattern<IREE::HAL::Loader::ExecutableLoadOp> {
  ExecutableLoadOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                             TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }
  LogicalResult
  matchAndRewrite(IREE::HAL::Loader::ExecutableLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get format string as a rodata blob.
    auto executableFormatStr = rewriter.create<IREE::VM::RodataInlineOp>(
        loadOp.getLoc(), loadOp.getFormatAttr());

    // Pack constants, if any.
    auto constantBuffer = createPackedConstantBuffer(
        loadOp.getLoc(), adaptor.getConstants(), rewriter);

    auto importType = importOp.getFunctionType();
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        loadOp, SymbolRefAttr::get(importOp), importType.getResults(),
        ValueRange{
            executableFormatStr,
            adaptor.getData(),
            constantBuffer,
        });
    copyImportAttrs(importOp, callOp);

    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

struct ExecutableDispatchOpConversion
    : public OpConversionPattern<IREE::HAL::Loader::ExecutableDispatchOp> {
  ExecutableDispatchOpConversion(MLIRContext *context,
                                 SymbolTable &importSymbols,
                                 TypeConverter &typeConverter,
                                 StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }
  LogicalResult
  matchAndRewrite(IREE::HAL::Loader::ExecutableDispatchOp dispatchOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto entryPoint = rewriter.create<IREE::VM::ConstI32Op>(
        dispatchOp.getLoc(),
        static_cast<int32_t>(adaptor.getEntryPoint().getZExtValue()));
    SmallVector<Value, 8> callOperands = {
        adaptor.getExecutable(),
        entryPoint,
        castToI32(adaptor.getWorkgroupX(), rewriter),
        castToI32(adaptor.getWorkgroupY(), rewriter),
        castToI32(adaptor.getWorkgroupZ(), rewriter),
    };
    auto pushConstants = adaptor.getPushConstants();
    SmallVector<int16_t, 5> segmentSizes = {
        /*executable=*/-1,
        /*entry_point=*/-1,
        /*workgroup_x=*/-1,
        /*workgroup_y=*/-1,
        /*workgroup_z=*/-1,
        /*push_constants=*/
        static_cast<int16_t>(pushConstants.size()),
        /*bindings=*/
        static_cast<int16_t>(adaptor.getBindingBuffers().size()),
    };
    callOperands.append(pushConstants.begin(), pushConstants.end());
    for (auto [bindingBuffer, bindingOffset, bindingLength] : llvm::zip_equal(
             adaptor.getBindingBuffers(), adaptor.getBindingOffsets(),
             adaptor.getBindingLengths())) {
      callOperands.push_back(bindingBuffer);
      callOperands.push_back(
          castToImportType(bindingOffset, rewriter.getI64Type(), rewriter));
      callOperands.push_back(
          castToImportType(bindingLength, rewriter.getI64Type(), rewriter));
    }
    auto importType = importOp.getFunctionType();
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        dispatchOp, SymbolRefAttr::get(importOp), importType.getResults(),
        segmentSizes, importType.getInputs(), callOperands);
    copyImportAttrs(importOp, callOp);
    return success();
  }

private:
  mutable IREE::VM::ImportOp importOp;
};

} // namespace

void populateHALLoaderToVMPatterns(MLIRContext *context,
                                   ConversionTarget &conversionTarget,
                                   TypeConverter &typeConverter,
                                   SymbolTable &importSymbols,
                                   RewritePatternSet &patterns) {
  patterns.insert<
      VMImportOpConversion<IREE::HAL::Loader::ExecutableQuerySupportOp>>(
      context, importSymbols, typeConverter,
      "hal_loader.executable.query_support");
  patterns.insert<ExecutableLoadOpConversion>(
      context, importSymbols, typeConverter, "hal_loader.executable.load");
  patterns.insert<ExecutableDispatchOpConversion>(
      context, importSymbols, typeConverter, "hal_loader.executable.dispatch");
}

} // namespace mlir::iree_compiler
