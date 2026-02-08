// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/UtilToVM/Patterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/BuiltinRegistry.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

void populateUtilAlignmentToVMPatterns(MLIRContext *context,
                                       ConversionTarget &conversionTarget,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns);
void populateUtilAssignmentToVMPatterns(MLIRContext *context,
                                        ConversionTarget &conversionTarget,
                                        TypeConverter &typeConverter,
                                        RewritePatternSet &patterns);
void populateUtilBufferToVMPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns);
void populateUtilGlobalToVMPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns);
void populateUtilListToVMPatterns(MLIRContext *context,
                                  ConversionTarget &conversionTarget,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);
void populateUtilStatusToVMPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns);
void populateUtilStringToVMPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    BuiltinRegistry &builtins,
                                    RewritePatternSet &patterns);
void populateUtilStructuralToVMPatterns(MLIRContext *context,
                                        ConversionTarget &conversionTarget,
                                        TypeConverter &typeConverter,
                                        ImportTable &importTable,
                                        RewritePatternSet &patterns);

namespace {

//===----------------------------------------------------------------------===//
// util.null
//===----------------------------------------------------------------------===//

struct NullOpConversion : public OpConversionPattern<IREE::Util::NullOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Util::NullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ConstRefZeroOp>(
        op, getTypeConverter()->convertType(op.getType()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// util.cmp.eq
//===----------------------------------------------------------------------===//

struct CmpEQOpConversion : public OpConversionPattern<IREE::Util::CmpEQOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Util::CmpEQOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = adaptor.getLhs().getType();
    if (isa<IREE::VM::RefType>(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::CmpEQRefOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    return failure(); // not used for non-ref types currently
  }
};

//===----------------------------------------------------------------------===//
// util.cmp.ne
//===----------------------------------------------------------------------===//

struct CmpNEOpConversion : public OpConversionPattern<IREE::Util::CmpNEOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(IREE::Util::CmpNEOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = adaptor.getLhs().getType();
    if (isa<IREE::VM::RefType>(operandType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::CmpNERefOp>(
          op, rewriter.getI32Type(), adaptor.getLhs(), adaptor.getRhs());
      return success();
    }
    return failure(); // not used for non-ref types currently
  }
};

//===----------------------------------------------------------------------===//
// util.optimization_barrier
//===----------------------------------------------------------------------===//

struct OptimizationBarrierOpConversion
    : public OpConversionPattern<IREE::Util::OptimizationBarrierOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::OptimizationBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::OptimizationBarrierOp>(
        op, adaptor.getOperands());
    return success();
  }
};

} // namespace

void populateUtilToVMPatterns(MLIRContext *context,
                              ConversionTarget &conversionTarget,
                              TypeConverter &typeConverter,
                              ImportTable &importTable,
                              BuiltinRegistry &builtins,
                              RewritePatternSet &patterns) {
  patterns.insert<NullOpConversion>(typeConverter, context);
  patterns.insert<CmpEQOpConversion>(typeConverter, context);
  patterns.insert<CmpNEOpConversion>(typeConverter, context);
  patterns.insert<OptimizationBarrierOpConversion>(typeConverter, context);

  populateUtilAlignmentToVMPatterns(context, conversionTarget, typeConverter,
                                    patterns);
  populateUtilAssignmentToVMPatterns(context, conversionTarget, typeConverter,
                                     patterns);
  populateUtilBufferToVMPatterns(context, conversionTarget, typeConverter,
                                 patterns);
  populateUtilGlobalToVMPatterns(context, conversionTarget, typeConverter,
                                 patterns);
  populateUtilListToVMPatterns(context, conversionTarget, typeConverter,
                               patterns);
  populateUtilStatusToVMPatterns(context, conversionTarget, typeConverter,
                                 patterns);
  populateUtilStringToVMPatterns(context, conversionTarget, typeConverter,
                                 builtins, patterns);
  populateUtilStructuralToVMPatterns(context, conversionTarget, typeConverter,
                                     importTable, patterns);
}

} // namespace mlir::iree_compiler
