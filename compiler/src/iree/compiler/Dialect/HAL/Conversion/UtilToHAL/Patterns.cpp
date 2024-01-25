// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/UtilToHAL/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct GlobalConversionPattern
    : public OpConversionPattern<IREE::Util::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newType = getTypeConverter()->convertType(op.getType());
    if (newType == op.getType())
      return failure();
    rewriter.modifyOpInPlace(op, [&]() {
      // NOTE: the initial value may be invalid here! We rely on
      // dialect-specific conversions to handle it.
      op.setTypeAttr(TypeAttr::get(newType));
    });
    return success();
  }
};

} // namespace

void populateUtilToHALPatterns(MLIRContext *context,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter,
                               RewritePatternSet &patterns) {
  conversionTarget.addDynamicallyLegalOp<IREE::Util::GlobalOp>(
      [&](IREE::Util::GlobalOp op) {
        return typeConverter.isLegal(op.getType()) &&
               (!op.getInitialValue().has_value() ||
                typeConverter.isLegal(op.getInitialValueAttr().getType()));
      });
  addGenericLegalOp<IREE::Util::GlobalLoadOp>(conversionTarget, typeConverter);
  addGenericLegalOp<IREE::Util::GlobalStoreOp>(conversionTarget, typeConverter);

  patterns.insert<GlobalConversionPattern,
                  GenericConvertTypesPattern<IREE::Util::GlobalLoadOp>,
                  GenericConvertTypesPattern<IREE::Util::GlobalStoreOp>>(
      typeConverter, context);

  populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                 patterns);
  populateGenericStructuralConversionPatterns(context, conversionTarget,
                                              typeConverter, patterns);
}

} // namespace mlir::iree_compiler
