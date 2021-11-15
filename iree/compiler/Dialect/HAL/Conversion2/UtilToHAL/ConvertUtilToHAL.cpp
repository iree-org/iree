// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion2/UtilToHAL/ConvertUtilToHAL.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct GlobalConversionPattern
    : public OpConversionPattern<IREE::Util::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Util::GlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto newType = getTypeConverter()->convertType(op.type());
    if (newType == op.type()) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      // NOTE: the initial value may be invalid here! We rely on
      // dialect-specific conversions to handle it.
      op.typeAttr(TypeAttr::get(newType));
    });
    return success();
  }
};

}  // namespace

void populateUtilToHALPatterns(MLIRContext *context,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter,
                               OwningRewritePatternList &patterns) {
  conversionTarget.addDynamicallyLegalOp<IREE::Util::GlobalOp>(
      [&](IREE::Util::GlobalOp op) {
        return typeConverter.isLegal(op.type()) &&
               (!op.initial_value().hasValue() ||
                typeConverter.isLegal(op.initial_valueAttr().getType()));
      });
  addGenericLegalOp<IREE::Util::GlobalLoadOp>(conversionTarget, typeConverter);
  addGenericLegalOp<IREE::Util::GlobalStoreOp>(conversionTarget, typeConverter);

  patterns.insert<GlobalConversionPattern,
                  GenericConvertTypesPattern<IREE::Util::GlobalLoadOp>,
                  GenericConvertTypesPattern<IREE::Util::GlobalStoreOp>>(
      typeConverter, context);

  populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                 patterns);
}

}  // namespace iree_compiler
}  // namespace mlir
