// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_
#define IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

template <typename T>
struct GenericConvertTypesPattern : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    for (auto oldType : op.getOperation()->getResultTypes()) {
      SmallVector<Type> newTypes;
      if (failed(this->getTypeConverter()->convertType(oldType, newTypes))) {
        return rewriter.notifyMatchFailure(op, "unsupported result type");
      }
      // TODO(benvanik): figure out this silly expansion stuff. Seems broken.
      // resultTypes.append(newTypes);
      resultTypes.push_back(newTypes.front());
    }
    auto newOp = rewriter.create<T>(op.getLoc(), resultTypes,
                                    adaptor.getOperands(), op->getAttrs());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

template <typename OpT>
inline void addGenericLegalOp(ConversionTarget &conversionTarget,
                              TypeConverter &typeConverter) {
  conversionTarget.addDynamicallyLegalOp<OpT>([&](OpT op) {
    return llvm::all_of(
               op->getOperandTypes(),
               [&typeConverter](Type t) { return typeConverter.isLegal(t); }) &&
           llvm::all_of(op->getResultTypes(), [&typeConverter](Type t) {
             return typeConverter.isLegal(t);
           });
  });
}

// Populates conversion patterns that perform conversion on util dialect ops.
// These patterns ensure that nested types are run through the provided
// |typeConverter|.
void populateUtilConversionPatterns(MLIRContext *context,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns);
void populateUtilConversionPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns);

// Populates conversion patterns for generic structural ops (func, scf, etc).
// The ops will be made dynamically legal based on whether all types can be
// converted using the provided |typeConverter|.
void populateGenericStructuralConversionPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_
