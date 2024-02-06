// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_
#define IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

template <typename T>
struct GenericConvertTypesPattern : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    for (auto oldType : op.getOperation()->getResultTypes()) {
      SmallVector<Type> newTypes;
      if (failed(this->getTypeConverter()->convertType(oldType, newTypes))) {
        return rewriter.notifyMatchFailure(op, "unsupported result type");
      }
      // TODO(benvanik): figure out this silly expansion stuff. Seems broken.
      // resultTypes.append(newTypes);
      newResultTypes.push_back(newTypes.front());
    }

    SmallVector<NamedAttribute> newAttrs;
    if (failed(convertTypeAttributes(op->getAttrs(), newAttrs))) {
      return rewriter.notifyMatchFailure(op,
                                         "failed converting type attributes");
    }

    if (newResultTypes == op->getResultTypes() &&
        op->getOperands() == adaptor.getOperands() &&
        newAttrs == op->getAttrs()) {
      return rewriter.notifyMatchFailure(op, "op does not need transformation");
    }

    auto newOp = rewriter.create<T>(op.getLoc(), newResultTypes,
                                    adaptor.getOperands(), newAttrs);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

protected:
  LogicalResult convertTypeAttributes(ArrayRef<NamedAttribute> attrs,
                                      SmallVector<NamedAttribute> &res) const {
    for (NamedAttribute attr : attrs) {
      TypeAttr oldType = attr.getValue().dyn_cast<TypeAttr>();
      if (!oldType) {
        res.push_back(attr);
        continue;
      }

      Type newType = this->getTypeConverter()->convertType(oldType.getValue());
      if (!newType) {
        return failure();
      }
      res.push_back(NamedAttribute(attr.getName(), TypeAttr::get(newType)));
    }
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
