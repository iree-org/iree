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
struct GenericConvertTypesPattern : OpConversionPattern<T> {
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

    ArrayRef<NamedAttribute> oldAttrs =
        op->getDiscardableAttrDictionary().getValue();
    // Null for ops with empty properties; such ops reject a properties attr.
    auto propsDict =
        dyn_cast_if_present<DictionaryAttr>(op->getPropertiesAsAttribute());
    ArrayRef<NamedAttribute> oldProps;
    if (propsDict) {
      oldProps = propsDict.getValue();
    }
    SmallVector<NamedAttribute> newAttrs, newProps;
    if (failed(convertTypeAttributes(oldAttrs, newAttrs)) ||
        failed(convertTypeAttributes(oldProps, newProps))) {
      return rewriter.notifyMatchFailure(op,
                                         "failed converting type attributes");
    }

    if (newResultTypes == op->getResultTypes() &&
        op->getOperands() == adaptor.getOperands() && newAttrs == oldAttrs &&
        newProps == oldProps) {
      return rewriter.notifyMatchFailure(op, "op does not need transformation");
    }

    OperationState state(op.getLoc(), T::getOperationName(),
                         adaptor.getOperands(), newResultTypes, newAttrs);
    if (propsDict) {
      state.propertiesAttr = rewriter.getDictionaryAttr(newProps);
    }
    rewriter.replaceOp(op, rewriter.create(state));
    return success();
  }

protected:
  LogicalResult convertTypeAttributes(ArrayRef<NamedAttribute> attrs,
                                      SmallVector<NamedAttribute> &res) const {
    for (NamedAttribute attr : attrs) {
      TypeAttr oldType = dyn_cast<TypeAttr>(attr.getValue());
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
