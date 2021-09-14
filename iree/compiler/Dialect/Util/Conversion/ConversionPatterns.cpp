// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

template <typename T>
class GenericConvertTypesConversion : public OpConversionPattern<T> {
 public:
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      T op, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    bool anyChanged = false;
    for (auto oldType : op.getOperation()->getResultTypes()) {
      SmallVector<Type> newTypes;
      if (failed(this->getTypeConverter()->convertType(oldType, newTypes))) {
        return rewriter.notifyMatchFailure(op, "unsupported result type");
      }
      if (newTypes.size() != 1 || newTypes.front() != oldType) {
        anyChanged = true;
      }
      // TODO(benvanik): figure out this silly expansion stuff. Seems broken.
      // resultTypes.append(newTypes);
      resultTypes.push_back(newTypes.front());
    }
    if (!anyChanged) return failure();
    auto newOp = rewriter.create<T>(op.getLoc(), resultTypes, newOperands,
                                    op->getAttrs());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

}  // namespace

void populateUtilConversionPatterns(MLIRContext *context,
                                    TypeConverter &typeConverter,
                                    OwningRewritePatternList &patterns) {
  patterns.insert<GenericConvertTypesConversion<IREE::Util::DoNotOptimizeOp>>(
      typeConverter, context);

  typeConverter.addConversion([&](IREE::Util::PtrType type,
                                  SmallVectorImpl<Type> &results) {
    SmallVector<Type> targetTypes;
    if (failed(typeConverter.convertType(type.getTargetType(), targetTypes))) {
      return failure();
    }
    results.reserve(targetTypes.size());
    for (auto targetType : targetTypes) {
      results.push_back(IREE::Util::PtrType::get(targetType));
    }
    return success();
  });

  typeConverter.addConversion([&](IREE::Util::ListType type) {
    auto elementType = typeConverter.convertType(type.getElementType());
    return IREE::Util::ListType::get(elementType);
  });
  patterns.insert<GenericConvertTypesConversion<IREE::Util::ListCreateOp>,
                  GenericConvertTypesConversion<IREE::Util::ListGetOp>,
                  GenericConvertTypesConversion<IREE::Util::ListSetOp>>(
      typeConverter, context);
}

void populateUtilConversionPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    OwningRewritePatternList &patterns) {
  conversionTarget.addDynamicallyLegalOp<IREE::Util::DoNotOptimizeOp>(
      [&](IREE::Util::DoNotOptimizeOp op) {
        return llvm::all_of(op.getOperandTypes(),
                            [&typeConverter](Type t) {
                              return typeConverter.isLegal(t);
                            }) &&
               llvm::all_of(op.getResultTypes(), [&typeConverter](Type t) {
                 return typeConverter.isLegal(t);
               });
      });

  conversionTarget.addDynamicallyLegalOp<IREE::Util::ListCreateOp>(
      [&](IREE::Util::ListCreateOp op) {
        return typeConverter.isLegal(op.getType());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::Util::ListGetOp>(
      [&](IREE::Util::ListGetOp op) {
        return typeConverter.isLegal(op.getType());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::Util::ListSetOp>(
      [&](IREE::Util::ListSetOp op) {
        return typeConverter.isLegal(op.value().getType());
      });

  populateUtilConversionPatterns(context, typeConverter, patterns);
}

}  // namespace iree_compiler
}  // namespace mlir
