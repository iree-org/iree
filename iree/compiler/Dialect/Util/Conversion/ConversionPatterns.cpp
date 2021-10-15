// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"

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

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

void populateUtilConversionPatterns(MLIRContext *context,
                                    TypeConverter &typeConverter,
                                    OwningRewritePatternList &patterns) {
  patterns.insert<GenericConvertTypesPattern<IREE::Util::DoNotOptimizeOp>>(
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
  patterns.insert<GenericConvertTypesPattern<IREE::Util::ListCreateOp>,
                  GenericConvertTypesPattern<IREE::Util::ListGetOp>,
                  GenericConvertTypesPattern<IREE::Util::ListSetOp>>(
      typeConverter, context);
}

void populateUtilConversionPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    OwningRewritePatternList &patterns) {
  addGenericLegalOp<IREE::Util::DoNotOptimizeOp>(conversionTarget,
                                                 typeConverter);
  addGenericLegalOp<IREE::Util::ListCreateOp>(conversionTarget, typeConverter);
  addGenericLegalOp<IREE::Util::ListGetOp>(conversionTarget, typeConverter);
  addGenericLegalOp<IREE::Util::ListSetOp>(conversionTarget, typeConverter);

  populateUtilConversionPatterns(context, typeConverter, patterns);
}

}  // namespace iree_compiler
}  // namespace mlir
