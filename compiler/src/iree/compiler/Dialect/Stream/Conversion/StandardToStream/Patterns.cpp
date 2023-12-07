// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/StandardToStream/Patterns.h"

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

void populateStandardConstantToStreamPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, RewritePatternSet &patterns);

void populateStandardStructuralToStreamPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, RewritePatternSet &patterns);

void populateStandardToStreamConversionPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  typeConverter.addConversion([](IndexType type) { return type; });
  typeConverter.addConversion([](IntegerType type) { return type; });
  typeConverter.addConversion([](FloatType type) { return type; });

  // Ensure all shape related ops are fully converted as we should no longer
  // have any types they are valid to be used on after this conversion.
  conversionTarget.addIllegalOp<memref::DimOp, memref::RankOp, tensor::DimOp,
                                tensor::RankOp>();

  populateStandardConstantToStreamPatterns(context, conversionTarget,
                                           typeConverter, patterns);
  populateStandardStructuralToStreamPatterns(context, conversionTarget,
                                             typeConverter, patterns);
}

} // namespace mlir::iree_compiler
