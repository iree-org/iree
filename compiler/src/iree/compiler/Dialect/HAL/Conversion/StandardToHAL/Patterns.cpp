// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/StandardToHAL/Patterns.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

void populateStandardShapeToHALPatterns(MLIRContext *context,
                                        ConversionTarget &conversionTarget,
                                        RewritePatternSet &patterns,
                                        TypeConverter &converter);

void populateStandardToHALPatterns(MLIRContext *context,
                                   ConversionTarget &conversionTarget,
                                   TypeConverter &typeConverter,
                                   RewritePatternSet &patterns) {
  populateStandardShapeToHALPatterns(context, conversionTarget, patterns,
                                     typeConverter);
}

} // namespace mlir::iree_compiler
