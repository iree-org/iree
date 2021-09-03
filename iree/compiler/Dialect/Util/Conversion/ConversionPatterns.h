// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_
#define IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Populates conversion patterns that perform conversion on util dialect ops.
// These patterns ensure that nested types are run through the provided
// |typeConverter|.
void populateUtilConversionPatterns(MLIRContext *context,
                                    TypeConverter &typeConverter,
                                    OwningRewritePatternList &patterns);
void populateUtilConversionPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_
