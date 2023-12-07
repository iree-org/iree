// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_STANDARDTOHAL_PATTERNS_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_STANDARDTOHAL_PATTERNS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Populates conversion patterns for std->HAL.
void populateStandardToHALPatterns(MLIRContext *context,
                                   ConversionTarget &conversionTarget,
                                   TypeConverter &typeConverter,
                                   RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_HAL_CONVERSION_STANDARDTOHAL_PATTERNS_H_
