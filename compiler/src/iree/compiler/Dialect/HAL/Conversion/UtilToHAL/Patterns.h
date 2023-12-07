// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_UTILTOHAL_PATTERNS_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_UTILTOHAL_PATTERNS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Appends all patterns for lowering IREE ops to HAL buffer ops and sets their
// legality.
void populateUtilToHALPatterns(MLIRContext *context,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter,
                               RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_HAL_CONVERSION_UTILTOHAL_PATTERNS_H_
