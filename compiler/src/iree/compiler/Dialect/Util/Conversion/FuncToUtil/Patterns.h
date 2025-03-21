// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_CONVERSION_FUNCTOUTIL_PATTERN_H_
#define IREE_COMPILER_DIALECT_UTIL_CONVERSION_FUNCTOUTIL_PATTERN_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Appends func dialect to util dialect patterns to the given pattern list.
void populateFuncToUtilPatterns(MLIRContext *context,
                                ConversionTarget &conversionTarget,
                                TypeConverter &typeConverter,
                                RewritePatternSet &patterns,
                                mlir::ModuleOp rootModuleOp);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_UTIL_CONVERSION_FUNCTOUTIL_PATTERN_H_
