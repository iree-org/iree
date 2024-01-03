// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_CONVERSION_MEMREFTOUTIL_PATTERN_H_
#define IREE_COMPILER_DIALECT_UTIL_CONVERSION_MEMREFTOUTIL_PATTERN_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Appends memref dialect to vm dialect patterns to the given pattern list.
// Because these patterns are often used in A->B->C lowerings, we allow the
// final buffer type to be specialized (this must be the buffer type that
// is valid in the 'C' dialect). If null, the a Util::BufferType is used.
void populateMemRefToUtilPatterns(MLIRContext *context,
                                  ConversionTarget &conversionTarget,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  Type convertedBufferType = {});

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_UTIL_CONVERSION_MEMREFTOUTIL_PATTERN_H_
