// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_
#define IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Indexing {

// Populates patterns to drop indexing assertions.
void populateStripIndexingAssertionPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::IREE::Indexing

#endif // IREE_COMPILER_DIALECT_UTIL_CONVERSION_CONVERSIONPATTERNS_H_
