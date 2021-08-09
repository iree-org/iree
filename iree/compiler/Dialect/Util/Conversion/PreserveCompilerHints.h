// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_CONVERSION_PRESERVECOMPILERHINTS_H_
#define IREE_COMPILER_DIALECT_IREE_CONVERSION_PRESERVECOMPILERHINTS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

// Adds op legality rules to |conversionTarget| to preserve compiler hints
// that satisfy the type constraints of |typeConverter|.
void setupCompilerHintsLegality(MLIRContext *context,
                                ConversionTarget &conversionTarget,
                                TypeConverter &typeConverter);

// Appends all patterns for preserving compiler hints while they are transformed
// by the dialect conversion framework.
void populatePreserveCompilerHintsPatterns(MLIRContext *context,
                                           OwningRewritePatternList &patterns);

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_CONVERSION_PRESERVECOMPILERHINTS_H_
