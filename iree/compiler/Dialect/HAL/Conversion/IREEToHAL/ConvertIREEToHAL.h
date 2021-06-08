// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERTIREETOHAL_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERTIREETOHAL_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// TODO(gcmn): Use conversion interfaces. Requires breaking circular dependency
// between HAL and IREE dialects.

// Appends all patterns for lowering IREE ops to HAL buffer ops.
void populateIREEToHALPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns);

// Setup the |conversionTarget| op legality to ensure helpful error messages for
// IREE ops we know should always be converted.
void setupIREEToHALLegality(MLIRContext *context, ConversionTarget &target);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_CONVERSION_CONVERTIREETOHAL_H_
