// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_HALTOHAL_CONVERTHALTOHAL_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_HALTOHAL_CONVERTHALTOHAL_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// TODO(#7277): remove when switched to streams (happens there now).

// Adds op legality rules to |conversionTarget| to ensure all incoming HAL
// pseudo ops are removed during HAL->HAL lowering.
void setupHALToHALLegality(MLIRContext *context,
                           ConversionTarget &conversionTarget,
                           TypeConverter &typeConverter);

// Populates conversion patterns for HAL->HAL (pseudo ops, etc).
void populateHALToHALPatterns(MLIRContext *context,
                              OwningRewritePatternList &patterns,
                              TypeConverter &typeConverter);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_CONVERSION_HALTOHAL_CONVERTHALTOHAL_H_
