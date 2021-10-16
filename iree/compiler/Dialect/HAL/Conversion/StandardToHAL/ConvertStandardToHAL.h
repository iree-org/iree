// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_STANDARDTOHAL_CONVERTSTANDARDTOHAL_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_STANDARDTOHAL_CONVERTSTANDARDTOHAL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// TODO(#7277): remove when switched to streams (happens there now).

// Adds op legality rules to |conversionTarget| to ensure all incoming std ops
// are removed during ->HAL lowering.
void setupStandardToHALLegality(MLIRContext *context,
                                ConversionTarget &conversionTarget,
                                TypeConverter &typeConverter);

// Populates conversion patterns for std->HAL.
void populateStandardToHALPatterns(MLIRContext *context,
                                   OwningRewritePatternList &patterns,
                                   TypeConverter &typeConverter);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_CONVERSION_STANDARDTOHAL_CONVERTSTANDARDTOHAL_H_
