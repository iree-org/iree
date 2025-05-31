// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_STREAMTOHAL_PATTERNS_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_STREAMTOHAL_PATTERNS_H_

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Populates conversion patterns for stream->HAL.
void populateStreamToHALPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    IREE::HAL::DeviceAnalysis &deviceAnalysis,
    const IREE::HAL::TargetRegistry &targetRegistry);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_HAL_CONVERSION_STREAMTOHAL_PATTERNS_H_
