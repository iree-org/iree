// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_REWRITERS_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_REWRITERS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::stablehlo {

//===----------------------------------------------------------------------===//
// General StableHLO/CHLO lowering patterns.
//===----------------------------------------------------------------------===//

/// Collection of rewrite patterns for lowering of CHLO ops to StableHLO and
/// Shape ops.
void populateLegalizeChloPatterns(MLIRContext *context,
                                  RewritePatternSet *patterns);

/// Collection of rewrite patterns for lowering of StableHLO ops to SCF control
/// flow ops.
void populateLegalizeControlFlowPatterns(MLIRContext *context,
                                         RewritePatternSet *patterns);

/// Collection of rewrite patterns for lowering of StableHLO dim operations.
void populateLegalizeShapeComputationPatterns(MLIRContext *context,
                                              RewritePatternSet *patterns);

//===----------------------------------------------------------------------===//
// IREE-specific patterns.
//===----------------------------------------------------------------------===//

/// Populates the patterns that convert from StableHLO to LinalgExt.
void populateStableHloToLinalgExtConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns);

/// Populates the patterns that convert from StableHLO collective ops to Flow
/// ops.
void populateStableHloCollectivesConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_REWRITERS_H_
