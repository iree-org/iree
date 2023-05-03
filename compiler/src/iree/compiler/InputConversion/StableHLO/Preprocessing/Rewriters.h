// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PREPROCESSING_REWRITERS_H_
#define IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PREPROCESSING_REWRITERS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::stablehlo {

//===----------------------------------------------------------------------===//
// General StableHLO/CHLO preprocessing patterns.
//===----------------------------------------------------------------------===//

/// Collection of rewrite patterns for lowering of StableHLO dim operations.
void populatePreprocessingEinsumToDotGeneralPatterns(
    MLIRContext *context, RewritePatternSet *patterns);

/// Collection of rewrite patterns for lowering of StableHLO complex
/// operations.
void populatePreprocessingComplexPatterns(MLIRContext *context,
                                          RewritePatternSet *patterns);

/// Collection of rewrite patterns to materialize 'batch_dimension' attributes.
void populatePreprocessingUnfuseBatchNormPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns);

}  // namespace mlir::iree_compiler::stablehlo

#endif  // IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PREPROCESSING_REWRITERS_H_
