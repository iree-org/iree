// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef STABLEHLO_IREE_INPUTCONVERSION_PREPROCESSING_REWRITERS_H_
#define STABLEHLO_IREE_INPUTCONVERSION_PREPROCESSING_REWRITERS_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::stablehlo {

//===----------------------------------------------------------------------===//
// General StableHLO/CHLO preprocessing patterns.
//===----------------------------------------------------------------------===//

/// Collection of canonicalization patterns for StableHLO.
void populateCanonicalizationPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns,
                                      PatternBenefit benefit = 1);

/// Collection of rewrite patterns for lowering of StableHLO dot general
/// operations.
void populatePreprocessingDotGeneralToDotPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns,
                                                  PatternBenefit benefit = 1);

/// Collection of rewrite patterns for lowering of StableHLO einsum operations.
void populatePreprocessingEinsumToDotGeneralPatterns(
    MLIRContext *context, RewritePatternSet *patterns);

/// Collection of rewrite patterns for lowering of StableHLO complex
/// operations.
void populatePreprocessingComplexPatterns(MLIRContext *context,
                                          RewritePatternSet *patterns);

/// Collection of rewrite patterns for lowering of StableHLO gather operations.
void populatePreprocessingGatherToTorchIndexSelectPatterns(
    MLIRContext *context, RewritePatternSet *patterns);

/// Collection of rewrite patterns to materialize 'batch_dimension' attributes.
void populatePreprocessingUnfuseBatchNormPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns);

} // namespace mlir::iree_compiler::stablehlo

#endif // STABLEHLO_IREE_INPUTCONVERSION_PREPROCESSING_REWRITERS_H_
