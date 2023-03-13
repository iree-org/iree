// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_MHLO_REWRITER_H_
#define IREE_COMPILER_INPUTCONVERSION_MHLO_REWRITER_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

/// Populates the patterns that convert from MHLO to Linalg on tensors. Imports
/// patterns from XLA, as well as some IREE specific modifications.
void populateMHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet &patterns);

/// Populates IREE specific patterns to convert HLO broadcasting ops to Linalg.
/// These are being maintained separately because they are a standalone unit
/// that is both intricate and possible to upstream, should there be alignment
/// to do so.
void populateMHLOBroadcastingToLinalgPatterns(MLIRContext *context,
                                              TypeConverter &typeConverter,
                                              RewritePatternSet &patterns);

/// Populates patterns to convert MHLO collective ops to Stream ops.
void populateMHLOCollectiveOpsConversionPatterns(MLIRContext *context,
                                                 TypeConverter &typeConverter,
                                                 RewritePatternSet &patterns);

/// Populates patterns to convert MHLO/CHLO arithmetic on complex tensors to
/// equivalent HLO level real arithmetic.
void populateMHLOComplexToRealPatterns(MLIRContext *context,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns);

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_INPUTCONVERSION_MHLO_REWRITER_H_
