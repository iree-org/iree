// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_MHLO_CONVERTMHLOTOFLOW_H_
#define IREE_COMPILER_INPUTCONVERSION_MHLO_CONVERTMHLOTOFLOW_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

// Setup the |conversionTarget| op legality for early-phase direct-to-flow
// conversion from the MHLO dialect. This will make certain ops illegal that we
// know we have good patterns for such that we can be sure we catch them before
// they are outlined into dispatch regions.
void setupDirectMHLOToFlowLegality(MLIRContext *context,
                                   ConversionTarget &conversionTarget);

// Appends all patterns for converting MHLO ops to flow ops.
void populateMHLOToFlowPatterns(MLIRContext *context,
                                RewritePatternSet &patterns);

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_INPUTCONVERSION_MHLO_CONVERTMHLOTOFLOW_H_
