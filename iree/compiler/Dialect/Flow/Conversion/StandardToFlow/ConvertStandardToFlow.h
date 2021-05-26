// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_CONVERSION_STANDARDTOFLOW_CONVERTSTANDARDTOFLOW_H_
#define IREE_COMPILER_DIALECT_FLOW_CONVERSION_STANDARDTOFLOW_CONVERTSTANDARDTOFLOW_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Setup the |conversionTarget| op legality for conversion of standard ops
// which should be mapped to flow.tensor.load. This is maintained as a very
// specific legalization because flow.tensor.load represents a kind of host
// read-back and should be materialized at specific points.
void setupStandardToFlowTensorLoadLegality(MLIRContext *context,
                                           ConversionTarget &conversionTarget);

// Appends all patterns for converting to flow.tensor.load.
void populateStandardToFlowTensorLoadPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_CONVERSION_STANDARDTOFLOW_CONVERTSTANDARDTOFLOW_H_
