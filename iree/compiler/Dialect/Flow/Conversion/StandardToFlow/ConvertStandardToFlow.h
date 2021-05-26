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

// Setup the |conversionTarget| op legality for early-phase direct-to-flow
// conversion from the standard op dialect. This will make certain ops illegal
// that we know we have good patterns for such that we can be sure we catch them
// before they are outlined into dispatch regions.
void setupDirectStandardToFlowLegality(MLIRContext *context,
                                       ConversionTarget &conversionTarget);

// Appends all patterns for converting std ops to flow ops.
void populateStandardToFlowPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_CONVERSION_STANDARDTOFLOW_CONVERTSTANDARDTOFLOW_H_
