// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Populates only the flow.stream.* conversion patterns.
void populateFlowStreamToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &converter);

// Populates only the flow.tensor.* conversion patterns.
void populateFlowTensorToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &converter);

// Populates only the flow.variable.* conversion patterns.
void populateFlowVariableToHALPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns,
                                       TypeConverter &converter);

void setupFlowToHALLegality(MLIRContext *context,
                            ConversionTarget &conversionTarget,
                            TypeConverter &typeConverter) {
  conversionTarget.addIllegalDialect<IREE::Flow::FlowDialect>();
}

// Populates conversion patterns for Flow->HAL.
void populateFlowToHALPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter) {
  populateFlowStreamToHALPatterns(context, patterns, typeConverter);
  populateFlowTensorToHALPatterns(context, patterns, typeConverter);
  populateFlowVariableToHALPatterns(context, patterns, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
