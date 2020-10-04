// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

// Populates only the structural (module/function/etc) conversion patterns.
void populateFlowStructuralToHALPatterns(MLIRContext *context,
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

// Populates only the std.dim and std.rank conversion patterns.
void populateHalBufferViewShapePatterns(MLIRContext *context,
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
  populateFlowStructuralToHALPatterns(context, patterns, typeConverter);
  populateFlowTensorToHALPatterns(context, patterns, typeConverter);
  populateFlowVariableToHALPatterns(context, patterns, typeConverter);
  populateHalBufferViewShapePatterns(context, patterns, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
