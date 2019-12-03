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
