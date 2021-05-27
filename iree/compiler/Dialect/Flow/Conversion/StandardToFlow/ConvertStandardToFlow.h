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
