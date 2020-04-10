// Copyright 2020 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PATTERNS_H_
#define IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PATTERNS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Sets up legality for shape calculation materialization conversions.
void setupMaterializeShapeCalculationsLegality(ConversionTarget &target);

// Populates patterns that will materialize shape calculations for any
// GetRankedShape and related ops.
void populateMaterializeShapeCalculationsConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context);

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PATTERNS_H_
