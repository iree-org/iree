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

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_SHAPEOPS_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_SHAPEOPS_H_

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Populates conversion patterns that perform folding and canonicalization of
// shape ops. These patterns are intended to be used with the dialect conversion
// framework.
void populateFoldConversionPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns);

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h.inc"

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_SHAPEOPS_H_
