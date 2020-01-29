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

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Given an arbitrary list of inputs, builds IR to obtain their shapes and
// cast them to a given !shape.ranked_shape. Statically verifiable invariants
// will be checked within this call and runtime code will be emitted to verify
// the rest.
// Returns nullptr and emits an error on violation of an invariant.
Value buildCastInputsToResultShape(Location loc,
                                   RankedShapeType resultShapeType,
                                   ArrayRef<Value> inputs, OpBuilder &builder);

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_BUILDERS_H_
