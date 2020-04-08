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

#ifndef IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_

#include <memory>

#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

class OpPassManager;

namespace iree_compiler {

// Populates a pass manager with the pipeline to expand functions to include
// explicit shape calculations for all dynamic tensors.
void populateMaterializeDynamicShapesPipeline(OpPassManager &pm);

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OperationPass<FuncOp>> createExpandFunctionDynamicDimsPass();

// For any function which contains ranked_shape argument/result types,
// expands them to individual dynamic dimensions, inserting appropriate casts
// within the function.
std::unique_ptr<OperationPass<FuncOp>>
createExpandFunctionRankedShapeDimsPass();

// For any dynamically shaped edges in a function, introduces an appropriate
// get_ranked_shape and corresponding tie_shape op to make the association.
std::unique_ptr<OperationPass<FuncOp>> createTieDynamicShapesPass();

// Materializes shape calculations for any get_ranked_shape ops.
std::unique_ptr<OperationPass<FuncOp>> createMaterializeShapeCalculationsPass();

// Cleans up any unnecessary shape placeholder ops. Can be run after all
// shape calculation code has been lowered.
std::unique_ptr<OperationPass<FuncOp>> createCleanupShapePlaceholdersPass();

// Converts shape-sensitive HLOs to be based on facilities in the shape
// dialect.
std::unique_ptr<OperationPass<FuncOp>> createConvertHLOToShapePass();

// Best-effort hoisting of shape calculations to attempt to establish the
// invariant that shape.tie_shape second operand dominates the first operand.
std::unique_ptr<OperationPass<FuncOp>> createHoistShapeCalculationsPass();

// Register all Passes
inline void registerShapePasses() {
  createExpandFunctionDynamicDimsPass();
  createExpandFunctionRankedShapeDimsPass();
  createTieDynamicShapesPass();
  createMaterializeShapeCalculationsPass();
  createCleanupShapePlaceholdersPass();
  createConvertHLOToShapePass();
  createHoistShapeCalculationsPass();
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
