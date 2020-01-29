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

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OpPassBase<ModuleOp>> createExpandFunctionDynamicDimsPass();

// Materializes shape calculations for any get_ranked_shape ops.
std::unique_ptr<OpPassBase<FuncOp>> createMaterializeShapeCalculationsPass();

// Cleans up any unnecessary shape placeholder ops. Can be run after all
// shape calculation code has been lowered.
std::unique_ptr<OpPassBase<FuncOp>> createCleanupShapePlaceholdersPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
