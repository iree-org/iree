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

//===- Passes.h - Codegen pass to convert from XLA to Linalg on buffers----===//
//
// IREE specific passes used in the XLA to Linalg conversion
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CONVERSION_HLOTOLINALG_PASSES_H_
#define IREE_COMPILER_CONVERSION_HLOTOLINALG_PASSES_H_
#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Resolves shape related ops (std.dim, shapex.tie_shape, etc.) by tracing
/// them back to the original HAL interface bindings.
std::unique_ptr<OperationPass<FuncOp>> createResolveShapeOpsPass();
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_HLOTOLINALG_PASSES_H_
