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

#ifndef IREE_COMPILER_CONVERSION_CODEGENUTILS_FUNCTIONUTILS_H_
#define IREE_COMPILER_CONVERSION_CODEGENUTILS_FUNCTIONUTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

/// Returns true if the given `func` is a kernel dispatch entry point.
bool isEntryPoint(FuncOp func);

/// Returns the number of outer parallel loops of a linalgOp.
unsigned getNumOuterParallelLoops(linalg::LinalgOp op);

/// Returns the entry point op for the `funcOp`. Returns `nullptr` on failure.
IREE::HAL::ExecutableEntryPointOp getEntryPoint(FuncOp funcOp);

/// Gets the source type of ops that implement ViewOpInterface recursively. Can
/// be used to get the untiled operands from a tiled operation.
Value getViewSource(Value view);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_CODEGENUTILS_FUNCTIONUTILS_H_
