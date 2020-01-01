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

#ifndef IREE_COMPILER_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Tensor <-> MemRef and Type Legalization
//===----------------------------------------------------------------------===//

// Converts all function signatures to use memref's for args/results.
std::unique_ptr<OpPassBase<ModuleOp>>
createConvertToMemRefCallingConventionPass();

// Removes all uses of XLA Tuples as args/results from functions by expanding
// them (recursively) into individual args/results.
std::unique_ptr<OpPassBase<ModuleOp>>
createConvertFromTupleCallingConventionPass();

// Legalizes all types to ones supported by the IREE VM.
std::unique_ptr<OpPassBase<ModuleOp>> createLegalizeTypeStoragePass();

//===----------------------------------------------------------------------===//
// Cleanup and Dead Code Elimination
//===----------------------------------------------------------------------===//

// Removes ops that technically have side effects, but we still want to remove
// if their results are unused: e.g. alloc and load.
std::unique_ptr<OpPassBase<FuncOp>> createAggressiveOpEliminationPass();

// Drops functions from the executable module that are unreachable from any
// exported functions (iree.executable.export).
std::unique_ptr<OpPassBase<ModuleOp>>
createDropUnreachableExecutableFunctionsPass();

//===----------------------------------------------------------------------===//
// Module Analysis and Assignment
//===----------------------------------------------------------------------===//

// Assigns module-unique ordinals to functions within the module.
std::unique_ptr<OpPassBase<ModuleOp>> createAssignFunctionOrdinalsPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSFORMS_TRANSFORMS_H_
