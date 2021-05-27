// Copyright 2021 Google LLC
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

#ifndef IREE_COMPILER_BINDINGS_NATIVE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_BINDINGS_NATIVE_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace ABI {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that setup a module for use
// with bindings following the native IREE ABI.
void buildTransformPassPipeline(OpPassManager &passManager);

void registerTransformPassPipeline();

//===----------------------------------------------------------------------===//
// IREE native ABI bindings support
//===----------------------------------------------------------------------===//

// Wraps all entry points in a function that is compatible with the
// expected invocation semantics of bindings following the native IREE ABI.
std::unique_ptr<OperationPass<ModuleOp>> createWrapEntryPointsPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerPasses() { createWrapEntryPointsPass(); }

}  // namespace ABI
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_BINDINGS_NATIVE_TRANSFORMS_PASSES_H_
