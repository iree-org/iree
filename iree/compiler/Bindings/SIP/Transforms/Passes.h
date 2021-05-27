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

#ifndef IREE_COMPILER_BINDINGS_SIP_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_BINDINGS_SIP_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace SIP {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that setup a module for use
// with an IREE SIP-compatible runtime binding implementation (python, etc).
void buildTransformPassPipeline(OpPassManager &passManager);

void registerTransformPassPipeline();

//===----------------------------------------------------------------------===//
// SIP-compatible bindings support
//===----------------------------------------------------------------------===//

// Materializes reflection metadata on exported function arguments and results.
// This runs as close to the input processing as possible as it needs to
// annotate the ABI that the consumer is expecting to interop with.
std::unique_ptr<OperationPass<FuncOp>> createMaterializeReflectionAttrsPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerPasses() {
  registerTransformPassPipeline();
  createMaterializeReflectionAttrsPass();
}

}  // namespace SIP
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_BINDINGS_SIP_TRANSFORMS_PASSES_H_
