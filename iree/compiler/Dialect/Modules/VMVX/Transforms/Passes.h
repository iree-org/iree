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

#ifndef IREE_COMPILER_DIALECT_MODULES_VMVX_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_MODULES_VMVX_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMVX {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required VMVX
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion from TF/HLO/etc to flow>
//   buildVMVXTransformPassPipeline & run
//   <serialize VM module>
void buildVMVXTransformPassPipeline(OpPassManager &passManager);

void createVMVXTransformPassPipeline();

//===----------------------------------------------------------------------===//
// Dialect conversion
//===----------------------------------------------------------------------===//

// Converts from various dialects (HAL, standard, etc) to the VMVX dialect.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerVMVXPasses() { createVMVXTransformPassPipeline(); }

}  // namespace VMVX
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_MODULES_VMVX_TRANSFORMS_PASSES_H_
