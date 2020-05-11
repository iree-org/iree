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

#ifndef IREE_COMPILER_DIALECT_VMLA_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_VMLA_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required VMLA
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion from TF/HLO/etc to flow>
//   buildVMLATransformPassPipeline & run
//   <serialize VM module>
void buildVMLATransformPassPipeline(OpPassManager &passManager);

void createVMLATransformPassPipeline();

//===----------------------------------------------------------------------===//
// Input canonicalization and legalization
//===----------------------------------------------------------------------===//

// Unrolls multi-dimensional reduction operations into reductions along each
// dimension, from innermost to outermost.
std::unique_ptr<OperationPass<FuncOp>> createUnrollReductionsPass();

// Tensor-level pattern-based lowerings. Thrown into one pass for simplicity.
std::unique_ptr<OperationPass<FuncOp>> createPreConversionLoweringPass();

//===----------------------------------------------------------------------===//
// Dialect conversion
//===----------------------------------------------------------------------===//

// Converts from various dialects (standard, HLO, etc) to the VMLA dialect.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerVMLAPasses() {
  createVMLATransformPassPipeline();
  createUnrollReductionsPass();
  createConversionPass();
  createPreConversionLoweringPass();
}

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VMLA_TRANSFORMS_PASSES_H_
