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

#ifndef IREE_COMPILER_CONVERSION_LINALGTONVVM_PASSES_H_
#define IREE_COMPILER_CONVERSION_LINALGTONVVM_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Performs the final conversion to NNVM+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToNVVMPass();

/// Convert Linalg ops to Vector.
std::unique_ptr<OperationPass<FuncOp>> createVectorizationPass();

/// Perform tiling and distribution to threads.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createTileAndDistributeToThreads();

std::unique_ptr<OperationPass<FuncOp>> createRemoveSingleIterationLoopPass();

/// Populates passes needed to lower a XLA HLO op to NVVM dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildNVVMTransformPassPipeline(OpPassManager &pm);
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTONVVM_PASSES_H_
