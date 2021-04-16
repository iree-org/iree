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

#ifndef IREE_COMPILER_CONVERSION_LINALGTOLLVM_PASSES_H_
#define IREE_COMPILER_CONVERSION_LINALGTOLLVM_PASSES_H_

#include "iree/compiler/Conversion/LinalgToLLVM/LLVMCodeGenOptions.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Converts linalg.conv into linalg.generic with a CPU-friendly iteration
/// order.
std::unique_ptr<FunctionPass> createPlanConvLoopOrderPass();

/// Distributes linalg ops among hal.interface.workgroup logical threads.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLinalgTileAndDistributePass();

/// Vectorizes linalg ops executed in the same hal.interface.workgroup.
std::unique_ptr<FunctionPass> createLinalgTileAndVectorizeWorkgroupsPass();

/// Replaces llvm.intr.fma with its unfused mul and add ops.
std::unique_ptr<FunctionPass> createUnfusedFMAOpsPass();

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns);

/// Performs the final conversion to LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass(
    LLVMCodegenOptions options);

/// Pass to convert Linalg ops into vector operations.
std::unique_ptr<FunctionPass> createLinalgVectorizePass();

/// Pass to materialize static launch information for a dispatch region when
/// using the linalg on tensors path.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createMaterializeCPULaunchConfigurationPass();

/// After running the upstream TensorConstantBufferize pass, remove tensor_loads
/// introduced for use only in tensor_extract. These can be folded to use a load
/// of the created memref object that holds the constant values.
std::unique_ptr<OperationPass<>> createFoldTensorExtractOpPass();

/// Populates passes needed to lower a XLA HLO op to LLVM dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMTransformPassPipeline(OpPassManager &passManager,
                                    LLVMCodegenOptions options);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOLLVM_PASSES_H_
