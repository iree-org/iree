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

/// Pad linalg ops workgroup tiles into the next integer multiple of the target
/// vector size.
std::unique_ptr<OperationPass<FuncOp>> createPadLinalgWorkgroupTilesPass();

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

//===----------------------------------------------------------------------===//
// Pass Pipelines for CPU Lowering
//===----------------------------------------------------------------------===//

/// Populates the passes to lower to scalars operations for linalg based
/// code-generation. This pipeline does not vectorize, but instead just converts
/// to memrefs
void addCPUDefaultPassPipeline(OpPassManager &passManager,
                               LLVMCodegenOptions options);

/// Populates the passes needed to lower to vector operations using linalg based
/// progressive lowering with vectorization after bufferization.
void addCPUVectorizationPassPipeline(OpPassManager &passManager,
                                     LLVMCodegenOptions options);

/// Populates the passes needed to lower scalar/native vector code to LLVM
/// Dialect.
void addLowerToLLVMPasses(OpPassManager &passManager,
                          LLVMCodegenOptions options);

/// Pass to lower the module an hal.executable.target operation to external
/// dialect. Currently this pass lowers to LLVM dialect, but could be
/// generalized to lower to any "final" dialect like SPIR-V/NVVM, etc.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLowerExecutableTargetPass(LLVMCodegenOptions options);

/// Populates passes needed to lower a XLA HLO op to LLVM dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMTransformPassPipeline(OpPassManager &passManager,
                                    LLVMCodegenOptions options);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOLLVM_PASSES_H_
