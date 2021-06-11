// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_LINALGTOLLVM_PASSES_H_
#define IREE_COMPILER_CONVERSION_LINALGTOLLVM_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace iree_compiler {

/// Converts linalg.conv into linalg.generic with a CPU-friendly iteration
/// order.
std::unique_ptr<FunctionPass> createPlanConvLoopOrderPass();

/// Pad linalg ops workgroup tiles into the next integer multiple of the target
/// vector size.
std::unique_ptr<OperationPass<FuncOp>> createPadLinalgWorkgroupTilesPass();

/// Multi-level tiling, padding and vectorization of  linalg ops on tensors.
std::unique_ptr<FunctionPass> createTilePadAndVectorizeWorkgroupsPass();

/// Vectorizes linalg ops executed in the same hal.interface.workgroup.
std::unique_ptr<FunctionPass> createLinalgTileAndVectorizeWorkgroupsPass(
    bool lowerToVectors = true);

/// Replaces llvm.intr.fma with its unfused mul and add ops.
std::unique_ptr<FunctionPass> createUnfusedFMAOpsPass();

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns);

/// Performs the final conversion to LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass(
    bool unfuseFMAOps = false);

/// Pass to convert Linalg ops into vector operations.
std::unique_ptr<FunctionPass> createLinalgVectorizePass();

//===----------------------------------------------------------------------===//
// Pass Pipelines for CPU Lowering.
//===----------------------------------------------------------------------===//

/// Populates the passes to lower to scalars operations for linalg based
/// code-generation. This pipeline does not vectorize, but instead just converts
/// to memrefs
void addCPUDefaultPassPipeline(OpPassManager &passManager);

/// Populates the passes needed to lower to vector operations using linalg based
/// progressive lowering with vectorization after bufferization.
void addCPUVectorizationPassPipeline(OpPassManager &passManager,
                                     bool lowerToVectors = true);

/// Pass to lower the module an hal.executable.target operation to external
/// dialect. Currently this pass lowers to LLVM dialect, but could be
/// generalized to lower to any "final" dialect like SPIR-V/NVVM, etc.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLowerExecutableTargetPass(bool lowerToVectors = true);

//===----------------------------------------------------------------------===//
// Pass Pipelines for lowering to LLVM dialect.
//===----------------------------------------------------------------------===//

/// Options for LLVM pipeline.
struct LLVMTransformPassPipelineOptions
    : public PassPipelineOptions<LLVMTransformPassPipelineOptions> {
  Option<bool> unfuseFMAOps{
      *this, "unfuse-fma-ops",
      llvm::cl::desc("Enable rewriting llvm.fma to its unfused version."),
      llvm::cl::init(false)};
};

/// Populates passes needed to lower a XLA HLO op to LLVM dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMTransformPassPipeline(
    OpPassManager &passManager,
    const LLVMTransformPassPipelineOptions &options);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOLLVM_PASSES_H_
