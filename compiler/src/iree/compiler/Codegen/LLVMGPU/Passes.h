// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the LLVMGPU Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_PASSES_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_PASSES_H_

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

using IREE::GPU::GPUPipelineOptions;

//----------------------------------------------------------------------------//
// LLVMGPU backend Pass Pipelines.
//----------------------------------------------------------------------------//

/// Lowering using SIMT CUDA core operations.
void addGPUMatmulSimtPassPipeline(OpPassManager &funcPassManager,
                                  const GPUPipelineOptions &options);

/// Lowering using mma.sync Tensor Core operations.
void addGPUMatmulTensorCoreMmaSyncPassPipeline(
    OpPassManager &funcPassManager, const GPUPipelineOptions &options,
    unsigned pipelineDepth);

/// Lowering using wmma Tensor Core operations.
void addGPUMatmulTensorCorePassPipeline(OpPassManager &funcPassManager,
                                        const GPUPipelineOptions &options,
                                        unsigned pipelineDepth);

void addGPUPackUnPackPasses(OpPassManager &funcPassManager);

/// Simple lowering only distributute linalg ops on blocks and threads. This
/// will result in scalar operations. Expects pass manager to be a
/// module-level pass manager.
void addGPUSimpleDistributePassPipeline(OpPassManager &funcPassManager);

/// Lowering config driven pipeline that uses greedy tile + fuse to distribute
/// to threads.
void addGPUTileAndFusePassPipeline(OpPassManager &funcPassManager,
                                   const GPUPipelineOptions &pipelineOptions);

/// Transform dialect-based path.
void addGPUTransformDialectPasses(OpPassManager &funcPassManager,
                                  StringRef entryPoint);

/// Lowering transpose using shared memory.
void addGPUTransposePassPipeline(OpPassManager &funcPassManager,
                                 const GPUPipelineOptions &options);

/// Lowering calling vectorization patterns. Expects pass manager to be a
/// module-level pass manager.
void addGPUVectorizationPassPipeline(OpPassManager &funcPassManager);

/// Lowering for winograd transform ops. Follows `VectorizationPassPipeline`
/// with different tiling and distribution passes.
void addGPUWinogradVectorizePassPipeline(OpPassManager &funcPassManager);

/// Lowering based on vector distribution patterns.
void addGPUVectorDistributePassPipeline(OpPassManager &funcPassManager,
                                        const GPUPipelineOptions &options,
                                        bool usePadToModelSharedMemcpy);

/// Lowering reductions to warp reductions.
void addGPUWarpReductionPassPipeline(OpPassManager &funcPassManager);

/// Default pass pipeline on GPU, currently used only for the ukernel path.
void addGPUDefaultPassPipeline(OpPassManager &funcPassManager,
                               const GPUPipelineOptions &options);

/// Pass pipeline to lower IREE HAL executables without tiling and distribution.
void addGPUBaseLoweringPassPipeline(OpPassManager &pm);

/// Populates passes needed to preprocess and select the translation strategy.
void buildLLVMGPUCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManagery);

/// Populates passes needed to lower a XLA HLO op to NVVM/ROCDL dialect via
/// the structured ops path. The pass manager `pm` in here should operate on
/// the module within the IREE::HAL::ExecutableOp.
void buildLLVMGPUCodegenPassPipeline(OpPassManager &variantPassManagery,
                                     bool useROCM);

/// Lowering calling vectorization patterns.
LogicalResult
verifyGPUMatmulPipeline(Operation *op,
                        IREE::Codegen::LoweringConfigAttr loweringConfig,
                        IREE::Codegen::TranslationInfoAttr translationInfo,
                        ArrayRef<int64_t> workgroupSize);

//------------------------------------------------------------------------------
// Wrappers that not use tablegen options.
//------------------------------------------------------------------------------

enum class LLVMGPUMatmulPadOption { ParallelDims, ReductionDims };
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUPromoteMatmulToFitMMAPass(LLVMGPUMatmulPadOption option);

enum class GPUTensorCoreType {
  WMMA = 0,
  MMA_SYNC = 1,
};

std::unique_ptr<InterfacePass<FunctionOpInterface>>

createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType tensorCoreType);
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUVectorToGPUPass(GPUTensorCoreType tensorCoreType);

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUTileAndDistributePass(bool distributeToWarp);

//----------------------------------------------------------------------------//
// Register LLVMGPU Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc" // IWYU pragma: keep

void registerCodegenLLVMGPUPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_PASSES_H_
