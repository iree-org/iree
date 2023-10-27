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

#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Lowering using SIMT CUDA core operations.
void addGPUMatmulSimtPassPipeline(OpPassManager &pm);

/// Lowering using mma.sync Tensor Core operations.
void addGPUMatmulTensorCoreMmaSyncPassPipeline(OpPassManager &pm,
                                               unsigned pipelineDepth);

/// Lowering using wmma Tensor Core operations.
void addGPUMatmulTensorCorePassPipeline(OpPassManager &pm,
                                        unsigned pipelineDepth);

void addGPUPackUnPackPasses(OpPassManager &pm);

/// Simple lowering only distributute linalg ops on blocks and threads. This
/// will result in scalar operations. Expects pass manager to be a
/// module-level pass manager.
void addGPUSimpleDistributePassPipeline(OpPassManager &pm);

/// Transform dialect-based path.
void addGPUTransformDialectPasses(OpPassManager &pm);

/// Lowering transpose using shared memory.
void addGPUTransposePassPipeline(OpPassManager &pm);

/// Lowering calling vectorization patterns. Expects pass manager to be a
/// module-level pass manager.
void addGPUVectorizationPassPipeline(OpPassManager &pm);

/// Lowering reductions to warp reductions.
void addGPUWarpReductionPassPipeline(OpPassManager &pm);

/// Default pass pipeline on GPU, currently used only for the ukernel path.
void addGPUDefaultPassPipeline(OpPassManager &pm);

/// Populates passes needed to preprocess and select the strategy for lowering.
void buildLLVMGPUTransformSelectionPassPipeline(OpPassManager &pm);

/// Populates passes needed to lower a XLA HLO op to NVVM/ROCDL dialect via
/// the structured ops path. The pass manager `pm` in here should operate on
/// the module within the IREE::HAL::ExecutableOp.
void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM);

/// Performs the final conversion to NNVM+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToNVVMPass();

/// Performs the final conversion to ROCDL+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToROCDLPass();

/// Cast address space to generic in CallOp and FuncOp
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMGPUCastAddressSpaceFunction();

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUDistribute();

/// Create pass selecting the lowering strategy for LLVMGPU.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMGPUSelectLoweringStrategyPass();

/// Create pass calling the dynamic pipeline for LLVMGPU.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMGPULowerExecutableTargetPass();

// Pass to pack shared memory allocations in order to reduce shared memory
// usage.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUPackSharedMemoryAlloc();

enum class GPUTensorCoreType {
  WMMA = 0,
  MMA_SYNC = 1,
};

/// Convert Linalg ops to Vector and prepare converstion to GPU MMA ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUTensorCoreVectorizationPass(
    GPUTensorCoreType tensorCoreType = GPUTensorCoreType::WMMA);

//. Pass to pad out tensors up to static dimensions.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorPadPass();

/// Perform tiling and distribution to threads.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUTileAndDistribute(bool distributeToWarp = false);

/// Lower vector ops before convertion to LLVM.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorLoweringPass();

/// Converts vector ops to gpu dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorToGPU(
    GPUTensorCoreType tensorCoreType = GPUTensorCoreType::WMMA);

/// Lowering calling vectorization patterns.
LogicalResult
verifyGPUMatmulPipeline(Operation *op,
                        IREE::Codegen::LoweringConfigAttr loweringConfig,
                        IREE::Codegen::TranslationInfoAttr translationInfo,
                        ArrayRef<int64_t> workgroupSize);

//----------------------------------------------------------------------------//
// Register LLVMGPU Passes
//----------------------------------------------------------------------------//

void registerCodegenLLVMGPUPasses();

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

std::unique_ptr<OperationPass<ModuleOp>> createTestLLVMGPULegalizePass();

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_PASSES_H_
