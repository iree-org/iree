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

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

/// Lowering calling vectorization patterns. Expects pass manager to be a
/// module-level pass manager.
void addGPUVectorizationPassPipeline(OpPassManager &pm);

/// Lowering calling vectorization patterns.
LogicalResult verifyGPUMatmulPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);

/// Lowering using SIMT CUDA core operations.
void addGPUMatmulSimtPassPipeline(OpPassManager &pm);

/// Lowering using wmma Tensor Core operations.
void addGPUMatmulTensorCorePassPipeline(OpPassManager &pm,
                                        unsigned pipelineDepth);

/// Lowering using mma.sync Tensor Core operations.
void addGPUMatmulTensorCoreMmaSyncPassPipeline(OpPassManager &pm,
                                               unsigned pipelineDepth);

enum class GPUPromoteSharedMemPattern {
  ContractionOpPattern = 0,
  TransposeOpPattern = 1,
};

/// Lowering transpose using shared memory.
void addGPUTransposePassPipeline(OpPassManager &pm);

/// Lowering reductions to warp reductions.
void addGPUWarpReductionPassPipeline(OpPassManager &pm);

/// Transform dialect-based path.
void addGPUTransformDialectPasses(OpPassManager &pm);

/// Simple lowering only distributute linalg ops on blocks and threads. This
/// will result in scalar operations. Expects pass manager to be a
/// module-level pass manager.
void addGPUSimpleDistributePassPipeline(OpPassManager &pm);

void addGPUPackUnPackPasses(OpPassManager &pm);

/// Populates passes needed to lower a XLA HLO op to NVVM/ROCDL dialect via
/// the structured ops path. The pass manager `pm` in here should operate on
/// the module within the IREE::HAL::ExecutableOp.
void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM);

/// Performs the final conversion to NNVM+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToNVVMPass();

/// Performs the final conversion to ROCDL+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToROCDLPass();

/// Perform tiling and distribution to threads.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTileAndDistribute(
    bool distributeToWarp = false);

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTileTensor(
    bool distributeToWarp = false);

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUDistribute();

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorAlloc(
    GPUPromoteSharedMemPattern promoteSharedMemPattern =
        GPUPromoteSharedMemPattern::ContractionOpPattern);

/// Create pass calling the dynamic pipeline for LLVMGPU.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMGPULowerExecutableTargetPass();

enum class GPUTensorCoreType {
  WMMA = 0,
  MMA_SYNC = 1,
};

/// Convert Linalg ops to Vector and prepare converstion to GPU MMA ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUTensorCoreVectorizationPass(
    GPUTensorCoreType tensorCoreType = GPUTensorCoreType::WMMA);

/// Lower vector ops before convertion to LLVM.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorLoweringPass();

/// Converts vector ops to gpu dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorToGPU(
    GPUTensorCoreType tensorCoreType = GPUTensorCoreType::WMMA);

//. Pass to pad out tensors up to static dimensions.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorPadPass();

// Pass to pack shared memory allocations in order to reduce shared memory
// usage.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUPackSharedMemoryAlloc();

/// Checks GPU backend specific IR constraints such as shared memory limits.
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMGPUCheckIRBeforeLLVMConversionPass();

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

std::unique_ptr<OperationPass<ModuleOp>> createTestLLVMGPULegalizePass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_PASSES_H_
