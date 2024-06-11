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

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Pass Pipeline Options
//===----------------------------------------------------------------------===//

/// Named attributes used in the `translation_info`'s config dictionary
/// attribute. These are used to override default pass heuristics at the
/// function granularity.
namespace LLVMGPUAttrNames {
inline constexpr StringLiteral kReorderWorkgroups = "reorder_workgroups";
inline constexpr StringLiteral kNoReduceSharedMemoryBankConflicts =
    "no_reduce_shared_memory_bank_conflicts";
} //  namespace LLVMGPUAttrNames

struct LLVMGPUPipelineOptions {
  enum reorderWorkGroupOption { None, Transpose, Swizzle };

  bool enableReduceSharedMemoryBankConflicts = true;
  bool enableReorderWorkgroups = false;
  bool enableUkernels = false;

  reorderWorkGroupOption reorderOption = None;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const LLVMGPUPipelineOptions &options);

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Lowering using SIMT CUDA core operations.
void addGPUMatmulSimtPassPipeline(OpPassManager &funcPassManager,
                                  const LLVMGPUPipelineOptions &options);

/// Lowering using mma.sync Tensor Core operations.
void addGPUMatmulTensorCoreMmaSyncPassPipeline(
    OpPassManager &funcPassManager, const LLVMGPUPipelineOptions &options,
    unsigned pipelineDepth);

/// Lowering using wmma Tensor Core operations.
void addGPUMatmulTensorCorePassPipeline(OpPassManager &funcPassManager,
                                        const LLVMGPUPipelineOptions &options,
                                        unsigned pipelineDepth);

void addGPUPackUnPackPasses(OpPassManager &funcPassManager);

/// Simple lowering only distributute linalg ops on blocks and threads. This
/// will result in scalar operations. Expects pass manager to be a
/// module-level pass manager.
void addGPUSimpleDistributePassPipeline(OpPassManager &funcPassManager);

/// Transform dialect-based path.
void addGPUTransformDialectPasses(OpPassManager &funcPassManager,
                                  StringRef entryPoint);

/// Lowering transpose using shared memory.
void addGPUTransposePassPipeline(OpPassManager &funcPassManager,
                                 const LLVMGPUPipelineOptions &options);

/// Lowering calling vectorization patterns. Expects pass manager to be a
/// module-level pass manager.
void addGPUVectorizationPassPipeline(OpPassManager &funcPassManager);

/// Lowering for winograd transform ops. Follows `VectorizationPassPipeline`
/// with different tiling and distribution passes.
void addGPUWinogradVectorizePassPipeline(OpPassManager &funcPassManager);

/// Lowering based on vector distribution patterns.
void addGPUVectorDistributePassPipeline(OpPassManager &funcPassManager,
                                        const LLVMGPUPipelineOptions &options,
                                        bool usePadToModelSharedMemcpy);

/// Lowering reductions to warp reductions.
void addGPUWarpReductionPassPipeline(OpPassManager &funcPassManager);

/// Default pass pipeline on GPU, currently used only for the ukernel path.
void addGPUDefaultPassPipeline(OpPassManager &funcPassManager,
                               const LLVMGPUPipelineOptions &options);

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

/// Performs the final conversion to NNVM+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToNVVMPass();

/// Performs the final conversion to ROCDL+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToROCDLPass();

/// Cast address space to generic in CallOp and FuncOp
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMGPUCastAddressSpaceFunction();

/// Perform type extension/truncation over vector.contract types to target GPU
/// MMA intrinsics.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUCastTypeToFitMMAPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUDistribute();

/// Create pass selecting the lowering strategy for LLVMGPU.
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMGPUSelectLoweringStrategyPass();

/// Create pass calling the dynamic pipeline for LLVMGPU.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPULowerExecutableTargetPass();

// Pass to pack shared memory allocations in order to reduce shared memory
// usage.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUPackSharedMemoryAlloc();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUPrefetchSharedMemoryPass();

/// Pass to pad operations on tensors in top-down order.
enum class LLVMGPUMatmulPadOption { ParallelDims, ReductionDims };
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUPromoteMatmulToFitMMAPass(
    LLVMGPUMatmulPadOption option = LLVMGPUMatmulPadOption::ParallelDims);

enum class GPUTensorCoreType {
  WMMA = 0,
  MMA_SYNC = 1,
};

/// Convert Linalg ops to Vector and prepare converstion to GPU MMA ops.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUTensorCoreVectorizationPass(
    GPUTensorCoreType tensorCoreType = GPUTensorCoreType::WMMA);

//. Pass to pad out tensors up to static dimensions.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUTensorPadPass();

/// Perform tiling and distribution to threads.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUTileAndDistribute(bool distributeToWarp = false);

// Pass to distribute vectorized functions.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUVectorDistribute();

/// Lower vector ops before convertion to LLVM.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUVectorLoweringPass();

/// Converts vector ops to gpu dialect.
std::unique_ptr<InterfacePass<FunctionOpInterface>> createLLVMGPUVectorToGPU(
    GPUTensorCoreType tensorCoreType = GPUTensorCoreType::WMMA);

/// Lowering calling vectorization patterns.
LogicalResult
verifyGPUMatmulPipeline(Operation *op,
                        IREE::Codegen::LoweringConfigAttr loweringConfig,
                        IREE::Codegen::TranslationInfoAttr translationInfo,
                        ArrayRef<int64_t> workgroupSize);

/// Given a chain of matmuls with some or no operations
/// in between, like
///
/// d = matmul_transpose_b(a, b) + c
/// ...
/// e = matmul_transpose_b(d, f) + g
///
/// this pattern transforms the above IR to
///
/// c.t = transpose c
/// d = matmul_transpose_b(b, a) + c.t
/// d.t = transpose d
/// ...
/// g.t = transpose g
/// e = matmul_transpose_b(f, d.t) + g.t
/// e.t = transpose e
///
/// On CDNA architectures, where the layouts of the RHS and result
/// are the same and transposed from the LHS layout, this type
/// of transformation can avoid trips to shared memory/shuffle instructions
/// on operators like Flash Attention.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAMDGPUPrepareForChainedMatmulPass();

//----------------------------------------------------------------------------//
// Register LLVMGPU Passes
//----------------------------------------------------------------------------//

void registerCodegenLLVMGPUPasses();

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

std::unique_ptr<OperationPass<ModuleOp>> createTestLLVMGPULegalizePass();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_PASSES_H_
