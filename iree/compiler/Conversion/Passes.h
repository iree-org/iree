// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_PASSES_H_
#define IREE_COMPILER_CONVERSION_PASSES_H_

#include <memory>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace iree_compiler {

// Registers all conversion passes in this directory.
void registerConversionPasses();

//------------------------------------------------------------------------------
// Misc/common conversions
//------------------------------------------------------------------------------

/// Create a pass to convert a model using f32 type to the equivalent one
/// using f16.
std::unique_ptr<OperationPass<ModuleOp>> createDemoteF32ToF16Pass();

/// Alias for callback function that allocates workgroup level memory. The
/// callback expects the static shape and element-type of the result memref
/// type. Also expects values for the dynamic dimension of the allocated memref,
/// where each dynamic dimension corresponds to a `ShapedType::kDynamicSize`
/// value in `staticShape`.
using WorkgroupMemoryAllocationFn = std::function<Value(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> staticShape,
    Type elementType, ArrayRef<Value> dynamicSizes)>;

/// Adds passes to convert tiled+distributed linalg on tensors code to linalg on
/// buffers.
void addLinalgBufferizePasses(
    OpPassManager &passManager,
    WorkgroupMemoryAllocationFn allocationFn = nullptr);

/// Pass to perform linalg on tensor bufferization. The function passed into the
/// pass through the `allocationFn` argument is invoked whenever a new buffer is
/// to be created. The callback will be passed the Values for the dynamic
/// dimensions in the memref type that is to be allocated.  The callback is
/// expected to return a MemRefType Value.  When no `allocationFn` is specified,
/// the default allocator generates an `std.alloc` instruction with the
/// allocated MemRefType having no stride map (i.e. default row-major striding)
/// and default memory space.
std::unique_ptr<OperationPass<FuncOp>> createLinalgBufferizePass(
    WorkgroupMemoryAllocationFn allocationFn = nullptr);

/// Pass to optimize vector transfer_read and transfer_write.
std::unique_ptr<OperationPass<FuncOp>> createOptimizeVectorTransferPass();

/// An ad-hoc pass to canonicalize selected loop carried dependencies on
/// scf.for.
std::unique_ptr<OperationPass<FuncOp>> createForOpCanonicalizationPass();

/// Pass to perform canonicalizations/cleanups related to HAL interface/buffer
/// allocations and view operations.
std::unique_ptr<OperationPass<FuncOp>> createCleanupBufferAllocViewPass();

/// Flattens n-D MemRef subspan ops to 1-D MemRef and folds the byte offsets on
/// subspan ops to the consumer load/store ops, in preparation for lowering to
/// backends that require linearized access.
std::unique_ptr<OperationPass<ModuleOp>> createFlattenMemRefSubspanPass();

/// Sets the number of workgroups to use for each entry point in the dispatch
/// region.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createSetNumWorkgroupsPass(ArrayRef<int64_t> workgroupSize = {});

/// After running the upstream TensorConstantBufferize pass, remove tensor_loads
/// introduced for use only in tensor_extract. These can be folded to use a load
/// of the created memref object that holds the constant values.
std::unique_ptr<OperationPass<>> createFoldTensorExtractOpPass();

//------------------------------------------------------------------------------
// LinalgToLinalg
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// LinalgToLLVM
//------------------------------------------------------------------------------

/// Converts linalg.conv into linalg.generic with a CPU-friendly iteration
/// order.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgToLLVMPlanConvLoopOrderPass();

/// Pad linalg ops workgroup tiles into the next integer multiple of the target
/// vector size.
std::unique_ptr<OperationPass<FuncOp>> createLLVMPadLinalgWorkgroupTilesPass();

/// Multi-level tiling, padding and vectorization of  linalg ops on tensors.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgToLLVMTilePadAndVectorizeWorkgroupsPass();

/// Vectorizes linalg ops executed in the same hal.interface.workgroup.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgToLLVMWorkgroupsVectorizationPass(bool lowerToVectors = true);

/// Replaces llvm.intr.fma with its unfused mul and add ops.
std::unique_ptr<OperationPass<FuncOp>> createLinalgToLLVMUnfuseFMAOpsPass();

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns);

/// Performs the final conversion to LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass(
    bool unfuseFMAOps = false);

/// Pass to convert Linalg ops into vector operations.
std::unique_ptr<OperationPass<FuncOp>> createLinalgVectorizePass();

//----------------------------------------------------------------------------//
// LinalgToLLVM Pass Pipelines for CPU Lowering.
//----------------------------------------------------------------------------//

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

//----------------------------------------------------------------------------//
// LinalgToLLVM  Pass Pipelines for lowering to LLVM dialect.
//----------------------------------------------------------------------------//

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

//------------------------------------------------------------------------------
// LinalgToGPU
//------------------------------------------------------------------------------

/// Performs the final conversion to NNVM+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>>
createLinalgToLLVMGPUConvertToNVVMPass();

/// Performs the final conversion to ROCDL+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>>
createLinalgToLLVMGPUConvertToROCDLPass();

/// Convert Linalg ops to Vector.
std::unique_ptr<OperationPass<FuncOp>> createLinalgToLLVMGPUVectorizationPass();

/// Perform tiling and distribution to threads.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLinalgToLLVMGPUTileAndDistributeToThreads();

std::unique_ptr<OperationPass<FuncOp>>
createLinalgToLLVMGPURemoveSingleIterationLoopPass();

/// Create pass calling the dynamic pipeline for LLVMGPU.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLinalgToLLVMGPULowerExecutableTargetPass();

/// Lowering calling vectorization patterns.
void addGPUVectorizationPassPipeline(OpPassManager &passManager);

/// Simple lowering only distributute linalg ops on blocks and threads. This
/// will result in scalar operations.
void addGPUSimpleDistributePassPipeline(OpPassManager &passManager);

/// Populates passes needed to lower a XLA HLO op to NVVM/ROCDL dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM);

//------------------------------------------------------------------------------
// LinalgToSPIRV Passes
//------------------------------------------------------------------------------

// Options that can be used to configure SPIR-V code generation.
struct SPIRVCodegenOptions {
  llvm::SmallVector<unsigned, 3> workgroupSize = {};
  llvm::SmallVector<unsigned, 3> workgroupTileSizes = {};
  llvm::SmallVector<unsigned, 3> invocationTileSizes = {};

  bool useWorkgroupMemory = false;

  static SPIRVCodegenOptions getFromCLOptions();
};

/// Pass to tile and vectorize Linalg operations on buffers in a single
/// workgroup.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLinalgToSPIRVTileAndVectorizeOneWorkgroupPass(
    const SPIRVCodegenOptions &options);

/// Pass to add the synchronizations and attributes needed to lower from PLoops
/// to GPU dialect.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLinalgToSPIRVConvertToGPUPass();

/// Pass to perform the final conversion to SPIR-V dialect.
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>>
createLinalgToSPIRVConvertToSPIRVPass();

/// Pass to convert vector operations to GPU level operations. Instructions of
/// vector size equal to subgroup size are distributed across the subgroup.
std::unique_ptr<OperationPass<FuncOp>> createLinalgToSPIRVConvertVectorToGPU();

/// Pass to convert vector read/write/arithmetic operations to the corresponding
/// cooperative matrix ops when possible.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgToSPIRVVectorToCooperativeMatrixPass();

/// Converts memref of scalar to memref of vector of efficent size. This will
/// allow to convert memory accesses to vector load/store in SPIR-V without
/// having pointer bitcast.
std::unique_ptr<OperationPass<ModuleOp>>
createLinalgToSPIRVVectorizeMemRefLoadStore();

/// Creates a pass to fold processor ID uses where possible.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLinalgToSPIRVFoldProcessorIDUsesPass();

/// Pass that materializes new hal.executable.entry_point ops for
/// spv.EntryPoints that are added by other passes.
/// To be removed along with SplitDispatchFunctionPass.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createMaterializeEntryPointsPass();

/// Creates a pass to concretize hal.interface.workgroup.* ops with concrete
/// tiling and distribution scheme.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLinalgToSPIRVConcretizeTileAmongWorkgroupsPass(
    const SPIRVCodegenOptions &options);

//----------------------------------------------------------------------------//
// LinalgToSPIRV Pipelines
//----------------------------------------------------------------------------//

/// Populates passes need to lower from Linalf to SPIR-V.
void buildLinalgToSPIRVPassPipeline(OpPassManager &pm,
                                    const SPIRVCodegenOptions &options);

/// Populates passes needed to lower a XLA HLO op to SPIR-V dialect via the
/// structured ops path. The pass manager `pm` in here operate on the module
/// within the IREE::HAL::ExecutableOp. The `workGroupSize` can be used to
/// control the work group size used in the code generation and is intended for
/// testing purposes only. The pass pipeline will set an appropriate workgroup
/// size.
/// TODO: Are both of these needed and does this one still work on HLO?
void buildSPIRVTransformPassPipeline(OpPassManager &pm,
                                     const SPIRVCodegenOptions &options);

//----------------------------------------------------------------------------//
// LinalgToSPIRV Patterns
//----------------------------------------------------------------------------//

/// Populates patterns to tile and distribute linalg.copy operations.
void populateTileAndDistributeLinalgCopyPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

/// Populates patterns to fold processor ID uses by using processor counts
/// information where possible.
void populateFoldGPUProcessorIDUsesPatterns(MLIRContext *context,
                                            OwningRewritePatternList &patterns);

//------------------------------------------------------------------------------
// LinalgToVector
//------------------------------------------------------------------------------

/// Creates a pass to vectorize a very specific form of linalg.conv ops.
std::unique_ptr<OperationPass<FuncOp>> createLinalgToVectorVectorizeConvPass();

/// Populates `patterns` with a very specific pattern that vectorizes a
/// linalg.conv op for a single thread. The linalg.conv should compute on
/// static-sized subviews. To match, output shape must be 1x1xWoxCo, where Co
/// Co is a multiple of 4, and filter shape must be 1x1x4xCo.
void populateLinalgToVectorVectorizeConvPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

//------------------------------------------------------------------------------
// VectorToLLVM
//------------------------------------------------------------------------------

/// A pass that converts vector dialect operations to inline assembly
std::unique_ptr<OperationPass<FuncOp>>
createVectorToAArch64InlineAssemblyPass();

/// Populates `patterns` to convert vector.contract op to a sequence
/// of AArch64 inline assembly operations.
void populateVectorContractToAArch64InlineAsm(
    OwningRewritePatternList &patterns, MLIRContext *context);

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

std::unique_ptr<OperationPass<FuncOp>>
createTestLinalgToLLVMGPUScalarizeMathOpPass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_PASSES_H_
