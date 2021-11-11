// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_PASSES_H_
#define IREE_COMPILER_CODEGEN_PASSES_H_

#include <memory>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Registers all conversion passes in this directory.
void registerCodegenPasses();

//------------------------------------------------------------------------------
// Misc/common conversions
//------------------------------------------------------------------------------

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
void addIREEComprehensiveBufferizePasses(
    OpPassManager &passManager,
    linalg::AllocationCallbacks allocationFn = linalg::AllocationCallbacks(
        linalg::defaultAllocationFn, linalg::defaultDeallocationFn,
        linalg::defaultMemCpyFn));

/// Pass to perform canonicalizations/cleanups related to HAL interface/buffer
/// allocations and view operations.
std::unique_ptr<OperationPass<FuncOp>> createCleanupBufferAllocViewPass();

/// Create a pass to convert a model using f32 type to the equivalent one
/// using f16.
std::unique_ptr<OperationPass<ModuleOp>> createDemoteF32ToF16Pass();

/// Flattens n-D MemRef subspan ops to 1-D MemRef and folds the byte offsets on
/// subspan ops to the consumer load/store ops, in preparation for lowering to
/// backends that require linearized access.
std::unique_ptr<OperationPass<ModuleOp>> createFlattenMemRefSubspanPass();

/// Creates a pass to to fold `affine.min` ops in tiled and distributed loops.
std::unique_ptr<OperationPass<FuncOp>>
createFoldAffineMinInDistributedLoopsPass();

/// After running the upstream TensorConstantBufferize pass, remove tensor_loads
/// introduced for use only in tensor_extract. These can be folded to use a load
/// of the created memref object that holds the constant values.
std::unique_ptr<OperationPass<>> createFoldTensorExtractOpPass();

/// An ad-hoc pass to canonicalize selected loop carried dependencies on
/// scf.for.
std::unique_ptr<OperationPass<FuncOp>> createForOpCanonicalizationPass();

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
std::unique_ptr<OperationPass<ModuleOp>> createIREEComprehensiveBufferizePass(
    linalg::AllocationCallbacks = linalg::AllocationCallbacks(
        linalg::defaultAllocationFn, linalg::defaultDeallocationFn,
        linalg::defaultMemCpyFn));

/// Creates a pass to remove single iteration distributed loops.
std::unique_ptr<OperationPass<FuncOp>> createRemoveSingleIterationLoopPass();

/// Creates a pass to vectorize a very specific form of linalg.conv ops.
std::unique_ptr<OperationPass<FuncOp>> createLinalgToVectorVectorizeConvPass();

/// Creates a pass to vectorize a very specific form of linalg.conv ops.
std::unique_ptr<OperationPass<FuncOp>> createLinalgToVectorVectorizeMMT4dPass();

/// Pass to optimize vector transfer_read and transfer_write.
std::unique_ptr<OperationPass<FuncOp>> createOptimizeVectorTransferPass();

/// Sets the number of workgroups to use for each entry point in the dispatch
/// region.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSetNumWorkgroupsPass(ArrayRef<int64_t> workgroupSize = {});

//----------------------------------------------------------------------------//
// Common codegen patterns.
//----------------------------------------------------------------------------//

/// Populates `patterns` with patterns to fold `affine.min` ops in tiled and
/// distributed loops.
void populateFoldAffineMinInDistributedLoopsPatterns(
    RewritePatternSet &patterns);

/// Populates `patterns` with a very specific pattern that vectorizes a
/// linalg.conv op for a single thread. The linalg.conv should compute on
/// static-sized subviews. To match, output shape must be 1x1xWoxCo, where Co
/// Co is a multiple of 4, and filter shape must be 1x1x4xCo.
void populateLinalgToVectorVectorizeConvPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

/// Populates `patterns` to convert linalg.mmt4d to vector.contract.
void populateLinalgToVectorVectorizeMMT4dPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

//------------------------------------------------------------------------------
// LLVMCPU
//------------------------------------------------------------------------------

/// Performs the final conversion to LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass();

/// Pass to lower the module an hal.executable.variant operation to external
/// dialect. Currently this pass lowers to LLVM dialect, but could be
/// generalized to lower to any "final" dialect like SPIR-V/NVVM, etc.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPULowerExecutableTargetPass(bool lowerToVectors = true);

/// Synchronizes LLVM linkage with MLIR symbol visibility.
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUSynchronizeSymbolVisibilityPass();

/// Multi-level tiling and vectorization of linalg ops on tensors.
std::unique_ptr<OperationPass<FuncOp>> createLLVMCPUTileAndVectorizePass(
    bool lowerToVectors = true);
std::unique_ptr<OperationPass<FuncOp>> createLLVMCPUTileFuseAndVectorizePass(
    bool lowerToVectors = true);

/// Vectorizes linalg ops executed in the same hal.interface.workgroup.
std::unique_ptr<OperationPass<FuncOp>> createLLVMCPUVectorizationPass(
    bool lowerToVectors = true);

/// Replaces llvm.intr.fma with its unfused mul and add ops.
std::unique_ptr<OperationPass<FuncOp>> createLLVMCPUUnfuseFMAOpsPass();

/// A pass that converts vector dialect operations to inline assembly
std::unique_ptr<OperationPass<FuncOp>>
createVectorToAArch64InlineAssemblyPass();

//------------------------------------------------------------------------------
// LLVMCPU Codegen specific patterns.
//------------------------------------------------------------------------------

/// Populates `patterns` to convert vector.contract op to a sequence
/// of AArch64 inline assembly operations.
void populateVectorContractToAArch64InlineAsm(
    OwningRewritePatternList &patterns, MLIRContext *context);

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns);

//----------------------------------------------------------------------------//
// LLVMCPU backend Pass Pipelines.
//----------------------------------------------------------------------------//

/// Populates the passes to lower to scalars operations for linalg based
/// code-generation. This pipeline does not vectorize, but instead just converts
/// to memrefs
void addCPUDefaultPassPipeline(OpPassManager &passManager);

/// Populates the passes needed to lower to vector operations using linalg based
/// progressive lowering with vectorization after bufferization.
void addCPUVectorizationPassPipeline(OpPassManager &passManager,
                                     bool lowerToVectors = true);

/// Populates the passes needed to multi level tile and lowering of linalg ops
/// on tensors to vectors operations.
void addTensorToVectorsPassPipeline(OpPassManager &passManager,
                                    bool lowerToVectors = true,
                                    bool useTileAndVectorizeV2 = false);

//----------------------------------------------------------------------------//
// LLVMCPU Pass Pipelines for lowering to LLVM dialect.
//----------------------------------------------------------------------------//

/// Populates passes needed to lower a XLA HLO op to LLVM dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMCPUCodegenPassPipeline(OpPassManager &passManager);

//------------------------------------------------------------------------------
// LLVMGPU
//------------------------------------------------------------------------------

/// Lowering calling vectorization patterns. Expects pass manager to be a
/// module-level pass manager.
void addGPUVectorizationPassPipeline(OpPassManager &pm);

/// Lowering calling vectorization patterns.
void addGPUMatmulSimtPassPipeline(OpPassManager &pm);

/// Simple lowering only distributute linalg ops on blocks and threads. This
/// will result in scalar operations. Expects pass manager to be a module-level
/// pass manager.
void addGPUSimpleDistributePassPipeline(OpPassManager &pm);

/// Populates passes needed to lower a XLA HLO op to NVVM/ROCDL dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM);

/// Performs the final conversion to NNVM+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToNVVMPass();

/// Performs the final conversion to ROCDL+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToROCDLPass();

/// Perform tiling and distribution to threads.
std::unique_ptr<OperationPass<FuncOp>>
createLLVMGPUTileAndDistributeToThreads();

/// Create pass calling the dynamic pipeline for LLVMGPU.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMGPULowerExecutableTargetPass();

/// Convert Linalg ops to Vector.
std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUVectorizationPass();

/// Lower vector ops before convertion to LLVM.
std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUVectorLoweringPass();

/// Convert shared memory copies to distributed transfer_read/transfer_write.
std::unique_ptr<OperationPass<FuncOp>>
createLLVMGPUDistributeSharedMemoryCopy();

/// Apply software pipelining.
std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUPipeliningPass();

//------------------------------------------------------------------------------
// SPIR-V Passes
//------------------------------------------------------------------------------

/// Pass pipeline to lower IREE HAL executables with workgroup tiled and
/// distributed Linalg ops to SPIR-V scalar code. Additionally performs
/// distribution to threads without vectorization.
void addSPIRVTileAndDistributePassPipeline(OpPassManager &pm);

/// Pass pipeline to lower IREE HAL executables with workgroup tiled and
/// distributed Linalg ops to SPIR-V scalar and vector code. Additionally
/// performs distribution to threads with vectorization.
void addSPIRVTileAndVectorizePassPipeline(OpPassManager &pm);

/// Pass pipeline to lower IREE HAL executables with workgroup tiled and
/// distributed Linalg ops to SPIR-V cooperative matrix code. Additionally
/// performs distribution to threads with vectorization.
void addSPIRVTileAndVectorizeToCooperativeOpsPassPipeline(OpPassManager &pm);

/// Pass to perform the final conversion to SPIR-V dialect.
///
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToSPIRVPass();

/// Creates a pass to fold processor ID uses where possible.
std::unique_ptr<OperationPass<FuncOp>> createSPIRVFoldProcessorIDUsesPass();

/// Main pass to lower executables to scalar + vector code on SPIR-V path.
/// Invokes one of the pass pipelines that translate the executable to
/// scalar + vector code.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVLowerExecutableTargetPass();

/// Pass to tile and distribute Linalg ops with buffer semantics to invocations.
std::unique_ptr<OperationPass<FuncOp>> createSPIRVTileAndDistributePass();

/// Pass to tile Linalg ops with buffer semantics to subgroups and vectorize to
/// vector ops suitable for lowering to SPIR-V cooperative ops.
std::unique_ptr<OperationPass<FuncOp>>
createSPIRVTileAndVectorizeToCooperativeOpsPass();

/// Pass to convert vector read/write/arithmetic operations to the corresponding
/// cooperative matrix ops when possible.
std::unique_ptr<OperationPass<FuncOp>> createSPIRVVectorToCooperativeOpsPass();

/// Pass to lower linalg.copy for copying data to workgroup memory.
std::unique_ptr<OperationPass<FuncOp>> createSPIRVCopyToWorkgroupMemoryPass();

/// Pass to vectorize Linalg ops with buffer semantics.
std::unique_ptr<OperationPass<FuncOp>> createSPIRVVectorizePass();

/// Converts memref of scalar to memref of vector of efficent size. This will
/// allow to convert memory accesses to vector load/store in SPIR-V without
/// having pointer bitcast.
std::unique_ptr<OperationPass<ModuleOp>> createSPIRVVectorizeLoadStore();

//----------------------------------------------------------------------------//
// SPIRV Codegen Pass Pipelines.
//----------------------------------------------------------------------------//

/// Populates passes needed to lower a XLA HLO op to SPIR-V dialect via the
/// structured ops path. The pass manager `pm` in here operate on the module
/// within the IREE::HAL::ExecutableOp. The `workGroupSize` can be used to
/// control the work group size used in the code generation and is intended for
/// testing purposes only. The pass pipeline will set an appropriate workgroup
/// size.
/// TODO: Are both of these needed and does this one still work on HLO?
void buildSPIRVCodegenPassPipeline(OpPassManager &pm);

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

std::unique_ptr<OperationPass<ModuleOp>> createTestLLVMGPULegalizePass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_PASSES_H_
