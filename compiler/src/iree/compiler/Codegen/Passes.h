// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_PASSES_H_
#define IREE_COMPILER_CODEGEN_PASSES_H_

#include <memory>

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

/// Registers all conversion passes in this directory.
void registerCodegenPasses();

/// Verify that the configuration used for compilation is valid.
LogicalResult verifyLoweringConfiguration(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});

//------------------------------------------------------------------------------
// Misc/common conversions
//------------------------------------------------------------------------------

/// Post-bufferization passes run to cleanup the IR
/// (ResolveShapedTypeResultDims, Canonicalization/CSE and
/// CleanupBufferAllocView).
void addIREEPostBufferizationPasses(OpPassManager &passManager);

using bufferization::BufferizationOptions;
void addIREEComprehensiveBufferizePasses(
    OpPassManager &passManager,
    Optional<BufferizationOptions::AllocationFn> allocationFn = std::nullopt,
    Optional<BufferizationOptions::DeallocationFn> deallocationFn =
        std::nullopt,
    Optional<BufferizationOptions::MemCpyFn> memCpyFn = std::nullopt);

/// Pass to perform canonicalizations/cleanups related to HAL interface/buffer
/// allocations and view operations.
std::unique_ptr<OperationPass<func::FuncOp>> createCleanupBufferAllocViewPass();

/// Pass to bufferize dispatches that are copying from one interface to
/// another. This will create a `linalg.generic` op which is a copy that can
/// then be used by backends to handle appropriately.
std::unique_ptr<OperationPass<ModuleOp>>
createBufferizeCopyOnlyDispatchesPass();

// Decomposes linalg generics on tensors into generics containing no more than
// one op in the body.
std::unique_ptr<Pass> createDecomposeLinalgGenericPass();

// Fixes resturn types of `hal.interface.binding.subspan` ops with non-zero
// offsets.
std::unique_ptr<OperationPass<func::FuncOp>>
createFixupSubspanWithOffsetsPass();

/// Flattens n-D MemRef subspan ops to 1-D MemRef and folds the byte offsets
/// on subspan ops to the consumer load/store ops, in preparation for lowering
/// to backends that require linearized access.
std::unique_ptr<OperationPass<ModuleOp>> createFlattenMemRefSubspanPass();

/// Creates a pass to fold `affine.min` ops in tiled and distributed loops.
std::unique_ptr<OperationPass<func::FuncOp>>
createFoldAffineMinInDistributedLoopsPass();

/// After running the upstream TensorConstantBufferize pass, remove
/// tensor_loads introduced for use only in tensor_extract. These can be
/// folded to use a load of the created memref object that holds the constant
/// values.
std::unique_ptr<OperationPass<>> createFoldTensorExtractOpPass();

/// An ad-hoc pass to canonicalize selected loop carried dependencies on
/// scf.for.
std::unique_ptr<OperationPass<func::FuncOp>> createForOpCanonicalizationPass();

/// A pass to eliminate tensor.empty ops that could turn into allocations
/// during bufferization.
std::unique_ptr<OperationPass<ModuleOp>> createEliminateEmptyTensorsPass();

/// Pass to perform linalg on tensor bufferization. The function passed into
/// the pass through the `allocationFn` argument is invoked whenever a new
/// buffer is to be created. The callback will be passed the Values for the
/// dynamic dimensions in the memref type that is to be allocated.  The
/// callback is expected to return a MemRefType Value.  When no `allocationFn`
/// is specified, the default allocator generates an `std.alloc` instruction
/// with the allocated MemRefType having no stride map (i.e. default row-major
/// striding) and default memory space.
std::unique_ptr<OperationPass<ModuleOp>> createIREEComprehensiveBufferizePass(
    Optional<BufferizationOptions::AllocationFn> allocationFn = std::nullopt,
    Optional<BufferizationOptions::DeallocationFn> deallocationFn =
        std::nullopt,
    Optional<BufferizationOptions::MemCpyFn> memCpyFn = std::nullopt);

std::unique_ptr<OperationPass<func::FuncOp>>
createHoistStaticallyBoundAllocationsPass();

/// Creates a pass to remove single iteration distributed loops.
std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveSingleIterationLoopPass();

/// Converts entry point function within dispatch regions to use
/// destination-passing style, which is better suited for the upstream
/// comprehensive bufferization pass.
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertToDestinationPassingStylePass(
    bool useWARForCooperativeMatrixCodegen = false);

/// Creates a pass to vectorize a very specific form of tensor.pad ops with
/// control flows.
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizePadPass();

/// Creates a pass to vectorize tensor.pack and tensor.unpack ops. The pass does
/// tiling, generalization, and kicking in the generic vectorizer. See
/// implementation for more details.
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizePackUnPackOpsPass();

/// Pass to optimize vector transfer_read and transfer_write.
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeVectorTransferPass(
    bool flatten = false);

/// Pass to lower ukernel operations into their defined function calls.
std::unique_ptr<OperationPass<ModuleOp>> createLowerUKernelOpsToCallsPass();

/// Pass to optimize vector transfer_read and transfer_write. See Passes.td for
/// `option` details.
std::unique_ptr<OperationPass<func::FuncOp>>
createSplitFullPartialTransferPass();
std::unique_ptr<OperationPass<func::FuncOp>> createSplitFullPartialTransferPass(
    StringRef option);

/// Tests iree-hal-preprocess-executables-with behavior.
std::unique_ptr<OperationPass<void>> createTestExecutablePreprocessingPass();

/// Pass to test Partitionable loop interface
std::unique_ptr<OperationPass<void>>
createTestPartitionableLoopsInterfacePass();

/// Pass to tile and distribute to workgroups.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createTileAndDistributeToWorkgroupsPass(
    int32_t maxWorkgroupParallelDims = kNumMaxParallelDims);

/// Pass to specialize workgroup distribution loops
std::unique_ptr<OperationPass<func::FuncOp>>
createWorkgroupSpecializationPass();

/// Pass to propagate type to avoid generating load/stores of illegal types.
std::unique_ptr<OperationPass<func::FuncOp>> createTypePropagationPass();

/// Pass to convert math operations to their polynomial approximation.
std::unique_ptr<OperationPass<>> createPolynomialApproximationPass();

/// Creates a pass to convert memref.copy to linalg op.
std::unique_ptr<OperationPass<func::FuncOp>> createMemrefCopyToLinalgPass();

/// Convert GPU shared memory copies to distributed
/// transfer_read/transfer_write.
std::unique_ptr<OperationPass<func::FuncOp>>
createGPUDistributeSharedMemoryCopy();

/// Apply multi-buffering transformation.
std::unique_ptr<OperationPass<func::FuncOp>> createGPUMultiBuffering(
    unsigned numBuffers = 5);

/// Pipeline shared memory copy by apply software pipelining scheduling where
/// copy to shared memory is in stage 0 and the rest of the operations are in
/// stage `depth - 1`.
enum class PipeliningSchedulingStrategy {
  // Schedule the load from global memory into stage 0 and the associated store
  // will be in stage depth - 1.
  loadGlobalStage0 = 0,
  // Schedule both the load from global and the store to shared memory in stage
  // 0. The compute operations will be in stage depth-1. This means there won't
  // be vector registers carried between stages.
  loadStoreStage0 = 1,
  // Schedule optimized when using nvidia tensorcore with async copies. It will
  // set all the copies in stage 0 then it will prefecth part of loads in `depth
  // - 2` stage and keep the rest of the load and compute into `depth - 1`.
  nvidiaTensorCore = 2,
};

/// Apply software pipelining.
std::unique_ptr<OperationPass<func::FuncOp>> createGPUPipeliningPass(
    bool epiloguePeeling = true, unsigned depth = 1,
    PipeliningSchedulingStrategy schedule =
        PipeliningSchedulingStrategy::loadGlobalStage0);

/// Converts vector ops to gpu dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createWorkGroupSwizzle(
    unsigned swizzleLogTile = 0);

/// Pad dynamic alloc op to convert them into static one.
std::unique_ptr<OperationPass<func::FuncOp>> createPadDynamicAlloc();

/// Create an IREE-specific Transform dialect interpreter pass with all
/// registrations necessary for IREE.
std::unique_ptr<Pass> createTransformDialectInterpreterPass(
    llvm::StringRef transformFileName = llvm::StringRef(),
    llvm::StringRef debugPayloadRootTag = llvm::StringRef(),
    llvm::StringRef debugTransformRootTag = llvm::StringRef());

/// Convert Linalg ops to Vector.
std::unique_ptr<OperationPass<func::FuncOp>> createGPUVectorizationPass(
    bool generateContract = true, int64_t maxVectorSize = 4096);

/// Tile reductions and generate serial loops around reductions.
std::unique_ptr<OperationPass<func::FuncOp>> createGPUTileReductionPass();

// Distributes vector ops to all threads/warps in a GPU workgroup.
// `getWarpSize` is for deciding the warp size to use; it takes the
// current function containing those vector ops as the argument.
// If nullptr, warp size 32 will be used.
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertVectorReductionToGPUPass(
    std::function<int(func::FuncOp)> getWarpSize = nullptr);

/// Fuses tensor.pad ops into their consumer ops' tiled loop nests.
std::unique_ptr<OperationPass<func::FuncOp>>
createFuseTensorPadWithConsumerPass();

/// Concretizes tensor.pad op's result shape if its source op implements
/// OffsetSizeAndStrideOpInterface. For example, pad(extract_slice).
std::unique_ptr<OperationPass<func::FuncOp>>
createConcretizePadResultShapePass();

/// Erases #hal.descriptor_type as MemRef memory space.
LogicalResult eraseHALDescriptorTypeFromMemRef(func::FuncOp funcOp);
std::unique_ptr<OperationPass<func::FuncOp>>
createEraseHALDescriptorTypeFromMemRefPass();

/// Pass to merge parallel linalg operations.
std::unique_ptr<OperationPass<func::FuncOp>>
createRematerializeParallelOpsPass();

/// Instruments memory reads and writes for address tracking.
std::unique_ptr<OperationPass<func::FuncOp>>
createInstrumentMemoryAccessesPass();

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
void populateLinalgToVectorVectorizeConvPatterns(MLIRContext *context,
                                                 RewritePatternSet &patterns);

/// Populates `patterns` with patterns that vectorize tensor.pad with static
/// result shape by generating control flows to guard against vector transfer
/// read ops to make sure they are in bounds.
///
/// Such conversions are needed for correctness when the tensor.pad op has
/// dynamic low padding values and also beneficial for eventually lowering to
/// hardware targets without native support for vector transfer read ops with
/// out of bound semantics.
void populateVectorizePadPatterns(RewritePatternSet &patterns,
                                  PatternBenefit baseBenefit = 1);

/// Populates patterns with patterns to concretize tensor.pad op'ss result
/// shape.
void populateConcretizePadResultShapePatterns(MLIRContext *context,
                                              RewritePatternSet &patterns);

//------------------------------------------------------------------------------
// LLVMCPU
//------------------------------------------------------------------------------

// Verifies that only supported IR constructs are passed to the compiler (like
// no Linalg transform markers are set).
std::unique_ptr<OperationPass<ModuleOp>>
createVerifyLinalgTransformLegalityPass();

/// Pass to tile and fuse TilingInterface ops with given tilingLevel.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUTileAndFusePass(
    int64_t tilingLevel = -1);

/// Pass to pad operations on tensors in top-down order.
enum class LLVMCPUTensorPadOption { ParallelDims, ReductionDims };
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUTensorPadPass(
    LLVMCPUTensorPadOption option = LLVMCPUTensorPadOption::ParallelDims);

/// Performs the final conversion to LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass(
    bool reassociateFpReordering = false);

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUEmitVectorizationRemarksPass();

/// Checks CPU backend specific IR constraints (like no stack allocations)
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUCheckIRBeforeLLVMConversionPass();

/// Pass to lower the module an hal.executable.variant operation to external
/// dialect. Currently this pass lowers to LLVM dialect, but could be
/// generalized to lower to any "final" dialect like SPIR-V/NVVM, etc.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPULowerExecutableTargetPass();

/// Pass to lower a sequence of operations to a iree_codegen.ukernel.*
/// operation.
std::unique_ptr<OperationPass<>> createLLVMCPULowerToUKernelsPass();

/// Materialize the encoding of operations. The layout to use for the encoded
/// operations are LLVMCPU specific.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUMaterializeEncodingPass();

/// Synchronizes LLVM linkage with MLIR symbol visibility.
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUSynchronizeSymbolVisibilityPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUMmt4dVectorLoweringPass();

/// Replaces llvm.intr.fma with its unfused mul and add ops.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUUnfuseFMAOpsPass();

/// A pass that converts certain vector.contract ops to custom kernels.
std::unique_ptr<OperationPass<func::FuncOp>>
createVectorContractCustomKernelsPass();

//------------------------------------------------------------------------------
// LLVMCPU Codegen specific patterns.
//------------------------------------------------------------------------------

/// Populates `patterns` to convert certain vector.contract ops to special
/// "kernels" written either in SIMD intrinsics or inline assembly.
void populateVectorContractCustomKernelsPatterns(
    IREE::HAL::ExecutableTargetAttr target, RewritePatternSet &patterns);

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       RewritePatternSet &patterns);

//----------------------------------------------------------------------------//
// LLVMCPU backend Pass Pipelines.
//----------------------------------------------------------------------------//

/// Populates the passes to lower to scalars operations for linalg based
/// code-generation. This pipeline does not vectorize, but instead just
/// converts to memrefs
void addCPUDefaultPassPipeline(OpPassManager &passManager);

/// Populates the passes to lower ops through data tiling transformations.
void addCPUDataTilingPipeline(OpPassManager &passManager);

/// Populates the passes to lower to tiled/distributed/bufferized ops,
/// suitable for library call dispatch and lowering to loops.
void addVMVXDefaultPassPipeline(OpPassManager &passManager,
                                bool enableMicrokernels);

/// Populates the passes to lower linalg ops on buffers. Currenly this
/// pipeline is only used for dispatches that just copy data from input
/// interfaces to output interface.
void addCPUBufferOpsTileAndVectorizePipeline(OpPassManager &passManager,
                                             bool enableVectorMasking);

/// Populates the passes needed to multi level tile and lowering of linalg ops
/// on tensors to vectors operations.
LogicalResult verifyTensorToVectorsPassPipelineConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});
void addTensorToVectorsPassPipeline(OpPassManager &passManager,
                                    bool lowerToVectors = true);

/// Populates the passes needed to do two-level tile + vectorize of linalg ops
/// using the Codegen drivers from sandbox.
LogicalResult verifyDoubleTilingExpertPassPipelineConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});
void addMultiTilingExpertPassPipeline(OpPassManager &passManager,
                                      int64_t numLevels, bool enablePeeling,
                                      bool enableVectorMasking,
                                      bool lowerToAVX2);
void addDoubleTilingPadExpertPassPipeline(OpPassManager &passManager,
                                          bool enableVectorMasking);

// Populates the passes needed to do tiling, decomposing, and vectorizing the
// convolution ops using the Codegen drivers from sandbox.
LogicalResult verifyConvTileAndDecomposeExpertConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});
void addConvTileAndDecomposeExpertPassPipeline(OpPassManager &passManager,
                                               bool enableVectorMasking);

/// Transform dialect-based common.
void addTransformDialectPasses(OpPassManager &passManager);

/// Populates the passes needed to multi level tile, fuse and vectorize
/// lowering of linalg ops on tensors to vectors operations.
void addMmt4dTilingExpertPassPipeline(OpPassManager &passManager,
                                      bool enableVectorMasking);

//----------------------------------------------------------------------------//
// LLVMCPU Pass Pipelines for lowering to LLVM dialect.
//----------------------------------------------------------------------------//

/// Populates passes needed to lower a XLA HLO op to LLVM dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMCPUCodegenPassPipeline(OpPassManager &passManager);

//----------------------------------------------------------------------------//
// LLVMCPU Linking Passes and Pipelines
//----------------------------------------------------------------------------//

/// Links LLVMCPU HAL executables within the top-level program module.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createLLVMCPULinkExecutablesPass();

/// Assigns executable constant ordinals across all LLVMCPU variants.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUAssignConstantOrdinalsPass();

/// Assigns executable import ordinals across all LLVMCPU variants.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUAssignImportOrdinalsPass();

/// Populates passes needed to link HAL executables across LLVMCPU targets.
void buildLLVMCPULinkingPassPipeline(OpPassManager &passManager);

//------------------------------------------------------------------------------
// LLVMGPU
//------------------------------------------------------------------------------

/// Lowering calling vectorization patterns. Expects pass manager to be a
/// module-level pass manager.
void addGPUVectorizationPassPipeline(OpPassManager &pm);

/// Lowering calling vectorization patterns.
LogicalResult verifyGPUMatmulSimtPassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);
void addGPUMatmulSimtPassPipeline(OpPassManager &pm);

LogicalResult verifyGPUMatmulTensorCorePipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);
/// Lowering using wmma tensorcore operations.
void addGPUMatmulTensorCorePassPipeline(OpPassManager &pm,
                                        unsigned pipelineDepth);

/// Lowering using mma.sync tensorcore operations.
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

/// Apply transformation to reduce the number of bank conflicts when accessing
/// shared memory by padding fastest moving dimension with the specified size.
std::unique_ptr<OperationPass<func::FuncOp>>
createGPUReduceSharedMemoryBankConflicts(int64_t paddingSizeBits = 128);

/// Converts vector ops to gpu dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorToGPU(
    GPUTensorCoreType tensorCoreType = GPUTensorCoreType::WMMA);

//. Pass to pad out tensors up to static dimensions.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorPadPass();

// Pass to pack shared memory allocations in order to reduce shared memory
// usage.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUPackSharedMemoryAlloc();

//------------------------------------------------------------------------------
// SPIR-V Passes
//------------------------------------------------------------------------------

/// Pass pipeline to lower IREE HAL executables by tiling and distributing to
/// workgroups and invocations. Each invocation handles a scalar.
void addSPIRVBaseDistributePassPipeline(OpPassManager &pm);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing to
/// workgroups and invocations and vectorizing. Each invocation handles a
/// vector.
LogicalResult verifySPIRVBaseVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);
void addSPIRVBaseVectorizePassPipeline(OpPassManager &pm);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing
/// to workgroups and subgroups and then vectorizing to SPIR-V cooperative
/// matrix code.
LogicalResult verifySPIRVCooperativeMatrixVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);
void addSPIRVCooperativeMatrixVectorizePassPipeline(OpPassManager &pm,
                                                    unsigned pipelineDepth,
                                                    unsigned storeStage);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing to
/// workgroups, promoting to use workgroup memory, and then tiling and
/// distributing to invocations and vectorizing. Each invocation handles a
/// vector.
LogicalResult verifySPIRVMatmulPromoteVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);
void addSPIRVMatmulPromoteVectorizePassPipeline(OpPassManager &pm,
                                                unsigned pipelineDepth,
                                                unsigned storeStage);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing
/// reduction to workgroups and then subgroups.
void addSPIRVSubgroupReducePassPipeline(OpPassManager &pm);

/// Pass to perform the final conversion to SPIR-V dialect.
///
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToSPIRVPass(
    bool enableFastMath = false, unsigned indexWidth = 32);

/// Creates a pass to fold processor ID uses where possible.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVFoldProcessorIDUsesPass();

/// Main pass to lower executables to scalar + vector code on SPIR-V path.
/// Invokes one of the pass pipelines that translate the executable to
/// scalar + vector code.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVLowerExecutableTargetPass();

/// Pass to tile and distribute Linalg ops with buffer semantics to
/// invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTileAndDistributePass();

/// Pass to promote Linalg ops with buffer semantics to use workgroup memory
/// and then tile to invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTileAndPromotePass(
    bool promoteCMatrix = false, bool skipThreadLevel = false);

/// Pass to tile Linalg ops with buffer semantics suitable for lowering to
/// SPIR-V cooperative ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVTileToCooperativeOpsPass();

/// Pass to do vectorization suitable for lowering to SPIR-V cooperative ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVVectorizeToCooperativeOpsPass();

/// Converts vector ops to gpu subgroup MMA ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVVectorToGPUSubgroupMMAOpsPass();

/// Pass to tile Linalg ops with tensor semantics to invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTilePass();

/// Pass to distribute tiled loop nests to invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVDistributePass();

/// Pass to vectorize Linalg ops with buffer semantics.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVVectorizePass();

/// Converts memref of scalar to memref of vector of efficent size. This will
/// allow to convert memory accesses to vector load/store in SPIR-V without
/// having pointer bitcast.
std::unique_ptr<OperationPass<ModuleOp>> createSPIRVVectorizeLoadStore();

/// Breaks down large vectors not natively supported by SPIR-V.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVBreakDownLargeVectorPass();

// Uses `tensor.pad` ops as anchors to create separate fast and slow paths
// inside the kernel. The fast path is for inner tiles where we don't need
// padding, while the slow path is for boundary tiles where we do need
// padding.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVCreateFastSlowPathPass();

/// Emulates 64-bit integer ops with 32-bit integer ops.
std::unique_ptr<OperationPass<ModuleOp>> createSPIRVEmulateI64Pass();

/// Turns static shaped storage buffer subspan ops into dynamic shaped ones.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVEraseStorageBufferStaticShapePass();

/// Pass to map MemRef memory spaces to SPIR-V storage classes.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVMapMemRefStorageClassPass();

/// Pass pipeline to lower winograd ops. This pipeline follows the
/// SPIRVBaseVectorize pipeline with the following exception:
/// Since the ops are already tiled, we skip tiling and instead
/// just annotate the loops with the spirv distribute attribute.
///
void addSPIRVWinogradVectorizePassPipeline(OpPassManager &pm);

/// Annotates the innermost Winograd loops with the spirv distribute attribute.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVAnnotateWinogradLoopsPass();

//----------------------------------------------------------------------------//
// SPIRV Codegen Pass Pipelines.
//----------------------------------------------------------------------------//

/// Populates passes needed to lower linalg/arith/math ops to SPIR-V ops via
/// the structured ops path. The pass manager `pm` here operate on the module
/// within the IREE::HAL::ExecutableOp.
void buildSPIRVCodegenPassPipeline(OpPassManager &pm, bool enableFastMath);

//------------------------------------------------------------------------------
// VMVX passes
//------------------------------------------------------------------------------

/// Materialize the encoding of operations. The layout to use for the encoded
/// operations are VMVX specific.
std::unique_ptr<OperationPass<func::FuncOp>>
createVMVXMaterializeEncodingPass();

// Lowers high level library calls from named ops and generics. This operates
// at the bufferized linalg level.
std::unique_ptr<Pass> createVMVXLowerLinalgMicrokernelsPass();

//----------------------------------------------------------------------------//
// VMVX Linking Passes and Pipelines
//----------------------------------------------------------------------------//

/// Links VMVX HAL executables within the top-level program module.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createVMVXLinkExecutablesPass();

/// Assigns executable constant ordinals across all VMVX variants.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createVMVXAssignConstantOrdinalsPass();

/// Populates passes needed to link HAL executables across VMVX targets.
void buildVMVXLinkingPassPipeline(OpPassManager &passManager);

//------------------------------------------------------------------------------
// WGSL passes
//------------------------------------------------------------------------------

// Removes push constants by replacing hal.interface.constant.loads with
// hal.interface.binding.subspan + flow.dispatch.tensor.load.
std::unique_ptr<OperationPass<func::FuncOp>>
createWGSLReplacePushConstantsPass();

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

std::unique_ptr<OperationPass<ModuleOp>> createTestLLVMGPULegalizePass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_PASSES_H_
