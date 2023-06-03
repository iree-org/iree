// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the Common Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_COMMON_PASSES_H_
#define IREE_COMPILER_CODEGEN_COMMON_PASSES_H_

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
/// Passes that are done on all backends before target-specific code-generation
/// kicks in.
void addCommonTargetExecutablePreprocessingPasses(OpPassManager &passManager);

/// Post-bufferization passes run to cleanup the IR
/// (ResolveShapedTypeResultDims, Canonicalization/CSE and
/// CleanupBufferAllocView).
void addIREEPostBufferizationPasses(OpPassManager &passManager);

using bufferization::BufferizationOptions;
void addIREEComprehensiveBufferizePasses(
    OpPassManager &passManager,
    std::optional<BufferizationOptions::AllocationFn> allocationFn =
        std::nullopt,
    std::optional<BufferizationOptions::DeallocationFn> deallocationFn =
        std::nullopt,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn = std::nullopt);

/// Pass to bubble up ordinal operations to allow workgroup count computation
/// based on slices to correlate back to workload computation.
std::unique_ptr<Pass> createBubbleUpOrdinalOpsPass();

/// Pass to perform canonicalizations/cleanups related to HAL interface/buffer
/// allocations and view operations.
std::unique_ptr<OperationPass<func::FuncOp>> createCleanupBufferAllocViewPass();
/// Pass to bufferize dispatches that are copying from one interface to
/// another. This will create a `linalg.generic` op which is a copy that can
/// then be used by backends to handle appropriately.
std::unique_ptr<OperationPass<ModuleOp>>
createBufferizeCopyOnlyDispatchesPass();

/// Pass to perform canonicalizations/cleanups related to HAL interface/buffer
/// allocations and view operations.
std::unique_ptr<OperationPass<func::FuncOp>> createCleanupBufferAllocViewPass();

/// Concretizes tensor.pad op's result shape if its source op implements
/// OffsetSizeAndStrideOpInterface. For example, pad(extract_slice).
std::unique_ptr<OperationPass<func::FuncOp>>
createConcretizePadResultShapePass();

/// Convert BF16 buffer ops and conversions to simulated behavior with uint16.
std::unique_ptr<OperationPass<ModuleOp>> createConvertBf16ToUInt16BuffersPass();

/// Convert BF16 buffer ops and conversions to simulated behavior with uint16.
std::unique_ptr<OperationPass<ModuleOp>> createConvertBf16ArithToF32Pass();

/// Converts entry point function within dispatch regions to use
/// destination-passing style, which is better suited for the upstream
/// comprehensive bufferization pass.
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertToDestinationPassingStylePass(
    bool useWARForCooperativeMatrixCodegen = false);

// Decompose affine.apply operations into sub affine.apply that can be
// hoisted in different loops.
std::unique_ptr<Pass> createDecomposeAffineOpsPass();

// Decomposes high-D convolution ops into low-D ones.
std::unique_ptr<Pass> createDecomposeConvolutionToLowerDimOpsPass();

// Decomposes linalg generics on tensors into generics containing no more than
// one op in the body.
std::unique_ptr<Pass> createDecomposeLinalgGenericPass();

/// Creates a pass to decompose tensor.pack and tensor.unpack ops. The pass does
/// tiling and generalization. See implementation for more details.
std::unique_ptr<OperationPass<func::FuncOp>> createDecomposePackUnPackOpsPass(
    bool tileOuterToOne = false);

/// A pass to eliminate tensor.empty ops that could turn into allocations
/// during bufferization.
std::unique_ptr<OperationPass<ModuleOp>> createEliminateEmptyTensorsPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createEraseHALDescriptorTypeFromMemRefPass();

// Extract address computations into their own separate instructions.
std::unique_ptr<Pass> createExtractAddressComputationPass();

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

/// Fuses tensor.pad ops into their consumer ops' tiled loop nests.
std::unique_ptr<OperationPass<func::FuncOp>>
createFuseTensorPadWithConsumerPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createHoistStaticallyBoundAllocationsPass();

/// Pass to perform linalg on tensor bufferization. The function passed into
/// the pass through the `allocationFn` argument is invoked whenever a new
/// buffer is to be created. The callback will be passed the Values for the
/// dynamic dimensions in the memref type that is to be allocated.  The
/// callback is expected to return a MemRefType Value.  When no `allocationFn`
/// is specified, the default allocator generates an `std.alloc` instruction
/// with the allocated MemRefType having no stride map (i.e. default row-major
/// striding) and default memory space.
std::unique_ptr<OperationPass<ModuleOp>> createIREEComprehensiveBufferizePass(
    std::optional<BufferizationOptions::AllocationFn> allocationFn =
        std::nullopt,
    std::optional<BufferizationOptions::DeallocationFn> deallocationFn =
        std::nullopt,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn = std::nullopt);

/// Pass to resolve `memref.expand_strided_metadata` operations.
std::unique_ptr<Pass> createIREEExpandStridedMetadataPass();

/// Instruments memory reads and writes for address tracking.
std::unique_ptr<OperationPass<func::FuncOp>>
createInstrumentMemoryAccessesPass();

/// Pass to lower ukernel operations into their defined function calls.
std::unique_ptr<OperationPass<ModuleOp>> createLowerUKernelOpsToCallsPass();

/// Creates a pass to convert memref.copy to linalg op.
std::unique_ptr<OperationPass<func::FuncOp>> createMemrefCopyToLinalgPass();

/// Pass to optimize vector transfer_read and transfer_write.
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeVectorTransferPass(
    bool flatten = false);

/// Pad dynamic alloc op to convert them into static one.
std::unique_ptr<OperationPass<func::FuncOp>> createPadDynamicAlloc();

/// Pass to convert math operations to their polynomial approximation.
std::unique_ptr<OperationPass<>> createPolynomialApproximationPass();

/// Pass to merge parallel linalg operations.
std::unique_ptr<OperationPass<func::FuncOp>>
createRematerializeParallelOpsPass();

/// Creates a pass to remove single iteration distributed loops.
std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveSingleIterationLoopPass();

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
    int32_t maxWorkgroupParallelDims = kNumMaxParallelDims,
    linalg::DistributionMethod distributionMethod =
        linalg::DistributionMethod::Cyclic);

/// Create an IREE-specific Transform dialect interpreter pass with all
/// registrations necessary for IREE.
std::unique_ptr<Pass> createTransformDialectInterpreterPass(
    llvm::StringRef transformFileName = llvm::StringRef(),
    llvm::StringRef debugPayloadRootTag = llvm::StringRef(),
    llvm::StringRef debugTransformRootTag = llvm::StringRef());

/// Pass to propagate type to avoid generating load/stores of illegal types.
std::unique_ptr<OperationPass<func::FuncOp>> createTypePropagationPass();

/// Creates a pass to vectorize a very specific form of tensor.pad ops with
/// control flows.
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizePadPass();

/// Pass to specialize workgroup distribution loops
std::unique_ptr<OperationPass<func::FuncOp>>
createWorkgroupSpecializationPass();

/// Erases #hal.descriptor_type as MemRef memory space.
LogicalResult eraseHALDescriptorTypeFromMemRef(func::FuncOp funcOp);

/// Populates patterns with patterns to concretize tensor.pad op's result
/// shape. `numWorkgroups`, if not empty, will be used as bounds for simplifying
/// workgroup ID ops.
void populateConcretizePadResultShapePatterns(
    RewritePatternSet &patterns, ArrayRef<int64_t> numWorkgroups = {});

/// Populates `patterns` with patterns to fold `affine.min` ops in tiled and
/// distributed loops.
void populateFoldAffineMinInDistributedLoopsPatterns(
    RewritePatternSet &patterns, ArrayRef<int64_t> staticNumWorkgroup = {});

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

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_PASSES_H_
