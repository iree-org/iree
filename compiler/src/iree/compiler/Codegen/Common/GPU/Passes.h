
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_PASSES_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_PASSES_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Definitions and Utilities
//===----------------------------------------------------------------------===//

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

enum class GPUPromoteSharedMemPattern {
  ContractionOpPattern = 0,
  TransposeOpPattern = 1,
};

FailureOr<scf::ForOp>
pipelineSharedMemoryCopy(RewriterBase &rewriter, scf::ForOp forOp,
                         PipeliningSchedulingStrategy startegy,
                         bool peelEpilogue, int64_t depth);

/// Tiles Linalg ops in the given `funcOp` along reduction dimensions to serial
/// loops without distribution. If `fuseInputProducer` is true, input producers
/// will be fused into the serial loop.
LogicalResult tileReductionToSerialLoops(mlir::FunctionOpInterface funcOp,
                                         bool fuseInputProducer = false);

LogicalResult swizzleWorkgroupsInFunc(mlir::FunctionOpInterface funcOp,
                                      unsigned swizzleLogTile);

// Lowers workgroup memory copies to distributed transfer_read/transfer_write
// ops. Expects the memory copy to be marked with copy_to_workgroup_memory
// marker.
LogicalResult gpuDistributeSharedMemoryCopy(mlir::FunctionOpInterface funcOp);

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Checks GPU specific resource usage constraints like shared memory limits.
// `getSharedMemoryLimit` is for querying the shared memory limit (in bytes);
// it takes the current entry function as the argument. 64KB will be used if
// nullptr.
// `getIndexBitwidth` is for querying the bitwidth for the `index` type.
// This size is used to check the allocation space required for memrefs of
// indices. If this function is nullptr, this pass will query the datalayout to
// get the index size.
std::unique_ptr<OperationPass<ModuleOp>> createGPUCheckResourceUsagePass(
    std::function<unsigned(mlir::FunctionOpInterface)> getSharedMemoryLimit =
        nullptr,
    std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth =
        nullptr);

/// Creates a pass to distribute scf.forall ops to GPU processors.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>> createGPUDistribute();

/// Convert GPU shared memory copies to distributed
/// transfer_read/transfer_write.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUDistributeSharedMemoryCopy();

/// Apply multi-buffering transformation.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUMultiBuffering(unsigned numBuffers = 5);

/// Apply software pipelining.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUPipeliningPass(bool epiloguePeeling = true, unsigned depth = 1,
                        PipeliningSchedulingStrategy schedule =
                            PipeliningSchedulingStrategy::loadGlobalStage0);

/// Apply transformation to reduce the number of bank conflicts when accessing
/// shared memory by padding fastest moving dimension with the specified size.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUReduceSharedMemoryBankConflicts(int64_t paddingSizeBits = 128);

// Creates a pass to create allocations for some tensor values to use GPU
// shared memory.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUTensorAlloc(GPUPromoteSharedMemPattern promoteSharedMemPattern =
                         GPUPromoteSharedMemPattern::ContractionOpPattern);

// Creates a pass to tile tensor (linalg) ops within a GPU workgroup.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUTensorTile(bool distributeToWarp = false);

// Creates a pass to tile tensor (linalg) ops along reduction dimensions.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUTensorTileToSerialLoops();

/// Tile reductions and generate serial loops around reductions.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUTileReductionPass();

// Creates a pass to create allocations for some vector values to use GPU
// shared memory.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUVectorAlloc();

// Distributes vector ops to all threads/warps in a GPU workgroup.
// `getWarpSize` is for deciding the warp size to use; it takes the
// current function containing those vector ops as the argument.
// If nullptr, warp size 32 will be used.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertVectorReductionToGPUPass(
    bool expandSubgroupReduction = true,
    std::function<int(mlir::FunctionOpInterface)> getWarpSize = nullptr);

/// Pass to specialize workgroup distribution loops
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createWorkgroupSpecializationPass();

/// Converts vector ops to gpu dialect.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createWorkGroupSwizzle(unsigned swizzleLogTile = 0);

// This pass generalizes named Linalg convolution and contraction ops to allow
// for better folding of unit dimensions.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUGeneralizeNamedConvolutionAndContractionOpsPass();

// This pass generalizes named Linalg ops that are better off as generics.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUGeneralizeNamedOpsPass();

/// Pass to lower a sequence of operations to a iree_codegen.ukernel.*
/// operation.
std::unique_ptr<OperationPass<>> createGPULowerToUKernelsPass();

/// Register Common GPU passes.
void registerCodegenCommonGPUPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_PASSES_H_
