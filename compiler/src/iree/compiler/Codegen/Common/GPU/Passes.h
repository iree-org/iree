
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_PASSES_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

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
LogicalResult tileReductionToSerialLoops(func::FuncOp funcOp,
                                         bool fuseInputProducer = false);

LogicalResult swizzleWorkgroupsInFunc(func::FuncOp funcOp,
                                      unsigned swizzleLogTile);

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
    std::function<unsigned(func::FuncOp)> getSharedMemoryLimit = nullptr,
    std::function<unsigned(func::FuncOp)> getIndexBitwidth = nullptr);

/// Creates a pass to distribute scf.forall ops to GPU processors.
std::unique_ptr<OperationPass<func::FuncOp>> createGPUDistribute();

/// Convert GPU shared memory copies to distributed
/// transfer_read/transfer_write.
std::unique_ptr<OperationPass<func::FuncOp>>
createGPUDistributeSharedMemoryCopy();

/// Apply multi-buffering transformation.
std::unique_ptr<OperationPass<func::FuncOp>>
createGPUMultiBuffering(unsigned numBuffers = 5);

/// Apply software pipelining.
std::unique_ptr<OperationPass<func::FuncOp>>
createGPUPipeliningPass(bool epiloguePeeling = true, unsigned depth = 1,
                        PipeliningSchedulingStrategy schedule =
                            PipeliningSchedulingStrategy::loadGlobalStage0);

/// Apply transformation to reduce the number of bank conflicts when accessing
/// shared memory by padding fastest moving dimension with the specified size.
std::unique_ptr<OperationPass<func::FuncOp>>
createGPUReduceSharedMemoryBankConflicts(int64_t paddingSizeBits = 128);

// Creates a pass to tile reduction dimensions and create allocations for some
// tensor values to use GPU shared memory.
std::unique_ptr<OperationPass<func::FuncOp>>
createGPUTensorAlloc(GPUPromoteSharedMemPattern promoteSharedMemPattern =
                         GPUPromoteSharedMemPattern::ContractionOpPattern);

// Creates a pass to tile tensor (linalg) ops within a GPU workgroup.
std::unique_ptr<OperationPass<func::FuncOp>>
createGPUTensorTile(bool distributeToWarp = false);

/// Tile reductions and generate serial loops around reductions.
std::unique_ptr<OperationPass<func::FuncOp>> createGPUTileReductionPass();

// Distributes vector ops to all threads/warps in a GPU workgroup.
// `getWarpSize` is for deciding the warp size to use; it takes the
// current function containing those vector ops as the argument.
// If nullptr, warp size 32 will be used.
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertVectorReductionToGPUPass(
    bool expandSubgroupReduction = true,
    std::function<int(func::FuncOp)> getWarpSize = nullptr);

/// Pass to specialize workgroup distribution loops
std::unique_ptr<OperationPass<func::FuncOp>>
createWorkgroupSpecializationPass();

/// Converts vector ops to gpu dialect.
std::unique_ptr<OperationPass<func::FuncOp>>
createWorkGroupSwizzle(unsigned swizzleLogTile = 0);

// This pass generalizes named Linalg ops that are better off as generics.
std::unique_ptr<OperationPass<func::FuncOp>> createGPUGeneralizeNamedOpsPass();

/// Register Common GPU passes.
void registerCodegenCommonGPUPasses();

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_PASSES_H_
