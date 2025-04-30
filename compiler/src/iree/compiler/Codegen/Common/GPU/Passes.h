
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_PASSES_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_PASSES_H_

#include <cstdint>
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
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
enum class PipeliningSchedulingStrategy : int64_t {
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
                         PipeliningSchedulingStrategy strategy,
                         bool peelEpilogue, int64_t depth);

/// Tiles Linalg ops in the given `funcOp` along reduction dimensions to serial
/// loops without distribution. If `fuseInputProducer` is true, input producers
/// will be fused into the serial loop.
LogicalResult tileReductionToSerialLoops(mlir::FunctionOpInterface funcOp,
                                         bool fuseInputProducer = false,
                                         bool coalesceLoops = false);

/// Adds padding to `memref.alloc` ops to reduce shared memory bank conflicts.
/// The `paddingSizeBits` argument should be picked based on the target
/// architecture, striking balance between minimizing bank conflicts and keeping
/// the data aligned. Smaller values (close to the bank bitwidth) achieve the
/// former, while larger (~= widest load size) the latter. We want to
/// **misalign** the rows, but not too much.
LogicalResult reduceSharedMemoryBankConflicts(mlir::FunctionOpInterface funcOp,
                                              unsigned paddingSizeBits);

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
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createGPUCheckResourceUsagePass(
    std::function<unsigned(mlir::FunctionOpInterface)> getIndexBitwidth =
        nullptr);

// Creates a pass to create allocations for some tensor values to use GPU
// shared memory.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUTensorAlloc(GPUPromoteSharedMemPattern promoteSharedMemPattern =
                         GPUPromoteSharedMemPattern::ContractionOpPattern);

// Distributes vector ops to all threads/warps in a GPU workgroup.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertVectorReductionToGPUPass(bool expandSubgroupReduction = true);

using IREE::GPU::ReorderWorkgroupsStrategy;

/// Reorders workgroup IDs.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createReorderWorkgroups(
    ReorderWorkgroupsStrategy strategy = ReorderWorkgroupsStrategy::None,
    std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn = nullptr);

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc" // IWYU pragma: keep

/// Register Common GPU passes.
void registerCodegenCommonGPUPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_PASSES_H_
