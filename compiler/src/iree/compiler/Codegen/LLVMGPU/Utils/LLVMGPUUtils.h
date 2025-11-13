// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_LLVMGPUUTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_LLVMGPUUTILS_H_

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {

namespace linalg {
class LinalgOp;
} // namespace linalg

namespace vector {
class ContractionOp;
} // namespace vector

namespace iree_compiler {
class VectorContractOpInfo;

class ContractionVectorLayoutOptions : public VectorLayoutOptions {
public:
  ContractionVectorLayoutOptions(Operation *root, Value laneId,
                                 int64_t subgroupSize,
                                 ArrayRef<int64_t> workgroupSize);
  RewritePatternSet &getPatterns();
  VectorLayoutInterface getDefaultLayout(VectorType type) const override;

private:
  RewritePatternSet patterns;
};

/// Helper to convert copy to shared memory to async copy. This creates groups
/// of consecutive copies and emit wait operation right after.
void createAsyncGroups(RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
                       bool useMMASync);

/// Function to reorder transposes and elementwise ops.
void reorderTranspose(RewriterBase &rewriter, mlir::FunctionOpInterface funcOp);

/// Look for allocs in shared memory space with overlapping liveness,
/// group them, and then pack all the allocations in each group into one i8
/// alloc.
///
/// Also adds barriers to make sure we are done writing/reading
/// from the previous alias group before starting a new one.
void packSharedMemoryAlloc(mlir::FunctionOpInterface funcOp);

// Prefetches data written to shared memory for the next iteration. Returns the
// new loop on success or failure when the `forOp` is not supported.
//
// numStages: Controls the depth of prefetching/pipelining.
//   - 1: No pipelining (operations remain in original order)
//   - 2: Two-stage pipeline (default) - prefetches 1 iteration ahead
//   - 3+: Multi-stage pipeline - prefetches (numStages-1) iterations ahead
// Stream pipeliner always prefetches one less than the number of stages.
FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp,
                                               unsigned numStages = 2);

/// Insert barriers and wait operations if there are allocs of a different alias
/// group before the given alloc.
void addBarrier(mlir::FunctionOpInterface funcOp, Operation *alloc,
                ArrayRef<Operation *> aliasGroup, bool hasAsyncCopies = true);

} // namespace iree_compiler
} // namespace mlir

#endif
