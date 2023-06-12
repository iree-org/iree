// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_STAGED_REDUCTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_STAGED_REDUCTION_STRATEGY_H_

#include "iree/compiler/Codegen/TransformDialectStrategies/Common/AbstractReductionStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;
struct ReductionConfig;

/// Encode a 3-staged strategy for a 1-d reduction mapped to a block.
///
/// This happens in a staged fashion to encode good tradeoffs between amount
/// of parallelism, occupancy and granularity of the load/store operations.
/// The tradeoff is controlled at a distance by specifying a
/// `maxNumThreadsToUse` upper bound.
///
/// Bottom-up perspective:
/// ======================
/// Stage 3: second stage of the the warp shuffle step reduces a vector<k x T>
/// element to a single element. Only threadIdx == 0 commits to memory.
///
/// Stage 2: the second stage of the reduction is the first stage of the warp
/// shuffle step. It is normalized to reduce from a "k-warps" abstraction,
/// across all warps in parallel, to a k-element result. Only the first thread
/// within each warp (e.g. threadIdx % kCudaWarpSize == 0) commits to memory.
///
/// Stage 1: the first stage of the reduction is normalized to run on "k-warps"
/// of maximal vector size for both the hardware and the problem sizes.
/// The over-provisioning to "k-warps" allows multiple warps to run in parallel.
/// The `numThreadsXInBlock` is this "k-warps" quantity and is also the
/// number of threads (i.e. blockDim.x) used to parallelize the problem.
/// This also results in `numThreadsXInBlock` live values that are
/// allocated in shared memory and creates a tradeoff between parallelism and
/// occupancy.
/// The normalization guarantees that whatever the problem size P, we reduce
/// from `tensor<P x T>` to `tensor<numThreadsXInBlock x T>` by using the
/// largest possible `vector.transfer` operations. The vector size is chosen as
/// follows: when the `reductionDimensionSize` is a multiple of 4, choose 4;
/// otherwise try with 2; otherwise just use 1.
//
// TODO: Split to ensure 4 on most of the problem and use a 1-epilogue. This is
// best done if we can encode the future stride to ensure the 4 is aligned.
class StagedReductionStrategy : public AbstractReductionStrategy {
 public:
  StagedReductionStrategy(
      const transform_ext::MatchedReductionCaptures &captures,
      const ReductionConfig &reductionConfig);

  StagedReductionStrategy(const StagedReductionStrategy &) = default;
  StagedReductionStrategy &operator=(const StagedReductionStrategy &) = default;

  std::array<int64_t, 3> getNumThreadsInBlock() const {
    return {numThreadsXInBlock, 1, 1};
  }

  int64_t getVectorSize() const { return vectorSize; }

 private:
  /// Compute the staged strategy based on the reductionDimensionSize, the
  /// `maxNumThreadsToUse` and the `vectorSize`.
  /// The latter 2 numbers control the tradeoff between parallelism and shared
  /// memory consumption.
  // TODO: Characterize shared memory consumption and limit for good occupancy.
  void configure(const ReductionConfig &reductionConfig);

  /// Maximal vector size (among {1, 2, 4}) that divides the
  /// `reductionDimensionSize` and is used for vector transfers in Stage 1.
  int64_t vectorSize;

  /// Maximal "k-warp" size within the limits of the `maxNumThreadsToUse` and
  /// `reductionDimensionSize` parameters.
  /// This is also the blockDim.x of the kernel.
  int64_t numThreadsXInBlock;
};

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_STAGED_REDUCTION_STRATEGY_H_
