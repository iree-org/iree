// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_SMALL_REDUCTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_SMALL_REDUCTION_STRATEGY_H_

#include <array>

#include "iree/compiler/Codegen/TransformStrategies/Common/AbstractReductionStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"

namespace mlir::iree_compiler::gpu {

/// Encode a strategy targeted at (very) small reductions, for which other
/// strategies perform poorly.
///
/// In the case of small reductions, we cannot make an efficient use of warp
/// shuffles. Instead, take advantage of caches.
/// This strategy aims at running the reduction sequentially within each
/// thread and taking parallelism from outer dimensions that we would
/// otherwise use for block-level parallelism.
///
/// There are 2 cases:
///   1. we can find good divisors of outer parallel dimensions and avoid
///      creating dynamic tile sizes. We can then vectorize to the reduction
///      size.
///   2. we cannot find good divisors, we pay the price of dynamic loops.
///
// TODO: Refine 1. with linalg splitting on the reduction dimension.
// TODO: Refine 2. with linalg splitting on the parallel dimension.
//
// Note: All this is to be able to handle very small and small-ish
// reductions without catastrophic regressions.
// TODO: Add another strategy based on segmented scans, which can allow us
// to force sizes that don't divide properly into warp shuffles.
class SmallReductionStrategy : public AbstractReductionStrategy, GPUStrategy {
public:
  SmallReductionStrategy(
      const transform_ext::MatchedReductionCaptures &captures,
      const ReductionConfig &reductionConfig, const GPUModel &gpuModel);

  SmallReductionStrategy(const SmallReductionStrategy &) = default;
  SmallReductionStrategy &operator=(const SmallReductionStrategy &) = default;

  std::array<int64_t, 3> getNumThreadsInBlock() const {
    std::array<int64_t, 3> res{1, 1, 1};
    for (int64_t i = 0, e = workgroupTileSizes.size(); i < e; ++i)
      res[i] = workgroupTileSizes[i];
    return res;
  }

private:
  /// Compute the small strategy based on the problem size and the
  /// `maxNumThreadsToUse`.
  void configure(const ReductionConfig &reductionConfig);
};

} // namespace mlir::iree_compiler::gpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_SMALL_REDUCTION_STRATEGY_H_
