// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_SMALL_REDUCTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_SMALL_REDUCTION_STRATEGY_H_

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/AbstractReductionStrategy.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

/// Encode a strategy targeted at (very) small reductions, for which other
/// strategies perform poorly.
///
/// In the case of small reductions, we cannot make an efficient use of warp
/// shuffles. Instead, take advantage of caches.
/// This strategy aims at running the reduction sequentially within each thread
/// and taking parallelism from outer dimensions that we would otherwise use for
/// block-level parallelism.
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
// Note: All this is to be able to handle very small and small-ish reductions
// without catastrophic regressions.
// TODO: Add another strategy based on segmented scans, which can allow us to
// force sizes that don't divide properly into warp shuffles.
class SmallReductionStrategy : public AbstractReductionStrategy {
 public:
  static FailureOr<SmallReductionStrategy> create(
      MLIRContext *context,
      const transform_ext::MatchedReductionCaptures &captures);

  SmallReductionStrategy(const SmallReductionStrategy &) = default;
  SmallReductionStrategy &operator=(const SmallReductionStrategy &) = default;

  int64_t getNumThreadsXInBlock() const { return getNumThreadsInBlock()[0]; }
  int64_t getNumThreadsYInBlock() const { return getNumThreadsInBlock()[1]; }
  int64_t getNumThreadsZInBlock() const { return getNumThreadsInBlock()[2]; }
  std::array<int64_t, 3> getNumThreadsInBlock() const override {
    std::array<int64_t, 3> res{1, 1, 1};
    for (int64_t i = 0, e = workgroupTileSizes.size(); i < e; ++i)
      res[i] = workgroupTileSizes[i];
    return res;
  }

  /// Profitability is computed on construction and queried.
  bool isProfitable() override { return profitable; }

 private:
  /// `hasTrailingElementwise` is currently used to guard against pathological
  /// cases where IREE can't bound a buffer and crashes.
  // TODO: Fix codegen/Common/PadDynamicAlloc.cpp which calls into upstream
  // code that tries to compose affine maps too aggressively when it could
  // instead resolve bounding by being more eager.
  SmallReductionStrategy(
      MLIRContext *context,
      const transform_ext::MatchedReductionCaptures &captures)
      : AbstractReductionStrategy(context, captures) {}

  /// Compute the small strategy based on the problem size and the
  /// `maxNumThreadsToUse`.
  void configure(const ReductionConfig &reductionConfig);

  /// Encode whether the strategy is profitable.
  bool profitable = false;
};

/// The configuration below has been determined empirically by performing a
/// manual tradeoff between problem size, amount of parallelism and vector size
/// on a particular NVIDIA RTX2080Ti 12GB card.
/// This is a coarse tradeoff that should generally give reasonably good results
/// but that begs to be complemented by hardcoded known good configurations and
/// ultimately a database and/or a random forest compression of configurations
/// with guaranteed performance.
// TODO: Lift some of the strategy sizing logic as hints and/or heuristics to
// also work properly in the dynamic case.
// TODO: Support more HW configs and make it more pluggable.
ReductionConfig getSmallReductionConfig(
    const transform_ext::MatchedReductionCaptures &captures);

/// Build the transform IR tiling reductions for the whole GPU.
/// Supports reductions in the last dimension, with optional leading and
/// trailing elementwise operations.
void buildGpuSmallReductionStrategy(ImplicitLocOpBuilder &b, Value variantH,
                                    const SmallReductionStrategy &strategy);

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_SMALL_REDUCTION_STRATEGY_H_
