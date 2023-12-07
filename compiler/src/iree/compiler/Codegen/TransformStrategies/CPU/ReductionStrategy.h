// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_REDUCTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_REDUCTION_STRATEGY_H_

#include "iree/compiler/Codegen/TransformStrategies/Common/AbstractReductionStrategy.h"

namespace mlir::iree_compiler::cpu {

struct CPUModel;

/// Structure to hold a summary of HW-derived properties to configure the
/// reduction strategy.
/// The objective of this struct is to act as a minimal summary of key
/// properties derived from the hardware (e.g. by an oracle) and that are
/// sufficient to steer the strategy to produce a good version.
/// These can be thought of as latent variables or embeddings that directly
/// control the strategy and can be derived from the hardware by some procedure.
struct ReductionConfig {
  int64_t vectorSize;
};

/// A simple CPU ReductionStrategy.
class ReductionStrategy : public iree_compiler::AbstractReductionStrategy {
public:
  ReductionStrategy(const transform_ext::MatchedReductionCaptures &captures,
                    const ReductionConfig &reductionConfig);

  ReductionStrategy(const ReductionStrategy &) = default;
  ReductionStrategy &operator=(const ReductionStrategy &) = default;

  int64_t getVectorSize() const { return vectorSize; }

private:
  /// Compute the small strategy based on the problem size.
  void configure(const ReductionConfig &config);

  /// Vector size.
  int64_t vectorSize;
};

/// Entry point to build the transform IR corresponding to a reduction strategy.
/// This is used to map an N-D parallel, 1-D reduction operation with optional
/// leading and optional trailing elementwise operations.
/// The 1-D reduction dimension must be in the most minor dimension.
/// The innermost dimensions of the leading and trailing operations must be most
/// minor along all accesses.
void buildReductionStrategy(ImplicitLocOpBuilder &b, Value variantH,
                            const ReductionStrategy &strategy);

} // namespace mlir::iree_compiler::cpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_REDUCTION_STRATEGY_H_
