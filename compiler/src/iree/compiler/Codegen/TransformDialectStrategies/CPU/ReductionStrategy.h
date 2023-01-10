// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_REDUCTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_REDUCTION_STRATEGY_H_

#include "iree/compiler/Codegen/TransformDialectStrategies/Common/AbstractReductionStrategy.h"

namespace mlir {
namespace iree_compiler {
namespace cpu {

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
  static ReductionStrategy create(
      MLIRContext *context,
      const transform_ext::MatchedReductionCaptures &captures);

  ReductionStrategy(const ReductionStrategy &) = default;
  ReductionStrategy &operator=(const ReductionStrategy &) = default;

  // Always profitable.
  bool isProfitable() override { return true; }

  int64_t getVectorSize() const { return vectorSize; }

 private:
  ReductionStrategy(MLIRContext *context,
                    const transform_ext::MatchedReductionCaptures &captures)
      : AbstractReductionStrategy(context, captures) {}

  void configure(const ReductionConfig &config);

  /// Vector size.
  int64_t vectorSize;
};

/// Entry point to build the transform IR corresponding to a reduction
/// strategy. This is used for mapping a N-D parallel, 1-D reduction
/// operation. The 1-D reduction dimensions must be in the most minor
/// dimension. Supports an optional leading and an optional trailing
/// elementwise operation.
void buildReductionStrategy(ImplicitLocOpBuilder &b, Value variantH,
                            const ReductionStrategy &strategy);

ReductionConfig getReductionConfig(
    const transform_ext::MatchedReductionCaptures &captures);

}  // namespace cpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_CPU_REDUCTION_STRATEGY_H_
