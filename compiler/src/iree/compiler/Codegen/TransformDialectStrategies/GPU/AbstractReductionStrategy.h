// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_REDUCTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_REDUCTION_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/AbstractReductionStrategy.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

/// Structure to hold a summary of HW-derived properties to configure the
/// reduction strategy.
/// The objective of this struct is to act as a minimal summary of key
/// properties derived from the hardware (e.g. by an oracle) and that are
/// sufficient to steer the strategy to produce a good version.
/// These can be thought of as latent variables or embeddings that directly
/// control the strategy and can be derived from the hardware by some procedure.
enum class ReductionStrategy { Small, Staged };
struct ReductionConfig {
  int64_t maxNumThreads;
  int64_t vectorSize;
  ReductionStrategy strategy;
};

/// Structure to hold the parameters that control the reduction strategy.
struct AbstractReductionStrategy : iree_compiler::AbstractReductionStrategy {
  AbstractReductionStrategy(
      MLIRContext *context,
      const transform_ext::MatchedReductionCaptures &captures)
      : iree_compiler::AbstractReductionStrategy(context, captures) {
    auto threadX =
        mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimX);
    auto threadY =
        mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimY);
    auto threadZ =
        mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimZ);
    allThreadAttrs = SmallVector<Attribute>{threadX, threadY, threadZ};
  }

  virtual ~AbstractReductionStrategy() {}

  int64_t getNumThreadsXInBlock() const { return getNumThreadsInBlock()[0]; }
  int64_t getNumThreadsYInBlock() const { return getNumThreadsInBlock()[1]; }
  int64_t getNumThreadsZInBlock() const { return getNumThreadsInBlock()[2]; }
  virtual std::array<int64_t, 3> getNumThreadsInBlock() const = 0;

  /// Derived quantities.
  SmallVector<Attribute> allThreadAttrs;
};

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_REDUCTION_STRATEGY_H_
