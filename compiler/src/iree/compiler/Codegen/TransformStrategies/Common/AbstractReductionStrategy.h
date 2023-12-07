// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_COMMON_ABSTRACT_REDUCTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_COMMON_ABSTRACT_REDUCTION_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"

namespace mlir::iree_compiler {

/// Structure to hold the parameters that control the reduction strategy.
struct AbstractReductionStrategy {
  AbstractReductionStrategy(
      const transform_ext::MatchedReductionCaptures &captures,
      ArrayRef<int64_t> workgroupTileSizes)
      : captures(captures), workgroupTileSizes(workgroupTileSizes) {}

  /// Constructor quantities.
  transform_ext::MatchedReductionCaptures captures;

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies.
  SmallVector<int64_t> workgroupTileSizes;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_COMMON_ABSTRACT_REDUCTION_STRATEGY_H_
