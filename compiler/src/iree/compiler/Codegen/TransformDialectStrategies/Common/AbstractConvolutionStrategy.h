// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_COMMON_ABSTRACT_CONVOLUTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_COMMON_ABSTRACT_CONVOLUTION_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
// Needed until IREE builds its own gpu::GPUBlockMappingAttr / gpu::Blocks
// attributes that are reusable across all targets.
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir {
namespace iree_compiler {

/// Structure to hold the parameters that control the convolution strategy.
struct AbstractConvolutionStrategy {
  AbstractConvolutionStrategy(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures)
      : context(context), captures(captures) {
    // Needed until IREE builds its own gpu::GPUBlockMappingAttr / gpu::Blocks
    // attributes that are reusable across all targets.
    auto blockX =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimX);
    auto blockY =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimY);
    auto blockZ =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimZ);
    allBlockAttrs = SmallVector<Attribute>{blockX, blockY, blockZ};
  }

  /// Constructor quantities.
  MLIRContext *context;
  transform_ext::MatchedConvolutionCaptures captures;

  /// Derived quantities.
  SmallVector<Attribute> allBlockAttrs;

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies.
  SmallVector<int64_t> workgroupTileSizes;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_COMMON_ABSTRACT_CONVOLUTION_STRATEGY_H_
