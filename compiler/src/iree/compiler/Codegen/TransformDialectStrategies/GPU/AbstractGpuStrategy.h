// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GPU_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GPU_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/AbstractReductionStrategy.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

/// Base quantities generally useful for all GPU strategies.
struct StrategyBase {
  StrategyBase(MLIRContext *ctx) : ctx(ctx) {}

  /// Constructor quantities.
  MLIRContext *ctx;

  Attribute blockX() const {
    return mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::Blocks::DimX);
  }
  Attribute blockY() const {
    return mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::Blocks::DimY);
  }
  Attribute blockZ() const {
    return mlir::gpu::GPUBlockMappingAttr::get(ctx, mlir::gpu::Blocks::DimZ);
  }
  Attribute threadX() const {
    return mlir::gpu::GPUThreadMappingAttr::get(ctx, mlir::gpu::Threads::DimX);
  }
  Attribute threadY() const {
    return mlir::gpu::GPUThreadMappingAttr::get(ctx, mlir::gpu::Threads::DimY);
  }
  Attribute threadZ() const {
    return mlir::gpu::GPUThreadMappingAttr::get(ctx, mlir::gpu::Threads::DimZ);
  }
  Attribute warpX() const {
    return mlir::gpu::GPUWarpMappingAttr::get(ctx, mlir::gpu::Warps::DimX);
  }
  Attribute warpY() const {
    return mlir::gpu::GPUWarpMappingAttr::get(ctx, mlir::gpu::Warps::DimY);
  }
  Attribute warpZ() const {
    return mlir::gpu::GPUWarpMappingAttr::get(ctx, mlir::gpu::Warps::DimZ);
  }
  Attribute linearIdX() const {
    return mlir::gpu::GPULinearIdMappingAttr::get(ctx,
                                                  mlir::gpu::LinearId::DimX);
  }
  Attribute linearIdY() const {
    return mlir::gpu::GPULinearIdMappingAttr::get(ctx,
                                                  mlir::gpu::LinearId::DimY);
  }
  Attribute linearIdZ() const {
    return mlir::gpu::GPULinearIdMappingAttr::get(ctx,
                                                  mlir::gpu::LinearId::DimZ);
  }
};

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GPU_STRATEGY_H_
