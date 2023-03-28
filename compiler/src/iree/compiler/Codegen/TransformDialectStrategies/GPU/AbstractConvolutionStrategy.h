// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_CONVOLUTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_CONVOLUTION_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/AbstractConvolutionStrategy.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

enum class ConvolutionStrategy { ImplicitGemm };
struct ConvolutionConfig {
  int64_t maxNumThreads;
  int64_t vectorSize;
  int64_t subgroupSize;
  bool isSpirv;
  ConvolutionStrategy strategy;
};

struct AbstractConvolutionStrategy
    : iree_compiler::AbstractConvolutionStrategy {
  AbstractConvolutionStrategy(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures)
      : iree_compiler::AbstractConvolutionStrategy(context, captures) {
    auto threadX =
        mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimX);
    auto threadY =
        mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimY);
    auto threadZ =
        mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimZ);
    allThreadAttrs = SmallVector<Attribute>{threadX, threadY, threadZ};

    auto warpX =
        mlir::gpu::GPUWarpMappingAttr::get(context, mlir::gpu::Warps::DimX);
    auto warpY =
        mlir::gpu::GPUWarpMappingAttr::get(context, mlir::gpu::Warps::DimY);
    auto warpZ =
        mlir::gpu::GPUWarpMappingAttr::get(context, mlir::gpu::Warps::DimZ);
    allWarpAttrs = SmallVector<Attribute>{warpX, warpY, warpZ};
  }

  virtual ~AbstractConvolutionStrategy() {}

  int64_t getNumThreadsXInBlock() const { return getNumThreadsInBlock()[0]; }
  int64_t getNumThreadsYInBlock() const { return getNumThreadsInBlock()[1]; }
  int64_t getNumThreadsZInBlock() const { return getNumThreadsInBlock()[2]; }

  int64_t getNumWarpsXInBlock() const { return getNumWarpsInBlock()[0]; }
  int64_t getNumWarpsYInBlock() const { return getNumWarpsInBlock()[1]; }
  int64_t getNumWarpsZInBlock() const { return getNumWarpsInBlock()[2]; }

  virtual SmallVector<int64_t> getNumThreadsInBlock() const = 0;
  virtual SmallVector<int64_t> getNumWarpsInBlock() const = 0;
  virtual SmallVector<int64_t> getWarpsTileSizes() const = 0;
  virtual SmallVector<int64_t> getInnerLoopTileSizes() const = 0;

  /// Derived quantities.
  SmallVector<Attribute> allThreadAttrs;
  SmallVector<Attribute> allWarpAttrs;
};

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_CONVOLUTION_STRATEGY_H_
