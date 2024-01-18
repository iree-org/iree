// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_PAD_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_PAD_STRATEGY_H_

#include <array>

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"

namespace mlir::iree_compiler::gpu {

struct PadConfig {};

/// Simple padding strategy.
class PadStrategy : public GPUStrategy {
public:
  PadStrategy(MLIRContext *context,
              const transform_ext::MatchedPadCaptures &captures,
              const PadConfig &config, const GPUModel &gpuModel)
      : GPUStrategy(gpuModel), ctx(context), captures(captures) {
    initDefaultValues();
    (void)config;
  }

  PadStrategy(const PadStrategy &) = default;
  PadStrategy &operator=(const PadStrategy &) = default;

  void initDefaultValues();
  void configure(GPUModel gpuModel);

  int64_t blockTileSizeX() const { return blockTileSizes[0]; }
  int64_t blockTileSizeY() const { return blockTileSizes[1]; }
  int64_t blockTileSizeZ() const { return blockTileSizes[2]; }
  int64_t numThreadsX() const { return numThreads[0]; }
  int64_t numThreadsY() const { return numThreads[1]; }
  int64_t numThreadsZ() const { return numThreads[2]; }

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedPadCaptures captures;

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies.
  SmallVector<int64_t> blockTileSizes;
  SmallVector<int64_t> numThreads;
  SmallVector<int64_t> vectorSize;
  // TODO: implement this case.
  bool useAsyncCopies = false;
};

} // namespace mlir::iree_compiler::gpu

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_PAD_STRATEGY_H_
