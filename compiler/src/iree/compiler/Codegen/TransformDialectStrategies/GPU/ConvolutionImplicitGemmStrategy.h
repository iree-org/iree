// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/AbstractConvolutionStrategy.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

class ConvolutionImplicitGemmStrategy : public AbstractConvolutionStrategy {
 public:
  static ConvolutionImplicitGemmStrategy create(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures,
      const ConvolutionConfig &convolutionConfig);

  ConvolutionImplicitGemmStrategy(const ConvolutionImplicitGemmStrategy &) =
      default;
  ConvolutionImplicitGemmStrategy &operator=(
      const ConvolutionImplicitGemmStrategy &) = default;

  SmallVector<int64_t> getNumThreadsInBlock() const override {
    return {numThreadsXInBlock, 1, 1};
  }

  SmallVector<int64_t> getNumWarpsInBlock() const override {
    return {numWarpsXInBlock, 1, 1};
  }

  SmallVector<int64_t> getInputTileSizes() const {
    SmallVector<int64_t> tileSizes(captures.convolutionDims.batch.size(), 0);
    if (isNchw) tileSizes.push_back(0);
    tileSizes.push_back(numThreadsXForIm2Col);
    return tileSizes;
    // if (isNchw)
    //   return {0, numThreadsXForIm2Col, 0};
    // return {0, 0, numThreadsXInBlock};
  }

  SmallVector<int64_t> getOutputTileSizes() const {
    SmallVector<int64_t> tileSizes(captures.convolutionDims.batch.size(), 0);
    if (!tileM) tileSizes.push_back(0);
    tileSizes.push_back(numThreadsXToDistribute);
    return tileSizes;
    // if (tileM)
    //   return {0, numThreadsXToDistribute, 0};
    // return {0, 0, numThreadsXToDistribute};
  }

  SmallVector<int64_t> getWarpsTileSizes() const override {
    SmallVector<int64_t> tileSizes(captures.convolutionDims.batch.size(), 0);
    if (!tileM) tileSizes.push_back(0);
    tileSizes.push_back(numWarpsXInBlock);
    return tileSizes;
    //  return {0, numWarpsXInBlock, 0};
    // return {0, 0, numWarpsXInBlock};
  }

  SmallVector<int64_t> getInnerLoopTileSizes() const override {
    SmallVector<int64_t> tileSizes(captures.convolutionDims.batch.size() + 2,
                                   0);
    tileSizes.push_back(innerLoopTileSize);
    return tileSizes;
    // return {0, 0, 0, innerLoopTileSize};
  }

  int64_t getImplicitGemmFilterOperandIndex() const {
    if (captures.convolutionDims.outputChannel[0] <
        captures.convolutionDims.outputImage[0])
      return 0;
    return 1;
  }

  bool getIsSpirv() const { return isSpirv; }

 private:
  ConvolutionImplicitGemmStrategy(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures)
      : AbstractConvolutionStrategy(context, captures) {}

  void configure(const ConvolutionConfig &convolutionConfig);

  int64_t numThreadsXInBlock;
  int64_t numThreadsXToDistribute;
  int64_t numThreadsXForIm2Col;
  int64_t numWarpsXInBlock;
  int64_t innerLoopTileSize;

  bool tileM = false;
  bool isNchw = false;
  bool isSpirv = false;
};

void buildConvolutionImplicitGemmStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const ConvolutionImplicitGemmStrategy &strategy);

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_IMPLICIT_GEMM_STRATEGY_H_
