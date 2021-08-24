// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- MaliConfig.h - Mali CodeGen Configurations -------------------------===//
//
// This file contains CodeGen configurations for Mali GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

namespace {

struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  // The number of threads per workgroup along each dimension.
  std::array<int64_t, 3> workgroupSize;
};

//===----------------------------------------------------------------------===//
// Matmul
//===----------------------------------------------------------------------===//

/// Writes preferred matmul workgroup tile sizes and workgroup size into
/// `pairs` for the given matmul `scale` (MxNxK) and `elementType`.
void getMatmulTileAndWorkgroupSizes(
    int64_t scale, Type elementType,
    SmallVectorImpl<TileWorkgroupSizePair> &pairs) {
  if (elementType.isF16()) {
    const int64_t smallMatrixSizeThreshold = 512 * 512;
    // For smaller destination size we cannot fill out the GPU with bigger tile
    // sizes. Instead we pick smaller tiles along M and N to increase the number
    // of workgroups and a larger K tile size since we have lower pressure and
    // need extra instructions to hide latency.
    // TODO: The threshold needs to be fine tuned by doing exploration based on
    // matrix shapes.
    if (scale <= smallMatrixSizeThreshold) {
      pairs.push_back(TileWorkgroupSizePair({{16, 32, 16}, {8, 2, 1}}));
    } else {
      pairs.push_back(TileWorkgroupSizePair({{16, 64, 4}, {8, 2, 1}}));
      pairs.push_back(TileWorkgroupSizePair({{8, 128, 4}, {8, 2, 1}}));
      pairs.push_back(TileWorkgroupSizePair({{16, 32, 4}, {8, 2, 1}}));
    }
    return;
  }

  // TODO: Heuristic picked based on MobileNet performance. We need
  // auto-tuning to be able to make a smarter choice.
  const int64_t smallMatrixSizeThreshold = 20000;

  if (scale <= smallMatrixSizeThreshold) {
    pairs.push_back(TileWorkgroupSizePair({{4, 32, 16}, {8, 2, 1}}));
  }
  pairs.push_back(TileWorkgroupSizePair({{12, 32, 4}, {8, 2, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{14, 32, 4}, {8, 2, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{10, 32, 4}, {8, 2, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{7, 64, 4}, {16, 1, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{8, 32, 4}, {8, 2, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{6, 32, 4}, {8, 2, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{24, 16, 4}, {2, 8, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{16, 16, 4}, {2, 8, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{24, 8, 4}, {2, 8, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{40, 8, 4}, {2, 8, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{32, 8, 4}, {2, 8, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{16, 8, 4}, {2, 8, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{1, 32, 16}, {8, 1, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{1, 32, 8}, {8, 1, 1}}));
  pairs.push_back(TileWorkgroupSizePair({{1, 32, 4}, {8, 1, 1}}));
}

/// Launch configuration for Mali GPU configuration.
Optional<SPIRVCodeGenConfig> getOpConfig(linalg::BatchMatmulOp op) {
  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);

  if (llvm::any_of(lhsShape, ShapedType::isDynamic) ||
      llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return llvm::None;
  }

  // Get a vector of best tile size ordered from best to worst.
  Type elementType =
      op.inputs()[0].getType().cast<ShapedType>().getElementType();
  int64_t matmulScale = lhsShape[0] * lhsShape[1] * rhsShape[2];
  SmallVector<TileWorkgroupSizePair, 4> pairs;
  getMatmulTileAndWorkgroupSizes(matmulScale, elementType, pairs);

  for (TileWorkgroupSizePair pair : pairs) {
    if (lhsShape[1] % pair.tileSize[0] != 0 ||
        rhsShape[2] % pair.tileSize[1] != 0 ||
        lhsShape[2] % pair.tileSize[2] != 0) {
      continue;
    }

    SPIRVCodeGenConfig config = {};
    config.pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;

    SmallVector<int64_t, 4> numElementsPerWorkgroup;
    numElementsPerWorkgroup = {1, pair.tileSize[0], pair.tileSize[1],
                               pair.tileSize[2]};

    config.workgroupTileSizes = numElementsPerWorkgroup;

    // No tiling at the subgroup level since this target doesn't use subgroup op
    // or shared memory.

    config.invocationTileSizes = {
        numElementsPerWorkgroup[0],
        numElementsPerWorkgroup[1] / pair.workgroupSize[1],
        numElementsPerWorkgroup[2] / pair.workgroupSize[0],
        numElementsPerWorkgroup[3]};

    config.workgroupSize = pair.workgroupSize;

    return config;
  }
  return llvm::None;
}

Optional<SPIRVCodeGenConfig> getOpConfig(linalg::MatmulOp op) {
  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);

  if (llvm::any_of(lhsShape, ShapedType::isDynamic) ||
      llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return llvm::None;
  }

  Type elementType =
      op.inputs()[0].getType().cast<ShapedType>().getElementType();
  int64_t matmulScale = lhsShape[0] * rhsShape[1];
  SmallVector<TileWorkgroupSizePair, 4> pairs;
  getMatmulTileAndWorkgroupSizes(matmulScale, elementType, pairs);

  for (TileWorkgroupSizePair pair : pairs) {
    if (lhsShape[0] % pair.tileSize[0] != 0 ||
        rhsShape[1] % pair.tileSize[1] != 0 ||
        lhsShape[1] % pair.tileSize[2] != 0) {
      continue;
    }

    SPIRVCodeGenConfig config = {};
    config.pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;

    SmallVector<int64_t, 4> numElementsPerWorkgroup(pair.tileSize.begin(),
                                                    pair.tileSize.end());

    config.workgroupTileSizes = numElementsPerWorkgroup;

    // No tiling at the subgroup level since this target doesn't use subgroup op
    // or shared memory.

    config.invocationTileSizes = {
        numElementsPerWorkgroup[0] / pair.workgroupSize[1],
        numElementsPerWorkgroup[1] / pair.workgroupSize[0],
        numElementsPerWorkgroup[2]};

    config.workgroupSize = pair.workgroupSize;

    return config;
  }
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// Convolution
//===----------------------------------------------------------------------===//

Optional<SPIRVCodeGenConfig> getOpConfig(linalg::Conv2DNhwcHwcfOp op) {
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  ArrayRef<int64_t> inputShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> outputShape = getUntiledResultShape(linalgOp, 0);

  if (llvm::any_of(inputShape, ShapedType::isDynamic) ||
      llvm::any_of(outputShape, ShapedType::isDynamic)) {
    return llvm::None;
  }

  bool isInputTilable = inputShape[3] % 4 == 0 || inputShape[3] < 4;
  if (!isInputTilable) return llvm::None;

  // A list of preferred tile sizes and workgroup sizes.
  // TODO(antiagainst): This is for Valhall now; need to consider other Mali
  // architectures.
  static const TileWorkgroupSizePair tileWorkgroupSizePairs[] = {
      {{4, 4, 16}, {4, 4, 1}},
      {{2, 2, 64}, {16, 1, 1}},
      {{4, 8, 8}, {2, 4, 2}},
      {{2, 2, 32}, {8, 2, 1}},
      {{1, 1, 32}, {8, 1, 1}}};

  for (const auto &pair : tileWorkgroupSizePairs) {
    const std::array<int64_t, 3> &tileSize = pair.tileSize;
    const std::array<int64_t, 3> &workgroupSize = pair.workgroupSize;

    bool isOutputTilable = (outputShape[0] == 1) &&
                           (outputShape[1] % tileSize[0] == 0) &&
                           (outputShape[2] % tileSize[1] == 0) &&
                           (outputShape[3] % tileSize[2] == 0);
    if (!isOutputTilable) continue;

    SPIRVCodeGenConfig config = {};
    config.pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;

    config.workgroupTileSizes = {/*batch=*/0, /*output_height=*/tileSize[0],
                                 /*output_width=*/tileSize[1],
                                 /*output_channel=*/tileSize[2]};

    // No tiling at the subgroup level given that we don't use subgroup
    // level syncrhonization or shared memory.

    config.invocationTileSizes = {
        /*batch=*/0, /*output_height=*/tileSize[0] / workgroupSize[2],
        /*output_width=*/tileSize[1] / workgroupSize[1],
        /*output_channel=*/tileSize[2] / workgroupSize[0]};

    // Finally, for each invocation, we use tiling to generate loops to loop
    // over the filter's height (step 1), width (step 1), and input channel
    // (step 4) dimensions.
    config.convFilterTileSizes = {0, 0, 0, 0, 1, 1, 4};

    config.workgroupSize = workgroupSize;

    // Define fully static number of workgroups. This is needed for folding
    // `affine.min` ops to expose static-shaped tiled convolution for
    // vectorization.
    // TODO(#5034): Use a proper way to prove tilability and fold `affine.min`s.
    config.workgroupCount = std::array<int64_t, 3>();
    for (unsigned i = 0; i < 3; ++i) {
      (*config.workgroupCount)[2 - i] = outputShape[i + 1] / tileSize[i];
    }

    return config;
  }
  return llvm::None;
}

Optional<SPIRVCodeGenConfig> getOpConfig(linalg::DepthwiseConv2DNhwOp op) {
  auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
  ArrayRef<int64_t> inputShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> outputShape = getUntiledResultShape(linalgOp, 0);

  if (llvm::any_of(inputShape, ShapedType::isDynamic) ||
      llvm::any_of(outputShape, ShapedType::isDynamic)) {
    return llvm::None;
  }

  // A list of preferred tile sizes and workgroup sizes.
  // TODO(antiagainst): This is for Valhall now; need to consider other Mali
  // architectures.
  static const TileWorkgroupSizePair tileWorkgroupSizePairs[] = {
      {{2, 2, 32}, {8, 2, 2}},
      {{1, 4, 16}, {4, 4, 1}},
      {{1, 1, 64}, {16, 1, 1}},
      {{4, 4, 8}, {2, 4, 2}},
  };

  for (const auto &pair : tileWorkgroupSizePairs) {
    const std::array<int64_t, 3> &tileSize = pair.tileSize;
    const std::array<int64_t, 3> &workgroupSize = pair.workgroupSize;

    bool isOutputTilable = outputShape[0] == 1 &&
                           (outputShape[1] % tileSize[0] == 0) &&
                           (outputShape[2] % tileSize[1] == 0) &&
                           (outputShape[3] % tileSize[2] == 0);
    if (!isOutputTilable) continue;

    SPIRVCodeGenConfig config = {};
    config.pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;

    config.workgroupTileSizes = {/*batch=*/0,
                                 /*output_height=*/tileSize[0],
                                 /*output_width=*/tileSize[1],
                                 /*output_channel=*/tileSize[2]};

    // No tiling at the subgroup level given that we don't use subgroup
    // level syncrhonization  or shared memory.

    config.invocationTileSizes = {
        /*batch=*/0, /*output_height=*/tileSize[0] / workgroupSize[2],
        /*output_width=*/tileSize[1] / workgroupSize[1],
        /*output_channel=*/tileSize[2] / workgroupSize[0]};

    // Finally, for each invocation, we use tiling to generate loops to loop
    // over the filter's height (step 1) and width (step 1) dimensions.
    config.convFilterTileSizes = {0, 0, 0, 0, 1, 1};

    config.workgroupSize = workgroupSize;

    // Define fully static number of workgroups. This is needed for folding
    // `affine.min` ops to expose static-shaped tiled convolution for
    // vectorization.
    // TODO(#5034): Use a proper way to prove tilability and fold `affine.min`s.
    config.workgroupCount = std::array<int64_t, 3>();
    for (unsigned i = 0; i < 3; ++i) {
      (*config.workgroupCount)[2 - i] = outputShape[i + 1] / tileSize[i];
    }

    return config;
  }
  return llvm::None;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

Optional<SPIRVCodeGenConfig> getMaliCodeGenConfig(const spirv::TargetEnv &,
                                                  Operation *op) {
  if (auto matmulOp = dyn_cast<linalg::BatchMatmulOp>(op)) {
    return getOpConfig(matmulOp);
  }
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
    return getOpConfig(matmulOp);
  }
  if (auto convOp = dyn_cast<linalg::Conv2DNhwcHwcfOp>(op)) {
    return getOpConfig(convOp);
  }
  if (auto convOp = dyn_cast<linalg::DepthwiseConv2DNhwOp>(op)) {
    return getOpConfig(convOp);
  }
  return llvm::None;
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
