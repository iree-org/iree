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
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

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
static void getMatmulTileAndWorkgroupSizes(
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
static LogicalResult setOpConfig(linalg::BatchMatmulOp op) {
  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);

  if (llvm::any_of(lhsShape, ShapedType::isDynamic) ||
      llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return success();
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

    auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;

    SmallVector<int64_t, 4> numElementsPerWorkgroup;
    numElementsPerWorkgroup = {1, pair.tileSize[0], pair.tileSize[1],
                               pair.tileSize[2]};

    TileSizesListType tileSizes;
    // Workgroup level.
    tileSizes.push_back(numElementsPerWorkgroup);
    // No tiling at the subgroup level since this target doesn't use subgroup op
    // or shared memory.
    tileSizes.emplace_back();
    // Invocation level.
    tileSizes.push_back({numElementsPerWorkgroup[0],
                         numElementsPerWorkgroup[1] / pair.workgroupSize[1],
                         numElementsPerWorkgroup[2] / pair.workgroupSize[0],
                         numElementsPerWorkgroup[3]});

    return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                                 op, tileSizes, {}, pipeline,
                                                 pair.workgroupSize);
  }
  return success();
}

static LogicalResult setOpConfig(linalg::MatmulOp op) {
  ArrayRef<int64_t> lhsShape = getUntiledShape(op.inputs()[0]);
  ArrayRef<int64_t> rhsShape = getUntiledShape(op.inputs()[1]);

  if (llvm::any_of(lhsShape, ShapedType::isDynamic) ||
      llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return success();
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

    auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;

    SmallVector<int64_t, 4> numElementsPerWorkgroup(pair.tileSize.begin(),
                                                    pair.tileSize.end());

    TileSizesListType tileSizes;
    // Workgroup level.
    tileSizes.push_back(numElementsPerWorkgroup);
    // No tiling at the subgroup level since this target doesn't use subgroup op
    // or shared memory.
    tileSizes.emplace_back();
    // Invocation level.
    tileSizes.push_back({numElementsPerWorkgroup[0] / pair.workgroupSize[1],
                         numElementsPerWorkgroup[1] / pair.workgroupSize[0],
                         numElementsPerWorkgroup[2]});

    return setOpConfigAndEntryPointFnTranslation(op->getParentOfType<FuncOp>(),
                                                 op, tileSizes, {}, pipeline,
                                                 pair.workgroupSize);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setMaliCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                   Operation *rootOp) {
  int64_t subgroupSize = targetEnv.getResourceLimits().subgroup_size().getInt();
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>(
          [](auto op) { return setOpConfig(op); })
      .Case<linalg::Conv2DNhwcHwcfOp>([subgroupSize](auto op) {
        return setConvOpConfig(op, subgroupSize,
                               /*bestTilingFactor=*/16);
      })
      .Case<linalg::DepthwiseConv2DNhwOp>([subgroupSize](auto op) {
        return setConvOpConfig(op, subgroupSize,
                               /*bestTilingFactor=*/16);
      })
      .Default([](Operation *) { return success(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
