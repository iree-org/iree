// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/TileSizeSelection.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir {
namespace iree_compiler {

TilingConfig::TilingConfig(IREE::Codegen::LoweringConfigAttr lc)
    : loweringConfig(lc) {
  assert(lc && "Expected a valid lowering config");

  // Initialize indices to invalid.
  std::fill(tilingLevelToActualLevelMap.begin(),
            tilingLevelToActualLevelMap.end(), InvalidLevel);

  // Map the tiling levels that are defined in the actual configuration to
  // their corresponding incremental levels. We currently support the following
  // scenarios:
  //   1. [[distribution]]
  //   2. [[distribution], [vector-parallel]]
  //   3. [[distribution], [vector-parallel], [vector-reduction]]
  //   4. [[distribution], [cache-parallel], [cache-reduction],
  //       [vector-parallel], [vector-reduction]]
  int numTileLevels = loweringConfig.getTileSizes().size();
  switch (numTileLevels) {
  case 3:
    tilingLevelToActualLevelMap[VectorReductionTiles] = 2;
    [[fallthrough]];
  case 2:
    tilingLevelToActualLevelMap[VectorParallelTiles] = 1;
    [[fallthrough]];
  case 1:
    tilingLevelToActualLevelMap[DistributionTiles] = 0;
    break;
  case MaxNumTileLevels:
    for (int i = 0; i < MaxNumTileLevels; ++i) {
      tilingLevelToActualLevelMap[i] = i;
    }
    break;
  default:
    break;
  }
};

/// Returns the tile sizes of all the vector dimensions, including parallel
/// and reduction dimensions.
SmallVector<int64_t> TilingConfig::getVectorTileSizes() {
  unsigned numDims = getNumDimensions();
  SmallVector<int64_t> vectorSizes(numDims);
  SmallVector<int64_t> parallelSizes = getVectorParallelSizes();
  SmallVector<int64_t> reductionSizes = getVectorReductionSizes();
  for (int i = 0; i < numDims; ++i) {
    vectorSizes[i] =
        parallelSizes[i] != 0 ? parallelSizes[i] : reductionSizes[i];
  }

  return vectorSizes;
}

/// Returns a list with the tiling levels that can be fused for this
/// configuration.
SmallVector<int64_t> TilingConfig::getFusableLevels() {
  switch (getNumTilingLevels()) {
  case 0:
    return {};
  case 1:
    // Only distribution level.
    return {0};
  case 3:
    // Distribution + vector parallel levels.
    return {0, 1};
  case 5:
    // Distribution + cache parallel levels.
    return {0, 1};
  default:
    llvm_unreachable("Unexpected number of tiling levels");
  }
}

/// Returns the actual level in the configuration for this level of tiling.
unsigned TilingConfig::getActualLevel(TilingLevel level) {
  assert(level < InvalidLevel && "Unexpected invalid tiling level");
  unsigned actualLevel = tilingLevelToActualLevelMap[level];
  assert(actualLevel != InvalidLevel &&
         "Searching for unavailable tiling level");
  return actualLevel;
}

} // namespace iree_compiler
} // namespace mlir
