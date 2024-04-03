// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler {

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
  //   2. [[distribution], [vector-common-parallel]]
  //   3. [[distribution], [vector-common-parallel], [vector-reduction],
  //       [vector-inner-parallel]]
  //   4. [[distribution], [cache-parallel], [cache-reduction],
  //       [vector-common-parallel], [vector-reduction],
  //       [vector-inner-parallel]]
  int numTileLevels = loweringConfig.getTilingLevels().size();
  switch (numTileLevels) {
  case 4:
    tilingLevelToActualLevelMap[VectorInnerParallelTiles] = 3;
    [[fallthrough]];
  case 3:
    tilingLevelToActualLevelMap[VectorReductionTiles] = 2;
    [[fallthrough]];
  case 2:
    tilingLevelToActualLevelMap[VectorCommonParallelTiles] = 1;
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

/// Returns the tiling level that contains the vector dim at `dimPos` (which is
/// an index into the result of `getVectorTileSizes()`).
unsigned TilingConfig::getTilingLevelForVectorDimPosition(unsigned dimPos) {
  constexpr std::array vectorTilingLevels{VectorCommonParallelTiles,
                                          VectorReductionTiles,
                                          VectorInnerParallelTiles};
  ArrayRef<TilingLevel> possibleLevels = vectorTilingLevels;
  if (!hasVectorInnerParallelLevel())
    possibleLevels = possibleLevels.drop_back();
  std::optional<unsigned> foundLevel;
  auto tilingLevels = loweringConfig.getTilingLevels();
  for (TilingLevel level : possibleLevels) {
    auto tilingLevelIndex = getActualLevel(level);
    if (tilingLevels[tilingLevelIndex].getSizes()[dimPos] != 0) {
      assert(!foundLevel.has_value() &&
             "expected at most one tile size to be non-zero");
      foundLevel = tilingLevelIndex;
    }
  }
  assert(foundLevel.has_value() && "no vector size found for `dimPos`");
  return *foundLevel;
}

/// Returns the tile size (size + scalability pair) at `index`. The
/// `scalableFlags` can be empty.
static std::pair<int64_t, bool> getTileSizeAtIndex(ArrayRef<int64_t> sizes,
                                                   ArrayRef<bool> scalableFlags,
                                                   unsigned index) {
  return std::make_pair(sizes[index],
                        index < scalableFlags.size() && scalableFlags[index]);
}

/// Returns the tile sizes of all the vector dimensions, including parallel
/// and reduction dimensions.
SizesAndScalableFlags TilingConfig::getVectorTileSizes() {
  unsigned numDims = getNumDimensions();
  SmallVector<int64_t> vectorSizes(numDims, 0);
  SmallVector<bool> scalableFlags(numDims, false);
  auto tilingLevels = loweringConfig.getTilingLevels();
  for (int dimPos = 0; dimPos < numDims; ++dimPos) {
    unsigned dimTilingLevel = getTilingLevelForVectorDimPosition(dimPos);
    std::tie(vectorSizes[dimPos], scalableFlags[dimPos]) = getTileSizeAtIndex(
        tilingLevels[dimTilingLevel].getSizes(),
        tilingLevels[dimTilingLevel].getScalableFlags(), dimPos);
  }
  return std::make_pair(vectorSizes, scalableFlags);
}

/// Returns a new `LoweringConfigAttr`, with the tile sizes of vector
/// dimensions, set to `sizes`, and the corresponding scalability set to
/// `scalableFlags`.
IREE::Codegen::LoweringConfigAttr
TilingConfig::getLoweringConfigWithNewVectorSizes(
    ArrayRef<int64_t> sizes, ArrayRef<bool> scalableFlags) {
  unsigned numDims = getNumDimensions();
  assert(sizes.size() == numDims &&
         "expected `sizes` to match number of dimensions");
  assert((scalableFlags.empty() || scalableFlags.size() == numDims) &&
         "expected `scalableFlags` to match "
         "number of dimensions (or be empty)");

  // Make a map from tiling levels to vector dims at that level.
  std::array<SmallVector<unsigned, 4>, MaxNumTileLevels> tilingLevelToDimsMap;
  for (unsigned dimPos = 0; dimPos < numDims; ++dimPos) {
    auto tilingLevelIndex = getTilingLevelForVectorDimPosition(dimPos);
    tilingLevelToDimsMap[tilingLevelIndex].push_back(dimPos);
  }

  MLIRContext *context = loweringConfig.getContext();
  auto tilingLevels = loweringConfig.getTilingLevels();
  SmallVector<IREE::Codegen::LoweringConfigTilingLevelAttr> newTilingLevelsList(
      tilingLevels.begin(), tilingLevels.end());

  // For each vector tiling level:
  for (auto [tilingLevelIndex, tilingLevelDims] :
       llvm::enumerate(tilingLevelToDimsMap)) {
    if (tilingLevelDims.empty())
      continue;
    auto level = tilingLevels[tilingLevelIndex];
    SmallVector<int64_t> newSizes(level.getSizes());
    SmallVector<bool> newScalableFlags(level.getScalableFlags());
    newScalableFlags.resize(numDims);
    // 1. Update all the vector sizes within that tiling level.
    for (unsigned dimPos : tilingLevelDims) {
      std::tie(newSizes[dimPos], newScalableFlags[dimPos]) =
          getTileSizeAtIndex(sizes, scalableFlags, dimPos);
    }
    // 2. Then create a new tiling level attribute for that level.
    auto newLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
        context, newSizes, level.getInterchange(), newScalableFlags);
    newTilingLevelsList[tilingLevelIndex] = newLevel;
  }

  // Create a new `lowering_config` attribute.
  auto newTilingLevels = IREE::Codegen::LoweringConfigTilingLevelsAttr::get(
      context, newTilingLevelsList);
  return IREE::Codegen::LoweringConfigAttr::get(
      context, newTilingLevels, loweringConfig.getNativeVectorSize());
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
  case 4:
    // Distribution + vector common parallel levels + vector inner parallel
    // levels.
    return {0, 1, 3};
  case 6:
    // Distribution + cache parallel levels.
    return {0, 1, 3, 5};
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

} // namespace mlir::iree_compiler
