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

/// Returns the tile sizes of all the vector dimensions, including parallel
/// and reduction dimensions.
SizesAndScalableFlags TilingConfig::getVectorTileSizes() {
  unsigned numDims = getNumDimensions();
  SmallVector<int64_t> vectorSizes(numDims, 0);
  SmallVector<bool> scalableFlags(numDims, false);
  auto [parallelCommonSizes, parallelCommonScalableFlags] =
      getVectorCommonParallelSizes();
  auto [reductionSizes, reductionScalableFlags] = getVectorReductionSizes();
  SizesAndScalableFlags parallelInnerTiles;
  if (hasVectorInnerParallelLevel()) {
    parallelInnerTiles = getVectorInnerParallelSizes();
  }

  for (int i = 0; i < numDims; ++i) {
    SmallVector<bool> dimSizes;
    dimSizes.push_back(!!parallelCommonSizes[i] ||
                       parallelCommonScalableFlags[i]);
    dimSizes.push_back(!!reductionSizes[i] || reductionScalableFlags[i]);
    if (hasVectorInnerParallelLevel())
      dimSizes.push_back(!!parallelInnerTiles.first[i] ||
                         parallelInnerTiles.second[i]);

    unsigned nonZeroCnt = llvm::count(dimSizes, true);
    assert(nonZeroCnt <= 1 && "expected one tile size at most to be non-zero");
    (void)nonZeroCnt;

    vectorSizes[i] = parallelCommonSizes[i] ^ reductionSizes[i];
    if (hasVectorInnerParallelLevel())
      vectorSizes[i] ^= parallelInnerTiles.first[i];

    scalableFlags[i] =
        parallelCommonScalableFlags[i] || reductionScalableFlags[i];
    if (hasVectorInnerParallelLevel())
      scalableFlags[i] |= parallelInnerTiles.second[i];
  }

  return std::make_pair(vectorSizes, scalableFlags);
}

IREE::Codegen::LoweringConfigAttr
TilingConfig::withNewVectorSizes(ArrayRef<int64_t> newSizes,
                                 ArrayRef<bool> newScalableFlags) {
  unsigned numDims = getNumDimensions();
  assert(newSizes.size() == numDims &&
         "expected `sizes` to match number of dimensions");
  assert((newScalableFlags.empty() || newScalableFlags.size() == numDims) &&
         "expected `scalableFlags` to match "
         "number of dimensions (or be empty)");

  auto tilingLevels = loweringConfig.getTilingLevels();
  auto tilingLevelHasSize = [tilingLevels](unsigned tilingLevelIndex,
                                           unsigned dimPos) mutable {
    auto level = tilingLevels[tilingLevelIndex];
    return level.getSizes()[dimPos] != 0;
  };
  auto tilingLevelForDimPos = [&](unsigned dimPos) -> unsigned {
    unsigned parellelCommonLevel = getVectorCommonParallelLevel();
    unsigned reductionLevel = getVectorReductionLevel();
    if (tilingLevelHasSize(parellelCommonLevel, dimPos))
      return parellelCommonLevel;
    if (tilingLevelHasSize(reductionLevel, dimPos))
      return reductionLevel;
    if (hasVectorInnerParallelLevel()) {
      unsigned parallelInnerLevel = getVectorInnerParallelLevel();
      if (tilingLevelHasSize(parallelInnerLevel, dimPos))
        return parallelInnerLevel;
    }
    assert(false && "no vector size found for `dimPos`");
  };

  // Make a map from tiling levels to vector dims at that level.
  std::array<SmallVector<unsigned, 4>, MaxNumTileLevels> tilingLevelToDimsMap;
  for (unsigned dimPos = 0; dimPos < numDims; ++dimPos) {
    auto tilingLevelIndex = tilingLevelForDimPos(dimPos);
    tilingLevelToDimsMap[tilingLevelIndex].push_back(dimPos);
  }

  MLIRContext *context = loweringConfig.getContext();
  SmallVector<IREE::Codegen::LoweringConfigTilingLevelAttr>
      updatedTilingLevelsList(tilingLevels.begin(), tilingLevels.end());

  // For each vector tiling level:
  for (auto [tilingLevelIndex, tilingLevelDims] :
       llvm::enumerate(tilingLevelToDimsMap)) {
    if (tilingLevelDims.empty())
      continue;
    auto level = tilingLevels[tilingLevelIndex];
    SmallVector<int64_t> updatedSizes(level.getSizes());
    SmallVector<bool> updatedScalableFlags(level.getScalableFlags());
    updatedScalableFlags.resize(numDims);
    // 1. Update all the vector sizes within that tiling level.
    for (unsigned dimPos : tilingLevelDims) {
      updatedSizes[dimPos] = newSizes[dimPos];
      updatedScalableFlags[dimPos] =
          dimPos < newScalableFlags.size() && newScalableFlags[dimPos];
    }
    // 2. Then create an updated tiling level attribute for that level.
    auto updatedLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
        context, updatedSizes, level.getInterchange(), updatedScalableFlags);
    updatedTilingLevelsList[tilingLevelIndex] = updatedLevel;
  }

  // Create an updated `lowering_config` attribute.
  auto updatedTilingLevels = IREE::Codegen::LoweringConfigTilingLevelsAttr::get(
      context, updatedTilingLevelsList);
  return IREE::Codegen::LoweringConfigAttr::get(
      context, updatedTilingLevels, loweringConfig.getNativeVectorSize());
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
