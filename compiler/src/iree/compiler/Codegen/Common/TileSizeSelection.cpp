// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "tiling-config"
#define KD_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(KD_DBGS() << X << "\n")

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler {

std::unique_ptr<TilingConfig>
TilingConfig::create(IREE::Codegen::LoweringConfigAttrInterface lc) {
  if (llvm::isa_and_present<IREE::Codegen::LoweringConfigAttr,
                            IREE::CPU::LoweringConfigAttr>(lc)) {
    return std::unique_ptr<TilingConfig>(new TilingConfig(lc));
  }
  return nullptr;
}

TilingConfig::TilingConfig(IREE::Codegen::LoweringConfigAttrInterface lc)
    : loweringConfig(lc) {
  assert(lc && "Expected a valid lowering config");
  if (auto codegenLc = dyn_cast<IREE::Codegen::LoweringConfigAttr>(lc)) {
    initFromCodegenLoweringConfig(codegenLc);
  } else if (auto cpuLc = dyn_cast<IREE::CPU::LoweringConfigAttr>(lc)) {
    initFromCPULoweringConfig(cpuLc);
  } else {
    assert(false && "unknown lowering config is not supported");
  }
}

void TilingConfig::initFromCodegenLoweringConfig(
    IREE::Codegen::LoweringConfigAttr lc) {
  // Initialize indices to invalid.
  std::fill(tilingLevelToActualLevelMap.begin(),
            tilingLevelToActualLevelMap.end(), TilingLevel::InvalidLevel);

  // Map the tiling levels that are defined in the actual configuration to
  // their corresponding incremental levels. We currently support the following
  // scenarios:
  //   1. [[distribution]]
  //   2. [[distribution], [vector-common-parallel]]
  //   3. [[distribution], [vector-common-parallel], [vector-reduction]]
  //   4. [[distribution], [vector-common-parallel], [vector-reduction],
  //       [vector-inner-parallel]]
  //   5. [[distribution], [cache-parallel], [cache-reduction],
  //       [vector-common-parallel], [vector-reduction],
  //       [vector-inner-parallel]]
  unsigned numTileLevels = getNumTilingLevels();
  switch (numTileLevels) {
  case 4:
    tilingLevelToActualLevelMap[TilingLevel::VectorInnerParallelTiles] = 3;
    [[fallthrough]];
  case 3:
    tilingLevelToActualLevelMap[TilingLevel::VectorReductionTiles] = 2;
    [[fallthrough]];
  case 2:
    tilingLevelToActualLevelMap[TilingLevel::VectorCommonParallelTiles] = 1;
    [[fallthrough]];
  case 1:
    tilingLevelToActualLevelMap[TilingLevel::DistributionTiles] = 0;
    break;
  case TilingLevel::MaxNumTileLevels:
    for (int i = 0; i < TilingLevel::MaxNumTileLevels; ++i) {
      tilingLevelToActualLevelMap[i] = i;
    }
    break;
  default:
    break;
  }
}

void TilingConfig::initFromCPULoweringConfig(IREE::CPU::LoweringConfigAttr lc) {
  std::fill(tilingLevelToActualLevelMap.begin(),
            tilingLevelToActualLevelMap.end(), TilingLevel::InvalidLevel);
  DictionaryAttr dictAttr = lc.getConfig();
  for (size_t i = 0, e = tilingLevelToActualLevelMap.size(); i < e; ++i) {
    if (!dictAttr || !dictAttr.contains(IREE::CPU::getTilingLevelName(
                         static_cast<IREE::CPU::TilingLevel>(i)))) {
      continue;
    }
    tilingLevelToActualLevelMap[i] = i;
  }
}

SmallVector<IREE::CPU::LoweringConfigLevelInfo>
TilingConfig::getTilingLevelInfo() {
  SmallVector<IREE::CPU::LoweringConfigLevelInfo> result;
  TileSizesListType tileSizesList = getTileSizes();
  ScalableTileFlagsListType scalableFlagsList = getScalableTileFlags();
  int64_t mappedIdx = 0;
  for (auto [idx, actualLevel] : llvm::enumerate(tilingLevelToActualLevelMap)) {
    if (actualLevel == IREE::CPU::TilingLevel::InvalidLevel) {
      continue;
    }
    result.push_back(IREE::CPU::LoweringConfigLevelInfo{
        static_cast<IREE::CPU::TilingLevel>(idx), tileSizesList[mappedIdx],
        scalableFlagsList[mappedIdx]});
    mappedIdx++;
  }
  return result;
}

/// Returns the tiling level that contains the vector dim at `dimPos` (which is
/// an index into the result of `getVectorTileSizes()`).
std::optional<unsigned>
TilingConfig::getTilingLevelForVectorDimPosition(unsigned dimPos) const {
  constexpr std::array vectorTilingLevels{
      TilingLevel::VectorCommonParallelTiles, TilingLevel::VectorReductionTiles,
      TilingLevel::VectorInnerParallelTiles};
  std::optional<unsigned> foundLevel;
  for (TilingLevel level : vectorTilingLevels) {
    auto tilingLevelIndex = tilingLevelToActualLevelMap[level];
    if (tilingLevelIndex == TilingLevel::InvalidLevel) {
      continue;
    }
    auto tilingLevel = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        loweringConfig.getTilingLevelAttr(tilingLevelIndex));
    if (tilingLevel.getSizes()[dimPos] != 0) {
      assert(!foundLevel.has_value() &&
             "expected at most one tile size to be non-zero");
      foundLevel = tilingLevelIndex;
    }
  }
  return foundLevel;
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
  for (int dimPos = 0; dimPos < numDims; ++dimPos) {
    auto dimTilingLevel = getTilingLevelForVectorDimPosition(dimPos);
    if (!dimTilingLevel.has_value())
      continue; // The size for this dim is zero in all vector tiling levels.
    auto tilingLevel = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        loweringConfig.getTilingLevelAttr(dimTilingLevel.value()));
    std::tie(vectorSizes[dimPos], scalableFlags[dimPos]) = getTileSizeAtIndex(
        tilingLevel.getSizes(), tilingLevel.getScalableFlags(), dimPos);
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
  std::array<SmallVector<unsigned, 4>, TilingLevel::MaxNumTileLevels>
      tilingLevelToDimsMap;
  for (unsigned dimPos = 0; dimPos < numDims; ++dimPos) {
    auto tilingLevelIndex = getTilingLevelForVectorDimPosition(dimPos);
    assert((tilingLevelIndex.has_value() || sizes[dimPos] == 0) &&
           "attempting to set vector size for dim with underspecified tiling "
           "level (zero is the only valid tile size)");
    if (tilingLevelIndex.has_value())
      tilingLevelToDimsMap[*tilingLevelIndex].push_back(dimPos);
  }

  MLIRContext *context = loweringConfig.getContext();
  SmallVector<IREE::Codegen::LoweringConfigTilingLevelAttr> tilingLevels;
  for (unsigned i = 0, e = getNumTilingLevels(); i < e; ++i) {
    tilingLevels.push_back(cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        loweringConfig.getTilingLevelAttr(i)));
  }
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
  return IREE::Codegen::LoweringConfigAttr::get(context, newTilingLevels);
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
    // Only distribution level + vector common parallel levels.
    return {0, 1};
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
  assert(level < TilingLevel::InvalidLevel &&
         "Unexpected invalid tiling level");
  unsigned actualLevel = tilingLevelToActualLevelMap[level];
  assert(actualLevel != TilingLevel::InvalidLevel &&
         "Searching for unavailable tiling level");
  return actualLevel;
}

} // namespace mlir::iree_compiler
