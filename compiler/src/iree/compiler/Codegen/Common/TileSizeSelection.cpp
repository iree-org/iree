// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "tiling-config"

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

bool TilingConfig::isValidLevel(IREE::CPU::TilingLevel level) {
  return tilingLevelToActualLevelMap[static_cast<int64_t>(level)] !=
         IREE::CPU::TilingLevel::InvalidLevel;
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

IREE::CPU::LoweringConfigAttr TilingConfig::getLoweringConfigWithNewVectorSizes(
    ArrayRef<int64_t> sizes, ArrayRef<bool> scalableFlags) {
  unsigned numDims = getNumDimensions();
  (void)numDims;
  assert(sizes.size() == numDims &&
         "expected `sizes` to match number of dimensions");
  assert((scalableFlags.empty() || scalableFlags.size() == numDims) &&
         "expected `scalableFlags` to match "
         "number of dimensions (or be empty)");

  MLIRContext *ctx = loweringConfig.getContext();
  SmallVector<NamedAttribute> items;
  for (unsigned i = 0, e = TilingLevel::MaxNumTileLevels; i < e; ++i) {
    auto level = static_cast<TilingLevel>(i);
    if (!isValidLevel(level)) {
      continue;
    }
    switch (level) {
    case TilingLevel::DistributionTiles:
    case TilingLevel::CacheParallelTiles:
    case TilingLevel::CacheReductionTiles: {
      items.emplace_back(IREE::CPU::getTilingLevelName(level),
                         getTilingLevelAttr(i));
      break;
    }
    case TilingLevel::VectorCommonParallelTiles:
    case TilingLevel::VectorReductionTiles:
    case TilingLevel::VectorInnerParallelTiles: {
      auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
          loweringConfig.getTilingLevelAttr(i));
      SmallVector<int64_t> newSizes(attr.getSizes());
      SmallVector<bool> newScalableFlags(attr.getScalableFlags());
      newScalableFlags.resize(newSizes.size(), false);
      for (auto [idx, size] : llvm::enumerate(newSizes)) {
        if (size == 0) {
          continue;
        }
        newSizes[idx] = sizes[idx];
        newScalableFlags[idx] = scalableFlags[idx];
      }
      auto newLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
          ctx, newSizes, attr.getInterchange(), newScalableFlags);
      items.emplace_back(IREE::CPU::getTilingLevelName(level), newLevel);
      break;
    }
    case TilingLevel::MaxNumTileLevels:
    case TilingLevel::InvalidLevel:
      break;
    };
  }
  return IREE::CPU::LoweringConfigAttr::get(ctx, items);
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
