// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_TILESIZESELECTION_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_TILESIZESELECTION_H_

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

/// Provides unified API to get access to all the tile size needed during the
/// CPU lowering process, while abstracting the representation and verification
/// details of such information in the IR.
class TilingConfig {
 public:
  TilingConfig(IREE::Codegen::LoweringConfigAttr lc) : loweringConfig(lc) {
    // Initialize indices to invalid.
    constexpr unsigned invalidIdx = std::numeric_limits<unsigned>::max();
    int numTileLevels = loweringConfig.getTileSizes().size();
    for (int i = 0; i < (int)TilingLevel::MaxNumTileLevels; ++i) {
      tilingLevelToActualLevelMap[i] = (unsigned)invalidIdx;
    }

    // Map the tiling levels that are defined in the actual configuration to
    // their corresponding incremental levels. We currently support the
    // following scenarios:
    //   1. [[distribution]]
    //   2. [[distribution], [vector-parallel]]
    //   3. [[distribution], [vector-parallel], [vector-reduction]]
    //   4. [[distribution], [cache-parallel], [cache-reduction],
    //       [vector-parallel], [vector-reduction]]
    switch (numTileLevels) {
      case 3:
        tilingLevelToActualLevelMap[(int)TilingLevel::VectorReductionTiles] = 2;
        [[fallthrough]];
      case 2:
        tilingLevelToActualLevelMap[(int)TilingLevel::VectorParallelTiles] = 1;
        [[fallthrough]];
      case 1:
        tilingLevelToActualLevelMap[(int)TilingLevel::DistributionTiles] = 0;
        break;
      case (int)TilingLevel::MaxNumTileLevels:
        for (int i = 0; i < (int)TilingLevel::MaxNumTileLevels; ++i) {
          tilingLevelToActualLevelMap[i] = i;
        }
        break;
      default:
        break;
    }
  };

  /// Returns the number of tiling levels of the configuration.
  unsigned getNumTilingLevels() {
    return loweringConfig.getTileSizes().size();
  };

  /// Returns the number of dimensions of the configuration. All the tiling
  /// levels must have the same number of dimensions.
  unsigned getNumDimensions() { return getDistributionTileSizes().size(); }

  /// Returns the number of parallel dimensions to tile at vector level.
  unsigned getNumVectorParallelTiles() {
    return llvm::count_if(getVectorParallelSizes(),
                          [](int64_t tileSize) { return tileSize != 0; });
  }

  /// Returns the tiling level for cache parallel dimensions.
  unsigned getCacheParallelLevel() {
    return getActualLevel(TilingLevel::CacheParallelTiles);
  }

  /// Returns the tiling level for cache reduction dimensions.
  unsigned getCacheReductionLevel() {
    return getActualLevel(TilingLevel::CacheReductionTiles);
  }

  /// Returns the tiling level for vector parallel dimensions.
  unsigned getVectorParallelLevel() {
    return getActualLevel(TilingLevel::VectorParallelTiles);
  }

  /// Returns the tiling level for vector parallel dimensions.
  unsigned getVectorReductionLevel() {
    return getActualLevel(TilingLevel::VectorReductionTiles);
  }

  /// Returns all the tile sizes of all the levels of the configuration.
  TileSizesListType getTileSizes() { return loweringConfig.getTileSizeVals(); }

  /// Returns the distribution tile sizes of the configuration.
  SmallVector<int64_t> getDistributionTileSizes() {
    return loweringConfig.getTileSizeVals(
        getActualLevel(TilingLevel::DistributionTiles));
  }

  SmallVector<int64_t> getCacheReductionSizes() {
    return loweringConfig.getTileSizeVals(getCacheReductionLevel());
  }

  SmallVector<int64_t> getVectorParallelSizes() {
    return loweringConfig.getTileSizeVals(getVectorParallelLevel());
  }

  SmallVector<int64_t> getVectorReductionSizes() {
    return loweringConfig.getTileSizeVals(getVectorReductionLevel());
  }

  /// Returns the tile sizes of all the vector dimensions, including parallel
  /// and reduction dimensions.
  SmallVector<int64_t> getVectorTileSizes() {
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
  SmallVector<int64_t> getFusableLevels() {
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

  // TODO(dcaballe): Revisit if these features are ever used.
  ArrayAttr getTileInterchange() { return loweringConfig.getTileInterchange(); }
  SmallVector<int64_t> getTileInterchangeSizes(unsigned level) {
    return loweringConfig.getTileInterchangeVals(level);
  }
  SmallVector<int64_t> getNativeVectorSizes() {
    return loweringConfig.getNativeVectorSizeVals();
  }

 private:
  /// Internal representation for all the supported tiling levels. All or just
  /// a subset of them may be available in a valid configuration.
  enum class TilingLevel {
    DistributionTiles = 0,
    CacheParallelTiles = 1,
    CacheReductionTiles = 2,
    VectorParallelTiles = 3,
    VectorReductionTiles = 4,
    MaxNumTileLevels = 5,
    InvalidLevel = 6,
  };

  /// Returns the actual level in the configuration for this level of tiling.
  unsigned getActualLevel(TilingLevel level) {
    assert(level < TilingLevel::InvalidLevel &&
           "Unexpected invalid tiling level");
    unsigned actualLevel = tilingLevelToActualLevelMap[(unsigned)level];
    assert(actualLevel != (unsigned)TilingLevel::InvalidLevel &&
           "Searching for unavailable tiling level");
    return actualLevel;
  }

  /// Holds the lowering config that provides the configuration.
  IREE::Codegen::LoweringConfigAttr loweringConfig;

  /// Maps `TilingLevel`'s to the actual number of levels in this configuration.
  std::array<unsigned, (size_t)TilingLevel::MaxNumTileLevels>
      tilingLevelToActualLevelMap;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMCPU_TILESIZESELECTION_H_
