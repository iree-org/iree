// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_TILESIZESELECTION_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_TILESIZESELECTION_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler {

using SizesAndScalableFlags =
    std::pair<SmallVector<int64_t>, SmallVector<bool>>;

/// Provides unified API to get access to all the tile size needed during the
/// CPU lowering process, while abstracting the representation and verification
/// details of such information in the IR.
///
/// We currently support the following scenarios:
///   1. [[distribution]]
///   2. [[distribution], [vector-common-parallel]]
///   3. [[distribution], [vector-common-parallel], [vector-reduction],
///       [vector-inner-parallel]]
///   4. [[distribution], [cache-parallel], [cache-reduction],
///       [vector-parallel], [vector-reduction]]
class TilingConfig {
public:
  TilingConfig(IREE::Codegen::LoweringConfigAttr lc);

  /// Returns the number of tiling levels of the configuration.
  unsigned getNumTilingLevels() {
    return loweringConfig.getTilingLevels().size();
  };

  /// Returns the number of dimensions of the configuration. All the tiling
  /// levels must have the same number of dimensions.
  unsigned getNumDimensions() {
    return getNumTilingLevels() > 0
               ? loweringConfig.getTilingLevels()[0].getSizes().size()
               : 0;
  }

  /// Returns the number of parallel dimensions to tile at vector level.
  unsigned getNumVectorParallelTiles() {
    unsigned parallelLevel = getVectorCommonParallelLevel();
    if (parallelLevel <= getNumTilingLevels())
      return 0;
    return llvm::count_if(
        loweringConfig.getTilingLevels()[parallelLevel].getSizes(),
        [](int64_t tileSize) { return tileSize != 0; });
  }

  /// Returns the tiling level for cache parallel dimensions.
  unsigned getDistributionLevel() { return getActualLevel(DistributionTiles); }

  /// Returns the tiling level for cache parallel dimensions.
  unsigned getCacheParallelLevel() {
    return getActualLevel(CacheParallelTiles);
  }

  /// Returns the tiling level for cache reduction dimensions.
  unsigned getCacheReductionLevel() {
    return getActualLevel(CacheReductionTiles);
  }

  /// Returns the tiling level for vector common parallel dimensions.
  unsigned getVectorCommonParallelLevel() {
    return getActualLevel(VectorCommonParallelTiles);
  }

  /// Returns true if the tiling configuration has vector inner parallel
  /// dimensions
  bool hasVectorInnerParallelLevel() { return getNumTilingLevels() > 3; }

  /// Returns the tiling level for vector inner parallel dimensions.
  unsigned getVectorInnerParallelLevel() {
    return getActualLevel(VectorInnerParallelTiles);
  }

  /// Returns the tiling level for vector parallel dimensions.
  unsigned getVectorReductionLevel() {
    return getActualLevel(VectorReductionTiles);
  }

  /// Returns all the tile sizes of all the levels of the configuration.
  TileSizesListType getTileSizes() { return loweringConfig.getTileSizeVals(); }

  /// Returns the distribution tile sizes of the configuration.
  SmallVector<int64_t> getDistributionTileSizes() {
    return getTileSizesForLevel(getActualLevel(DistributionTiles));
  }

  SmallVector<int64_t> getCacheReductionSizes() {
    return getTileSizesForLevel(getCacheReductionLevel());
  }

  SizesAndScalableFlags getVectorCommonParallelSizes() {
    return getVectorSizesForLevel(getVectorCommonParallelLevel());
  }

  SizesAndScalableFlags getVectorReductionSizes() {
    return getVectorSizesForLevel(getVectorReductionLevel());
  }

  SizesAndScalableFlags getVectorInnerParallelSizes() {
    return getVectorSizesForLevel(getVectorInnerParallelLevel());
  }

  /// Returns the tile sizes of all the vector dimensions, including parallel
  /// and reduction dimensions.
  SizesAndScalableFlags getVectorTileSizes();

  /// Returns a list with the tiling levels that can be fused for this
  /// configuration.
  SmallVector<int64_t> getFusableLevels();

  // TODO(dcaballe): Revisit if these features are ever used.
  SmallVector<int64_t> getTileInterchangeSizes(unsigned level) {
    return loweringConfig.getTileInterchangeVals(level);
  }
  SmallVector<int64_t> getNativeVectorSizes() {
    return loweringConfig.getNativeVectorSizeVals();
  }

private:
  SizesAndScalableFlags getVectorSizesForLevel(unsigned level) {
    return std::make_pair(loweringConfig.getTileSizeVals(level),
                          loweringConfig.getScalableTileFlagVals(level));
  }

  SmallVector<int64_t> getTileSizesForLevel(unsigned level) {
    return loweringConfig.getTileSizeVals(level);
  }

  /// Internal representation for all the supported tiling levels. All or just
  /// a subset of them may be available in a valid configuration.
  enum TilingLevel : unsigned {
    DistributionTiles = 0,
    CacheParallelTiles = 1,
    CacheReductionTiles = 2,
    VectorCommonParallelTiles = 3,
    VectorReductionTiles = 4,
    VectorInnerParallelTiles = 5,
    MaxNumTileLevels = 6,
    InvalidLevel = 7,
  };

  /// Returns the actual level in the configuration for this level of tiling.
  unsigned getActualLevel(TilingLevel level);

  /// Holds the lowering config that provides the configuration.
  IREE::Codegen::LoweringConfigAttr loweringConfig;

  /// Maps `TilingLevel`'s to the actual number of levels in this configuration.
  std::array<unsigned, TilingLevel::MaxNumTileLevels>
      tilingLevelToActualLevelMap;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_TILESIZESELECTION_H_
