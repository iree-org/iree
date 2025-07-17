// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_TILESIZESELECTION_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_TILESIZESELECTION_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"

namespace mlir::iree_compiler {

using SizesAndScalableFlags =
    std::pair<SmallVector<int64_t>, SmallVector<bool>>;

/// Provides unified API to get access to all the tile size needed during the
/// CPU lowering process, while abstracting the representation and verification
/// details of such information in the IR.
///
/// We currently support the following scenarios, if
/// IREE::Codegen::LoweringConfigAttr is used:
///   1. [[distribution]]
///   2. [[distribution], [vector-common-parallel]]
///   3. [[distribution], [vector-common-parallel], [vector-reduction]]
///   4. [[distribution], [vector-common-parallel], [vector-reduction],
///       [vector-inner-parallel]]
///   5. [[distribution], [cache-parallel], [cache-reduction],
///       [vector-parallel], [vector-reduction]]
class TilingConfig {
public:
  /// Internal representation for all the supported tiling levels. All or just
  /// a subset of them may be available in a valid configuration.
  using TilingLevel = IREE::CPU::TilingLevel;

  static std::unique_ptr<TilingConfig>
  create(IREE::Codegen::LoweringConfigAttrInterface lc);
  TilingConfig() = delete;
  ~TilingConfig() {}

  /// Returns the number of tiling levels of the configuration.
  unsigned getNumTilingLevels() const {
    std::optional<unsigned> maybeResult = loweringConfig.getNumTilingLevels();
    assert(maybeResult.has_value() &&
           "invalid loweringConfig that fails to get number of tiling levels");
    return maybeResult.value();
  };

  /// Returns the number of dimensions of the configuration. All the tiling
  /// levels must have the same number of dimensions.
  unsigned getNumDimensions() {
    for (unsigned level : tilingLevelToActualLevelMap) {
      if (level == TilingLevel::InvalidLevel) {
        continue;
      }
      return loweringConfig.getStaticTilingLevelSizes(level, /*target=*/nullptr)
          .size();
    }
    return 0;
  }

  /// Returns the number of parallel dimensions to tile at vector level.
  unsigned getNumVectorParallelTiles() {
    unsigned parallelLevel = getVectorCommonParallelLevel();
    if (parallelLevel <= getNumTilingLevels())
      return 0;
    return llvm::count_if(loweringConfig.getStaticTilingLevelSizes(
                              parallelLevel, /*target=*/nullptr),
                          [](int64_t tileSize) { return tileSize != 0; });
  }

  /// Returns a list of tiling information for each level. Each value is a valid
  /// level in the TilingConfig.
  /// Different from attribute variant, the method materialize the attribute
  /// content to the `IREE::CPU::LoweringConfigLevelInfo` contrainer.
  SmallVector<IREE::CPU::LoweringConfigLevelInfo> getTilingLevelInfo();

  /// Returns all the tile sizes of all the levels of the configuration.
  TileSizesListType getTileSizes() const {
    TileSizesListType result;
    for (auto i : tilingLevelToActualLevelMap) {
      if (i == TilingLevel::InvalidLevel) {
        continue;
      }
      Attribute attr = loweringConfig.getTilingLevelAttr(i);
      assert(attr && "failed to get tiling level attribute");
      result.emplace_back(
          cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(attr).getSizes());
    }
    return result;
  }

  /// Returns a list that contains all the scalable tile flags in TilingLevel
  /// order.
  ScalableTileFlagsListType getScalableTileFlags() const {
    ScalableTileFlagsListType result;
    for (auto i : tilingLevelToActualLevelMap) {
      if (i == TilingLevel::InvalidLevel) {
        continue;
      }
      Attribute attr = loweringConfig.getTilingLevelAttr(i);
      assert(attr && "failed to get tiling level attribute");
      auto tilingLevelAttr =
          cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(attr);
      result.emplace_back(tilingLevelAttr.getScalableFlags());
      // Extend the scalable flags with `false` to match the length of the
      // sizes.
      result.back().resize(tilingLevelAttr.getSizes().size());
    }
    return result;
  }

  /// Returns the tiling level for cache parallel dimensions.
  unsigned getDistributionLevel() {
    return getActualLevel(TilingLevel::DistributionTiles);
  }

  /// Returns the tiling level for cache parallel dimensions.
  unsigned getCacheParallelLevel() {
    return getActualLevel(TilingLevel::CacheParallelTiles);
  }

  /// Returns the tiling level for cache reduction dimensions.
  unsigned getCacheReductionLevel() {
    return getActualLevel(TilingLevel::CacheReductionTiles);
  }

  /// Returns the tiling level for vector common parallel dimensions.
  unsigned getVectorCommonParallelLevel() {
    return getActualLevel(TilingLevel::VectorCommonParallelTiles);
  }

  /// Returns true if the tiling configuration has vector inner parallel
  /// dimensions
  bool hasVectorInnerParallelLevel() { return getNumTilingLevels() > 3; }

  /// Returns the tiling level for vector inner parallel dimensions.
  unsigned getVectorInnerParallelLevel() {
    return getActualLevel(TilingLevel::VectorInnerParallelTiles);
  }

  /// Returns the tiling level for vector parallel dimensions.
  unsigned getVectorReductionLevel() {
    return getActualLevel(TilingLevel::VectorReductionTiles);
  }

  /// Returns the distribution tile sizes of the configuration.
  SmallVector<int64_t> getDistributionTileSizes() {
    return getTileSizesForLevel(getActualLevel(TilingLevel::DistributionTiles));
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

  /// Returns a new `LoweringConfigAttr`, with the tile sizes of vector
  /// dimensions, set to `sizes`, and the corresponding scalability set to
  /// `scalableFlags`.
  IREE::Codegen::LoweringConfigAttr
  getLoweringConfigWithNewVectorSizes(ArrayRef<int64_t> sizes,
                                      ArrayRef<bool> scalableFlags = {});

  /// Returns a list with the tiling levels that can be fused for this
  /// configuration.
  SmallVector<int64_t> getFusableLevels();

  // TODO(dcaballe): Revisit if these features are ever used.
  SmallVector<int64_t> getTileInterchangeSizes(unsigned level) {
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        loweringConfig.getTilingLevelAttr(level));
    return SmallVector<int64_t>(attr.getInterchange());
  }

private:
  // Initialize the TilingConfig with given LoweringConfigAttr attribute
  // details.
  explicit TilingConfig(IREE::Codegen::LoweringConfigAttrInterface lc);
  void initFromCodegenLoweringConfig(IREE::Codegen::LoweringConfigAttr lc);
  void initFromCPULoweringConfig(IREE::CPU::LoweringConfigAttr lc);

  SizesAndScalableFlags getVectorSizesForLevel(unsigned level) {
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        loweringConfig.getTilingLevelAttr(level));
    return {SmallVector<int64_t>(attr.getSizes()),
            SmallVector<bool>(attr.getScalableFlags())};
  }

  SmallVector<int64_t> getTileSizesForLevel(unsigned level) {
    return loweringConfig.getStaticTilingLevelSizes(level, /*target=*/nullptr);
  }

  /// Returns the tiling level that contains the vector dim at `dimPos` (which
  /// is an index into the result of `getVectorTileSizes()`).
  std::optional<unsigned>
  getTilingLevelForVectorDimPosition(unsigned dimPos) const;

  /// Returns the actual level in the configuration for this level of tiling.
  unsigned getActualLevel(TilingLevel level);

  /// Holds the lowering config that provides the configuration.
  IREE::Codegen::LoweringConfigAttrInterface loweringConfig;

  /// Maps `TilingLevel`'s to the actual number of levels in this configuration.
  std::array<unsigned, TilingLevel::MaxNumTileLevels>
      tilingLevelToActualLevelMap;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_TILESIZESELECTION_H_
