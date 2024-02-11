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

using SizeAndScalableFlag = std::tuple<int64_t &, bool &>;

/// A tuple that encapsulates two quantities describing tile sizes:
///   * regular tile sizes (integers) - that's always required
///   * scalable tile flags (bool - only used/required for scalable
///     vectorisation.
/// Use this wrapper to make sure that both quantities are upated when
/// manipulating tile sizes.
struct SizesAndScalableFlagsTuple {
  SmallVector<int64_t> sizes;
  SmallVector<bool> flags;

  // Represents a pair of references to a size and a scalable flag at the given
  // index. Due to various implementation details of vector of bools, it's much
  // easier to store a reference to a whole container and an index. While this
  // increases the size of this wrapper, it also simplifies the implementation.
  struct ReferencePair {
    SmallVector<int64_t> &sizesVec;
    SmallVector<bool> &flagsVec;
    // Index of this pair within the vectors
    size_t index;

    explicit ReferencePair(const ReferencePair &a) = default;
    ReferencePair(SmallVector<int64_t> &sizesVecRef,
                  SmallVector<bool> &boolVectorRef, size_t indexRef)
        : sizesVec(sizesVecRef), flagsVec(boolVectorRef), index(indexRef) {}

    // Update this pair based on the input integer + bool
    ReferencePair &operator=(const std::pair<int64_t, bool> &values) {
      sizesVec[index] = values.first;
      flagsVec[index] = values.second;
      return *this;
    }

    // Update this pair based on the input integer. Assume that the scalable
    // size is false. This is safe to use in cases where no scalable
    // vectorisation/tiling is used/supported.
    ReferencePair &operator=(int64_t size) {
      sizesVec[index] = size;
      flagsVec[index] = false;
      return *this;
    }

    // Update this pair based on the input ReferencePair
    ReferencePair &operator=(const ReferencePair &pair) {
      sizesVec[index] = pair.sizesVec[index];
      flagsVec[index] = pair.flagsVec[index];
      return *this;
    }
  };

  SizesAndScalableFlagsTuple(SmallVector<int64_t> s, SmallVector<bool> f)
      : sizes(s), flags(f) {}

  // Initialise to {0, false} for all sizes
  SizesAndScalableFlagsTuple(size_t numElements)
      : sizes(SmallVector<int64_t>(numElements, 0)),
        flags(SmallVector<bool>(numElements, false)) {}

  SizesAndScalableFlags get() {
    return std::pair<SmallVector<int64_t>, SmallVector<bool>>(sizes, flags);
  }

  ReferencePair operator[](size_t index) {
    // A new pair requires a reference to sizes, scalable flags and an index.
    return {sizes, flags, index};
  }
};

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
    return SizesAndScalableFlagsTuple(
               loweringConfig.getTileSizeVals(level),
               loweringConfig.getScalableTileFlagVals(level))
        .get();
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
