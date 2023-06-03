// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

// TODO(hanchung): Create a pass to handle detailed logic about splitting tiling
// sizes for parallel dims and reduction dims.
// We have to fuse the fill + named_op + generic ops along parallel dims
// firstly. At this stage, we do not apply vectorization. The reduction dim
// won't get tiled if the case is matmul + generic op. In this case, we have to
// tile along reduction dim again, which needs them to be TilingInterface ops.
// TODO: Update doc.
class TilingConfig {
 public:
  TilingConfig(IREE::Codegen::LoweringConfigAttr lc) : loweringConfig(lc) {
    int numTileLevels = loweringConfig.getTileSizes().size();

    // Initialize indices as invalid.
    constexpr unsigned invalidIdx = std::numeric_limits<unsigned>::max();
    for (int i = 0; i < MaxNumTileLevels; ++i) {
      levelToIdxMap[i] = invalidIdx;
    }

    switch (numTileLevels) {
      case 3:
        levelToIdxMap[ReductionVectorTiles] = 2;
        // Fall through.
      case 2:
        levelToIdxMap[ParallelVectorTiles] = 1;
        // Fall through.
      case 1:
        levelToIdxMap[DistributionTiles] = 0;
        break;
      case MaxNumTileLevels:
        for (int i = 0; i < MaxNumTileLevels; ++i) {
          levelToIdxMap[i] = i;
        }
        break;
      default:
        llvm_unreachable("Unexpected number of tiling levels");
    };
  };

  unsigned getNumLevels() { return loweringConfig.getTileSizes().size(); };

  TileSizesListType getAllTileSizes() {
    return loweringConfig.getTileSizeVals();
  }

  SmallVector<int64_t> getDistributionVectorSizes() {
    return loweringConfig.getTileSizeVals(getIdx(DistributionTiles));
  }

  unsigned getParallelCacheIdx() { return getIdx(ParallelCacheTiles); }

  unsigned getParallelVectorIdx() { return getIdx(ParallelVectorTiles); }

  unsigned getNumParallelVectorTiles() {
    return llvm::count_if(getParallelVectorSizes(),
                          [](int64_t tileSize) { return tileSize != 0; });
  }

  SmallVector<int64_t> getParallelVectorSizes() {
    return loweringConfig.getTileSizeVals(getParallelVectorIdx());
  }

  unsigned getReductionCacheIdx() { return getIdx(ReductionCacheTiles); }
  SmallVector<int64_t> getReductionCacheSizes() {
    return loweringConfig.getTileSizeVals(getReductionCacheIdx());
  }

  unsigned getReductionVectorIdx() { return getIdx(ReductionVectorTiles); }

  SmallVector<int64_t> getReductionVectorSizes() {
    return loweringConfig.getTileSizeVals(getReductionVectorIdx());
  }

  SmallVector<int64_t> getFusableLevels() {
    switch (getNumLevels()) {
      case 0:
        return {};
      case 1:
        // Only distribution level.
        return {0};
      case 3:
        // Distribution + parallel vectorization levels.
        return {0, 1};
      case 5:
        // Distribution + parallel cache and vectorization levels.
        return {0, 1, 2};
      default:
        llvm_unreachable("Unexpected number of tiling levels");
    }
  }

 private:
  enum TilingLevel {
    // Tile TilingInterface operations to threads.
    DistributionTiles = 0,
    ParallelCacheTiles = 1,
    // Tile TilingInterface operation on workgroup thread for parallel dims.
    ParallelVectorTiles = 2,
    ReductionCacheTiles = 3,
    // Tile TilingInterface operations on workgroup thread for reduction dims.
    ReductionVectorTiles = 4,
    MaxNumTileLevels = 5,
    InvalidLevel = 6,
  };

  unsigned getIdx(TilingLevel level) {
    unsigned idx = levelToIdxMap[level];
    assert(idx != InvalidLevel && "Searching for unavailable tiling level");
    return idx;
  }

  IREE::Codegen::LoweringConfigAttr loweringConfig;
  std::array<unsigned, MaxNumTileLevels> levelToIdxMap;
};

LogicalResult initCPULaunchConfig(ModuleOp moduleOp);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMCPU_KERNELDISPATCH_H_
