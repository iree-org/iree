// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

class TileSizeSelection : public ::testing::Test {
protected:
  TileSizeSelection() {
    reg.insert<IREE::Codegen::IREECodegenDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  // Initialize `loweringConfig` to contain `numTilingLevels` tiling levels.
  // The actual tile sizes are not set.
  void initLoweringConfig(unsigned numTilingLevels) {
    SmallVector<IREE::Codegen::LoweringConfigTilingLevelAttr>
        newTilingLevelsList;
    for (size_t i = 0; i < numTilingLevels; i++) {
      SmallVector<int64_t> sizes;
      SmallVector<bool> scalableFlags;

      auto newLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
          &ctx, sizes, /*interchange=*/ArrayRef<int64_t>{}, scalableFlags);
      newTilingLevelsList.push_back(newLevel);
    }

    auto newTilingLevels = IREE::Codegen::LoweringConfigTilingLevelsAttr::get(
        &ctx, newTilingLevelsList);
    loweringConfig =
        IREE::Codegen::LoweringConfigAttr::get(&ctx, newTilingLevels);
  }

  ~TileSizeSelection() override {}

  MLIRContext ctx;
  DialectRegistry reg;
  IREE::Codegen::LoweringConfigAttr loweringConfig;
};

class CPUTileSizeSelection : public ::testing::Test {
protected:
  CPUTileSizeSelection() {
    reg.insert<IREE::CPU::IREECPUDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  // Initialize `loweringConfig` to contain the config for `targets`. The actual
  // tile sizes are not set in each target, i.e., they are empty lists.
  void initLoweringConfig(SmallVector<IREE::CPU::TilingLevel> targets) {
    SmallVector<NamedAttribute> configItems;
    for (auto level : targets) {
      SmallVector<int64_t> sizes;
      SmallVector<bool> scalableFlags;
      configItems.emplace_back(
          IREE::CPU::getTilingLevelName(level),
          IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(&ctx, sizes,
                                                            scalableFlags));
    }
    loweringConfig = IREE::CPU::LoweringConfigAttr::get(
        &ctx, DictionaryAttr::get(&ctx, configItems));
  }

  ~CPUTileSizeSelection() override {}

  MLIRContext ctx;
  DialectRegistry reg;
  IREE::CPU::LoweringConfigAttr loweringConfig;
};

TEST_F(TileSizeSelection, NumTilingLevels) {
  const unsigned kMaxNumTilingLevels = 7;

  // 1. Initialize Lowering Config
  initLoweringConfig(kMaxNumTilingLevels);

  // 2. Create TilingConfig and check if the number of tiling levels match.
  TilingConfig tilingConfig(loweringConfig);
  EXPECT_EQ(tilingConfig.getNumTilingLevels(), kMaxNumTilingLevels);
}

TEST_F(CPUTileSizeSelection, WithAllFields) {
  // Initialize Lowering Config
  SmallVector<IREE::CPU::TilingLevel> targets = {
      IREE::CPU::TilingLevel::DistributionTiles,
      IREE::CPU::CacheParallelTiles,
      IREE::CPU::CacheReductionTiles,
      IREE::CPU::VectorCommonParallelTiles,
      IREE::CPU::VectorReductionTiles,
      IREE::CPU::VectorInnerParallelTiles};
  initLoweringConfig(targets);

  // Create TilingConfig and check if the number of tiling levels match.
  TilingConfig tilingConfig(loweringConfig);

  // There are no re-mapping between the TilingConfig and the original config.
  EXPECT_EQ(tilingConfig.getNumTilingLevels(), targets.size());
  EXPECT_EQ(tilingConfig.getDistributionLevel(),
            IREE::CPU::TilingLevel::DistributionTiles);
  EXPECT_EQ(tilingConfig.getCacheParallelLevel(),
            IREE::CPU::CacheParallelTiles);
  EXPECT_EQ(tilingConfig.getCacheReductionLevel(),
            IREE::CPU::CacheReductionTiles);
  EXPECT_EQ(tilingConfig.getVectorCommonParallelLevel(),
            IREE::CPU::VectorCommonParallelTiles);
  EXPECT_EQ(tilingConfig.getVectorReductionLevel(),
            IREE::CPU::VectorReductionTiles);
  EXPECT_EQ(tilingConfig.getVectorInnerParallelLevel(),
            IREE::CPU::VectorInnerParallelTiles);
}

TEST_F(TileSizeSelection, getLevel_4_levels) {
  // 1. Initialize Lowering Config
  initLoweringConfig(/*numTilingLevels=*/4);

  // 2. Create TilingConfig and verify the actual tiling level numbers.
  TilingConfig tilingConfig(loweringConfig);
  EXPECT_EQ(tilingConfig.getVectorInnerParallelLevel(), 3);
  EXPECT_EQ(tilingConfig.getVectorReductionLevel(), 2);
  EXPECT_EQ(tilingConfig.getVectorCommonParallelLevel(), 1);
  EXPECT_EQ(tilingConfig.getDistributionLevel(), 0);
}

TEST_F(TileSizeSelection, getLevel_1_level) {
  // 1. Initialize Lowering Config
  initLoweringConfig(/*numTilingLevels=*/1);

  // 2. Create TilingConfig and verify the actual tiling level numbers.
  TilingConfig tilingConfig(loweringConfig);
  EXPECT_EQ(tilingConfig.getDistributionLevel(), 0);
}

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)

using TileSizeSelectionDeathTest = TileSizeSelection;

TEST_F(TileSizeSelectionDeathTest, getLevel_out_of_bounds) {
  // 1. Initialize Lowering Config
  initLoweringConfig(/*numTilingLevels=*/3);

  // 2. Create TilingConfig and verify that the "vector-inner-parallel" tiling
  // level does not exist (it's out of bounds).
  TilingConfig tilingConfig(loweringConfig);
  ASSERT_DEATH_IF_SUPPORTED(
      { tilingConfig.getVectorInnerParallelLevel(); },
      "Searching for unavailable tiling level");
}

#endif
} // namespace mlir::iree_compiler
