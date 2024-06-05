// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

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
        IREE::Codegen::LoweringConfigAttr::get(&ctx, newTilingLevels,
                                               /*nativeVectorSize=*/4);
  }

  ~TileSizeSelection() override {}

  MLIRContext ctx;
  DialectRegistry reg;
  IREE::Codegen::LoweringConfigAttr loweringConfig;
};

TEST_F(TileSizeSelection, NumTilingLevels) {
  const unsigned kMaxNumTilingLevels = 7;

  // 1. Initialize Lowering Config
  initLoweringConfig(kMaxNumTilingLevels);

  // 2. Create TilingConfig and check if the number of tiling levels match.
  TilingConfig tilingConfig(loweringConfig);
  EXPECT_EQ(tilingConfig.getNumTilingLevels(), kMaxNumTilingLevels);
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
