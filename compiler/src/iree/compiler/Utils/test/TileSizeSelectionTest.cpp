// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

using namespace mlir::iree_compiler;
using namespace llvm;
using namespace mlir;
using namespace testing;

TEST(TileSizeSelection, Basic) {
  // 0. Init MLIR
  // TODO: Move this to a test fixture
  mlir::MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<IREE::Codegen::IREECodegenDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  // 1. Create Lowering Config
  const unsigned kMaxNumTilingLevels = 7;
  SmallVector<IREE::Codegen::LoweringConfigTilingLevelAttr> newTilingLevelsList;
  for (size_t i = 0; i < kMaxNumTilingLevels; i++) {
    SmallVector<int64_t> sizes;
    SmallVector<bool> scalableFlags;

    llvm::SmallVector<int64_t> interchange = {};

    auto newLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
        &ctx, sizes, ArrayRef<int64_t>{}, scalableFlags);
    newTilingLevelsList.push_back(newLevel);
  }

  auto newTilingLevels = IREE::Codegen::LoweringConfigTilingLevelsAttr::get(
      &ctx, newTilingLevelsList);
  IREE::Codegen::LoweringConfigAttr loweringConfig =
      IREE::Codegen::LoweringConfigAttr::get(&ctx, newTilingLevels,
                                             /*nativeVectorSize=*/4);
  // 2. Create Tiling Config
  TilingConfig tilingConfig(loweringConfig);

  // 3. Test
  EXPECT_EQ(tilingConfig.getNumTilingLevels(), kMaxNumTilingLevels);
}
