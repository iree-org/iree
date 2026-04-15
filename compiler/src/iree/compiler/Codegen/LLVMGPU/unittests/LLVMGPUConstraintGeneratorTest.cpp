// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUConstraintGenerator.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir::iree_compiler {
namespace {

class GetCompatibleMMAAttrsTest : public ::testing::Test {
protected:
  GetCompatibleMMAAttrsTest() {
    DialectRegistry reg;
    reg.insert<IREE::GPU::IREEGPUDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  MLIRContext *getContext() { return &ctx; }

  IREE::GPU::TargetAttr getGfx942Target() {
    return IREE::GPU::getHIPTargetDetails("gfx942", "", &ctx);
  }

  IREE::GPU::TargetAttr getGfx1201Target() {
    return IREE::GPU::getHIPTargetDetails("gfx1201", "", &ctx);
  }

private:
  MLIRContext ctx;
};

TEST_F(GetCompatibleMMAAttrsTest, F16InputF32AccOnGfx942) {
  MLIRContext *ctx = getContext();
  auto target = getGfx942Target();
  ASSERT_TRUE(target);

  Type f16 = Float16Type::get(ctx);
  Type f32 = Float32Type::get(ctx);
  auto result = getCompatibleMMAAttrs(target, f16, f16, f32, ctx);
  EXPECT_FALSE(result.empty());

  for (Attribute attr : result) {
    auto mma = dyn_cast<IREE::GPU::MmaInterfaceAttr>(attr);
    ASSERT_TRUE(!!mma);
    auto [aType, bType, cType] = mma.getABCElementTypes();
    EXPECT_EQ(aType, f16);
    EXPECT_EQ(bType, f16);
    EXPECT_EQ(cType, f32);
  }
}

TEST_F(GetCompatibleMMAAttrsTest, IncludesVirtualMMA) {
  MLIRContext *ctx = getContext();
  auto target = getGfx942Target();
  ASSERT_TRUE(target);

  Type f16 = Float16Type::get(ctx);
  Type f32 = Float32Type::get(ctx);
  auto withoutVirtual = getCompatibleMMAAttrs(target, f16, f16, f32, ctx);
  auto withVirtual =
      getCompatibleMMAAttrs(target, f16, f16, f32, ctx, /*includeVirtual=*/true);
  EXPECT_GE(withVirtual.size(), withoutVirtual.size());
}

TEST_F(GetCompatibleMMAAttrsTest, I8InputI32AccOnRDNA4) {
  MLIRContext *ctx = getContext();
  auto target = getGfx1201Target();
  ASSERT_TRUE(target);

  Type i8 = IntegerType::get(ctx, 8);
  Type i32 = IntegerType::get(ctx, 32);
  auto result = getCompatibleMMAAttrs(target, i8, i8, i32, ctx);
  EXPECT_FALSE(result.empty());

  for (Attribute attr : result) {
    auto mma = dyn_cast<IREE::GPU::MmaInterfaceAttr>(attr);
    ASSERT_TRUE(!!mma);
    auto [aType, bType, cType] = mma.getABCElementTypes();
    EXPECT_EQ(aType, i8);
    EXPECT_EQ(bType, i8);
  }
}

TEST_F(GetCompatibleMMAAttrsTest, IncompatibleTypesReturnEmpty) {
  MLIRContext *ctx = getContext();
  auto target = getGfx942Target();
  ASSERT_TRUE(target);

  Type i1 = IntegerType::get(ctx, 1);
  auto result = getCompatibleMMAAttrs(target, i1, i1, i1, ctx);
  EXPECT_TRUE(result.empty());
}

TEST_F(GetCompatibleMMAAttrsTest, NoDuplicates) {
  MLIRContext *ctx = getContext();
  auto target = getGfx942Target();
  ASSERT_TRUE(target);

  Type f16 = Float16Type::get(ctx);
  Type f32 = Float32Type::get(ctx);
  auto result =
      getCompatibleMMAAttrs(target, f16, f16, f32, ctx, /*includeVirtual=*/true);
  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = i + 1; j < result.size(); ++j) {
      EXPECT_NE(result[i], result[j]);
    }
  }
}

} // namespace
} // namespace mlir::iree_compiler