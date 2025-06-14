// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"

namespace mlir::iree_compiler::IREE::Encoding {
namespace {

class EncodingAttrsTest : public ::testing::Test {
protected:
  EncodingAttrsTest() {
    reg.insert<IREEEncodingDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }
  ~EncodingAttrsTest() override {}

  MLIRContext *getContext() { return &ctx; }

private:
  MLIRContext ctx;
  DialectRegistry reg;
};

TEST_F(EncodingAttrsTest, EncodingAttr) {
  MLIRContext *ctx = getContext();
  Builder builder(ctx);
  SmallVector<Type> elemTypes(3, builder.getF32Type());
  auto attr = cast<SerializableAttr>(EncodingAttr::get(
      ctx, /*operandIndex=*/0, EncodingOpType::matmul, elemTypes));
  EXPECT_FALSE(attr.isIdentityLayout());

  attr = cast<SerializableAttr>(attr.cloneWithLayouts(
      PadEncodingLayoutAttr::getIdentityAttr(ctx, /*rank=*/2)));
  EXPECT_TRUE(attr.isIdentityLayout());
}

TEST_F(EncodingAttrsTest, MatulKAttr) {
  MLIRContext *ctx = getContext();
  Builder builder(ctx);
  auto attr = cast<SerializableAttr>(MatmulKAttr::get(ctx, /*k_dims=*/{1}));
  EXPECT_FALSE(attr.isIdentityLayout());

  attr = cast<SerializableAttr>(attr.cloneWithLayouts(
      PadEncodingLayoutAttr::getIdentityAttr(ctx, /*rank=*/2)));
  EXPECT_TRUE(attr.isIdentityLayout());
}

TEST_F(EncodingAttrsTest, PadEncodingLayoutAttr) {
  MLIRContext *ctx = getContext();
  auto zeroPaddingAttr =
      PadEncodingLayoutAttr::getIdentityAttr(ctx, /*rank=*/2);
  EXPECT_TRUE(cast<SerializableAttr>(zeroPaddingAttr).isIdentityLayout());

  SmallVector<int64_t> paddings = {4, 2};
  auto nonZeroPaddingAttr = PadEncodingLayoutAttr::get(ctx, paddings);
  EXPECT_FALSE(cast<SerializableAttr>(nonZeroPaddingAttr).isIdentityLayout());
}

} // namespace
} // namespace mlir::iree_compiler::IREE::Encoding
