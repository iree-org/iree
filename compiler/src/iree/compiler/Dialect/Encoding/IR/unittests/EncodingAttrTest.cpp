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

  attr = cast<SerializableAttr>(
      attr.cloneWithLayouts(PaddingAttr::getIdentityAttr(ctx, /*rank=*/2)));
  EXPECT_TRUE(attr.isIdentityLayout());
}

TEST_F(EncodingAttrsTest, PaddingAttr) {
  MLIRContext *ctx = getContext();
  auto zeroPaddingAttr = PaddingAttr::getIdentityAttr(ctx, /*rank=*/2);
  EXPECT_TRUE(cast<SerializableAttr>(zeroPaddingAttr).isIdentityLayout());

  SmallVector<int64_t> paddings = {4, 2};
  auto nonZeroPaddingAttr = PaddingAttr::get(ctx, paddings);
  EXPECT_FALSE(cast<SerializableAttr>(nonZeroPaddingAttr).isIdentityLayout());
}

TEST_F(EncodingAttrsTest, PackedStorageCapabilities) {
  auto packedStorage = PackedStorageAttr::get(getContext());
  auto encoding = cast<SerializableAttr>(packedStorage);
  EXPECT_TRUE(encoding.canDropWithoutTensorMaterialization());
  EXPECT_TRUE(encoding.canSkipBufferViewAssertions());
  EXPECT_TRUE(encoding.isInvariantUnderDimCollapse());
  EXPECT_FALSE(encoding.isIdentityLayout());
  EXPECT_EQ(encoding.getCollapsedEncoding({2, 4}, {8}), packedStorage);
  EXPECT_EQ(encoding.getCollapsedEncoding({ShapedType::kDynamic, 4},
                                         {ShapedType::kDynamic}),
            packedStorage);
}

TEST_F(EncodingAttrsTest, ConservativeCapabilities) {
  MLIRContext *ctx = getContext();
  Builder builder(ctx);
  SmallVector<Type> elemTypes(3, builder.getF32Type());
  SmallVector<Attribute> encodings = {
      TestingAttr::get(ctx, ArrayAttr{}, Type{}), IdentityAttr::get(ctx),
      PaddingAttr::get(ctx, ArrayRef<int64_t>{4, 2}),
      LayoutAttr::get(ctx, builder.getArrayAttr({PackedStorageAttr::get(ctx)})),
      EncodingAttr::get(ctx, /*operandIndex=*/0, EncodingOpType::matmul,
                        elemTypes)};
  for (Attribute attr : encodings) {
    auto encoding = cast<SerializableAttr>(attr);
    EXPECT_FALSE(encoding.canDropWithoutTensorMaterialization());
    EXPECT_FALSE(encoding.canSkipBufferViewAssertions());
    EXPECT_FALSE(encoding.isInvariantUnderDimCollapse());
  }
}

} // namespace
} // namespace mlir::iree_compiler::IREE::Encoding
