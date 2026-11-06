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

static int64_t evaluateOffset(AffineMap map, ArrayRef<int64_t> coordinates,
                              ArrayRef<int64_t> dynamicDims = {}) {
  Builder builder(map.getContext());
  SmallVector<Attribute> operands;
  for (int64_t coordinate : coordinates) {
    operands.push_back(builder.getIndexAttr(coordinate));
  }
  for (int64_t dim : dynamicDims) {
    operands.push_back(builder.getIndexAttr(dim));
  }
  SmallVector<Attribute> results;
  if (failed(map.constantFold(operands, results))) {
    ADD_FAILURE() << "offset did not fold";
    return -1;
  }
  return cast<IntegerAttr>(results.front()).getInt();
}

TEST_F(EncodingAttrsTest, PackedAndUnpackedOffsets) {
  Builder builder(getContext());
  auto packed = PackedStorageAttr::get(getContext());
  auto unpacked = UnpackedStorageAttr::get(getContext());
  for (int64_t bits : {1, 2, 4, 8, 16, 32, 64}) {
    Type elementType = builder.getIntegerType(bits);
    auto type = RankedTensorType::get({2, 16}, elementType);
    AffineMap packedMap = packed.getStorageElementOffsetMap(type);
    AffineMap unpackedMap = unpacked.getStorageElementOffsetMap(type);
    EXPECT_EQ(packed.getStorageElementType(elementType), elementType);
    EXPECT_EQ(unpacked.getStorageElementType(elementType),
              bits < 8 ? builder.getI8Type() : elementType);
    for (int64_t column : {0, 1, 7, 8, 15}) {
      EXPECT_EQ(evaluateOffset(packedMap, {1, column}),
                (16 + column) * bits / 8);
      EXPECT_EQ(evaluateOffset(unpackedMap, {1, column}),
                (16 + column) * (bits < 8 ? 1 : bits / 8));
    }
  }
  EXPECT_EQ(unpacked.getStorageElementType(builder.getIntegerType(3)),
            builder.getI8Type());
  EXPECT_EQ(unpacked.getStorageElementType(builder.getIntegerType(9)),
            builder.getI16Type());
  EXPECT_EQ(unpacked.getStorageElementType(builder.getIntegerType(24)),
            builder.getI32Type());
  EXPECT_EQ(unpacked.getStorageElementType(builder.getIntegerType(33)),
            builder.getI64Type());
  auto scalar = RankedTensorType::get({}, builder.getIntegerType(5));
  EXPECT_EQ(evaluateOffset(unpacked.getStorageElementOffsetMap(scalar), {}), 0);
}

TEST_F(EncodingAttrsTest, FloatingPointStorageOffsets) {
  Builder builder(getContext());
  auto packed = PackedStorageAttr::get(getContext());
  auto unpacked = UnpackedStorageAttr::get(getContext());
  SmallVector<Type> types = {
      Float4E2M1FNType::get(getContext()), Float8E4M3FNType::get(getContext()),
      builder.getF16Type(), builder.getF32Type(), builder.getF64Type()};
  for (Type elementType : types) {
    auto type = RankedTensorType::get({2, 4}, elementType);
    int64_t bits = elementType.getIntOrFloatBitWidth();
    EXPECT_EQ(packed.getStorageElementType(elementType), elementType);
    EXPECT_EQ(unpacked.getStorageElementType(elementType),
              bits < 8 ? builder.getI8Type() : elementType);
    EXPECT_EQ(evaluateOffset(packed.getStorageElementOffsetMap(type), {1, 1}),
              5 * bits / 8);
    EXPECT_EQ(evaluateOffset(unpacked.getStorageElementOffsetMap(type), {1, 1}),
              5 * (bits < 8 ? 1 : bits / 8));
  }
  auto fp6Type = Float6E3M2FNType::get(getContext());
  EXPECT_EQ(unpacked.getStorageElementType(fp6Type), builder.getI8Type());
  auto type = RankedTensorType::get({2, 4}, fp6Type);
  EXPECT_EQ(evaluateOffset(unpacked.getStorageElementOffsetMap(type), {1, 1}),
            5);
}

TEST_F(EncodingAttrsTest, PaddingOffsetUsesPhysicalStrides) {
  Builder builder(getContext());
  auto padding = PaddingAttr::get(getContext(), ArrayRef<int64_t>{0, 1});
  auto type = RankedTensorType::get({2, 3}, builder.getI32Type());
  EXPECT_EQ(evaluateOffset(padding.getStorageElementOffsetMap(type), {1, 0}),
            16);
  auto dynamicType = RankedTensorType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, builder.getI32Type());
  EXPECT_EQ(evaluateOffset(padding.getStorageElementOffsetMap(dynamicType),
                           {1, 2}, {2, 3}),
            24);
  auto unresolved = PaddingAttr::get(
      getContext(), ArrayRef<int64_t>{0, ShapedType::kDynamic});
  EXPECT_FALSE(unresolved.getStorageElementOffsetMap(type));
}

TEST_F(EncodingAttrsTest, LayoutOffsetsMustAgree) {
  Builder builder(getContext());
  auto identity = IdentityAttr::get(getContext());
  auto padding = PaddingAttr::get(getContext(), ArrayRef<int64_t>{0, 1});
  auto zeroPadding = PaddingAttr::get(getContext(), ArrayRef<int64_t>{0, 0});
  auto type = RankedTensorType::get({2, 3}, builder.getI32Type());
  auto singleton =
      LayoutAttr::get(getContext(), builder.getArrayAttr({padding}));
  EXPECT_EQ(evaluateOffset(singleton.getStorageElementOffsetMap(type), {1, 0}),
            16);
  auto agreeing = LayoutAttr::get(
      getContext(), builder.getArrayAttr({identity, zeroPadding}));
  EXPECT_EQ(evaluateOffset(agreeing.getStorageElementOffsetMap(type), {1, 0}),
            12);
  auto conflicting =
      LayoutAttr::get(getContext(), builder.getArrayAttr({identity, padding}));
  EXPECT_FALSE(conflicting.getStorageElementOffsetMap(type));
  auto unresolved = cast<SerializableAttr>(
      TestingAttr::get(getContext(), ArrayAttr{}, Type{}));
  EXPECT_FALSE(unresolved.getStorageElementOffsetMap(type));
}

TEST_F(EncodingAttrsTest, UnavailableOffsetDoesNotEmitIR) {
  OpBuilder builder(getContext());
  Block block;
  builder.setInsertionPointToEnd(&block);
  auto type = RankedTensorType::get({2, 3}, builder.getI32Type());
  auto encoding = cast<SerializableAttr>(PaddingAttr::get(
      getContext(), ArrayRef<int64_t>{0, ShapedType::kDynamic}));
  EXPECT_FALSE(encoding.calculateStorageElementOffsetInBytes(
      builder.getUnknownLoc(), builder, type, {}, {}));
  EXPECT_TRUE(block.empty());
}

TEST_F(EncodingAttrsTest, UnpackedStorageCapabilities) {
  Builder builder(getContext());
  auto unpacked = UnpackedStorageAttr::get(getContext());
  auto encoding = cast<SerializableAttr>(unpacked);
  EXPECT_TRUE(encoding.canDropWithoutTensorMaterialization());
  EXPECT_FALSE(encoding.canSkipBufferViewAssertions());
  EXPECT_TRUE(encoding.isInvariantUnderDimCollapse());
  EXPECT_EQ(encoding.getCollapsedEncoding({2, 4}, {8}), unpacked);
  EXPECT_TRUE(failed(encoding.convertForBitcast({4}, builder.getIntegerType(4),
                                                builder.getI8Type())));
  auto sameType = encoding.convertForBitcast({4}, builder.getIntegerType(4),
                                             builder.getIntegerType(4));
  ASSERT_TRUE(succeeded(sameType));
  EXPECT_EQ(sameType->encoding, unpacked);
}

} // namespace
} // namespace mlir::iree_compiler::IREE::Encoding
