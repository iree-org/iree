// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"

namespace mlir::iree_compiler::IREE::Codegen {
namespace {

using Kind = TileSwizzle::Dim::Kind;

TEST(TileSwizzle, RelationalOperator) {
  TileSwizzle swizzle1;
  swizzle1.permutation = {1, 2, 0};
  TileSwizzle swizzle2;
  EXPECT_NE(swizzle1, swizzle2);
  swizzle2.permutation = swizzle1.permutation;
  EXPECT_EQ(swizzle1, swizzle2);
  swizzle1.expandShape.push_back({TileSwizzle::Dim(Kind::CrossThread, 16)});
  swizzle2.expandShape.push_back({TileSwizzle::Dim(Kind::Internal, 16)});
  EXPECT_NE(swizzle1, swizzle2);
  swizzle2.expandShape[0][0].kind = Kind::CrossThread;
  EXPECT_EQ(swizzle1, swizzle2);
}

TEST(TileSwizzle, Serialization) {
  TileSwizzle swizzle;
  swizzle.expandShape.push_back({TileSwizzle::Dim(Kind::CrossThread, 16)});
  swizzle.expandShape.push_back({TileSwizzle::Dim(Kind::CrossIntrinsic, 4),
                                 TileSwizzle::Dim(Kind::Internal, 4)});
  swizzle.permutation = {1, 2, 0};

  MLIRContext ctx;
  DictionaryAttr dictAttr = serializeTileSwizzle(&ctx, swizzle);

  EXPECT_TRUE(dictAttr.contains("expandShape"));
  EXPECT_TRUE(dictAttr.contains("permutation"));

  auto expandShapeArrayAttr =
      dyn_cast<ArrayAttr>(dictAttr.getNamed("expandShape")->getValue());
  EXPECT_EQ(expandShapeArrayAttr.size(), swizzle.expandShape.size());
  for (auto [expectedShape, actualShape] : llvm::zip_equal(
           swizzle.expandShape, expandShapeArrayAttr.getAsRange<ArrayAttr>())) {
    EXPECT_EQ(expectedShape.size(), actualShape.size());
    for (auto [expectedDim, actualDim] :
         llvm::zip_equal(expectedShape, actualShape)) {
      SmallVector<int16_t> dimValue =
          extractFromIntegerArrayAttr<int16_t>(actualDim);
      EXPECT_EQ(expectedDim.kind, static_cast<Kind>(dimValue[0]));
      EXPECT_EQ(expectedDim.size, dimValue[1]);
    }
  }

  SmallVector<int64_t> extractedPerm = extractFromIntegerArrayAttr<int64_t>(
      dictAttr.getNamed("permutation")->getValue());
  EXPECT_EQ(extractedPerm, swizzle.permutation);

  std::optional<TileSwizzle> deserializedSwizzle =
      deserializeTileSwizzle(dictAttr);
  EXPECT_TRUE(deserializedSwizzle.has_value());
  EXPECT_EQ(swizzle, deserializedSwizzle);
}

TEST(TileSwizzle, Deserialization) {
  MLIRContext ctx;
  Builder b(&ctx);

  auto emptyDictAttr = b.getDictionaryAttr({});
  EXPECT_FALSE(deserializeTileSwizzle(emptyDictAttr).has_value());

  SmallVector<NamedAttribute> items;
  items.emplace_back(b.getStringAttr("expandShape"), b.getArrayAttr({}));
  EXPECT_FALSE(deserializeTileSwizzle(b.getDictionaryAttr(items)).has_value());

  items.emplace_back(b.getStringAttr("permutation"), b.getArrayAttr({}));
  EXPECT_TRUE(deserializeTileSwizzle(b.getDictionaryAttr(items)).has_value());

  items.back().setValue(b.getUnitAttr());
  EXPECT_FALSE(deserializeTileSwizzle(b.getDictionaryAttr(items)).has_value());
}

} // namespace
} // namespace mlir::iree_compiler::IREE::Codegen
