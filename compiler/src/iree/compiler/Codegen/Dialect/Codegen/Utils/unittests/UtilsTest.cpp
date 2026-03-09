// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"

namespace mlir::iree_compiler::IREE::Codegen {
namespace {

using testing::Optional;
using Dim = TileSwizzle::Dim;
using Kind = Dim::Kind;
using SymbolicMultiplier = Dim::SymbolicMultiplier;

TEST(TileSwizzle, RelationalOperator) {
  TileSwizzle swizzle1;
  swizzle1.permutation() = {1, 2, 0};
  TileSwizzle swizzle2;
  EXPECT_NE(swizzle1, swizzle2);
  swizzle2.permutation() = swizzle1.permutation();
  EXPECT_EQ(swizzle1, swizzle2);
  swizzle1.expandShape().push_back({Dim::crossThread(16)});
  swizzle2.expandShape().push_back({Dim::internal(16)});
  EXPECT_NE(swizzle1, swizzle2);
  swizzle2.expandShape()[0][0] =
      Dim::crossThread(swizzle2.expandShape()[0][0].size());
  EXPECT_EQ(swizzle1, swizzle2);
}

TEST(TileSwizzle, DimKindToString) {
  EXPECT_EQ(convertSwizzleKindToString(Kind::Internal), "Internal");
  EXPECT_EQ(convertSwizzleKindToString(Kind::CrossIntrinsic), "CrossIntrinsic");
  EXPECT_EQ(convertSwizzleKindToString(Kind::CrossThread), "CrossThread");
}

TEST(TileSwizzle, SymbolicMultiplierToString) {
  EXPECT_EQ(convertSymbolicMultiplierToString(SymbolicMultiplier::One), "One");
  EXPECT_EQ(convertSymbolicMultiplierToString(SymbolicMultiplier::ArmVscale),
            "ArmVscale");
  EXPECT_EQ(convertSymbolicMultiplierToString(
                SymbolicMultiplier::RiscvVlenIn128bitUnits),
            "RiscvVlenIn128bitUnits");
}

TEST(TileSwizzle, StringToDimKind) {
  std::optional<Kind> maybeKind;
  maybeKind = convertStringToSwizzleKind("Internal");
  EXPECT_THAT(maybeKind, Optional(Kind::Internal));
  maybeKind = convertStringToSwizzleKind("CrossIntrinsic");
  EXPECT_THAT(maybeKind, Optional(Kind::CrossIntrinsic));
  maybeKind = convertStringToSwizzleKind("CrossThread");
  EXPECT_THAT(maybeKind, Optional(Kind::CrossThread));
  maybeKind = convertStringToSwizzleKind("deadbeef");
  EXPECT_FALSE(maybeKind.has_value());
}

TEST(TileSwizzle, StringToSymbolicMultiplier) {
  std::optional<SymbolicMultiplier> maybeSymbolicMultiplier;
  maybeSymbolicMultiplier = convertStringToSymbolicMultiplier("One");
  EXPECT_THAT(maybeSymbolicMultiplier, Optional(SymbolicMultiplier::One));
  maybeSymbolicMultiplier = convertStringToSymbolicMultiplier("ArmVscale");
  EXPECT_THAT(maybeSymbolicMultiplier, Optional(SymbolicMultiplier::ArmVscale));
  maybeSymbolicMultiplier =
      convertStringToSymbolicMultiplier("RiscvVlenIn128bitUnits");
  EXPECT_THAT(maybeSymbolicMultiplier,
              Optional(SymbolicMultiplier::RiscvVlenIn128bitUnits));
  maybeSymbolicMultiplier = convertStringToSymbolicMultiplier("deadbeef");
  EXPECT_FALSE(maybeSymbolicMultiplier.has_value());
}

TEST(TileSwizzle, SerializationDeserialization) {
  using Dim = Dim;
  using SymbolicMultiplier = SymbolicMultiplier;
  TileSwizzle swizzle;
  swizzle.expandShape().push_back(
      {Dim::crossThread(4), Dim::internal(2, SymbolicMultiplier::ArmVscale)});
  swizzle.expandShape().push_back(
      {Dim::crossThread(16, /*distributionFactor=*/2), Dim::crossIntrinsic(4),
       Dim::internal(4)});
  SmallVector<int64_t> permutation = {3, 2, 4, 1, 0};
  swizzle.permutation() = permutation;

  MLIRContext ctx;
  DictionaryAttr dictAttr = serializeTileSwizzle(&ctx, swizzle);

  EXPECT_TRUE(dictAttr.contains("expandShape"));
  EXPECT_TRUE(dictAttr.contains("permutation"));

  // Verify if the sizes match. The check of values is done by the comparison
  // between deserialization result and the original struct.
  auto expandShapeArrayAttr =
      dyn_cast<ArrayAttr>(dictAttr.getNamed("expandShape")->getValue());
  EXPECT_EQ(expandShapeArrayAttr.size(), swizzle.expandShape().size());
  for (auto [expectedShape, actualShape] :
       llvm::zip_equal(swizzle.expandShape(),
                       expandShapeArrayAttr.getAsRange<ArrayAttr>())) {
    EXPECT_EQ(expectedShape.size(), actualShape.size());
  }

  SmallVector<int64_t> extractedPerm = extractFromIntegerArrayAttr<int64_t>(
      dictAttr.getNamed("permutation")->getValue());
  EXPECT_EQ(extractedPerm, swizzle.permutation());

  // Verify that deserialization+serialization roundtrip works.
  std::optional<TileSwizzle> deserializedSwizzle =
      deserializeTileSwizzle(dictAttr);
  EXPECT_THAT(deserializedSwizzle, Optional(swizzle));

  // Just because deserialization+serialization roundtrip works, does not mean
  // the values are preserved. Check them all.
  auto deserializedExpandShape = deserializedSwizzle.value().expandShape();
  EXPECT_EQ(deserializedExpandShape[0][0].kind(), Kind::CrossThread);
  EXPECT_EQ(deserializedExpandShape[0][0].size(), 4);
  EXPECT_EQ(deserializedExpandShape[0][1].kind(), Kind::Internal);
  EXPECT_EQ(deserializedExpandShape[0][1].size(), 2);
  EXPECT_EQ(deserializedExpandShape[0][1].symbolicMultiplier(),
            SymbolicMultiplier::ArmVscale);
  EXPECT_EQ(deserializedExpandShape[1][0].kind(), Kind::CrossThread);
  EXPECT_EQ(deserializedExpandShape[1][0].size(), 16);
  EXPECT_EQ(deserializedExpandShape[1][0].distributionFactor(), 2);
  EXPECT_EQ(deserializedExpandShape[1][1].kind(), Kind::CrossIntrinsic);
  EXPECT_EQ(deserializedExpandShape[1][1].size(), 4);
  EXPECT_EQ(deserializedExpandShape[1][2].kind(), Kind::Internal);
  EXPECT_EQ(deserializedExpandShape[1][2].size(), 4);
  EXPECT_EQ(deserializedSwizzle.value().permutation(), permutation);
}

TEST(TileSwizzle, DeserializationEdgeCases) {
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

TEST(MaterializeEncodingInfo, RelationalOperator) {
  MaterializeEncodingInfo info1;
  info1.innerDimsPos = {0, 1};
  info1.innerTileSizes = {16, 1};
  info1.outerDimsPerm = {0, 2, 1, 3};

  MaterializeEncodingInfo info2;
  info2.innerDimsPos = {1, 0};
  info2.innerTileSizes = {16, 1};
  info2.outerDimsPerm = {0, 2, 1, 3};

  EXPECT_EQ(info1, info1);
  EXPECT_EQ(info2, info2);
  EXPECT_NE(info1, info2);

  // They mismatch if one has a swizzle, but not the other.
  info2 = info1;
  info1.swizzle = TileSwizzle();
  EXPECT_NE(info1, info2);

  // They match because they all have an empty swizzle.
  info2.swizzle = TileSwizzle();
  EXPECT_EQ(info1, info2);

  // They mismatch if one has scalableTiles, but not the other.
  info1.scalableTiles = ScalableTileFlags();
  EXPECT_NE(info1, info2);

  // They match because they all have empty scalableTiles.
  info2.scalableTiles = ScalableTileFlags();
  EXPECT_EQ(info1, info2);
}

TEST(MaterializeEncodingInfo, Serialization) {
  MaterializeEncodingInfo info;
  info.innerDimsPos = {0, 1};
  info.innerTileSizes = {16, 1};
  info.outerDimsPerm = {0, 2, 1, 3};

  MLIRContext ctx;
  DictionaryAttr dictAttr = serializeEncodingInfo(&ctx, info);

  EXPECT_TRUE(dictAttr.contains("innerDimsPos"));
  EXPECT_TRUE(dictAttr.contains("innerTileSizes"));
  EXPECT_TRUE(dictAttr.contains("outerDimsPerm"));
  EXPECT_FALSE(dictAttr.contains("swizzle"));
  EXPECT_FALSE(dictAttr.contains("scalableTiles"));

  info.scalableTiles = {true, false};
  dictAttr = serializeEncodingInfo(&ctx, info);
  EXPECT_TRUE(dictAttr.contains("scalableTiles"));

  EXPECT_TRUE(isa<ArrayAttr>(dictAttr.getNamed("innerDimsPos")->getValue()));
  EXPECT_TRUE(isa<ArrayAttr>(dictAttr.getNamed("innerTileSizes")->getValue()));
  EXPECT_TRUE(isa<ArrayAttr>(dictAttr.getNamed("outerDimsPerm")->getValue()));
  EXPECT_TRUE(isa<ArrayAttr>(dictAttr.getNamed("scalableTiles")->getValue()));

  auto extractedInnerDimsPos = extractFromIntegerArrayAttr<int64_t>(
      dictAttr.getNamed("innerDimsPos")->getValue());
  EXPECT_EQ(extractedInnerDimsPos, info.innerDimsPos);
  auto extractedInnerTileSizes = extractFromIntegerArrayAttr<int64_t>(
      dictAttr.getNamed("innerTileSizes")->getValue());
  EXPECT_EQ(extractedInnerTileSizes, info.innerTileSizes);
  auto extractedOuterDimsPerm = extractFromIntegerArrayAttr<int64_t>(
      dictAttr.getNamed("outerDimsPerm")->getValue());
  EXPECT_EQ(extractedOuterDimsPerm, info.outerDimsPerm);
  auto extractedScalableTiles = llvm::map_to_vector(
      cast<ArrayAttr>(dictAttr.getNamed("scalableTiles")->getValue()),
      [](Attribute a) { return cast<BoolAttr>(a).getValue(); });
  EXPECT_EQ(extractedScalableTiles, info.scalableTiles);

  std::optional<MaterializeEncodingInfo> deserializedInfo =
      deserializeEncodingInfo(dictAttr);
  EXPECT_THAT(deserializedInfo, Optional(info));
}

TEST(MaterializeEncodingInfo, Deserialization) {
  MLIRContext ctx;
  Builder b(&ctx);

  auto emptyDictAttr = b.getDictionaryAttr({});
  EXPECT_FALSE(deserializeEncodingInfo(emptyDictAttr).has_value());

  SmallVector<NamedAttribute> items;
  items.emplace_back(b.getStringAttr("innerDimsPos"),
                     b.getI64ArrayAttr({0, 1}));
  EXPECT_FALSE(deserializeEncodingInfo(b.getDictionaryAttr(items)).has_value());

  items.emplace_back(b.getStringAttr("innerTileSizes"),
                     b.getI64ArrayAttr({16, 1}));
  EXPECT_FALSE(deserializeEncodingInfo(b.getDictionaryAttr(items)).has_value());

  items.emplace_back(b.getStringAttr("outerDimsPerm"),
                     b.getI64ArrayAttr({0, 2, 1, 3}));
  EXPECT_TRUE(deserializeEncodingInfo(b.getDictionaryAttr(items)).has_value());

  // If the swizzle presents, it needs to be deserializable to TileSwizzle.
  items.emplace_back(b.getStringAttr("swizzle"), b.getUnitAttr());
  EXPECT_FALSE(deserializeEncodingInfo(b.getDictionaryAttr(items)).has_value());

  TileSwizzle swizzle;
  items.back().setValue(serializeTileSwizzle(&ctx, swizzle));
  EXPECT_TRUE(deserializeEncodingInfo(b.getDictionaryAttr(items)).has_value());

  items.emplace_back(b.getStringAttr("scalableTiles"),
                     b.getBoolArrayAttr({true, false}));
  EXPECT_TRUE(deserializeEncodingInfo(b.getDictionaryAttr(items)).has_value());
}

TEST(MaterializeEncodingInfo, IdentityLayout) {
  MaterializeEncodingInfo info;
  EXPECT_TRUE(isIdentityLayout(info));
  info.swizzle = TileSwizzle();
  EXPECT_FALSE(isIdentityLayout(info));
}

} // namespace
} // namespace mlir::iree_compiler::IREE::Codegen
