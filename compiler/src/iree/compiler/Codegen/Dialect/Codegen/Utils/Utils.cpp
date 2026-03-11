// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-codegen-dialect-codegen-utils"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// Relational operator and IOstream implementations for Layout Structs.
//===----------------------------------------------------------------------===//

bool operator==(TileSwizzle::Dim lhs, TileSwizzle::Dim rhs) {
  return !memcmp(&lhs, &rhs, sizeof lhs);
}

bool operator!=(TileSwizzle::Dim lhs, TileSwizzle::Dim rhs) {
  return !(lhs == rhs);
}

bool operator==(const TileSwizzle &lhs, const TileSwizzle &rhs) {
  return lhs.expandShape() == rhs.expandShape() &&
         lhs.permutation() == rhs.permutation();
}

bool operator!=(const TileSwizzle &lhs, const TileSwizzle &rhs) {
  return !(lhs == rhs);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              TileSwizzle::Dim::Kind kind) {
  return os << convertSwizzleKindToString(kind);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, TileSwizzle::Dim dim) {
  os << dim.size();
  if (dim.kind() == TileSwizzle::Dim::Kind::CrossThread &&
      dim.distributionFactor() != 1) {
    os << "*" << dim.distributionFactor();
  }
  if (dim.kind() == TileSwizzle::Dim::Kind::Internal &&
      dim.symbolicMultiplier() != TileSwizzle::Dim::SymbolicMultiplier::One) {
    os << "*" << convertSymbolicMultiplierToString(dim.symbolicMultiplier());
  }
  return os << "(" << dim.kind() << ")";
}

static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const TileSwizzle::ExpandShapeDimVectorType &expandShapeDimVector) {
  return os << llvm::interleaved_array(expandShapeDimVector);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TileSwizzle &swizzle) {
  return os << "{expandShape = "
            << llvm::interleaved_array(swizzle.expandShape())
            << ", permutation = "
            << llvm::interleaved_array(swizzle.permutation()) << "}";
}

static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const ScalableTileFlags &scalableTileFlags) {
  if (scalableTileFlags.empty()) {
    return os;
  }
  os << "scalableTiles = [";
  for (unsigned i = 0; i < scalableTileFlags.size(); ++i) {
    os << (scalableTileFlags[i] ? "true" : "false");
    if (i + 1 < scalableTileFlags.size()) {
      os << ", ";
    }
  }
  return os;
}

bool operator==(const MaterializeEncodingInfo &lhs,
                const MaterializeEncodingInfo &rhs) {
  return lhs.innerDimsPos == rhs.innerDimsPos &&
         lhs.innerTileSizes == rhs.innerTileSizes &&
         lhs.outerDimsPerm == rhs.outerDimsPerm && lhs.swizzle == rhs.swizzle &&
         lhs.scalableTiles == rhs.scalableTiles;
}

bool operator!=(const MaterializeEncodingInfo &lhs,
                const MaterializeEncodingInfo &rhs) {
  return !(lhs == rhs);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const MaterializeEncodingInfo &encodingInfo) {
  os << "{innerDimsPos = [" << llvm::interleaved(encodingInfo.innerDimsPos)
     << "], innerTileSizes = ["
     << llvm::interleaved(encodingInfo.innerTileSizes) << "], outerDimsPerm = ["
     << llvm::interleaved(encodingInfo.outerDimsPerm);

  if (encodingInfo.swizzle) {
    os << "], swizzle = " << encodingInfo.swizzle.value();
  }
  if (encodingInfo.scalableTiles) {
    os << "], " << encodingInfo.scalableTiles.value();
  }
  os << "]}";
  return os;
}

//===----------------------------------------------------------------------===//
// Layout Utilities.
//===----------------------------------------------------------------------===//

std::string convertSwizzleKindToString(TileSwizzle::Dim::Kind kind) {
  switch (kind) {
  case TileSwizzle::Dim::Kind::Internal:
    return "Internal";
  case TileSwizzle::Dim::Kind::CrossThread:
    return "CrossThread";
  case TileSwizzle::Dim::Kind::CrossIntrinsic:
    return "CrossIntrinsic";
  default:
    assert(false && "unhandled enum value");
  }
  return "";
}

std::optional<TileSwizzle::Dim::Kind>
convertStringToSwizzleKind(StringRef str) {
  if (str == "Internal") {
    return TileSwizzle::Dim::Kind::Internal;
  }
  if (str == "CrossThread") {
    return TileSwizzle::Dim::Kind::CrossThread;
  }
  if (str == "CrossIntrinsic") {
    return TileSwizzle::Dim::Kind::CrossIntrinsic;
  }
  return std::nullopt;
}

std::string convertSymbolicMultiplierToString(
    TileSwizzle::Dim::SymbolicMultiplier symbolicMultiplier) {
  switch (symbolicMultiplier) {
  case TileSwizzle::Dim::SymbolicMultiplier::One:
    return "One";
  case TileSwizzle::Dim::SymbolicMultiplier::ArmSveVLIn128bitUnits:
    return "ArmSveVLIn128bitUnits";
  case TileSwizzle::Dim::SymbolicMultiplier::RiscvVlenIn64bitUnits:
    return "RiscvVlenIn64bitUnits";
  }
  assert(false && "unhandled enum value");
  return "";
}

std::optional<TileSwizzle::Dim::SymbolicMultiplier>
convertStringToSymbolicMultiplier(StringRef str) {
  if (str == "One") {
    return TileSwizzle::Dim::SymbolicMultiplier::One;
  }
  if (str == "ArmSveVLIn128bitUnits") {
    return TileSwizzle::Dim::SymbolicMultiplier::ArmSveVLIn128bitUnits;
  }
  if (str == "RiscvVlenIn64bitUnits") {
    return TileSwizzle::Dim::SymbolicMultiplier::RiscvVlenIn64bitUnits;
  }
  return std::nullopt;
}

static ArrayAttr swizzleDimToArrayAttr(MLIRContext *ctx, TileSwizzle::Dim dim) {
  Builder b(ctx);
  SmallVector<Attribute> attrs;
  attrs.push_back(b.getStringAttr(convertSwizzleKindToString(dim.kind())));
  attrs.push_back(b.getI16IntegerAttr(dim.size()));
  if (dim.kind() == TileSwizzle::Dim::Kind::CrossThread) {
    attrs.push_back(b.getI8IntegerAttr(dim.distributionFactor()));
  }
  if (dim.kind() == TileSwizzle::Dim::Kind::Internal) {
    attrs.push_back(b.getStringAttr(
        convertSymbolicMultiplierToString(dim.symbolicMultiplier())));
  }
  return b.getArrayAttr(attrs);
}

static std::optional<TileSwizzle::Dim> arrayAttrToSwizzleDim(Attribute attr) {
  auto arrayAttr = dyn_cast<ArrayAttr>(attr);
  if (!arrayAttr) {
    return std::nullopt;
  }
  ArrayRef<Attribute> attrs = arrayAttr.getValue();
  if (attrs.size() < 2) {
    return std::nullopt;
  }
  auto kindAttr = dyn_cast<StringAttr>(attrs[0]);
  auto sizeAttr = dyn_cast<IntegerAttr>(attrs[1]);
  if (!kindAttr || !sizeAttr) {
    return std::nullopt;
  }
  std::optional<TileSwizzle::Dim::Kind> maybeKind =
      convertStringToSwizzleKind(kindAttr.getValue());
  if (!maybeKind) {
    return std::nullopt;
  }
  const int64_t size = sizeAttr.getInt();
  if (size <= 0 || size > std::numeric_limits<int16_t>::max()) {
    return std::nullopt;
  }
  if (maybeKind.value() == TileSwizzle::Dim::Kind::Internal) {
    TileSwizzle::Dim::SymbolicMultiplier symbolicMultiplier =
        TileSwizzle::Dim::SymbolicMultiplier::One;
    if (attrs.size() > 2) {
      auto symbolicMultiplierAttr = dyn_cast<StringAttr>(attrs[2]);
      if (!symbolicMultiplierAttr) {
        return std::nullopt;
      }
      if (auto maybeSymbolicMultiplier = convertStringToSymbolicMultiplier(
              symbolicMultiplierAttr.getValue())) {
        symbolicMultiplier = maybeSymbolicMultiplier.value();
      }
    }
    return TileSwizzle::Dim::internal(size, symbolicMultiplier);
  }
  if (maybeKind.value() == TileSwizzle::Dim::Kind::CrossIntrinsic) {
    return TileSwizzle::Dim::crossIntrinsic(size);
  }
  if (maybeKind.value() == TileSwizzle::Dim::Kind::CrossThread) {
    int64_t distributionFactor = 1;
    if (attrs.size() > 2) {
      auto distributionFactorAttr = dyn_cast<IntegerAttr>(attrs[2]);
      if (!distributionFactorAttr) {
        return std::nullopt;
      }
      distributionFactor = distributionFactorAttr.getInt();
      if (distributionFactor <= 0 ||
          distributionFactor > std::numeric_limits<int8_t>::max()) {
        return std::nullopt;
      }
    }
    return TileSwizzle::Dim::crossThread(size, distributionFactor);
  }
  return std::nullopt;
}

DictionaryAttr serializeTileSwizzle(MLIRContext *ctx,
                                    const TileSwizzle &swizzle) {
  Builder b(ctx);
  SmallVector<NamedAttribute> items;

  SmallVector<Attribute> expandShape;
  for (auto expandConfig : swizzle.expandShape()) {
    Attribute expandAttr = b.getArrayAttr(
        llvm::map_to_vector(expandConfig, [&](TileSwizzle::Dim dim) {
          return cast<Attribute>(swizzleDimToArrayAttr(ctx, dim));
        }));
    expandShape.push_back(expandAttr);
  }

  items.emplace_back(b.getStringAttr("expandShape"),
                     b.getArrayAttr(expandShape));
  items.emplace_back(b.getStringAttr("permutation"),
                     b.getI64ArrayAttr(swizzle.permutation()));

  return b.getDictionaryAttr(items);
}

std::optional<TileSwizzle> deserializeTileSwizzle(DictionaryAttr attr) {
  TileSwizzle swizzle;

  auto expandShapeAttr = attr.getNamed("expandShape");
  if (!expandShapeAttr) {
    return std::nullopt;
  }
  auto expandShapeArrayAttr = dyn_cast<ArrayAttr>(expandShapeAttr->getValue());
  if (!expandShapeArrayAttr) {
    return std::nullopt;
  }

  for (auto expandConfig : expandShapeArrayAttr.getAsRange<ArrayAttr>()) {
    TileSwizzle::ExpandShapeDimVectorType vec;
    for (auto dimAttr : expandConfig.getAsRange<ArrayAttr>()) {
      auto maybeDim = arrayAttrToSwizzleDim(dimAttr);
      if (!maybeDim) {
        return std::nullopt;
      }
      vec.push_back(maybeDim.value());
    }
    swizzle.expandShape().push_back(vec);
  }

  auto permAttr = attr.getNamed("permutation");
  if (!permAttr || !isa<ArrayAttr>(permAttr->getValue())) {
    return std::nullopt;
  }
  swizzle.permutation() =
      extractFromIntegerArrayAttr<int64_t>(permAttr->getValue());

  return swizzle;
}

DictionaryAttr serializeEncodingInfo(MLIRContext *ctx,
                                     const MaterializeEncodingInfo &info) {
  Builder b(ctx);
  SmallVector<NamedAttribute> items;
  items.emplace_back(b.getStringAttr("innerDimsPos"),
                     b.getI64ArrayAttr(info.innerDimsPos));
  items.emplace_back(b.getStringAttr("innerTileSizes"),
                     b.getI64ArrayAttr(info.innerTileSizes));
  items.emplace_back(b.getStringAttr("outerDimsPerm"),
                     b.getI64ArrayAttr(info.outerDimsPerm));
  if (info.swizzle) {
    items.emplace_back(b.getStringAttr("swizzle"),
                       serializeTileSwizzle(ctx, info.swizzle.value()));
  }
  if (info.scalableTiles) {
    items.emplace_back(b.getStringAttr("scalableTiles"),
                       b.getBoolArrayAttr(info.scalableTiles.value()));
  }

  return b.getDictionaryAttr(items);
}

std::optional<MaterializeEncodingInfo>
deserializeEncodingInfo(DictionaryAttr attr) {
  MaterializeEncodingInfo info;

#define extractArrayAttrItem(name)                                             \
  {                                                                            \
    auto value = attr.getNamed(#name);                                         \
    if (!value || !isa<ArrayAttr>(value->getValue())) {                        \
      return std::nullopt;                                                     \
    }                                                                          \
    info.name = extractFromIntegerArrayAttr<int64_t>(value->getValue());       \
  }

  extractArrayAttrItem(innerDimsPos);
  extractArrayAttrItem(innerTileSizes);
  extractArrayAttrItem(outerDimsPerm);
#undef extractArrayAttrItem

  if (attr.contains("swizzle")) {
    auto dictAttr =
        dyn_cast<DictionaryAttr>(attr.getNamed("swizzle")->getValue());
    if (!dictAttr) {
      return std::nullopt;
    }
    info.swizzle = deserializeTileSwizzle(dictAttr);
    if (!info.swizzle) {
      return std::nullopt;
    }
  }
  if (attr.contains("scalableTiles")) {
    auto value = attr.getNamed("scalableTiles");
    if (!value || !isa<ArrayAttr>(value->getValue())) {
      return std::nullopt;
    }
    ScalableTileFlags res = llvm::map_to_vector(
        cast<ArrayAttr>(value->getValue()),
        [](Attribute a) { return cast<BoolAttr>(a).getValue(); });
    info.scalableTiles = std::move(res);
  }

  return info;
}

bool isIdentityLayout(const MaterializeEncodingInfo &info) {
  // It is not an identity layout if swizzle is present. The swizzle and
  // scalableTiles are optional variables. User should not set the fields when
  // they do not need them.
  return info.innerDimsPos.empty() && info.innerTileSizes.empty() &&
         info.outerDimsPerm.empty() && !info.swizzle && !info.scalableTiles;
}

SmallVector<int64_t>
getExpandedTileShape(const TileSwizzle::ExpandShapeType &expandShape) {
  SmallVector<int64_t> result;
  for (auto e : expandShape) {
    for (auto d : e) {
      result.push_back(d.size());
    }
  }
  return result;
}

/// Returns the EncodingContractionLikeDimInfo for an encoding with scaled
/// contraction user_indexing_maps, or failure if the scaled contraction
/// dimensions can not be inferred.
static FailureOr<EncodingContractionLikeDimInfo>
getScaledContractionLikeDimInfo(Encoding::EncodingAttr encoding) {
  FailureOr<IREE::LinalgExt::ScaledContractionDimensions> maybeScaledCDims =
      Encoding::getEncodingScaledContractionDims(encoding);
  if (failed(maybeScaledCDims)) {
    return failure();
  }
  IREE::LinalgExt::ScaledContractionDimensions scaledCDims =
      maybeScaledCDims.value();
  EncodingContractionLikeDimInfo dimInfo;
  assert(scaledCDims.m.size() <= 1 && scaledCDims.n.size() <= 1 &&
         scaledCDims.k.size() == 1 && scaledCDims.batch.size() <= 1 &&
         scaledCDims.kB.size() == 1 &&
         "Expected at most one M, N, K, Kb, and Batch dimension");
  int64_t operandIdx = encoding.getOperandIndex().getInt();
  bool isLhs = operandIdx == IREE::Encoding::SCALED_MATMUL_LHS;
  bool isRhs = operandIdx == IREE::Encoding::SCALED_MATMUL_RHS;
  bool isLhsScales = operandIdx == IREE::Encoding::SCALED_MATMUL_LHS_SCALES;
  bool isRhsScales = operandIdx == IREE::Encoding::SCALED_MATMUL_RHS_SCALES;
  bool isResult = operandIdx == IREE::Encoding::SCALED_MATMUL_RESULT;
  if (!scaledCDims.batch.empty()) {
    dimInfo.batchDim = {/*shouldHaveDim=*/true,
                        encoding.mapDimToOperandIndex(scaledCDims.batch[0])};
  }
  if (!scaledCDims.m.empty()) {
    dimInfo.mDim = {/*shouldHaveDim=*/isLhs || isResult,
                    encoding.mapDimToOperandIndex(scaledCDims.m[0])};
  }
  if (!scaledCDims.n.empty()) {
    dimInfo.nDim = {/*shouldHaveDim=*/isRhs || isResult,
                    encoding.mapDimToOperandIndex(scaledCDims.n[0])};
  }
  dimInfo.kDim = {/*shouldHaveDim=*/isLhs || isRhs || isLhsScales ||
                      isRhsScales,
                  encoding.mapDimToOperandIndex(scaledCDims.k[0])};
  dimInfo.kBDim = {/*shouldHaveDim=*/isLhs || isRhs,
                   encoding.mapDimToOperandIndex(scaledCDims.kB[0])};
  return dimInfo;
}

/// Returns the EncodingContractionLikeDimInfo for an encoding with contraction
/// user_indexing_maps, or failure if the contraction dimensions can not be
/// inferred.
static FailureOr<EncodingContractionLikeDimInfo>
getContractionLikeDimInfo(Encoding::EncodingAttr encoding) {
  FailureOr<linalg::ContractionDimensions> maybeCDims =
      Encoding::getEncodingContractionDims(encoding);
  if (failed(maybeCDims)) {
    return failure();
  }
  linalg::ContractionDimensions cDims = maybeCDims.value();
  EncodingContractionLikeDimInfo dimInfo;
  assert(cDims.m.size() <= 1 && cDims.n.size() <= 1 && cDims.k.size() == 1 &&
         cDims.batch.size() <= 1 &&
         "Expected at most one M, N, K, and Batch dimension");
  int64_t operandIdx = encoding.getOperandIndex().getInt();
  bool isLhs = operandIdx == IREE::Encoding::MATMUL_LHS;
  bool isRhs = operandIdx == IREE::Encoding::MATMUL_RHS;
  bool isResult = operandIdx == IREE::Encoding::MATMUL_RESULT;
  if (!cDims.batch.empty()) {
    dimInfo.batchDim = {/*shouldHaveDim=*/true,
                        encoding.mapDimToOperandIndex(cDims.batch[0])};
  }
  if (!cDims.m.empty()) {
    dimInfo.mDim = {/*shouldHaveDim=*/isLhs || isResult,
                    encoding.mapDimToOperandIndex(cDims.m[0])};
  }
  if (!cDims.n.empty()) {
    dimInfo.nDim = {/*shouldHaveDim=*/isRhs || isResult,
                    encoding.mapDimToOperandIndex(cDims.n[0])};
  }
  dimInfo.kDim = {/*shouldHaveDim=*/isLhs || isRhs,
                  encoding.mapDimToOperandIndex(cDims.k[0])};
  return dimInfo;
}

FailureOr<EncodingContractionLikeDimInfo>
getEncodingContractionLikeDims(Encoding::EncodingAttr encoding) {
  FailureOr<EncodingContractionLikeDimInfo> maybeDimInfo =
      getContractionLikeDimInfo(encoding);
  if (succeeded(maybeDimInfo)) {
    return maybeDimInfo.value();
  }
  return getScaledContractionLikeDimInfo(encoding);
}

FailureOr<MaterializeEncodingInfo>
getEncodingInfoForMatmul(Encoding::EncodingAttr encoding, TileMxNxK tileMxNxK) {
  return getEncodingInfoForMatmul(
      encoding, TileMxNxKxKb{tileMxNxK.M, tileMxNxK.N, tileMxNxK.K});
}

FailureOr<MaterializeEncodingInfo>
getEncodingInfoForMatmul(Encoding::EncodingAttr encoding,
                         TileMxNxKxKb tileMxNxKxKb) {
  FailureOr<EncodingContractionLikeDimInfo> maybeDimInfo =
      getEncodingContractionLikeDims(encoding);
  if (failed(maybeDimInfo)) {
    return failure();
  }
  std::optional<unsigned> batchDim = maybeDimInfo->batchDim.operandIdx;
  std::optional<unsigned> mDim = maybeDimInfo->mDim.operandIdx;
  std::optional<unsigned> nDim = maybeDimInfo->nDim.operandIdx;
  std::optional<unsigned> kDim = maybeDimInfo->kDim.operandIdx;
  std::optional<unsigned> kBDim = maybeDimInfo->kBDim.operandIdx;
  MaterializeEncodingInfo encodingInfo;
  if (batchDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(batchDim.value());
  }
  if (mDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(mDim.value());
    encodingInfo.innerDimsPos.push_back(mDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxKxKb.M);
  }
  if (nDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(nDim.value());
    encodingInfo.innerDimsPos.push_back(nDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxKxKb.N);
  }
  if (kDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(kDim.value());
    encodingInfo.innerDimsPos.push_back(kDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxKxKb.K);
  }
  if (kBDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(kBDim.value());
    encodingInfo.innerDimsPos.push_back(kBDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxKxKb.KB);
  }
  return encodingInfo;
}

/// Finds a loop dimension that appears as a pure AffineDimExpr in exactly the
/// given set of operand maps (identified by index) and not in the others.
/// This is used to identify IC and OC dims from convolution indexing maps.
static std::optional<unsigned>
findConvDimInMaps(ArrayRef<AffineMap> maps, ArrayRef<unsigned> presentInMaps,
                  ArrayRef<unsigned> absentFromMaps) {
  unsigned numDims = maps[0].getNumDims();
  for (unsigned d = 0; d < numDims; ++d) {
    auto dimExpr = getAffineDimExpr(d, maps[0].getContext());
    bool allPresent = llvm::all_of(presentInMaps, [&](unsigned mapIdx) {
      return maps[mapIdx].getResultPosition(dimExpr).has_value();
    });
    bool allAbsent = llvm::all_of(absentFromMaps, [&](unsigned mapIdx) {
      return !maps[mapIdx].getResultPosition(dimExpr).has_value();
    });
    if (allPresent && allAbsent) {
      return d;
    }
  }
  return std::nullopt;
}

FailureOr<MaterializeEncodingInfo>
getEncodingInfoForConv(Encoding::EncodingAttr encoding, TileOCxIC tile) {
  int64_t operandIdx = encoding.getOperandIndex().getInt();
  SmallVector<AffineMap> maps = encoding.getRootMaps();
  if (maps.size() < 3) {
    return failure();
  }

  // IC: loop dim present in input (map 0) and filter (map 1), absent from
  //     output (map 2). This is the reduction dimension for input channels.
  // OC: loop dim present in filter (map 1) and output (map 2), absent from
  //     input (map 0). This is the parallel dimension for output channels.
  // Note: spatial dims (oh, ow) appear in input+output; filter loop dims
  //       (fh, fw) appear in input+filter. Neither matches IC or OC pattern.
  std::optional<unsigned> icLoopDim =
      findConvDimInMaps(maps, /*presentIn=*/{0, 1}, /*absentFrom=*/{2});
  std::optional<unsigned> ocLoopDim =
      findConvDimInMaps(maps, /*presentIn=*/{1, 2}, /*absentFrom=*/{0});
  if (!icLoopDim || !ocLoopDim) {
    return failure();
  }

  // Map the IC/OC loop dims to positions within this operand's tensor.
  std::optional<unsigned> icPos = encoding.mapDimToOperandIndex(*icLoopDim);
  std::optional<unsigned> ocPos = encoding.mapDimToOperandIndex(*ocLoopDim);

  MaterializeEncodingInfo info;

  if (operandIdx == Encoding::CONV_IN) {
    // Input [N, H, W, IC] → [N, H, W, IC/c0, c0]: pack IC only.
    if (!icPos) {
      return failure();
    }
    int64_t rank = maps[0].getNumResults();
    for (int64_t i = 0; i < rank; ++i) {
      info.outerDimsPerm.push_back(i);
    }
    info.innerDimsPos = {static_cast<int64_t>(*icPos)};
    info.innerTileSizes = {tile.IC};
  } else if (operandIdx == Encoding::CONV_FILTER) {
    // Filter [FH, FW, IC, OC] → [OC/k0, FH, FW, IC/c0, k0, c0]:
    //   promote OC to outermost, pack both OC and IC.
    if (!icPos || !ocPos) {
      return failure();
    }
    int64_t rank = maps[1].getNumResults();
    // outerDimsPerm: OC first, then remaining dims in original order.
    info.outerDimsPerm.push_back(static_cast<int64_t>(*ocPos));
    for (int64_t i = 0; i < rank; ++i) {
      if (i != static_cast<int64_t>(*ocPos)) {
        info.outerDimsPerm.push_back(i);
      }
    }
    // innerDimsPos/Sizes: OC (k0) then IC (c0).
    info.innerDimsPos = {static_cast<int64_t>(*ocPos),
                         static_cast<int64_t>(*icPos)};
    info.innerTileSizes = {tile.OC, tile.IC};
  } else if (operandIdx == Encoding::CONV_OUT) {
    // Output [N, OH, OW, OC] → [N, OH, OW, OC/k0, k0]: pack OC only.
    if (!ocPos) {
      return failure();
    }
    int64_t rank = maps[2].getNumResults();
    for (int64_t i = 0; i < rank; ++i) {
      info.outerDimsPerm.push_back(i);
    }
    info.innerDimsPos = {static_cast<int64_t>(*ocPos)};
    info.innerTileSizes = {tile.OC};
  } else {
    return failure();
  }

  return info;
}

} // namespace mlir::iree_compiler::IREE::Codegen
