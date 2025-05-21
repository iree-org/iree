// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
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
  return lhs.kind == rhs.kind && lhs.size == rhs.size;
}

bool operator!=(TileSwizzle::Dim lhs, TileSwizzle::Dim rhs) {
  return !(lhs == rhs);
}

bool operator==(const TileSwizzle &lhs, const TileSwizzle &rhs) {
  return lhs.expandShape == rhs.expandShape &&
         lhs.permutation == rhs.permutation;
}

bool operator!=(const TileSwizzle &lhs, const TileSwizzle &rhs) {
  return !(lhs == rhs);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              TileSwizzle::Dim::Kind kind) {
  return os << convertSwizzleKindToString(kind);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, TileSwizzle::Dim dim) {
  return os << dim.size << "(" << dim.kind << ")";
}

static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const TileSwizzle::ExpandShapeDimVectorType &expandShapeDimVector) {
  return os << llvm::interleaved_array(expandShapeDimVector);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TileSwizzle &swizzle) {
  return os << "{expandShape = " << llvm::interleaved_array(swizzle.expandShape)
            << ", permutation = "
            << llvm::interleaved_array(swizzle.permutation) << "}";
}

bool operator==(const MaterializeEncodingInfo &lhs,
                const MaterializeEncodingInfo &rhs) {
  return lhs.innerDimsPos == rhs.innerDimsPos &&
         lhs.innerTileSizes == rhs.innerTileSizes &&
         lhs.outerDimsPerm == rhs.outerDimsPerm && lhs.swizzle == rhs.swizzle;
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
    os << ", swizzle = " << encodingInfo.swizzle.value();
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
    assert(false && "unhandled enum type");
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

static ArrayAttr swizzleDimToArrayAttr(MLIRContext *ctx, TileSwizzle::Dim dim) {
  Builder b(ctx);
  return b.getArrayAttr({b.getStringAttr(convertSwizzleKindToString(dim.kind)),
                         b.getI16IntegerAttr(dim.size)});
}

static std::optional<TileSwizzle::Dim> arrayAttrToSwizzleDim(Attribute attr) {
  auto arrayAttr = dyn_cast<ArrayAttr>(attr);
  if (!arrayAttr) {
    return std::nullopt;
  }
  ArrayRef<Attribute> attrs = arrayAttr.getValue();
  if (attrs.size() != 2) {
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
  return TileSwizzle::Dim(maybeKind.value(), sizeAttr.getInt());
}

DictionaryAttr serializeTileSwizzle(MLIRContext *ctx,
                                    const TileSwizzle &swizzle) {
  Builder b(ctx);
  SmallVector<NamedAttribute> items;

  SmallVector<Attribute> expandShape;
  for (auto expandConfig : swizzle.expandShape) {
    Attribute expandAttr = b.getArrayAttr(
        llvm::map_to_vector(expandConfig, [&](TileSwizzle::Dim dim) {
          return cast<Attribute>(swizzleDimToArrayAttr(ctx, dim));
        }));
    expandShape.push_back(expandAttr);
  }

  items.emplace_back(b.getStringAttr("expandShape"),
                     b.getArrayAttr(expandShape));
  items.emplace_back(b.getStringAttr("permutation"),
                     b.getI64ArrayAttr(swizzle.permutation));

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
    swizzle.expandShape.push_back(vec);
  }

  auto permAttr = attr.getNamed("permutation");
  if (!permAttr || !isa<ArrayAttr>(permAttr->getValue())) {
    return std::nullopt;
  }
  swizzle.permutation =
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

  return info;
}

bool isIdentityLayout(const MaterializeEncodingInfo &info) {
  // It is not an identity layout if swizzle is present. The swizzle is an
  // optional variable. User should not set the field when they do not need
  // swizzle.
  return info.innerDimsPos.empty() && info.innerTileSizes.empty() &&
         info.outerDimsPerm.empty() && !info.swizzle;
}

SmallVector<int64_t>
getExpandedTileShape(const TileSwizzle::ExpandShapeType &expandShape) {
  SmallVector<int64_t> result;
  for (auto e : expandShape) {
    for (auto d : e) {
      result.push_back(d.size);
    }
  }
  return result;
}

FailureOr<MaterializeEncodingInfo>
getEncodingInfoForMatmul(Encoding::EncodingAttr encoding, TileMxNxK tileMxNxK) {
  MaterializeEncodingInfo encodingInfo;
  FailureOr<linalg::ContractionDimensions> cDims =
      Encoding::getEncodingContractionDims(encoding);
  if (failed(cDims)) {
    return failure();
  }
  // The following expects M, N, K, and Batch sizes of at most 1 for now
  assert(cDims->m.size() <= 1 && cDims->n.size() <= 1 && cDims->k.size() == 1 &&
         cDims->batch.size() <= 1 &&
         "Expected at most one M, N, K, and Batch dimension");
  std::optional<unsigned> batchDim =
      cDims->batch.empty() ? std::nullopt
                           : encoding.mapDimToOperandIndex(cDims->batch[0]);
  std::optional<unsigned> mDim =
      cDims->m.empty() ? std::nullopt
                       : encoding.mapDimToOperandIndex(cDims->m[0]);
  std::optional<unsigned> nDim =
      cDims->n.empty() ? std::nullopt
                       : encoding.mapDimToOperandIndex(cDims->n[0]);
  std::optional<unsigned> kDim = encoding.mapDimToOperandIndex(cDims->k[0]);
  if (batchDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(batchDim.value());
  }
  if (mDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(mDim.value());
    encodingInfo.innerDimsPos.push_back(mDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxK.M);
  }
  if (nDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(nDim.value());
    encodingInfo.innerDimsPos.push_back(nDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxK.N);
  }
  if (kDim.has_value()) {
    encodingInfo.outerDimsPerm.push_back(kDim.value());
    encodingInfo.innerDimsPos.push_back(kDim.value());
    encodingInfo.innerTileSizes.push_back(tileMxNxK.K);
  }
  return encodingInfo;
}

} // namespace mlir::iree_compiler::IREE::Codegen
