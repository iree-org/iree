// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSwizzle.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

static Attribute dimToArrayAttr(MLIRContext *ctx, TileSwizzle::Dim dim) {
  Builder b(ctx);
  return b.getDenseI16ArrayAttr({static_cast<int16_t>(dim.kind), dim.size});
}

DictionaryAttr serializeTileSwizzle(MLIRContext *ctx, TileSwizzle swizzle) {
  Builder b(ctx);
  SmallVector<NamedAttribute> items;

  SmallVector<Attribute> expandShape;
  for (auto expandConfig : swizzle.expandShape) {
    Attribute expandAttr = b.getArrayAttr(
        llvm::map_to_vector(expandConfig, [&](TileSwizzle::Dim dim) {
          return dimToArrayAttr(ctx, dim);
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
    // FIXME: The attribute that created by the builder can't be casted back to
    // the original attribute. Because it creates an ArrayAttr but not the
    // attribute that we specificed. It is not used in the prototype. For the
    // fix, see examples in the deserializeMaterializeEncodingInfo method.
    TileSwizzle::ExpandShapeDimVectorType vec;
    for (auto dimAttr : expandConfig.getAsRange<DenseI16ArrayAttr>()) {
      ArrayRef<int16_t> dimRef = dimAttr.asArrayRef();
      TileSwizzle::Dim dim;
      dim.kind = static_cast<TileSwizzle::Dim::Kind>(dimRef[0]);
      dim.size = dimRef[1];
      vec.push_back(dim);
    }
    swizzle.expandShape.push_back(vec);
  }

  auto permutation =
      cast<DenseI64ArrayAttr>(attr.getNamed("permutation")->getValue())
          .asArrayRef();
  swizzle.permutation.assign(permutation.begin(), permutation.end());

  return swizzle;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              TileSwizzle::Dim::Kind kind) {
  switch (kind) {
  case TileSwizzle::Dim::Kind::Internal:
    return os << "Internal";
  case TileSwizzle::Dim::Kind::CrossThread:
    return os << "CrossThread";
  case TileSwizzle::Dim::Kind::CrossIntrinsic:
    return os << "CrossIntrinsic";
  default:
    // Required by GCC.
    assert(false);
    return os;
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, TileSwizzle::Dim dim) {
  return os << dim.size << "(" << dim.kind << ")";
}

static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const TileSwizzle::ExpandShapeDimVectorType &expandShapeDimVector) {
  os << "[";
  llvm::interleaveComma(expandShapeDimVector, os);
  return os << "]";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TileSwizzle &swizzle) {
  os << "{expandShape = [";
  llvm::interleaveComma(swizzle.expandShape, os);
  os << "], permutation = [";
  llvm::interleaveComma(swizzle.permutation, os);
  os << "]}";
  return os;
}

} // namespace mlir::iree_compiler
