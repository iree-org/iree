// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_UTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_UTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// Relational operator and IOstream implementations for Layout Structs.
//===----------------------------------------------------------------------===//

bool operator==(TileSwizzle::Dim lhs, TileSwizzle::Dim rhs);
bool operator!=(TileSwizzle::Dim lhs, TileSwizzle::Dim rhs);
bool operator==(const TileSwizzle &lhs, const TileSwizzle &rhs);
bool operator!=(const TileSwizzle &lhs, const TileSwizzle &rhs);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              TileSwizzle::Dim::Kind kind);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, TileSwizzle::Dim dim);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TileSwizzle &swizzle);

bool operator==(const MaterializeEncodingInfo &lhs,
                const MaterializeEncodingInfo &rhs);
bool operator!=(const MaterializeEncodingInfo &lhs,
                const MaterializeEncodingInfo &rhs);

//===----------------------------------------------------------------------===//
// Layout Utilities.
//===----------------------------------------------------------------------===//

/// Conversion between TileSwizzle::Dim::Kind and string.
std::string convertSwizzleKindToString(TileSwizzle::Dim::Kind kind);
std::optional<TileSwizzle::Dim::Kind> convertStringToSwizzleKind(StringRef str);

/// Conversion between TileSwizzle struct and DictionaryAttr.
DictionaryAttr serializeTileSwizzle(MLIRContext *ctx,
                                    const TileSwizzle &swizzle);
std::optional<TileSwizzle> deserializeTileSwizzle(DictionaryAttr attr);

/// Conversion between MaterializeEncodingInfo struct and DictionaryAttr.
DictionaryAttr serializeEncodingInfo(MLIRContext *ctx,
                                     const MaterializeEncodingInfo &info);
std::optional<MaterializeEncodingInfo>
deserializeEncodingInfo(DictionaryAttr attr);

/// Returns true if the `info` denotes an identity layout, i.e., there is no
/// relayout requirement.
bool isIdentityLayout(const MaterializeEncodingInfo &info);

/// Concatenates the vectors.
SmallVector<int64_t>
getExpandedTileShape(const TileSwizzle::ExpandShapeType &expandShape);

struct TileMxNxK {
  int64_t M = 1;
  int64_t N = 1;
  int64_t K = 1;
};

MaterializeEncodingInfo
getEncodingInfoForMatmul(Encoding::EncodingAttr encoding, TileMxNxK tileMxNxK);

//===----------------------------------------------------------------------===//
// Operation Lowering Utilities.
//===----------------------------------------------------------------------===//

// TODO(hanchung): The below methods are exposed to public because they are
// shared between MaterializeEncodingIntoPackUnPack.cpp.cpp and
// CPUEncodingExternalModels.cpp. They will be moved to other places after all
// the CPU backends implement their layout attributes.

/// Returns the best TileMxNxK from `enumeratedTiles` pool. If the
/// `hostDefinedUpperBound` is not empty, the chosen tile sizes can not be
/// greater than the values.
/// TODO(#16933): Remove `hostDefinedUpperBound` once we can propagate such
/// information to host. For now, they are defined by host.
TileMxNxK chooseMatmulTile(ArrayRef<TileMxNxK> enumeratedTiles,
                           IREE::Encoding::MatmulNarrowDim narrowDim,
                           ArrayRef<int64_t> hostDefinedUpperBound = {});

FailureOr<Operation *>
lowerContractionOpWithEncoding(OpBuilder &builder, linalg::LinalgOp linalgOp,
                               ValueRange operands, bool transposeNarrowN,
                               LayoutAttrInterface layoutAttr);

} // namespace mlir::iree_compiler::IREE::Codegen

#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_UTILS_H_
