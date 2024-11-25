// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_UTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_UTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/ADT/SmallVector.h"
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

/// Concatenates the vectors.
SmallVector<int64_t>
getExpandedTileShape(const TileSwizzle::ExpandShapeType &expandShape);

} // namespace mlir::iree_compiler::IREE::Codegen

#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_UTILS_H_
