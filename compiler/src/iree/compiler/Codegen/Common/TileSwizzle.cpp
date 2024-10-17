// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSwizzle.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler {

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
