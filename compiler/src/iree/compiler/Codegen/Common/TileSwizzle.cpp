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
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, TileSwizzle::Dim dim) {
  return os << dim.size << "(" << dim.kind << ")";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TileSwizzle &swizzle) {
  os << "{expandShape = [";
  for (auto [i, e] : llvm::enumerate(swizzle.expandShape)) {
    if (i > 0) {
      os << ", ";
    }
    os << "[";
    for (auto [j, d] : llvm::enumerate(e)) {
      if (j > 0) {
        os << ", ";
      }
      os << d;
    }
    os << "]";
  }
  os << "], swizzle = [";
  for (auto [i, p] : llvm::enumerate(swizzle.permutation)) {
    if (i > 0) {
      os << ", ";
    }
    os << p;
  }
  os << "]}";
  return os;
}

} // namespace mlir::iree_compiler
