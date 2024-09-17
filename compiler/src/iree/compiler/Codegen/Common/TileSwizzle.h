// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_TILESWIZZLE_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_TILESWIZZLE_H_

#include <cstdint>
#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler {

// Metadata for a swizzle, that is, an (expand_shape -> transposition)
// pair of ops performing a change of layout within the tiles. This is used
// on GPU, where the tiles themselves can have an arbitrary layout.
struct TileSwizzle {
  // This vector-of-vectors contains all the information needed to generate
  // a `tensor.expand_shape` creating additional internal dimensions into the
  // tile. For example, expandShape = [[16], [4, 2]] means that the original
  // tile shape [16, 8] gets expanded such that the first dimension 16 is left
  // unchanged, and the second dimension 8 gets split into two internal dims
  // of size 4 and 2.
  llvm::SmallVector<llvm::SmallVector<int64_t>> expandShape;
  // This permutation vector applies to the expanded dimensions and is used
  // to generate a `linalg.transpose` changing the layout of the tile. For
  // example, permutation[0] dictates which of the expanded dimensions becomes
  // the leading dimension of the layout.
  llvm::SmallVector<int64_t> permutation;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_TILESWIZZLE_H_
