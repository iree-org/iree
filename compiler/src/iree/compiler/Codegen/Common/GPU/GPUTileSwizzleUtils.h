// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_GPU_GPUTILESWIZZLEUTILS_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_GPU_GPUTILESWIZZLEUTILS_H_

#include "iree/compiler/Codegen/Common/TileSwizzle.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"

namespace mlir::iree_compiler {

// Returns the TileSwizzle bringing a tile from row-major layout into the tiled
// layout consumed by the given `intrinsic` and `fragment`.
TileSwizzle getIntrinsicSwizzle(IREE::GPU::MMAIntrinsic intrinsic,
                                IREE::GPU::MMAFragment fragment);

// Unrolls the dimension given by `srcIndex` by the given `unrollFactor`.
// This is not interleaving layouts. The layout will consist of multiple copies
// of the input tile, side by side.
//
// Example:
//    Input swizzle = { expandShape = [[16], [4]], permutation = [1, 0] }
//    Input srcIndex = 1
//    Input unrollFactor = 4
// -> Output swizzle = { expandShape = [[16], [4, 4]], permutation = [1, 2, 0] }
//
void unroll(TileSwizzle &swizzle, int srcIndex, int unrollFactor);

// Interleaves the layout in `swizzle` by mutating `swizzle.permutation` to
// move permutation[0], the outer-most dimension (which the unroll() function
// created to be the unrolling dimension), to the inner dimension given by
// `expandedDimIndexToInterleaveAt`.
//
// Example:
//    Input swizzle = { expandShape = [[16], [4, 4]], permutation = [1, 2, 0] }
//    Input srcIndex = 1
//    Input expandedDimIndexToInterleaveAt = 1
// -> Output swizzle = { expandShape = [[16], [4, 4]], permutation = [2, 0, 1] }
//
void interleave(TileSwizzle &swizzle, int srcIndex,
                int expandedDimIndexToInterleaveAt);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_GPU_GPUTILESWIZZLEUTILS_H_
