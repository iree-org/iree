// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"

namespace mlir::iree_compiler::IREE::GPU {

// Returns the TileSwizzle bringing a tile from row-major layout into the tiled
// layout consumed by the given `intrinsic` and `fragment`.
Codegen::TileSwizzle getIntrinsicSwizzle(IREE::GPU::MMAIntrinsic intrinsic,
                                         IREE::GPU::MMAFragment fragment);

// Returns the swizzle for the full data-tiled-mma tile, including all the
// relevant unrolling and expansion factors.
Codegen::TileSwizzle getSwizzle(IREE::GPU::DataTiledMMAAttr mma,
                                IREE::GPU::MMAFragment fragment);

/// Moves all `Kind::CrossThread` dims of the Acc layout to the outermost
/// within their expand shape reassociation groups. This only moves the cross
/// thread dims of the Acc layout because we want to fuse the unset_encoding
/// ops with the data tiled matmul. In order to do this, the sliced dimensions
/// (CrossThread) for each thread need to be outermost in the final write out.
///
/// This transformation is for the Acc layout, but the Lhs and Rhs layouts need
/// to be transformed too, because the layouts need to match the Acc for their
/// respective M and N tile dimensions.
///
/// Example (CrossThread dims are denoted with surrounding {} braces):
///   Input:
///     Lhs swizzle:
///       expandShape = [[8, {16}], [2, {4}, 4]],
///       permutation = [0, 3, 1, 2, 4]
///     Rhs layout:
///       expandShape = [[{4}, 2, {16}], [2, {4}, 4]],
///       permutation = [0, 1, 4, 2, 3, 5]
///     Acc layout:
///       expandShape = [[8, {4}, 4], [{4}, 2, {16}]],
///       permutation = [3, 0, 4, 1, 5, 2]
///   Output:
///     Lhs layout:
///       expandShape = [[{4}, 8, {4}], [2, {4}, 4]],
///       permutation = [1, 4, 0, 2, 3, 5]
///     Rhs layout:
///       expandShape = [[{4}, {16}, 2], [2, {4}, 4]],
///       permutation = [0, 2, 4, 1, 3, 5]
///     Acc layout:
///       expandShape = [[{4}, 8, 4], [{4}, {16}, 2]],
///       permutation = [3, 1, 5, 0, 4, 2]
void moveCrossThreadOutermost(Codegen::TileSwizzle &swizzle,
                              Codegen::TileSwizzle accSwizzle,
                              MMAFragment fragment);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_
