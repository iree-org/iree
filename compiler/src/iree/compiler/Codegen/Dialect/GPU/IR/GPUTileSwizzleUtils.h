// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_

#include "iree/compiler/Codegen/Common/TileSwizzle.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"

namespace mlir::iree_compiler {

// Returns the TileSwizzle bringing a tile from row-major layout into the tiled
// layout consumed by the given `intrinsic` and `fragment`.
TileSwizzle getIntrinsicSwizzle(IREE::GPU::MMAIntrinsic intrinsic,
                                IREE::GPU::MMAFragment fragment);

// Returns the swizzle for the full data-tiled-mma tile, including all the
// relevant unrolling and expansion factors.
TileSwizzle getSwizzle(IREE::GPU::DataTiledMMAAttr mma,
                       IREE::GPU::MMAFragment fragment);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_
