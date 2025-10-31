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

/// Returns the swizzled tile shape, but with dim sizes overwritten with 1 if
/// `predicate` returns false.
SmallVector<int64_t> sliceSwizzledShape(
    const Codegen::TileSwizzle &swizzle,
    llvm::function_ref<bool(Codegen::TileSwizzle::Dim)> predicate);

/// Returns the swizzle for the full data-tiled-mma tile, including all the
/// relevant unrolling and expansion factors.
Codegen::TileSwizzle getSwizzle(IREE::GPU::DataTiledMMAAttr mma,
                                int operandIndex);

/// Returns the swizzle for the full data-tiled-scaled-mma tile, including all
/// the relevant unrolling and expansion factors.
Codegen::TileSwizzle getSwizzle(IREE::GPU::DataTiledScaledMMAAttr scaledMma,
                                unsigned operandIdx);

/// Returns the swizzle for the data-tiled-mma tile, based on the `fragment`
/// and contraction dimensions required from the `encoding`.
FailureOr<Codegen::TileSwizzle>
getEncodingSwizzle(IREE::Encoding::EncodingAttr encoding,
                   IREE::GPU::DataTiledMMAInterfaceAttr mma,
                   unsigned operandIndex);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPUTILESWIZZLEUTILS_H_
