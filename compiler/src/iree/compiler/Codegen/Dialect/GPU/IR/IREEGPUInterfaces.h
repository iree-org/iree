// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::GPU {
Value defaultPromotionImpl(OpBuilder &builder, OpOperand &operand,
                           Attribute attr);

/// Computes offsets/sizes/strides for a single operand tile from a swizzle
/// description and a thread ID. This is the shared implementation behind
/// DataTiledMMAInterfaceAttr::populateOperandOffsetsSizesStrides and can be
/// called directly when a custom swizzle is needed.
LogicalResult populateSwizzleBasedOffsetsSizesStrides(
    OpBuilder &builder, Location loc, const Codegen::TileSwizzle &swizzle,
    Value threadId, ArrayRef<int64_t> permutation,
    SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides);

} // namespace mlir::iree_compiler::IREE::GPU

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h.inc"
// clang-format on

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES_H_
