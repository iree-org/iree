// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUEnums.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.h.inc"

namespace mlir::iree_compiler::IREE::CPU {

// Returns the TileSwizzle for the given intrinsic and operand index.
Codegen::TileSwizzle getIntrinsicSwizzle(MMAIntrinsic mma, int operandIdx);

// Returns the TileSwizzle for the given MMA attr and operand index.
Codegen::TileSwizzle getSwizzle(DataTiledMMAAttr mma, int operandIdx);

} // namespace mlir::iree_compiler::IREE::CPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS_H_
