// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREEGPUAttrs.h - Codegen GPU dialect attributes --------------------===//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

// clang-format off
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h.inc"
// clang-format on

namespace mlir::linalg {
class LinalgOp;
} // namespace mlir::linalg

namespace mlir::iree_compiler {

// Returns an MmaAttr from the array of mmaTypes compatible with the given
// structured operation description. The conditions for compatibility are
//
// 1. The iteration bounds are aligned on the shape of the mma operation.
// 2. The element types of |inputTypes| match with `[aType, bType, cType]`
//
// Returns the first successful match.
FailureOr<IREE::GPU::MmaAttr>
getCompatibleMmaAttr(ArrayAttr mmaTypes, ArrayRef<AffineMap> indexingMaps,
                     ArrayRef<int64_t> iterationBounds, TypeRange inputTypes);
// Helper for contractions.
FailureOr<IREE::GPU::MmaAttr> getCompatibleMmaAttr(ArrayAttr mmaTypes,
                                                   vector::ContractionOp);
// Helper for linalg ops. Fails if the linalg op is not inferrable as a
// contraction op.
FailureOr<IREE::GPU::MmaAttr> getCompatibleMmaAttr(ArrayAttr mmaTypes,
                                                   linalg::LinalgOp);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS_H_
