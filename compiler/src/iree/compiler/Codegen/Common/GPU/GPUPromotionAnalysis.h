// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_GPUPROMOTIONANALYSIS_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_GPUPROMOTIONANALYSIS_H_

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler {

/// Discardable attribute name on `to_layout` ops that carries the promotion
/// type (e.g., UseGlobalLoadDMAAttr).
inline constexpr llvm::StringLiteral kPromotionTypeAttr =
    "iree_gpu.promotion_type";

/// Result of promotion type analysis. Maps SSA values to their inferred
/// promotion type attribute (e.g., UseGlobalLoadDMAAttr). Only values with a
/// concrete (non-overdefined) promotion type are included.
using PromotionTypeMap = llvm::DenseMap<Value, Attribute>;

/// Run backward promotion type analysis on `root`. Propagates promotion types
/// from `iree_gpu.promotion_type` discardable attributes on `to_layout` ops
/// backward through elementwise, transpose, broadcast, and shape_cast ops.
///
/// Returns a map from SSA values to their inferred promotion type. Values with
/// conflicting promotion types (overdefined) are not included.
PromotionTypeMap analyzePromotionTypes(Operation *root);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_GPUPROMOTIONANALYSIS_H_
