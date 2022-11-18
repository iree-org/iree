// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.h - Transformations common to all backends --------------===//
//
// Defines transformations that are common to backends
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMS_H_

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

/// Get the `offsets`, `sizes` and `strides` for a `storeOp` (or `loadOp`). This
/// method clones the operations that generate the `Value`s used for
/// specifying the offsets, sizesm strides and dynamic dims of the
/// `storeOp/loadOp` at the insertion point to avoid use-def violations.
void cloneOffsetsSizesAndStrides(OpBuilder &builder,
                                 IREE::Flow::DispatchTensorStoreOp storeOp,
                                 SmallVector<OpFoldResult> &offsets,
                                 SmallVector<OpFoldResult> &sizes,
                                 SmallVector<OpFoldResult> &strides,
                                 SmallVector<Value> &dynamicDims);
void cloneOffsetsSizesAndStrides(OpBuilder &builder,
                                 IREE::Flow::DispatchTensorLoadOp loadOp,
                                 SmallVector<OpFoldResult> &offsets,
                                 SmallVector<OpFoldResult> &sizes,
                                 SmallVector<OpFoldResult> &strides,
                                 SmallVector<Value> &dynamicDims);

/// Insert patterns to perform folding of AffineMinOp by matching the
/// pattern generated by tile and distribute. Try to fold a affine.min op by
/// matching the following form:
/// ```
/// scf.for %iv = %lb to %ub step %step
///   %affine.min affine_map<(d0, d1) -> (N, d0 - d1)>(%ub, %iv)
/// ```
/// With N a compile time constant. This operations can be replace by
/// `%cN = arith.constant N : index` if we can prove that %lb, %step and %ub
/// are divisible by N.
void populateAffineMinSCFCanonicalizationPattern(RewritePatternSet &patterns);

using GetMinMaxExprFn =
    std::function<Optional<std::pair<AffineExpr, AffineExpr>>(
        Value value, SmallVectorImpl<Value> &dims,
        SmallVectorImpl<Value> &symbols)>;

/// Insert pattern to remove single iteration loop. The pattern will detect
/// single iteration loops based on the range returned by the lambda
/// |getMinMaxFn| for some know values.
void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns,
                                              GetMinMaxExprFn getMinMaxFn);
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMS_H_
