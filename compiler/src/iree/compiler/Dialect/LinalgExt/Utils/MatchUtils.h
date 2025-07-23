// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_UTILS_MATCHUTILS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_UTILS_MATCHUTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace detail {
/// Result of matching a Linalg generic against the predicates of it being a
/// contraction.
enum class MatchContractionResult;
} // namespace detail

Value defaultPromotionImpl(OpBuilder &builder, OpOperand &operand,
                           Attribute attr);

/// Positions of a Linalg op loops that correspond to different kinds of
/// contraction dimension for scaled contractions.
struct ScaledContractionDimensions {
  SmallVector<unsigned, 2> batch;
  SmallVector<unsigned, 2> m;
  SmallVector<unsigned, 2> n;
  SmallVector<unsigned, 2> k;
  SmallVector<unsigned, 2> kB;
};

/// Find 2 parallel (m and n) and 2 reduction (k and k_B) dimension candidates
/// that form a scaled matmul subcomputation within `linalgOp`.
/// These dimensions are such that:
///   1. The m dimension is involved in an outer-product along LHS
///      (i.e. it is a permutation on RES and LHS and does not appear in RHS).
///   2. The n dimension is involved in an outer-product along RHS
///      (i.e. it is a permutation on RES and RHS and does not appear in LHS).
///   3. The k dimension appears as a permutation on LHS and RHS.
///   4. The k_B dimension appears as a permutation on LHS and RHS in scales
///   4. m, n, k, k_B appear only once in any given indexing.
///   5. Optional batch dimensions that appear in all operands are captured.
/// This allows e.g. detecting that some contraction is embedded within
/// `linalgOp` with some orthogonal heuristic.
FailureOr<ScaledContractionDimensions>
inferScaledContractionDims(linalg::LinalgOp linalgOp);
FailureOr<ScaledContractionDimensions>
inferScaledContractionDims(ArrayRef<AffineMap> indexingMaps);

// Checks whether `linalgOp` conforms to ScaledContractionOp.
bool isaScaledContractionOpInterface(linalg::LinalgOp linalgOp);

// Checks whether a given block corresponds to the body of a linalg op
// describing a scaled mma.
bool isScaledContractionBody(
    Block &block, function_ref<bool(Operation *, Operation *)> isaPair,
    llvm::raw_ostream &errs = mlir::thread_safe_nulls());

detail::MatchContractionResult
isScaledContractionImpl(Operation *op,
                        ScaledContractionDimensions *dimensions = nullptr);

} // namespace  mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_UTILS_MATCHUTILS_H_
