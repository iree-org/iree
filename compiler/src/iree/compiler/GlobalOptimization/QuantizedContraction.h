// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_QUANTIZEDCONTRACTION_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_QUANTIZEDCONTRACTION_H_

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"

// Internal analysis support for ConvertQDQToIntegerMath.
namespace mlir::iree_compiler::GlobalOptimization::detail {

using IREE::LinalgExt::DequantizeAffineOp;

/// One dequantized operand. All map domains are the original contraction's
/// iteration space.
struct QuantizedOperand {
  DequantizeAffineOp dequantize;
  /// Range: quantized input data space.
  AffineMap inputMap;
  /// Range: scale data space.
  AffineMap scaleMap;
  /// Range: zero-point data space. Null if absent.
  AffineMap zeroPointMap;
};

/// A matched contraction with a nonempty, statically bounded integer reduction.
/// Construction succeeds only for a plain multiply-add body and a zero init.
/// Dimension indices refer to the original contraction's iteration space.
struct QuantizedContraction {
  QuantizedOperand lhs;
  QuantizedOperand rhs;
  /// Original contraction's indexing map for the output.
  AffineMap outputMap;
  /// Reduction dims over which all quantization parameters are invariant.
  SmallVector<unsigned> integerReductionDims;
  /// Reduction dims retained as parallel dims until the scaling epilogue.
  SmallVector<unsigned> floatingReductionDims;
  /// Product of integer reduction extents, bounded for the i32 accumulator.
  int64_t reductionExtent = 1;

  // Each operand sum is multiplied by the opposite operand's zero point.
  bool needsLhsSum() const {
    auto quant = rhs.dequantize;
    return !quant.isSymmetric();
  }

  bool needsRhsSum() const {
    auto quant = lhs.dequantize;
    return !quant.isSymmetric();
  }
};

/// Analyzes a verified contraction without modifying its IR. Fails when the
/// contraction cannot be rewritten using a statically bounded integer
/// reduction.
FailureOr<QuantizedContraction> getQuantizedContraction(linalg::LinalgOp op);

/// Returns a map selecting domain dimensions that do not appear as bare results
/// in any of `maps`, a nonempty list with a shared domain. These dimensions
/// need explicit extents because Linalg cannot infer them from operand shapes.
AffineMap getExplicitExtentDimsMap(ArrayRef<AffineMap> maps);

} // namespace mlir::iree_compiler::GlobalOptimization::detail

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_QUANTIZEDCONTRACTION_H_
