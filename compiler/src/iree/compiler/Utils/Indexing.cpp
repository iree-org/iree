// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/Indexing.h"
#include "mlir/IR/AffineExprVisitor.h"

using namespace mlir;

namespace mlir::iree_compiler {
LogicalResult basisFromSizesStrides(ArrayRef<int64_t> sizes,
                                    ArrayRef<int64_t> strides,
                                    SmallVectorImpl<int64_t> &basis,
                                    SmallVectorImpl<size_t> &dimToResult) {
  assert(sizes.size() == strides.size());
  size_t numDims = sizes.size();
  basis.reserve(numDims);

  SmallVector<std::tuple<int64_t, int64_t, size_t>> terms =
      llvm::map_to_vector(llvm::enumerate(strides, sizes), [&](auto tuple) {
        auto [dim, stride, size] = tuple;
        return std::make_tuple(stride, size, dim);
      });
  llvm::sort(terms);

  int64_t previousSizes = 1;
  SmallVector<std::optional<size_t>> basisEntryToDim;
  basisEntryToDim.reserve(numDims);
  for (auto [stride, size, dim] : terms) {
    if (stride == 0) {
      stride = 1;
      size = 1;
    }
    if (stride % previousSizes != 0)
      return failure();

    // Handle casis like threads = {4, 8}, strides = {1, 16}, which need an
    // extra basis element.
    if (stride != previousSizes) {
      int64_t jumpSize = stride / previousSizes;
      basisEntryToDim.push_back(std::nullopt);
      basis.push_back(jumpSize);
      previousSizes *= jumpSize;
    }

    basisEntryToDim.push_back(dim);
    basis.push_back(size);
    previousSizes *= size;
  }

  // Post-process. The basis is backwards and the permutation
  // we've constructed is the inverse of what we need.
  std::reverse(basis.begin(), basis.end());
  size_t basisLength = basis.size();
  dimToResult.assign(numDims, ~0);
  for (auto [reverseBasisPos, dimPos] : llvm::enumerate(basisEntryToDim)) {
    if (!dimPos)
      continue;
    // There's an extra overflow term at the front of the delineraize results,
    // so this subtraction lands in the [1, basisLength] range we need it
    // to be in.
    dimToResult[*dimPos] = basisLength - reverseBasisPos;
  }
  return success();
}

/// Visit affine expressions recursively and calculate the coefficient of the
/// dimension `position`. If the dimension doesn't exist in the expression, this
/// returns `0`. If the coefficient can't be calculated, for example in case of
/// invalid expressions like modulo, this returns `std::nullopt`.
class CoefficientFinder
    : public AffineExprVisitor<CoefficientFinder, std::optional<int64_t>> {
public:
  CoefficientFinder(unsigned position) : position(position) {}

  std::optional<int64_t> visitConstantExpr(AffineConstantExpr expr) {
    // Coefficients are handled through the multiplication visitor.
    return 0;
  }

  std::optional<int64_t> visitDimExpr(AffineDimExpr expr) {
    if (expr.getPosition() == position) {
      return 1;
    }
    return 0;
  }

  std::optional<int64_t> visitAddExpr(AffineBinaryOpExpr expr) {
    std::optional<int64_t> lhsVal = visit(expr.getLHS());
    std::optional<int64_t> rhsVal = visit(expr.getRHS());
    if (!lhsVal.has_value() || !rhsVal.has_value()) {
      return std::nullopt;
    }
    return lhsVal.value() + rhsVal.value();
  }

  std::optional<int64_t> visitMulExpr(AffineBinaryOpExpr expr) {
    std::optional<int64_t> lhsVal = visit(expr.getLHS());
    std::optional<int64_t> rhsVal = visit(expr.getRHS());
    if (!lhsVal.has_value() || !rhsVal.has_value()) {
      return std::nullopt;
    }
    if (auto lhsConst = dyn_cast<AffineConstantExpr>(expr.getLHS())) {
      return lhsConst.getValue() * rhsVal.value();
    }
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      return rhsConst.getValue() * lhsVal.value();
    }
    return lhsVal.value() * rhsVal.value();
  }

  // We disallow mod, floor div and ceil div on the position.
  std::optional<int64_t> visitModExpr(AffineBinaryOpExpr expr) {
    return visitInvalidExpr(expr);
  }
  std::optional<int64_t> visitFloorDivExpr(AffineBinaryOpExpr expr) {
    return visitInvalidExpr(expr);
  }
  std::optional<int64_t> visitCeilDivExpr(AffineBinaryOpExpr expr) {
    return visitInvalidExpr(expr);
  }

private:
  /// Utility binary expression visitor method. Returns `std::nullopt` if
  /// either the lhs or rhs has an invalid operation on the dimension expression
  /// with the provided `position`.
  std::optional<int64_t> visitInvalidExpr(AffineBinaryOpExpr expr) {
    std::optional<int64_t> lhsVal = visit(expr.getLHS());
    std::optional<int64_t> rhsVal = visit(expr.getRHS());
    if ((lhsVal.has_value() && lhsVal.value() != 0) ||
        (rhsVal.has_value() && rhsVal.value() != 0)) {
      return std::nullopt;
    }
    return 0;
  }
  /// The dimension for which to return the coefficient.
  unsigned position;
};

std::optional<int64_t> getCoefficient(AffineExpr expr, unsigned position) {
  CoefficientFinder coefficientFinder(position);
  return coefficientFinder.visit(expr);
}

std::optional<int64_t> getCoefficient(Value a, Value b) {
  if (a == b) {
    return 1;
  }
  auto applyOp = dyn_cast_if_present<affine::AffineApplyOp>(a.getDefiningOp());
  if (!applyOp) {
    return std::nullopt;
  }
  AffineMap affineMap = applyOp.getAffineMap();
  if (affineMap.getNumResults() != 1) {
    return std::nullopt;
  }
  std::optional<unsigned> operandIndex;
  for (auto [i, mapOperand] : llvm::enumerate(applyOp.getMapOperands())) {
    if (mapOperand == b) {
      // Fail if the b is passed in multiple times to the AffineApplyOp.
      if (operandIndex.has_value()) {
        return std::nullopt;
      }
      operandIndex = i;
    }
  }
  if (!operandIndex.has_value()) {
    return std::nullopt;
  }
  return getCoefficient(affineMap.getResult(0), operandIndex.value());
}

bool isUnitFunctionOf(Value a, Value b) {
  std::optional<int64_t> coeff = getCoefficient(a, b);
  return coeff && *coeff == 1;
}

} // namespace mlir::iree_compiler
