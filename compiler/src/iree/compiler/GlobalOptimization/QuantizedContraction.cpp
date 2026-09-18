// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/QuantizedContraction.h"

#include "iree/compiler/GlobalOptimization/QuantizationUtils.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

namespace mlir::iree_compiler::GlobalOptimization::detail {

static std::optional<QuantizedOperandRanges>
getOperandRanges(DequantizeAffineOp dequantize) {
  unsigned bitWidth =
      cast<IntegerType>(dequantize.getInputType().getElementType()).getWidth();
  std::optional<QuantMinMax> quantMinMax;
  if (auto quantMin = dequantize.getQuantMin()) {
    quantMinMax = QuantMinMax{*quantMin, *dequantize.getQuantMax()};
  }
  return getQuantizedOperandRanges(bitWidth, dequantize.getInputUnsigned(),
                                   quantMinMax, dequantize.isSymmetric());
}

/// Matches exactly two scalar operations: input multiplication and accumulator
/// addition, in either operand order. Casts and other intervening operations
/// prevent matching.
static bool hasPlainFloatMulAddBody(linalg::LinalgOp op) {
  Block *body = op.getBlock();
  if (body->getNumArguments() != 3 || body->getOperations().size() != 3) {
    return false;
  }
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != 1) {
    return false;
  }
  auto addition = yield.getValues()[0].getDefiningOp<arith::AddFOp>();
  if (!addition) {
    return false;
  }
  Value accumulator = body->getArgument(2);
  Value product;
  if (addition.getLhs() == accumulator) {
    product = addition.getRhs();
  } else if (addition.getRhs() == accumulator) {
    product = addition.getLhs();
  } else {
    return false;
  }
  auto multiplication = product.getDefiningOp<arith::MulFOp>();
  if (!multiplication) {
    return false;
  }
  Value lhs = body->getArgument(0);
  Value rhs = body->getArgument(1);
  return (multiplication.getLhs() == lhs && multiplication.getRhs() == rhs) ||
         (multiplication.getLhs() == rhs && multiplication.getRhs() == lhs);
}

/// Rebase dequantization indexing maps onto the original contraction's
/// iteration space. The inverse output map maps dequantized data space back to
/// the dequantization iteration space, accounting for folded permutations.
static QuantizedOperand getQuantizedOperand(DequantizeAffineOp dequantize,
                                            AffineMap operandMap) {
  AffineMap toDequantizeIteration =
      inversePermutation(dequantize.getOutputMap()).compose(operandMap);
  return {dequantize, dequantize.getInputMap().compose(toDequantizeIteration),
          dequantize.getScaleMap().compose(toDequantizeIteration),
          dequantize.isSymmetric()
              ? AffineMap()
              : dequantize.getZeroPointMap().compose(toDequantizeIteration)};
}

FailureOr<QuantizedContraction> getQuantizedContraction(linalg::LinalgOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      !op.hasPureTensorSemantics() || !hasPlainFloatMulAddBody(op)) {
    return failure();
  }
  auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat())) {
    return failure();
  }

  auto lhs = op.getDpsInputs()[0].getDefiningOp<DequantizeAffineOp>();
  auto rhs = op.getDpsInputs()[1].getDefiningOp<DequantizeAffineOp>();
  if (!lhs || !rhs) {
    return failure();
  }
  QuantizedContraction detail;
  detail.lhs = getQuantizedOperand(
      lhs, op.getMatchingIndexingMap(op.getDpsInputOperand(0)));
  detail.rhs = getQuantizedOperand(
      rhs, op.getMatchingIndexingMap(op.getDpsInputOperand(1)));

  detail.outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  // The output map has to be a projected permutation for the epilogue to be
  // expressible over the output data space and remaining reduction dimensions.
  if (!detail.outputMap.isProjectedPermutation()) {
    return failure();
  }

  auto lhsRanges = getOperandRanges(lhs);
  auto rhsRanges = getOperandRanges(rhs);
  if (!lhsRanges || !rhsRanges) {
    return failure();
  }
  int64_t maxReductionExtent = getMaxReductionExtent(*lhsRanges, *rhsRanges);
  if (maxReductionExtent == 0) {
    return failure();
  }

  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  SmallVector<int64_t> integerReductionExtents;
  for (auto [dim, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != utils::IteratorType::reduction) {
      continue;
    }
    // Require each reduction dim to select elements from both inputs that
    // contribute to the same output element, as k does in
    // C[m, n] = sum_k A[m, k] * B[k, n]. Reject other reduction structures.
    if (detail.outputMap.isFunctionOfDim(dim) ||
        !detail.lhs.inputMap.isFunctionOfDim(dim) ||
        !detail.rhs.inputMap.isFunctionOfDim(dim)) {
      return failure();
    }
    // Integer reduction requires scales and zero points to be invariant along
    // this dim, so scales can be applied after summation and zero points can
    // be factored into the operand-sum corrections. For example:
    //   sum_k ((Aq[k] - zA) * sA) * ((Bq[k] - zB) * sB)
    //     = sA * sB * sum_k (Aq[k] - zA) * (Bq[k] - zB).
    // If a parameter varies, retain this dim as a parallel dim in the integer
    // contraction, then scale and reduce its partial results in the floating-
    // point epilogue. For block quantization, the block dim takes this path,
    // while reduction within each block can still use integer arithmetic.
    // Symmetric operands have no zero-point map to inspect.
    if (detail.lhs.scaleMap.isFunctionOfDim(dim) ||
        detail.rhs.scaleMap.isFunctionOfDim(dim) ||
        (detail.lhs.zeroPointMap &&
         detail.lhs.zeroPointMap.isFunctionOfDim(dim)) ||
        (detail.rhs.zeroPointMap &&
         detail.rhs.zeroPointMap.isFunctionOfDim(dim))) {
      detail.floatingReductionDims.push_back(dim);
      continue;
    }
    detail.integerReductionDims.push_back(dim);
    integerReductionExtents.push_back(loopRanges[dim]);
  }
  auto reductionExtent =
      getBoundedReductionExtent(integerReductionExtents, maxReductionExtent);
  if (!reductionExtent) {
    return failure();
  }
  detail.reductionExtent = *reductionExtent;
  return detail;
}

AffineMap getExplicitExtentDimsMap(ArrayRef<AffineMap> maps) {
  unsigned rank = maps.front().getNumDims();
  llvm::SmallBitVector namedDims(rank);
  for (AffineMap map : maps) {
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        namedDims.set(dimExpr.getPosition());
      }
    }
  }

  return AffineMap::getMultiDimIdentityMap(rank, maps.front().getContext())
      .dropResults(namedDims);
}

} // namespace mlir::iree_compiler::GlobalOptimization::detail
