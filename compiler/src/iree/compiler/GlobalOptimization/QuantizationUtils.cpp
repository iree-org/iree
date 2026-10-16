// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/QuantizationUtils.h"

#include <algorithm>

#include "llvm/Support/MathExtras.h"

namespace mlir::iree_compiler::GlobalOptimization {

int64_t getStorageMagnitude(unsigned bitWidth, bool isUnsigned) {
  if (bitWidth == 0 || bitWidth >= kAccumulatorWidth) {
    return 0;
  }
  // The signed minimum has magnitude one greater than the signed maximum.
  return isUnsigned ? llvm::maxUIntN(bitWidth) : llvm::maxIntN(bitWidth) + 1;
}

std::optional<QuantizedOperandRanges>
getQuantizedOperandRanges(unsigned storageBitWidth, bool storageIsUnsigned,
                          std::optional<QuantMinMax> quantMinMax,
                          bool isSymmetric) {
  int64_t magnitude = getStorageMagnitude(storageBitWidth, storageIsUnsigned);
  if (!magnitude) {
    return std::nullopt;
  }
  QuantMinMax storage = storageIsUnsigned
                            ? QuantMinMax{0, magnitude}
                            : QuantMinMax{-magnitude, magnitude - 1};
  QuantMinMax input = quantMinMax.value_or(storage);
  QuantMinMax zeroPoint = isSymmetric ? QuantMinMax{0, 0} : storage;
  return QuantizedOperandRanges{input, zeroPoint};
}

static QuantMinMax multiplyRanges(QuantMinMax lhs, QuantMinMax rhs) {
  auto products = {lhs.min * rhs.min, lhs.min * rhs.max, lhs.max * rhs.min,
                   lhs.max * rhs.max};
  return {*std::min_element(products.begin(), products.end()),
          *std::max_element(products.begin(), products.end())};
}

static QuantMinMax subtractRanges(QuantMinMax lhs, QuantMinMax rhs) {
  return {lhs.min - rhs.max, lhs.max - rhs.min};
}

int64_t getMaxReductionExtent(const QuantizedOperandRanges &lhs,
                              const QuantizedOperandRanges &rhs) {
  QuantMinMax a = lhs.input, za = lhs.zeroPoint;
  QuantMinMax b = rhs.input, zb = rhs.zeroPoint;
  QuantMinMax product = multiplyRanges(a, b);
  QuantMinMax lhsCorrection = multiplyRanges(zb, a);
  QuantMinMax rhsCorrection = multiplyRanges(za, b);
  QuantMinMax crossTerm = multiplyRanges(za, zb);
  QuantMinMax correctionSum = {lhsCorrection.min + rhsCorrection.min,
                               lhsCorrection.max + rhsCorrection.max};
  QuantMinMax correction = subtractRanges(correctionSum, crossTerm);
  // Bound the final result using (Aq-zA)*(Bq-zB), retaining the cancellation
  // lost by independently bounding D - correction.
  QuantMinMax corrected =
      multiplyRanges(subtractRanges(a, za), subtractRanges(b, zb));

  // Storage is at most 31 bits. Centered values have magnitude <= INT32_MAX;
  // the products and correction range endpoints above all fit in int64_t.
  // Include the operand sums as well as each product and correction stage.
  int64_t maxMagnitude = 0;
  for (QuantMinMax range : {a, b, product, lhsCorrection, rhsCorrection,
                            crossTerm, correctionSum, correction, corrected}) {
    maxMagnitude = std::max({maxMagnitude, -range.min, range.max});
  }
  return maxMagnitude > 0 ? llvm::maxIntN(kAccumulatorWidth) / maxMagnitude : 0;
}

std::optional<int64_t>
getBoundedReductionExtent(llvm::ArrayRef<int64_t> extents, int64_t maxExtent) {
  if (extents.empty() || maxExtent <= 0) {
    return std::nullopt;
  }
  int64_t product = 1;
  for (int64_t extent : extents) {
    if (extent <= 0 || product > maxExtent / extent) {
      return std::nullopt;
    }
    product *= extent;
  }
  return product;
}

} // namespace mlir::iree_compiler::GlobalOptimization
