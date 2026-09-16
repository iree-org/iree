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

int64_t getDifferenceMagnitude(unsigned storageBitWidth, bool storageIsUnsigned,
                               std::optional<QuantMinMax> quantMinMax,
                               bool isSymmetric) {
  int64_t storageMagnitude =
      getStorageMagnitude(storageBitWidth, storageIsUnsigned);
  if (!storageMagnitude) {
    return 0;
  }

  int64_t inputMagnitude = storageMagnitude;
  if (quantMinMax) {
    inputMagnitude = std::max(-quantMinMax->min, quantMinMax->max);
  }
  // A zero point is a value on the input's quantized grid. Its SSA carrier
  // type does not enlarge that grid; PT2E commonly uses i64 for i8 values.
  int64_t zeroPointMagnitude = isSymmetric ? 0 : storageMagnitude;
  return inputMagnitude + zeroPointMagnitude;
}

int64_t getMaxReductionExtent(int64_t lhsMagnitude, int64_t rhsMagnitude) {
  if (lhsMagnitude <= 0 || rhsMagnitude <= 0) {
    return 0;
  }
  return llvm::maxIntN(kAccumulatorWidth) / lhsMagnitude / rhsMagnitude;
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
