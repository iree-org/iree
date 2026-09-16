// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_QUANTIZATIONUTILS_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_QUANTIZATIONUTILS_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"

namespace mlir::iree_compiler::GlobalOptimization {

static constexpr unsigned kAccumulatorWidth = 32;

struct QuantMinMax {
  int64_t min;
  int64_t max;
};

/// Maximum magnitude representable by the quantized storage grid. Returns zero
/// when bitWidth is zero or the grid cannot be accumulated in i32.
/// For i8 -> 128
///   Minimum: -128 -> magnitude 128
///   Maximum: 127 -> magnitude 127
int64_t getStorageMagnitude(unsigned bitWidth, bool isUnsigned);

/// Conservative maximum magnitude of a dequantized operand's integer
/// difference = |input| + |zero_point|
/// Since: |input - zero_point| <= |input| + |zero_point|
/// this overestimates bounds because it assumes the worst-case alignment
/// of input and zero point magnitudes.
int64_t getDifferenceMagnitude(unsigned storageBitWidth, bool storageIsUnsigned,
                               std::optional<QuantMinMax> quantMinMax,
                               bool isSymmetric);

/// Maximum positive reduction extent N satisfying
/// N * lhsMagnitude * rhsMagnitude <= INT32_MAX. The magnitudes bound
/// |input| + |zero_point|, so this also bounds intermediate correction terms.
/// Returns zero for nonpositive magnitudes or when no positive N is safe.
/// Uses division to avoid overflowing when the magnitudes themselves are large.
int64_t getMaxReductionExtent(int64_t lhsMagnitude, int64_t rhsMagnitude);

/// Product of a nonempty list of positive static integer reduction extents,
/// provided it does not exceed maxExtent. Returns nullopt for empty lists,
/// nonpositive extents (including dynamic sentinels), or an exceeded limit.
/// Checks the limit before multiplying to avoid overflow.
std::optional<int64_t>
getBoundedReductionExtent(llvm::ArrayRef<int64_t> extents, int64_t maxExtent);

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_QUANTIZATIONUTILS_H_
