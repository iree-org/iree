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

struct QuantizedOperandRanges {
  QuantMinMax input;
  QuantMinMax zeroPoint;
};

/// Builds input and zero-point ranges from the storage grid. An optional input
/// range must be contained in that grid. Zero points use the full storage grid
/// independently of the input range and carrier type, or [0, 0] when symmetric.
/// Returns nullopt for zero-width storage or widths >= kAccumulatorWidth.
std::optional<QuantizedOperandRanges>
getQuantizedOperandRanges(unsigned storageBitWidth, bool storageIsUnsigned,
                          std::optional<QuantMinMax> quantMinMax,
                          bool isSymmetric);

/// Bounds every integer intermediate per reduction element. Signedness enters
/// through the operand ranges; the same calculation applies to all operands.
///
/// In general, taking |x| to mean the maximum magnitude over x's range, the
/// triangle inequality gives the sufficient condition
///
///   N * (|Aq|*|Bq| + |zB|*|Aq| + |zA|*|Bq| + |zA|*|zB|) <= INT32_MAX,
///
/// which factors exactly into
///
///   N * (|Aq| + |zA|) * (|Bq| + |zB|) <= INT32_MAX.
///
/// This bounds the expanded terms and their additions/subtractions, but loses
/// sign information. For unsigned inputs and zero points, D = sum(Aq*Bq),
/// P = zB*sum(Aq), Q = zA*sum(Bq), and R = N*zA*zB are nonnegative. Thus
/// |P+Q-R| <= max(P+Q, R), rather than P+Q+R. Also, |Aq-zA| is bounded by
/// max(|Aq|, |zA|), rather than |Aq|+|zA|. For full unsigned i8 ranges,
/// D, P, Q, R and the centered result have magnitude <= N*255^2, while
/// P+Q and the correction have magnitude <= 2*N*255^2. The sufficient bound
/// is therefore 2*N*255^2 <= INT32_MAX, versus 4*N*255^2 above.
///
/// Signed values can have opposing signs, so those unsigned inequalities do
/// not apply in general. Propagating intervals through the actual correction
/// stages handles both cases, including mixed signedness, without a special
/// case. The centered expression separately bounds the final result.
///
/// Operands must have ranges on supported storage grids, as produced by
/// getQuantizedOperandRanges. Returns the largest positive N allowed by these
/// bounds, or zero for an all-zero bound or when no positive N is safe.
int64_t getMaxReductionExtent(const QuantizedOperandRanges &lhs,
                              const QuantizedOperandRanges &rhs);

/// Product of a nonempty list of positive static integer reduction extents,
/// provided it does not exceed maxExtent. Returns nullopt for empty lists,
/// nonpositive extents (including dynamic sentinels), or an exceeded limit.
/// Checks the limit before multiplying to avoid overflow.
std::optional<int64_t>
getBoundedReductionExtent(llvm::ArrayRef<int64_t> extents, int64_t maxExtent);

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_QUANTIZATIONUTILS_H_
