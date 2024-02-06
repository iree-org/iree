// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_INDEXING_IR_INDEXINGINTERFACES_H_
#define IREE_COMPILER_DIALECT_INDEXING_IR_INDEXINGINTERFACES_H_

#include "llvm/ADT/APInt.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

using llvm::APInt;

namespace mlir::iree_compiler::IREE::Indexing {

//===----------------------------------------------------------------------===//
// SignedIndexRange
//===----------------------------------------------------------------------===//

/// Idiomatic saturated range-like type to represent saturated range arithmetic.
struct SaturatedIndexRange {
  bool saturatedMin = true;
  bool saturatedMax = true;
  int64_t minValue = 0;
  int64_t maxValue = 0;
  int64_t alignment = 1;

  static SaturatedIndexRange getConstantRange(int64_t cst);
  static SaturatedIndexRange getAligned(int64_t minValue, int64_t maxValue,
                                        int64_t alignment);
  static SaturatedIndexRange getAlignedMin(int64_t minValue, int64_t alignment);
  static SaturatedIndexRange getAlignedMax(int64_t maxValue, int64_t alignment);
  static SaturatedIndexRange getAligned(bool sMin, bool sMax, int64_t minValue,
                                        int64_t maxValue, int64_t alignment);

  SaturatedIndexRange()
      : saturatedMin(true), saturatedMax(true), minValue(0), maxValue(0),
        alignment(1) {}
  SaturatedIndexRange(int64_t alignment)
      : saturatedMin(true), saturatedMax(true), minValue(0), maxValue(0),
        alignment(alignment) {}
  SaturatedIndexRange(int64_t min, int64_t max, int64_t alignment)
      : saturatedMin(false), saturatedMax(false), minValue(min), maxValue(max),
        alignment(alignment) {}
  SaturatedIndexRange(bool sMin, bool sMax, int64_t min, int64_t max,
                      int64_t alignment)
      : saturatedMin(sMin), saturatedMax(sMax), minValue(min), maxValue(max),
        alignment(alignment) {}
  bool operator==(const SaturatedIndexRange &other) const {
    return saturatedMin == other.saturatedMin &&
           saturatedMax == other.saturatedMax && minValue == other.minValue &&
           maxValue == other.maxValue;
  }
  bool operator!=(const SaturatedIndexRange &other) const {
    return !(*this == other);
  }

  bool isPositiveRange() const { return !saturatedMin && minValue > 0; }
  bool isNonNegativeRange() const { return !saturatedMin && minValue >= 0; }

  bool isNegativeRange() const { return !saturatedMax && maxValue < 0; }
  bool isNonPositiveRange() const { return !saturatedMax && maxValue <= 0; }

  bool isConstant() const {
    return !saturatedMin && !saturatedMax && minValue == maxValue;
  }
  bool isZero() const { return isConstant() && minValue == 0; }

  SaturatedIndexRange getUnion(const SaturatedIndexRange &other) const;
  SaturatedIndexRange getIntersection(const SaturatedIndexRange &other) const;

  SaturatedIndexRange operator+(const SaturatedIndexRange &other) const;
  SaturatedIndexRange operator-(const SaturatedIndexRange &other) const;
  SaturatedIndexRange operator*(const SaturatedIndexRange &other) const;

  const std::string getAsStr() const;
};

using SaturatedIndexRangeList = SmallVector<SaturatedIndexRange>;
using SaturatedIndexRangeArray = ArrayRef<SaturatedIndexRange>;

using SaturatedValueRange =
    std::variant<SaturatedIndexRange, SaturatedIndexRangeArray>;

} // namespace mlir::iree_compiler::IREE::Indexing

#include "iree/compiler/Dialect/Indexing/IR/IndexingOpInterfaces.h.inc" // IWYU pragma: export

#endif // IREE_COMPILER_DIALECT_INDEXING_IR_INDEXINGINTERFACES_H_
