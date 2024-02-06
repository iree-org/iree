// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree/compiler/Dialect/Indexing/IR/IndexingInterfaces.h"
#include "iree/compiler/Dialect/Indexing/IR/IndexingOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::Indexing {

//===----------------------------------------------------------------------===//
// SaturatedIndexRange
//===----------------------------------------------------------------------===//

SaturatedIndexRange SaturatedIndexRange::getConstantRange(int64_t cst) {
  return SaturatedIndexRange(cst, cst, cst == 0 ? 1 : cst);
}

SaturatedIndexRange SaturatedIndexRange::getAligned(int64_t minValue,
                                                    int64_t maxValue,
                                                    int64_t alignment) {
  if (minValue == maxValue) {
    assert(minValue % alignment == 0 && "invalid alignment for constant value");
    return SaturatedIndexRange(minValue, minValue,
                               minValue == 0 ? alignment : minValue);
  }
  if (minValue % alignment != 0) {
    // Always align the minimum value upwards.
    minValue = minValue >= 0 ? (minValue / alignment) * alignment + alignment
                             : (minValue / alignment) * alignment;
  }
  if (maxValue % alignment != 0) {
    // Always align the maximum value downwards.
    maxValue = maxValue < 0 ? (maxValue / alignment) * alignment - alignment
                            : (maxValue / alignment) * alignment;
  }
  return SaturatedIndexRange(minValue, maxValue,
                             minValue == maxValue && minValue != 0 ? minValue
                                                                   : alignment);
}

SaturatedIndexRange SaturatedIndexRange::getAlignedMin(int64_t minValue,
                                                       int64_t alignment) {
  if (minValue % alignment != 0) {
    // Always align the minimum value upwards.
    minValue = minValue >= 0 ? (minValue / alignment) * alignment + alignment
                             : (minValue / alignment) * alignment;
  }
  return SaturatedIndexRange(/*saturatedMin=*/false, /*saturatedMax=*/true,
                             minValue, 0, alignment);
}

SaturatedIndexRange SaturatedIndexRange::getAlignedMax(int64_t maxValue,
                                                       int64_t alignment) {
  if (maxValue % alignment != 0) {
    // Always align the maximum value downwards.
    maxValue = maxValue < 0 ? (maxValue / alignment) * alignment - alignment
                            : (maxValue / alignment) * alignment;
  }
  return SaturatedIndexRange(/*saturatedMin=*/true, /*saturatedMax=*/false, 0,
                             maxValue, alignment);
}

SaturatedIndexRange SaturatedIndexRange::getAligned(bool sMin, bool sMax,
                                                    int64_t minValue,
                                                    int64_t maxValue,
                                                    int64_t alignment) {
  if (sMin && sMax) {
    return SaturatedIndexRange(alignment);
  }
  if (sMin) {
    return SaturatedIndexRange::getAlignedMax(maxValue, alignment);
  }
  if (sMax) {
    return SaturatedIndexRange::getAlignedMin(minValue, alignment);
  }
  return SaturatedIndexRange::getAligned(minValue, maxValue, alignment);
}

SaturatedIndexRange
SaturatedIndexRange::getUnion(const SaturatedIndexRange &other) const {
  bool sMin = saturatedMin && other.saturatedMin;
  bool sMax = saturatedMax && other.saturatedMax;
  int64_t newMin = 0;
  if (!sMin) {
    if (saturatedMin) {
      newMin = other.minValue;
    } else if (other.saturatedMin) {
      newMin = minValue;
    } else if (!sMin) {
      newMin = std::max(minValue, other.minValue);
    }
  }
  int64_t newMax = 0;
  if (!sMax) {
    if (saturatedMax) {
      newMax = other.maxValue;
    } else if (other.saturatedMax) {
      newMax = maxValue;
    } else {
      newMax = std::min(maxValue, other.maxValue);
    }
  }
  int64_t newAlignment = std::lcm(alignment, other.alignment);

  return SaturatedIndexRange::getAligned(sMin, sMax, newMin, newMax,
                                         newAlignment);
}

SaturatedIndexRange
SaturatedIndexRange::getIntersection(const SaturatedIndexRange &other) const {
  bool sMin = saturatedMin || other.saturatedMin;
  bool sMax = saturatedMax || other.saturatedMax;
  int64_t newMin = sMin ? 0 : std::min(minValue, other.minValue);
  int64_t newMax = sMax ? 0 : std::max(maxValue, other.maxValue);
  int64_t newAlignment = std::gcd(alignment, other.alignment);

  return SaturatedIndexRange::getAligned(sMin, sMax, newMin, newMax,
                                         newAlignment);
}

SaturatedIndexRange
SaturatedIndexRange::operator+(const SaturatedIndexRange &other) const {
  bool sMin = saturatedMin || other.saturatedMin;
  bool sMax = saturatedMax || other.saturatedMax;
  int64_t newMin = minValue + other.minValue;
  int64_t newMax = maxValue + other.maxValue;
  int64_t newAlignment = std::gcd(alignment, other.alignment);

  return SaturatedIndexRange::getAligned(sMin, sMax, newMin, newMax,
                                         newAlignment);
}

SaturatedIndexRange
SaturatedIndexRange::operator-(const SaturatedIndexRange &other) const {
  bool sMin = saturatedMin || other.saturatedMax;
  bool sMax = saturatedMax || other.saturatedMin;
  int64_t newMin = minValue - other.maxValue;
  int64_t newMax = maxValue - other.minValue;
  int64_t newAlignment = std::gcd(alignment, other.alignment);

  return SaturatedIndexRange::getAligned(sMin, sMax, newMin, newMax,
                                         newAlignment);
}

SaturatedIndexRange
SaturatedIndexRange::operator*(const SaturatedIndexRange &other) const {
  int64_t newAlignment = alignment * other.alignment;

  if (isZero() || other.isZero()) {
    return SaturatedIndexRange(0, 0, newAlignment);
  }

  if ((saturatedMin && saturatedMax) ||
      (other.saturatedMin && other.saturatedMax)) {
    return SaturatedIndexRange(newAlignment);
  }

  bool sMin = (saturatedMin && !other.isNonPositiveRange()) ||
              (other.saturatedMin && !isNonPositiveRange()) ||
              (saturatedMax && !isNonNegativeRange()) ||
              (other.saturatedMax && !isNonNegativeRange());

  bool sMax = (saturatedMax && !other.isNonPositiveRange()) ||
              (other.saturatedMax && !isNonPositiveRange()) ||
              (saturatedMin && !isNonNegativeRange()) ||
              (other.saturatedMin && !isNonNegativeRange());

  int64_t minMin = minValue * other.minValue;
  int64_t minMax = minValue * other.maxValue;
  int64_t maxMin = maxValue * other.minValue;
  int64_t maxMax = maxValue * other.maxValue;

  bool satMinMin = saturatedMin || other.saturatedMin;
  bool satMinMax = saturatedMin || other.saturatedMax;
  bool satMaxMin = saturatedMax || other.saturatedMin;
  bool satMaxMax = saturatedMax || other.saturatedMax;

  int64_t smallest = INT64_MAX;
  if (!sMin) {
    if (!satMinMin)
      smallest = std::min(smallest, minMin);
    if (!satMinMax)
      smallest = std::min(smallest, minMax);
    if (!satMaxMin)
      smallest = std::min(smallest, maxMin);
    if (!satMaxMax)
      smallest = std::min(smallest, maxMax);
  }

  int64_t largest = INT64_MIN;
  if (!sMax) {
    if (!satMinMin)
      largest = std::max(largest, minMin);
    if (!satMinMax)
      largest = std::max(largest, minMax);
    if (!satMaxMin)
      largest = std::max(largest, maxMin);
    if (!satMaxMax)
      largest = std::max(largest, maxMax);
  }

  return SaturatedIndexRange::getAligned(sMin, sMax, smallest, largest,
                                         newAlignment);
}

const std::string SaturatedIndexRange::getAsStr() const {
  std::stringstream ss;
  ss << "<[";
  if (saturatedMin) {
    ss << "UNBOUNDED";
  } else {
    ss << minValue;
  }
  ss << ", ";
  if (saturatedMax) {
    ss << "UNBOUNDED";
  } else {
    ss << maxValue;
  }
  ss << "], " << alignment << ">";
  return ss.str();
}

#include "iree/compiler/Dialect/Indexing/IR/IndexingOpInterfaces.cpp.inc"

} // namespace mlir::iree_compiler::IREE::Indexing
