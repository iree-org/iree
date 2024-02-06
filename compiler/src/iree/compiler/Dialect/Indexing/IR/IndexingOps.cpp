// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Indexing/IR/IndexingOps.h"

#include "iree/compiler/Dialect/Indexing/IR/IndexingInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::Indexing {

//===----------------------------------------------------------------------===//
// AssertAlignedRangeOp
//===----------------------------------------------------------------------===//

SaturatedIndexRange AssertAlignedRangeOp::getIndexRange(
    Value target, ArrayRef<SaturatedValueRange> operandRanges) {
  std::optional<int64_t> maybeMinValue = getMinValue();
  std::optional<int64_t> maybeMaxValue = getMaxValue();
  bool sMin = maybeMinValue == std::nullopt;
  bool sMax = maybeMaxValue == std::nullopt;
  int64_t minValue = sMin ? 0 : *maybeMinValue;
  int64_t maxValue = sMax ? 0 : *maybeMaxValue;
  SaturatedIndexRange thisRange(sMin, sMax, minValue, maxValue, getAlignment());
  auto newRange =
      thisRange.getUnion(std::get<SaturatedIndexRange>(operandRanges[0]));
  return newRange;
}

std::optional<SaturatedIndexRange>
AssertAlignedRangeOp::initializeRange(Value target, bool &isFixedPoint) {
  std::optional<int64_t> maybeMinValue = getMinValue();
  std::optional<int64_t> maybeMaxValue = getMaxValue();
  bool sMin = maybeMinValue == std::nullopt;
  bool sMax = maybeMaxValue == std::nullopt;
  int64_t minValue = sMin ? 0 : *maybeMinValue;
  int64_t maxValue = sMax ? 0 : *maybeMaxValue;
  SaturatedIndexRange thisRange(sMin, sMax, minValue, maxValue, getAlignment());
  isFixedPoint = false;
  return thisRange;
}

LogicalResult AssertAlignedRangeOp::verify() {
  if (getAlignment() <= 0) {
    emitOpError("invalid non-positive alignment");
    return failure();
  }
  if (getMinValue() && getMaxValue() && *getMinValue() > *getMaxValue()) {
    emitOpError("minimum range value ")
        << *getMinValue() << " must be less than maximum " << *getMaxValue();
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AssertDimRangeOp
//===----------------------------------------------------------------------===//

SaturatedIndexRangeList AssertDimRangeOp::getDynamicDimRanges(
    Value target, ArrayRef<SaturatedValueRange> operandRanges) {
  SaturatedIndexRangeList newRanges(
      std::get<SaturatedIndexRangeArray>(operandRanges[0]));

  int64_t targetDim = 0;
  for (auto [dim, size] :
       llvm::enumerate(llvm::cast<ShapedType>(target.getType()).getShape())) {
    if (ShapedType::isDynamic(size)) {
      if (dim == getDim())
        break;
      targetDim++;
    }
  }

  std::optional<int64_t> maybeMinValue = getMinValue();
  std::optional<int64_t> maybeMaxValue = getMaxValue();
  bool sMin = maybeMinValue == std::nullopt;
  bool sMax = maybeMaxValue == std::nullopt;
  int64_t minValue = sMin ? 0 : *maybeMinValue;
  int64_t maxValue = sMax ? 0 : *maybeMaxValue;

  SaturatedIndexRange thisRange(sMin, sMax, minValue, maxValue, getAlignment());
  newRanges[targetDim] = thisRange.getUnion(newRanges[targetDim]);
  return newRanges;
}

std::optional<SaturatedIndexRangeList>
AssertDimRangeOp::initializeDimRanges(Value target) {
  auto targetType = llvm::cast<ShapedType>(target.getType());
  // We can ignore static dims as we aren't expected to store any ranges
  // for them.
  if (!targetType.isDynamicDim(getDim())) {
    return std::nullopt;
  }
  int64_t numDynamicDims = 0;
  int64_t targetIndex = 0;
  for (auto [dim, size] : llvm::enumerate(targetType.getShape())) {
    if (ShapedType::isDynamic(size)) {
      if (dim == getDim()) {
        targetIndex = numDynamicDims;
      }
      numDynamicDims++;
    }
  }
  SaturatedIndexRangeList ranges(numDynamicDims, SaturatedIndexRange());

  std::optional<int64_t> maybeMinValue = getMinValue();
  std::optional<int64_t> maybeMaxValue = getMaxValue();
  bool sMin = maybeMinValue == std::nullopt;
  bool sMax = maybeMaxValue == std::nullopt;
  int64_t minValue = sMin ? 0 : *maybeMinValue;
  int64_t maxValue = sMax ? 0 : *maybeMaxValue;

  ranges[targetIndex] =
      SaturatedIndexRange(sMin, sMax, minValue, maxValue, getAlignment());
  return ranges;
}

LogicalResult AssertDimRangeOp::verify() {
  int64_t rank = getType().getRank();
  if (getDim() >= rank) {
    emitOpError("asserted dimension out of range");
    return failure();
  }
  if (getAlignment() <= 0) {
    emitOpError("invalid negative alignment");
    return failure();
  }
  if (getMinValue() && getMaxValue() && *getMinValue() > *getMaxValue()) {
    emitOpError("minimum range value ")
        << *getMinValue() << " must be less than maximum " << *getMaxValue();
    return failure();
  }
  if (getMinValue() && *getMinValue() < 0) {
    emitOpError("minimum range value ")
        << *getMinValue() << " must be non-negative";
    return failure();
  }
  return success();
}

} // namespace mlir::iree_compiler::IREE::Indexing

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Indexing/IR/IndexingOps.cpp.inc"
