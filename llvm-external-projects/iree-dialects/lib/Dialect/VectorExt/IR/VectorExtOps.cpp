// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include <numeric>

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

//===----------------------------------------------------------------------===//
// LayoutConflictResolutionOp
//===----------------------------------------------------------------------===//

LogicalResult validateLayout(Operation *op, StringRef label,
                             VectorLayoutInterface layout,
                             ArrayRef<int64_t> inputShape) {
  if (!layout.isValidLayout(inputShape)) {
    return op->emitError(
        "The " + label +
        " layout shape cannot be distributed over the given vector shape.");
  }
  return success();
}

// Validate that the desired layout has the same shape as the input.
LogicalResult LayoutConflictResolutionOp::verify() {
  Operation *op = getOperation();
  ArrayRef<int64_t> inputShape =
      cast<VectorType>(getInput().getType()).getShape();
  if (succeeded(validateLayout(op, "source", getSourceLayout(), inputShape)))
    return validateLayout(op, "desired", getDesiredLayout(), inputShape);
  return failure();
}

// to_simd -> to_simt
OpFoldResult ToSIMDOp::fold(FoldAdaptor) {
  if (auto simtOp = getOperand().getDefiningOp<ToSIMTOp>()) {
    return simtOp.getOperand();
  }
  return {};
}

// to_simt -> to_simd
OpFoldResult ToSIMTOp::fold(FoldAdaptor) {
  if (auto simdOp = getOperand().getDefiningOp<ToSIMDOp>()) {
    return simdOp.getOperand();
  }
  return {};
}

LayoutIterator &LayoutIterator::operator++() {
  for (auto &[dim, it] : state) {
    if (frozenDimensions.contains(dim))
      continue;
    if (it != ranges[dim].end()) {
      ++it;
      break;
    }
    it = ranges[dim].begin();
  }
  return *this;
}

void LayoutIterator::maybeFreezeAndConcatenate(
    const LayoutIterator &frozenIterator) {
  for (auto &[frozenDim, frozenIt] : frozenIterator.getState()) {
    if (!state.contains(frozenDim)) {
      frozenDimensions.insert(frozenDim);
      state[frozenDim] = frozenIt;
    }
  }
}

static bool isLaneDimension(LayoutDimension dim) {
  return (dim == LayoutDimension::LANEX) || (dim == LayoutDimension::LANEY) ||
         (dim == LayoutDimension::LANEZ);
}

void LayoutIterator::initialize(PerDimLayoutAttr &attr,
                                DenseMap<LayoutDimension, int64_t> strides) {
  auto reversedLabels = llvm::reverse(attr.getLabels());
  auto reversedShapes = llvm::reverse(attr.getShapes());
  for (auto [nameAttr, shape] : llvm::zip(reversedLabels, reversedShapes)) {
    LayoutDimension dim = nameAttr.getValue();
    if (isLaneDimension(dim))
      continue;
    int64_t stride = strides.contains(dim) ? strides[dim] : 1;
    ranges[dim] = DimensionalRange(0, shape - 1, stride);
    state[dim] = ranges[dim].begin();
  }
}

LayoutIterator::LayoutIterator(LayoutAttr &attr,
                               DenseMap<LayoutDimension, int64_t> strides) {
  for (PerDimLayoutAttr perDimAttr : attr.getLayouts()) {
    initialize(perDimAttr, strides);
  }
}

LayoutIterator::LayoutIterator(PerDimLayoutAttr &attr,
                               DenseMap<LayoutDimension, int64_t> strides) {
  initialize(attr, strides);
}

/// The iterator is done when it returns back to
/// its begin state.
bool LayoutIterator::iterationComplete() {
  bool complete{true};
  for (auto &[dim, it] : state) {
    if (frozenDimensions.contains(dim))
      continue;
    if (it != ranges[dim].begin()) {
      complete = false;
      break;
    }
  }
  return complete;
}

void LayoutIterator::apply(
    std::function<void(const LayoutIterator::State &)> callback) {
  do {
    callback(state);
    ++(*this);
  } while (!iterationComplete());
}

// clang-format off
#define GET_OP_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format on
