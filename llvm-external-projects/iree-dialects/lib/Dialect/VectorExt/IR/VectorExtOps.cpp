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

void LayoutIterator::maybeFreezeAndConcatenate(
    const LayoutIterator::State &frozenState) {
  for (auto &[frozenDim, frozenIt] : frozenState.iterators) {
    if (!state.contains(frozenDim)) {
      frozenDimensions.insert(frozenDim);
      state[frozenDim] = frozenIt;
    }
  }
}

void LayoutIterator::initialize(const PerDimLayoutAttr &attr,
                                DenseMap<LayoutDimension, int64_t> strides,
                                std::optional<int64_t> simdIndex) {
  auto reversedLabels = llvm::reverse(attr.getLabels());
  auto reversedShapes = llvm::reverse(attr.getShapes());
  for (auto [nameAttr, shape] : llvm::zip(reversedLabels, reversedShapes)) {
    LayoutDimension dim = nameAttr.getValue();
    if (isLaneDimension(dim))
      continue;
    int64_t stride = strides.contains(dim) ? strides[dim] : 1;
    state.ranges[dim] = DimensionalRange(0, shape, stride);
    state.iterators[dim] = state.ranges[dim].begin();
    maxIterations *= shape / stride;
    if (simdIndex) {
      int64_t index = simdIndex.value();
      if (!state.simdToLayoutDim.contains(index))
        state.simdToLayoutDim[index] = {};
      state.simdToLayoutDim[index].insert(dim);
    }
  }
}

LayoutIterator::LayoutIterator(LayoutAttr &attr,
                               DenseMap<LayoutDimension, int64_t> strides) {
  for (auto perDimAttr : llvm::enumerate(attr.getLayouts())) {
    initialize(perDimAttr.value(), strides, perDimAttr.index());
  }
}

LayoutIterator::LayoutIterator(LayoutAttr &attr) {
  DenseMap<LayoutDimension, int64_t> strides;
  for (auto perDimAttr : llvm::enumerate(attr.getLayouts())) {
    initialize(perDimAttr.value(), strides, perDimAttr.index());
  }
}

LayoutIterator::LayoutIterator(LayoutAttr &attr,
                               DenseMap<LayoutDimension, int64_t> strides,
                               int64_t simtIndex) {
  for (auto perDimAttr : llvm::enumerate(attr.getLayouts())) {
    if (perDimAttr.index() != simtIndex)
      continue;
    initialize(perDimAttr.value(), strides, perDimAttr.index());
  }
}

LayoutIterator::LayoutIterator(LayoutAttr &attr, int64_t simtIndex) {
  DenseMap<LayoutDimension, int64_t> strides;
  for (auto perDimAttr : llvm::enumerate(attr.getLayouts())) {
    if (perDimAttr.index() != simtIndex)
      continue;
    initialize(perDimAttr.value(), strides, perDimAttr.index());
  }
}

LayoutIterator::LayoutIterator(PerDimLayoutAttr &attr,
                               DenseMap<LayoutDimension, int64_t> strides) {
  initialize(attr, strides, std::nullopt);
}

LayoutIterator &LayoutIterator::operator++() {
  for (auto &[dim, it] : state.iterators) {
    if (frozenDimensions.contains(dim))
      continue;
    ++it;
    if (it == state.ranges[dim].end()) {
      it = state.ranges[dim].begin();
      continue;
    }
    break;
  }
  iterations++;
  return *this;
}

/// The iterator is done when all the loops are complete.
bool LayoutIterator::iterationComplete() { return iterations == maxIterations; }

void LayoutIterator::apply(
    std::function<void(const LayoutIterator::State &)> callback) {
  for (; !iterationComplete(); ++(*this)) {
    callback(state);
  }
}

// Get the offset into the SIMT vector corresponding to the incoming iterator.
// The returned offsets will always be the same shape as the labels array.
// Groups vector dimensions together. Assumes last dimension is vector
// dimension.
SmallVector<int64_t> LayoutIterator::State::computeSIMTIndex() const {
  SmallVector<int64_t> offset;
  std::optional<int64_t> vecOffset;
  for (auto label : labels) {
    for (auto [name, it] : iterators) {
      if (name != label)
        continue;
      if (isBatchDimension(name)) {
        offset.push_back(it.getPosition());
        continue;
      }
      if (isVectorDimension(name)) {
        int64_t step{1};
        if (name == LayoutDimension::VECTORY) {
          step = ranges.lookup(LayoutDimension::VECTORX).stop;
        }
        vecOffset = vecOffset.value_or(0) + it.getPosition() * step;
      }
    }
  }
  if (vecOffset)
    offset.push_back(vecOffset.value());
  return offset;
}

SmallVector<int64_t>
LayoutIterator::State::computeIteratorProjectedSIMTIndex() const {
  SmallVector<int64_t> indices = computeSIMTIndex();
  SmallVector<int64_t> projectedIndices;
  for (int i = 0; i < labels.size(); i++) {
    for (auto [name, it] : iterators) {
      if (name == labels[i])
        projectedIndices.push_back(indices[i]);
    }
  }
  return projectedIndices;
}

void LayoutIterator::erase(LayoutDimension dim) {
  if (state.contains(dim))
    state.erase(dim);
}

LayoutIterator LayoutIterator::getBatchIterator() const {
  LayoutIterator projectedIterator = *this;
  for (auto [dim, it] : state.iterators) {
    if (!isBatchDimension(dim)) {
      DimensionalRange range = state.ranges.lookup(dim);
      projectedIterator.maxIterations /= (range.stop / range.step);
      projectedIterator.erase(dim);
    }
  }
  return projectedIterator;
}

// clang-format off
#define GET_OP_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format on
