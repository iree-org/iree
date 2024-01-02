// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_
#define IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtInterfaces.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtEnums.h.inc" // IWYU pragma: export

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h.inc" // IWYU pragma: export

// clang-format on

namespace mlir::iree_compiler::IREE::VectorExt {

/// Dimensional Strided Iterator class used to represent
/// an iterator through a single dimension of the layout.
class DimensionalIterator {
public:
  DimensionalIterator(int64_t position = 0, int64_t stride = 1)
      : position(position), stride(stride) {}
  int64_t operator*() const { return position; }
  DimensionalIterator &operator++() {
    position += stride;
    return *this;
  }
  bool operator!=(const DimensionalIterator &other) const {
    return position != other.position;
  }

private:
  int64_t position, stride;
};

/// Dimensional Range class used to represent the range of
/// a particular dimension of the layout. Can be iterated on
/// using a DimensionalIterator.
class DimensionalRange {
public:
  DimensionalRange() {}
  DimensionalRange(int64_t start, int64_t stop, int64_t step = 1)
      : start(start), stop(stop), step(step) {}
  DimensionalIterator begin() const { return DimensionalIterator(start, step); }
  DimensionalIterator end() const { return DimensionalIterator(stop, step); }

private:
  int64_t start, stop, step;
};

// Iterator class for LayoutAttrs and PerDimLayoutAttrs.
// Provides O(1) access to state for any given dimension.
// Also preserves insertion order.
// Layout iterators skip lane dimensions as these are not
// required during distribution.
class LayoutIterator {
public:
  using State = llvm::MapVector<LayoutDimension, DimensionalIterator>;
  using DimensionMapping =
      llvm::DenseMap<int64_t, SmallVector<LayoutDimension>>;
  void maybeFreezeAndConcatenate(const LayoutIterator &frozenIterator);
  LayoutIterator(LayoutAttr &attr, DenseMap<LayoutDimension, int64_t> strides);
  LayoutIterator(PerDimLayoutAttr &attr,
                 DenseMap<LayoutDimension, int64_t> strides);
  void apply(std::function<void(const LayoutIterator::State &)>);
  LayoutIterator &operator++();
  State getState() const { return state; }

private:
  void initialize(PerDimLayoutAttr &attr,
                  DenseMap<LayoutDimension, int64_t> strides);
  bool iterationComplete();
  State state;
  llvm::MapVector<LayoutDimension, DimensionalRange> ranges;
  DimensionMapping simdDimensionToLayoutDimension;
  DenseSet<LayoutDimension> frozenDimensions;
};

} // namespace mlir::iree_compiler::IREE::VectorExt

#endif // IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_
