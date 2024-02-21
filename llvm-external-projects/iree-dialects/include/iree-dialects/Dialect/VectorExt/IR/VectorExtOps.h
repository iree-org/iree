// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_
#define IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtInterfaces.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

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

  bool operator==(const DimensionalIterator &other) const {
    return position == other.position;
  }
  bool operator!=(const DimensionalIterator &other) const {
    return !(*this == other);
  }
  bool operator<(const DimensionalIterator &other) const {
    return position < other.position;
  }

  int64_t getPosition() const { return position; }

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

  int64_t start, stop, step;
};

// Iterator class for LayoutAttrs and PerDimLayoutAttrs.
// Provides O(1) access to state for any given dimension.
// Also preserves insertion order.
// Layout iterators skip lane dimensions as these are not
// required during distribution.
class LayoutIterator {
public:
  struct State {
    SmallVector<int64_t> computeSIMTIndex() const;
    SmallVector<int64_t> computeIteratorProjectedSIMTIndex() const;
    bool contains(LayoutDimension dim) const { return iterators.contains(dim); }
    void erase(LayoutDimension dim) { iterators.erase(dim); }
    DimensionalIterator lookup(LayoutDimension dim) const {
      return iterators.lookup(dim);
    }
    DimensionalIterator &operator[](LayoutDimension dim) {
      return iterators[dim];
    }
    void print() const {
      for (const auto &[dim, it] : iterators) {
        llvm::outs() << stringifyLayoutDimension(dim).str() + ":" +
                            std::to_string(*it) + ", ";
      }
      llvm::outs() << "\n";
    }
    llvm::MapVector<LayoutDimension, DimensionalIterator> iterators;
    DenseMap<int64_t, DenseSet<LayoutDimension>> simdToLayoutDim;
    llvm::MapVector<LayoutDimension, DimensionalRange> ranges;
    SmallVector<LayoutDimension> labels{
        LayoutDimension::BATCHX, LayoutDimension::BATCHY,
        LayoutDimension::VECTORY, LayoutDimension::VECTORX};
  };
  void maybeFreezeAndConcatenate(const LayoutIterator::State &frozenState);
  LayoutIterator(LayoutAttr &attr);
  LayoutIterator(LayoutAttr &attr, int64_t simtIndex);
  LayoutIterator(LayoutAttr &attr, DenseMap<LayoutDimension, int64_t> strides);
  LayoutIterator(LayoutAttr &attr, DenseMap<LayoutDimension, int64_t> strides,
                 int64_t simtIndex);
  LayoutIterator(PerDimLayoutAttr &attr,
                 DenseMap<LayoutDimension, int64_t> strides);
  void apply(std::function<void(const LayoutIterator::State &)>);
  LayoutIterator &operator++();
  State getState() const { return state; }
  void erase(LayoutDimension dim);
  LayoutIterator getBatchIterator() const;
  bool iterationComplete();

private:
  void initialize(const PerDimLayoutAttr &attr,
                  DenseMap<LayoutDimension, int64_t> strides,
                  std::optional<int64_t> simdIndex);
  State state;
  DenseSet<LayoutDimension> frozenDimensions;
  int64_t iterations{0};
  int64_t maxIterations{1};
};

inline bool isBatchDimension(LayoutDimension dim) {
  return (dim == LayoutDimension::BATCHX) || (dim == LayoutDimension::BATCHY);
}

inline bool isLaneDimension(LayoutDimension dim) {
  return (dim == LayoutDimension::LANEX) || (dim == LayoutDimension::LANEY) ||
         (dim == LayoutDimension::LANEZ);
}

inline bool isVectorDimension(LayoutDimension dim) {
  return (dim == LayoutDimension::VECTORX) ||
         (dim == LayoutDimension::VECTORY) || (dim == LayoutDimension::VECTORZ);
}

} // namespace mlir::iree_compiler::IREE::VectorExt

#endif // IREE_DIALECTS_DIALECT_VECTOREXT_IR_VECTOREXTOPS_H_
