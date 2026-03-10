// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IntTuple utilities.
//
// An IntTuple is a recursive type: either a single integer (IntegerAttr, i64)
// or a tuple of IntTuples (ArrayAttr). These utilities operate on Attribute
// trees encoding this structure.

#ifndef IREE_COMPILER_CODEGEN_DIALECT_MAP_IR_INTTUPLE_H_
#define IREE_COMPILER_CODEGEN_DIALECT_MAP_IR_INTTUPLE_H_

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler::IREE::Map {

// --- Query functions ---

/// Returns true if attr is a valid IntTuple (IntegerAttr or ArrayAttr of
/// IntTuples, recursively).
bool isIntTuple(Attribute attr);

/// Returns true if attr is a leaf integer (IntegerAttr).
bool isLeaf(Attribute attr);

/// Returns the integer value of a leaf. Requires isLeaf(attr).
int64_t getLeafValue(Attribute attr);

/// Top-level element count. 1 for a leaf, N for an N-element ArrayAttr.
int64_t getRank(Attribute attr);

/// Maximum nesting depth. 0 for a leaf, 1 + max(child depths) for a tuple.
int64_t getDepth(Attribute attr);

/// Product of all leaf integers. May overflow for large layouts.
int64_t getSize(Attribute attr);

/// Get the i-th top-level element. For a leaf, i must be 0 (returns self).
Attribute getElement(Attribute attr, int64_t i);

/// Collect all leaves from an IntTuple into a flat vector.
SmallVector<int64_t> getLeaves(Attribute attr);

// --- Predicates ---

/// Returns true if a and b have identical tree structure (same nesting, same
/// ranks at each level). Leaf values may differ.
bool isCongruent(Attribute a, Attribute b);

// --- Builders ---

/// Create a leaf IntegerAttr(i64).
Attribute makeLeaf(MLIRContext *ctx, int64_t val);

/// Create a tuple (ArrayAttr) from elements.
Attribute makeTuple(MLIRContext *ctx, ArrayRef<Attribute> elements);

/// Flatten all nesting into a single-level tuple of leaves.
Attribute flatten(MLIRContext *ctx, Attribute tuple);

/// Recursively unwrap single-element tuples: (x) → simplify(x).
Attribute simplify(Attribute attr);

// --- Arithmetic ---

/// Recursive inner product: sum of (leaf_coord * leaf_stride) over all leaves.
int64_t innerProduct(Attribute coord, Attribute stride);

/// shape_div: divide shape by divisor, distributing left-to-right.
Attribute shapeDiv(MLIRContext *ctx, Attribute shape, int64_t divisor);

/// suffixProduct: compute row-major (lexicographic) strides for a flat shape.
/// For shape (M, N, K) returns (N*K, K, 1) as an ArrayAttr of i64 leaves.
Attribute suffixProduct(MLIRContext *ctx, Attribute shape);

// --- Coordinate conversion ---

/// idx2crd: convert a 1-D index to a natural coordinate matching shape.
/// Uses lexicographic (row-major) ordering.
SmallVector<int64_t> idx2crd(int64_t idx, Attribute shape);

/// crd2idx: convert a coordinate to a 1-D index via inner product with stride.
int64_t crd2idx(ArrayRef<int64_t> coord, Attribute stride);

// --- Filtering ---

/// Filter stride-0 and size-1 modes from parallel shape+stride tuples.
/// Returns (filteredShape, filteredStride) as a pair of attributes.
std::pair<Attribute, Attribute> filterZeros(MLIRContext *ctx, Attribute shape,
                                            Attribute stride);

// --- Leaf info ---

/// Info about a single leaf in a (shape, stride) mode pair.
/// `stride` is the layout stride (0 = broadcast).
/// `dataStride` is the lex data stride (product of all subsequent leaf sizes).
struct LeafInfo {
  int64_t size;
  int64_t stride;
  int64_t dataStride;
};

/// Walk leaves of parallel (shape, stride), computing lex data strides.
SmallVector<LeafInfo> getLeafInfos(Attribute shape, Attribute stride);

/// Filter leaf infos matching a predicate.
SmallVector<LeafInfo>
filterLeafInfos(Attribute shape, Attribute stride,
                llvm::function_ref<bool(const LeafInfo &)> pred);

/// Fold over leaf infos with an accumulator.
/// fn receives (accumulator, LeafInfo) and returns the new accumulator.
int64_t
foldLeafInfos(Attribute shape, Attribute stride, int64_t init,
              llvm::function_ref<int64_t(int64_t, const LeafInfo &)> fn);

} // namespace mlir::iree_compiler::IREE::Map

#endif // IREE_COMPILER_CODEGEN_DIALECT_MAP_IR_INTTUPLE_H_
