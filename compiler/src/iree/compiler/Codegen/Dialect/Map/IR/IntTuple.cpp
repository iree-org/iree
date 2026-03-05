// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::Map {

// --- Query functions ---

bool isIntTuple(Attribute attr) {
  if (!attr) {
    return false;
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return intAttr.getType().isInteger(32);
  }
  if (auto arrAttr = dyn_cast<ArrayAttr>(attr)) {
    return llvm::all_of(arrAttr,
                        [](Attribute elem) { return isIntTuple(elem); });
  }
  return false;
}

bool isLeaf(Attribute attr) { return isa<IntegerAttr>(attr); }

int32_t getLeafValue(Attribute attr) {
  return static_cast<int32_t>(cast<IntegerAttr>(attr).getInt());
}

int32_t getRank(Attribute attr) {
  if (isLeaf(attr)) {
    return 1;
  }
  return cast<ArrayAttr>(attr).size();
}

int32_t getDepth(Attribute attr) {
  if (isLeaf(attr)) {
    return 0;
  }
  int32_t maxChild = 0;
  for (Attribute child : cast<ArrayAttr>(attr)) {
    maxChild = std::max(maxChild, getDepth(child));
  }
  return 1 + maxChild;
}

int32_t getSize(Attribute attr) {
  if (isLeaf(attr)) {
    return getLeafValue(attr);
  }
  int32_t product = 1;
  for (Attribute child : cast<ArrayAttr>(attr)) {
    product *= getSize(child);
  }
  return product;
}

Attribute getElement(Attribute attr, int32_t i) {
  if (isLeaf(attr)) {
    assert(i == 0 && "leaf only has element 0");
    return attr;
  }
  return cast<ArrayAttr>(attr)[i];
}

// --- Predicates ---

bool isCongruent(Attribute a, Attribute b) {
  if (isLeaf(a) && isLeaf(b)) {
    return true;
  }
  if (isLeaf(a) != isLeaf(b)) {
    return false;
  }
  auto arrA = cast<ArrayAttr>(a);
  auto arrB = cast<ArrayAttr>(b);
  if (arrA.size() != arrB.size()) {
    return false;
  }
  for (auto [ea, eb] : llvm::zip_equal(arrA, arrB)) {
    if (!isCongruent(ea, eb)) {
      return false;
    }
  }
  return true;
}

// --- Builders ---

Attribute makeLeaf(MLIRContext *ctx, int32_t val) {
  return IntegerAttr::get(IntegerType::get(ctx, 32), val);
}

Attribute makeTuple(MLIRContext *ctx, ArrayRef<Attribute> elements) {
  return ArrayAttr::get(ctx, elements);
}

Attribute simplify(Attribute attr) {
  if (isLeaf(attr)) {
    return attr;
  }
  auto arr = cast<ArrayAttr>(attr);
  if (arr.size() == 1) {
    return simplify(arr[0]);
  }
  SmallVector<Attribute> result;
  for (auto elem : arr) {
    result.push_back(simplify(elem));
  }
  return makeTuple(attr.getContext(), result);
}

Attribute flatten(MLIRContext *ctx, Attribute tuple) {
  SmallVector<int32_t> leaves = getLeaves(tuple);
  SmallVector<Attribute> leafAttrs;
  leafAttrs.reserve(leaves.size());
  for (int32_t v : leaves) {
    leafAttrs.push_back(makeLeaf(ctx, v));
  }
  return makeTuple(ctx, leafAttrs);
}

// --- Arithmetic ---

int32_t innerProduct(Attribute coord, Attribute stride) {
  if (isLeaf(coord)) {
    assert(isLeaf(stride));
    return getLeafValue(coord) * getLeafValue(stride);
  }
  auto coordArr = cast<ArrayAttr>(coord);
  auto strideArr = cast<ArrayAttr>(stride);
  assert(coordArr.size() == strideArr.size());
  int32_t sum = 0;
  for (auto [c, s] : llvm::zip_equal(coordArr, strideArr)) {
    sum += innerProduct(c, s);
  }
  return sum;
}

/// Divide a shape by a divisor, distributing the division left-to-right
/// across the hierarchical structure (outermost mode first).
///
/// For a leaf: if leaf is divisible by divisor, return leaf / divisor.
/// Otherwise the divisor is larger than the leaf, so the leaf is fully
/// consumed (returns 1) and the remaining divisor carries forward.
///
/// For a tuple: fold left-to-right with a running remainder. Each child
/// either absorbs the remainder fully, partially, or is untouched once
/// the remainder is exhausted. Left-to-right is correct because in
/// lexicographic ordering, leftmost modes are outermost -- dividing
/// removes the outermost coordinates first.
Attribute shapeDiv(MLIRContext *ctx, Attribute shape, int32_t divisor) {
  if (divisor == 1) {
    return shape;
  }

  if (isLeaf(shape)) {
    int32_t s = getLeafValue(shape);
    if (s % divisor == 0) {
      return makeLeaf(ctx, s / divisor);
    }
    assert(divisor % s == 0 && "shapeDiv: divisibility constraint violated");
    return makeLeaf(ctx, 1);
  }

  // Tuple: fold left-to-right with running remainder.
  auto arr = cast<ArrayAttr>(shape);
  SmallVector<Attribute> result;
  int32_t rem = divisor;
  for (Attribute child : arr) {
    int32_t childSize = getSize(child);
    if (rem == 1) {
      result.push_back(child);
    } else if (childSize % rem == 0) {
      result.push_back(shapeDiv(ctx, child, rem));
      rem = 1;
    } else {
      assert(rem % childSize == 0 &&
             "shapeDiv: divisibility constraint violated");
      result.push_back(shapeDiv(ctx, child, childSize));
      rem /= childSize;
    }
  }
  return makeTuple(ctx, result);
}

// Local i32 ceil division. Intentionally not using llvm::divideCeil (which
// takes int64_t) to stay in the i32 domain used throughout the layout algebra.
int32_t ceilDiv(int32_t a, int32_t b) {
  assert(b > 0);
  return (a + b - 1) / b;
}

Attribute suffixProduct(MLIRContext *ctx, Attribute shape) {
  SmallVector<int32_t> leaves = getLeaves(shape);
  int n = static_cast<int>(leaves.size());
  SmallVector<int32_t> result(n);
  int32_t acc = 1;
  for (int i = n - 1; i >= 0; --i) {
    result[i] = acc;
    acc *= leaves[i];
  }
  SmallVector<Attribute> attrs;
  attrs.reserve(n);
  for (int32_t v : result) {
    attrs.push_back(makeLeaf(ctx, v));
  }
  return makeTuple(ctx, attrs);
}

// --- Coordinate conversion ---

// Lexicographic (row-major): rightmost dimension varies fastest.
SmallVector<int32_t> idx2crd(int32_t idx, Attribute shape) {
  SmallVector<int32_t> leaves = getLeaves(shape);
  int n = static_cast<int>(leaves.size());
  SmallVector<int32_t> coords(n);
  int32_t remaining = idx;
  for (int i = n - 1; i >= 0; --i) {
    coords[i] = remaining % leaves[i];
    remaining /= leaves[i];
  }
  return coords;
}

int32_t crd2idx(ArrayRef<int32_t> coord, Attribute shape, Attribute stride) {
  SmallVector<int32_t> strides = getLeaves(stride);
  assert(coord.size() == strides.size());
  int32_t result = 0;
  for (auto [c, s] : llvm::zip_equal(coord, strides)) {
    result += c * s;
  }
  return result;
}

// --- Filtering ---

std::pair<Attribute, Attribute> filterZeros(MLIRContext *ctx, Attribute shape,
                                            Attribute stride) {
  SmallVector<int32_t> flatShape = getLeaves(shape);
  SmallVector<int32_t> flatStride = getLeaves(stride);
  assert(flatShape.size() == flatStride.size());

  SmallVector<Attribute> filteredShape, filteredStride;
  for (auto [s, d] : llvm::zip_equal(flatShape, flatStride)) {
    if (d != 0 && s != 1) {
      filteredShape.push_back(makeLeaf(ctx, s));
      filteredStride.push_back(makeLeaf(ctx, d));
    }
  }

  if (filteredShape.empty()) {
    filteredShape.push_back(makeLeaf(ctx, 1));
    filteredStride.push_back(makeLeaf(ctx, 0));
  }

  return {makeTuple(ctx, filteredShape), makeTuple(ctx, filteredStride)};
}

SmallVector<int32_t> getLeaves(Attribute attr) {
  SmallVector<int32_t> result;
  if (isLeaf(attr)) {
    result.push_back(getLeafValue(attr));
    return result;
  }
  for (Attribute child : cast<ArrayAttr>(attr)) {
    auto childLeaves = getLeaves(child);
    result.append(childLeaves.begin(), childLeaves.end());
  }
  return result;
}

std::pair<Attribute, Attribute> coalesceMode(MLIRContext *ctx, Attribute shape,
                                             Attribute stride) {
  if (isLeaf(shape)) {
    if (getLeafValue(shape) == 1) {
      return {shape, makeLeaf(ctx, 0)};
    }
    return {shape, stride};
  }

  if (!isCongruent(shape, stride)) {
    return {shape, stride};
  }

  SmallVector<int32_t> shapes = getLeaves(shape);
  SmallVector<int32_t> strides = getLeaves(stride);
  assert(shapes.size() == strides.size());

  if (shapes.empty()) {
    return {shape, stride};
  }

  SmallVector<std::pair<int32_t, int32_t>> result;
  result.push_back({shapes.back(), strides.back()});

  for (int i = static_cast<int>(shapes.size()) - 2; i >= 0; --i) {
    int32_t si = shapes[i];
    int32_t di = strides[i];
    auto &[accShape, accStride] = result.back();

    if (si == 1) {
      continue;
    }
    if (accShape == 1) {
      accShape = si;
      accStride = di;
      continue;
    }
    if (accShape * accStride == di) {
      accShape = si * accShape;
      continue;
    }
    result.push_back({si, di});
  }

  std::reverse(result.begin(), result.end());

  for (auto &[s, d] : result) {
    if (s == 1) {
      d = 0;
    }
  }

  if (result.size() == 1) {
    return {makeLeaf(ctx, result[0].first), makeLeaf(ctx, result[0].second)};
  }

  SmallVector<Attribute> newShape, newStride;
  for (auto [s, d] : result) {
    newShape.push_back(makeLeaf(ctx, s));
    newStride.push_back(makeLeaf(ctx, d));
  }
  return {makeTuple(ctx, newShape), makeTuple(ctx, newStride)};
}

// --- Leaf info ---

SmallVector<LeafInfo> getLeafInfos(Attribute shape, Attribute stride) {
  SmallVector<int32_t> shapes = getLeaves(shape);
  SmallVector<int32_t> strides = getLeaves(stride);
  assert(shapes.size() == strides.size());
  int n = static_cast<int>(shapes.size());

  SmallVector<int64_t> dataStrides(n, 1);
  for (int i = n - 2; i >= 0; --i) {
    dataStrides[i] = dataStrides[i + 1] * shapes[i + 1];
  }

  SmallVector<LeafInfo> result;
  for (int i = 0; i < n; ++i) {
    result.push_back({shapes[i], strides[i], dataStrides[i]});
  }
  return result;
}

SmallVector<LeafInfo>
filterLeafInfos(Attribute shape, Attribute stride,
                llvm::function_ref<bool(const LeafInfo &)> pred) {
  SmallVector<LeafInfo> result;
  for (auto &leaf : getLeafInfos(shape, stride)) {
    if (pred(leaf)) {
      result.push_back(leaf);
    }
  }
  return result;
}

int64_t
foldLeafInfos(Attribute shape, Attribute stride, int64_t init,
              llvm::function_ref<int64_t(int64_t, const LeafInfo &)> fn) {
  int64_t acc = init;
  for (auto &leaf : getLeafInfos(shape, stride)) {
    acc = fn(acc, leaf);
  }
  return acc;
}

} // namespace mlir::iree_compiler::IREE::Map
