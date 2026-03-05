// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This PackMap is inspired by CuTe layouts
// (https://arxiv.org/pdf/2603.02298v1), but adapted for IREE's use case.
//
// The layout algebra implementation is derived from code:
// Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. with BSD-3-Clause license.
// https://github.com/pytorch/pytorch/blob/main/torch/distributed/_pycute/layout.py

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"

#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

#include <limits>

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Map;

//===----------------------------------------------------------------------===//
// IntTuple parsing/printing helpers
//===----------------------------------------------------------------------===//

/// Parse an IntTuple: either a bare integer or `(` IntTuple (`,` IntTuple)*
/// `)`.
static FailureOr<Attribute> parseIntTuple(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();

  if (succeeded(parser.parseOptionalLParen())) {
    SmallVector<Attribute> elements;
    auto result = parseIntTuple(parser);
    if (failed(result)) {
      return failure();
    }
    elements.push_back(*result);
    while (succeeded(parser.parseOptionalComma())) {
      result = parseIntTuple(parser);
      if (failed(result)) {
        return failure();
      }
      elements.push_back(*result);
    }
    if (failed(parser.parseRParen())) {
      return failure();
    }
    return makeTuple(ctx, elements);
  }

  int64_t val64;
  if (failed(parser.parseInteger(val64))) {
    return failure();
  }
  if (val64 < std::numeric_limits<int32_t>::min() ||
      val64 > std::numeric_limits<int32_t>::max()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "IntTuple value does not fit in i32");
  }
  return makeLeaf(ctx, static_cast<int32_t>(val64));
}

/// Print an IntTuple.
static void printIntTuple(AsmPrinter &printer, Attribute attr) {
  if (isLeaf(attr)) {
    printer << getLeafValue(attr);
    return;
  }
  auto arr = cast<ArrayAttr>(attr);
  printer << "(";
  llvm::interleaveComma(arr, printer,
                        [&](Attribute elem) { printIntTuple(printer, elem); });
  printer << ")";
}

//===----------------------------------------------------------------------===//
// PackMapAttr — parsing/printing
//===----------------------------------------------------------------------===//

Attribute PackMapAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }

  auto shape = parseIntTuple(parser);
  if (failed(shape)) {
    return {};
  }

  if (failed(parser.parseColon())) {
    return {};
  }

  auto stride = parseIntTuple(parser);
  if (failed(stride)) {
    return {};
  }

  if (failed(parser.parseGreater())) {
    return {};
  }

  return PackMapAttr::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), *shape, *stride);
}

void PackMapAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntTuple(printer, getShape());
  printer << " : ";
  printIntTuple(printer, getStride());
  printer << ">";
}

LogicalResult PackMapAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Attribute shape, Attribute stride) {
  if (!isIntTuple(shape)) {
    return emitError() << "shape must be a valid IntTuple";
  }
  if (!isIntTuple(stride)) {
    return emitError() << "stride must be a valid IntTuple";
  }

  auto checkPositive = [&](Attribute attr, StringRef name) -> LogicalResult {
    SmallVector<int32_t> leaves = getLeaves(attr);
    for (int32_t v : leaves) {
      if (v <= 0) {
        return emitError() << name << " leaf values must be positive, got "
                           << v;
      }
    }
    return success();
  };
  if (failed(checkPositive(shape, "shape"))) {
    return failure();
  }

  SmallVector<int32_t> strideLeaves = getLeaves(stride);
  for (int32_t v : strideLeaves) {
    if (v < 0) {
      return emitError() << "stride leaf values must be non-negative, got "
                         << v;
    }
  }

  if (!isCongruent(shape, stride)) {
    return emitError() << "shape and stride must be congruent (identical tree "
                          "structure)";
  }

  if (isLeaf(shape)) {
    return emitError()
           << "top-level shape must be a tuple (use (N) : (S) for rank-1)";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PackMapAttr — property methods
//===----------------------------------------------------------------------===//

int32_t PackMapAttr::getRank() { return Map::getRank(getShape()); }

int32_t PackMapAttr::getDepth() { return Map::getDepth(getShape()); }

int32_t PackMapAttr::getSize() { return Map::getSize(getShape()); }

/// Maximum index + 1: cosize = sum((s-1)*d for each leaf) + 1.
int32_t PackMapAttr::getCosize() {
  SmallVector<int32_t> shapes = getLeaves(getShape());
  SmallVector<int32_t> strides = getLeaves(getStride());
  int32_t result = 0;
  for (auto [s, d] : llvm::zip_equal(shapes, strides)) {
    result += (s - 1) * d;
  }
  return result + 1;
}

Attribute PackMapAttr::getShapeMode(int32_t i) {
  return Map::getElement(getShape(), i);
}

Attribute PackMapAttr::getStrideMode(int32_t i) {
  return Map::getElement(getStride(), i);
}

//===----------------------------------------------------------------------===//
// PackMapAttr — evaluation
//===----------------------------------------------------------------------===//

/// Evaluate the layout function: coord -> index.
///
/// The layout maps coordinates to indices via a weighted sum of per-leaf
/// coordinates and strides. The caller can provide coordinates at different
/// granularities -- a single flat index, full per-leaf coordinates, or
/// partial per-mode coordinates -- and we normalize to per-leaf natural
/// coordinates before computing the final inner product with strides.
int32_t PackMapAttr::evaluate(ArrayRef<int32_t> coord) {
  SmallVector<int32_t> strides = getLeaves(getStride());
  SmallVector<int32_t> shapes = getLeaves(getShape());
  assert(shapes.size() == strides.size());

  SmallVector<int32_t> natCoord;
  if (coord.size() == 1 && shapes.size() > 1) {
    natCoord = idx2crd(coord[0], getShape());
  } else if (coord.size() == shapes.size()) {
    natCoord.assign(coord.begin(), coord.end());
  } else {
    // Partial coordinate: fewer coords than leaves but more than 1.
    // Each coord indexes a top-level mode; encode as a flat index
    // using Horner's method left-to-right (lexicographic / row-major).
    int32_t flatIdx = 0;
    SmallVector<int32_t> modeShapes;
    for (int32_t i = 0; i < getRank(); ++i) {
      modeShapes.push_back(Map::getSize(getShapeMode(i)));
    }
    for (int32_t i = 0; i < static_cast<int32_t>(coord.size()); ++i) {
      flatIdx = flatIdx * modeShapes[i] + coord[i];
    }
    natCoord = idx2crd(flatIdx, getShape());
  }

  return crd2idx(natCoord, getShape(), getStride());
}

//===----------------------------------------------------------------------===//
// PackMapAttr — simplification
//===----------------------------------------------------------------------===//

/// Flatten all hierarchy into a single-level tuple of leaves.
/// ((2, 4), 8) : ((16, 1), 4) → (2, 4, 8) : (16, 1, 4).
PackMapAttr PackMapAttr::flatten() {
  MLIRContext *ctx = getContext();
  return PackMapAttr::get(ctx, Map::flatten(ctx, getShape()),
                          Map::flatten(ctx, getStride()));
}

/// Merge adjacent leaves that form a contiguous range.
///
/// Two adjacent leaves can be merged into one when they cover a contiguous
/// range of indices. Scanning right-to-left (innermost first), we accumulate
/// into (accShape, accStride) and check each new leaf (si, di):
///
///   - si == 1: trivial, contributes nothing -- skip.
///   - accShape == 1: accumulator is trivial -- replace it with (si, di).
///   - di == accShape * accStride: the new leaf starts exactly where the
///     accumulator ends, so they are contiguous -- merge into
///     (si * accShape, accStride).
///   - otherwise: non-contiguous -- push the accumulator and start fresh.
///
/// Example: (2, 4) : (4, 1) -> di=4 == 4*1 -> contiguous -> (8) : (1).
///          (4, 8) : (1, 4) -> di=1 != 8*4 -> not contiguous -> unchanged.
PackMapAttr PackMapAttr::coalesce() {
  MLIRContext *ctx = getContext();
  SmallVector<int32_t> shapes = getLeaves(getShape());
  SmallVector<int32_t> strides = getLeaves(getStride());
  assert(shapes.size() == strides.size());

  if (shapes.empty()) {
    return *this;
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

  SmallVector<Attribute> newShape, newStride;
  for (auto [s, d] : result) {
    newShape.push_back(makeLeaf(ctx, s));
    // Normalize: size-1 leaves always get stride 0 (they contribute 0 to the
    // index regardless of their original stride).
    newStride.push_back(makeLeaf(ctx, s == 1 ? 0 : d));
  }

  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

//===----------------------------------------------------------------------===//
// PackMapAttr — algebra
//===----------------------------------------------------------------------===//

/// Functional composition: result(c) = this(rhs(c)).
///
/// A coalesced LHS has leaves that form a mixed-radix decomposition of its
/// index space: the innermost (rightmost) leaf covers [0, s0), the next
/// covers multiples of s0, etc. Each RHS leaf with stride `d` selects every
/// d-th index from LHS. We need to figure out which LHS leaves each RHS leaf
/// "lands on" -- i.e., how to distribute the RHS leaf across the LHS
/// mixed-radix digits.
///
/// We walk LHS leaves right-to-left (innermost first). For each, we compute
///   newShape = min(max(1, lhsShape / restStride), restShape)
/// This determines how many positions the RHS leaf covers in this LHS digit:
///   - If restStride < lhsShape, the RHS steps within this digit (partial
///     coverage), so newShape = lhsShape / restStride (capped by restShape).
///   - If restStride >= lhsShape, the RHS skips this digit entirely
///     (newShape = 1, pruned from output).
/// The result stride for each piece is restStride * lhsStride, composing
/// the RHS step size with the LHS digit's stride. After each digit,
/// restShape shrinks by the positions consumed, and restStride is divided
/// by the digit's shape to shift into the next digit's scale.
///
/// Special cases: stride-0 RHS leaves broadcast (always map to index 0).
/// Single-leaf LHS is a simple stride scaling.
PackMapAttr PackMapAttr::compose(PackMapAttr rhs) {
  MLIRContext *ctx = getContext();
  PackMapAttr lhs = this->coalesce();
  SmallVector<int32_t> lhsShapes = getLeaves(lhs.getShape());
  SmallVector<int32_t> lhsStrides = getLeaves(lhs.getStride());

  SmallVector<int32_t> rhsShapes = getLeaves(rhs.getShape());
  SmallVector<int32_t> rhsStrides = getLeaves(rhs.getStride());

  SmallVector<Attribute> resultShapes, resultStrides;
  int n = static_cast<int>(lhsShapes.size());

  for (auto [rhsS, rhsD] : llvm::zip_equal(rhsShapes, rhsStrides)) {
    if (rhsD == 0) {
      resultShapes.push_back(makeLeaf(ctx, rhsS));
      resultStrides.push_back(makeLeaf(ctx, 0));
      continue;
    }

    if (n == 1) {
      resultShapes.push_back(makeLeaf(ctx, rhsS));
      resultStrides.push_back(makeLeaf(ctx, rhsD * lhsStrides[0]));
      continue;
    }

    SmallVector<Attribute> modeShapes, modeStrides;
    int32_t restShape = rhsS;
    int32_t restStride = rhsD;

    for (int j = n - 1; j >= 1; --j) {
      int32_t currShape = lhsShapes[j];
      int32_t currStride = lhsStrides[j];

      assert((currShape % restStride == 0 || restStride % currShape == 0) &&
             "Stride Divisibility Condition");

      int32_t newShape =
          std::min(std::max(1, currShape / restStride), restShape);

      if (newShape != 1) {
        modeShapes.push_back(makeLeaf(ctx, newShape));
        modeStrides.push_back(makeLeaf(ctx, restStride * currStride));
      }

      restShape = restShape / newShape;
      restStride = ceilDiv(restStride, currShape);
    }

    if (restShape != 1 || modeShapes.empty()) {
      modeShapes.push_back(makeLeaf(ctx, restShape));
      modeStrides.push_back(makeLeaf(ctx, restStride * lhsStrides[0]));
    }

    std::reverse(modeShapes.begin(), modeShapes.end());
    std::reverse(modeStrides.begin(), modeStrides.end());

    if (modeShapes.size() == 1) {
      resultShapes.push_back(modeShapes[0]);
      resultStrides.push_back(modeStrides[0]);
    } else {
      resultShapes.push_back(makeTuple(ctx, modeShapes));
      resultStrides.push_back(makeTuple(ctx, modeStrides));
    }
  }

  return PackMapAttr::get(ctx, makeTuple(ctx, resultShapes),
                          makeTuple(ctx, resultStrides));
}

/// Find a layout B such that A and B together tile [0, cotarget).
///
/// Think of A's modes as covering certain slices of the index space. The
/// complement needs to cover the gaps -- the indices A never produces.
///
/// Sorting A's modes by stride processes them finest-to-coarsest. We track
/// `accumulated` -- the contiguous block [0, accumulated) covered so far.
/// If the next mode has stride d > accumulated, the indices
/// [accumulated, d) are unreached -- a gap. We fill it with a complement
/// mode of shape d/accumulated and stride accumulated: this places
/// d/accumulated copies of the already-covered block at regular offsets,
/// exactly tiling the gap. After processing a mode (s, d), the frontier
/// advances to d * s. A final mode extends from the last frontier to
/// cotarget.
///
/// Example: A = (4):(1) covers [0,4), accumulated=4, cotarget=16.
///          Final mode: (4):(4) tiles {0,4,8,12} -- four copies of [0,4).
PackMapAttr PackMapAttr::complement(int32_t cotarget) {
  MLIRContext *ctx = getContext();

  auto [filtShape, filtStride] = filterZeros(ctx, getShape(), getStride());
  SmallVector<int32_t> shapes = getLeaves(filtShape);
  SmallVector<int32_t> strides = getLeaves(filtStride);

  SmallVector<std::pair<int32_t, int32_t>> modes;
  for (auto [s, d] : llvm::zip_equal(shapes, strides)) {
    modes.push_back({s, d});
  }
  llvm::sort(modes, [](auto &a, auto &b) { return a.second < b.second; });

  SmallVector<Attribute> compShape, compStride;
  int32_t accumulated = 1;
  for (auto [s, d] : modes) {
    int32_t gap = d / accumulated;
    if (gap > 1) {
      compShape.push_back(makeLeaf(ctx, gap));
      compStride.push_back(makeLeaf(ctx, accumulated));
    }
    accumulated = d * s;
  }

  {
    int32_t remaining = ceilDiv(cotarget, accumulated);
    compShape.push_back(makeLeaf(ctx, remaining));
    compStride.push_back(makeLeaf(ctx, accumulated));
  }

  std::reverse(compShape.begin(), compShape.end());
  std::reverse(compStride.begin(), compStride.end());

  auto result = PackMapAttr::get(ctx, makeTuple(ctx, compShape),
                                 makeTuple(ctx, compStride));
  return result.coalesce();
}

/// Factor layout into (inner, outer) using a tiler.
///
/// The tiler covers a subset of the index space (the "tile"), and
/// tiler.complement covers the rest. Composing the layout with each gives
/// two independent views: `inner` sees only within-tile coordinates, `outer`
/// sees only which-tile coordinates. Together they preserve the full layout.
///
/// Example: (32):(1) divided by tiler (4):(1)
///          -> inner = (4):(1), outer = (8):(4).
PackMapAttr PackMapAttr::logicalDivide(PackMapAttr tiler) {
  MLIRContext *ctx = getContext();

  PackMapAttr coal = this->coalesce();
  PackMapAttr tilerComp = tiler.complement(coal.getSize());

  PackMapAttr innerResult = coal.compose(tiler);
  PackMapAttr outerResult = coal.compose(tilerComp);

  SmallVector<Attribute> resShape = {innerResult.getShape(),
                                     outerResult.getShape()};
  SmallVector<Attribute> resStride = {innerResult.getStride(),
                                      outerResult.getStride()};
  return PackMapAttr::get(ctx, makeTuple(ctx, resShape),
                          makeTuple(ctx, resStride));
}

/// Reorder modes: result mode i = original mode perm[i].
PackMapAttr PackMapAttr::permute(ArrayRef<int64_t> perm) {
  MLIRContext *ctx = getContext();
  SmallVector<Attribute> newShape, newStride;
  for (int64_t i : perm) {
    newShape.push_back(getShapeMode(i));
    newStride.push_back(getStrideMode(i));
  }
  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

/// Drop modes where droppedDims[i] is true.
PackMapAttr PackMapAttr::project(ArrayRef<bool> droppedDims) {
  MLIRContext *ctx = getContext();
  SmallVector<Attribute> newShape, newStride;
  for (size_t i = 0; i < droppedDims.size(); ++i) {
    if (!droppedDims[i]) {
      newShape.push_back(getShapeMode(i));
      newStride.push_back(getStrideMode(i));
    }
  }
  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

/// Replicate this layout using a tiler pattern.
///
/// The original layout covers [0, size). To create copies, we enlarge the
/// space to target = size * tiler.cosize and take this.complement(target).
/// The complement covers exactly the indices between copies of `this` --
/// the "gaps" where new copies can be placed without overlap. Composing
/// the complement with the tiler parameterizes which copy to access using
/// the tiler's coordinate system. Mode 0 is the original, mode 1 selects
/// which copy.
///
/// Example: (4):(1) product with (8):(1)
///          -> ((4), (8)) : ((1), (4)), total size 32.
PackMapAttr PackMapAttr::logicalProduct(PackMapAttr tiler) {
  MLIRContext *ctx = getContext();

  int32_t target = getSize() * tiler.getCosize();
  PackMapAttr comp = this->complement(target);
  PackMapAttr mode1 = comp.compose(tiler);

  SmallVector<Attribute> combinedShape = {getShape(), mode1.getShape()};
  SmallVector<Attribute> combinedStride = {getStride(), mode1.getStride()};
  return PackMapAttr::get(ctx, makeTuple(ctx, combinedShape),
                          makeTuple(ctx, combinedStride));
}

/// Remove trivial modes (size-1 or stride-0 leaves), then coalesce.
/// Returns (1):(0) if all modes are trivial.
PackMapAttr PackMapAttr::filter() {
  MLIRContext *ctx = getContext();
  SmallVector<int32_t> shapes = getLeaves(getShape());
  SmallVector<int32_t> strides = getLeaves(getStride());

  SmallVector<Attribute> filtShape, filtStride;
  for (auto [s, d] : llvm::zip_equal(shapes, strides)) {
    if (s != 1 && d != 0) {
      filtShape.push_back(makeLeaf(ctx, s));
      filtStride.push_back(makeLeaf(ctx, d));
    }
  }

  if (filtShape.empty()) {
    filtShape.push_back(makeLeaf(ctx, 1));
    filtStride.push_back(makeLeaf(ctx, 0));
  }

  return PackMapAttr::get(ctx, makeTuple(ctx, filtShape),
                          makeTuple(ctx, filtStride))
      .coalesce();
}

/// Find R such that A(R(x)) = x for all x in [0, injective range).
///
/// A is a bijection (when injective) that reorders indices -- it maps
/// coordinates to indices via A's strides. To invert this, R must undo
/// A's reordering.
///
/// Sorting A's modes by stride reveals A's "digit structure": the mode
/// with stride 1 is the least significant digit (finest granularity), the
/// next mode is the next digit, and so on. The natural (row-major) strides
/// of A's shape define the identity layout's digit structure.
///
/// The rightInverse pairs each of A's digits with the corresponding
/// natural stride: same shapes (same digit sizes), but natural strides
/// instead of A's strides. This creates a layout that converts from A's
/// output ordering back to natural ordering -- when A is applied on top,
/// the two reorderings cancel, yielding the original index x.
///
/// We collect contiguously from stride=1 because a gap means A skips
/// some indices (non-injective), limiting the inverse to the contiguous
/// prefix where A is a bijection.
PackMapAttr PackMapAttr::rightInverse() {
  MLIRContext *ctx = getContext();
  SmallVector<int32_t> shapes = getLeaves(getShape());
  SmallVector<int32_t> strides = getLeaves(getStride());

  int n = static_cast<int>(shapes.size());
  SmallVector<int32_t> rStrides(n);
  {
    int32_t acc = 1;
    for (int i = n - 1; i >= 0; --i) {
      rStrides[i] = acc;
      acc *= shapes[i];
    }
  }

  SmallVector<std::tuple<int32_t, int32_t, int32_t>> sorted;
  for (int i = 0; i < n; ++i) {
    sorted.push_back({strides[i], shapes[i], rStrides[i]});
  }
  llvm::sort(sorted,
             [](auto &a, auto &b) { return std::get<0>(a) < std::get<0>(b); });

  SmallVector<Attribute> resShapes, resStrides;
  int32_t currentIdx = 1;
  for (auto [stride, shape, rStride] : sorted) {
    if (shape == 1) {
      continue;
    }
    if (currentIdx != stride) {
      break;
    }
    resShapes.push_back(makeLeaf(ctx, shape));
    resStrides.push_back(makeLeaf(ctx, rStride));
    currentIdx = shape * stride;
  }

  if (resShapes.empty()) {
    resShapes.push_back(makeLeaf(ctx, 1));
    resStrides.push_back(makeLeaf(ctx, 0));
  }

  std::reverse(resShapes.begin(), resShapes.end());
  std::reverse(resStrides.begin(), resStrides.end());

  return PackMapAttr::get(ctx, makeTuple(ctx, resShapes),
                          makeTuple(ctx, resStrides))
      .coalesce();
}

/// Find L such that L(A(x)) = x for all x in [0, size).
///
/// If A is not surjective, its range has gaps, so rightInverse alone cannot
/// cover the full output domain. The fix: complement fills those gaps,
/// creating a combined layout (comp, A) that IS injective over [0, size).
/// The rightInverse of this combined layout then works for all outputs,
/// and when restricted to A's actual outputs, it recovers x.
PackMapAttr PackMapAttr::leftInverse() {
  MLIRContext *ctx = getContext();
  PackMapAttr comp = this->complement(getSize());

  SmallVector<Attribute> combinedShape = {comp.getShape(), getShape()};
  SmallVector<Attribute> combinedStride = {comp.getStride(), getStride()};
  PackMapAttr combined = PackMapAttr::get(ctx, makeTuple(ctx, combinedShape),
                                          makeTuple(ctx, combinedStride));
  return combined.rightInverse();
}

//===----------------------------------------------------------------------===//
// PackMapAttr — tiled divide and product
//===----------------------------------------------------------------------===//

/// Take a rank-2 result and flatten mode 1 into top-level modes.
static PackMapAttr flattenRestModes(PackMapAttr divided) {
  assert(divided.getRank() == 2 && "expected rank-2 layout");
  MLIRContext *ctx = divided.getContext();
  SmallVector<Attribute> newShape = {divided.getShapeMode(0)};
  SmallVector<Attribute> newStride = {divided.getStrideMode(0)};
  Attribute restShape = divided.getShapeMode(1);
  Attribute restStride = divided.getStrideMode(1);
  if (isLeaf(restShape)) {
    newShape.push_back(restShape);
    newStride.push_back(restStride);
  } else {
    for (Attribute s : cast<ArrayAttr>(restShape)) {
      newShape.push_back(s);
    }
    for (Attribute d : cast<ArrayAttr>(restStride)) {
      newStride.push_back(d);
    }
  }
  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

PackMapAttr PackMapAttr::tiledDivide(PackMapAttr tiler) {
  return flattenRestModes(logicalDivide(tiler));
}

PackMapAttr PackMapAttr::tiledProduct(PackMapAttr tiler) {
  return flattenRestModes(logicalProduct(tiler));
}

//===----------------------------------------------------------------------===//
// PackMapAttr — factory
//===----------------------------------------------------------------------===//

/// Create the row-major identity layout: strides are suffix products.
/// shape (M, N, K) → (M, N, K) : (N*K, K, 1).
PackMapAttr PackMapAttr::makeIdentity(MLIRContext *ctx,
                                      ArrayRef<int64_t> shape) {
  SmallVector<Attribute> leaves;
  for (int64_t s : shape) {
    leaves.push_back(makeLeaf(ctx, static_cast<int32_t>(s)));
  }
  Attribute shapeAttr = makeTuple(ctx, leaves);
  return PackMapAttr::get(ctx, shapeAttr, suffixProduct(ctx, shapeAttr));
}

//===----------------------------------------------------------------------===//
// Dialect attribute registration
//===----------------------------------------------------------------------===//

void IREEMapDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen generated definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.cpp.inc"
