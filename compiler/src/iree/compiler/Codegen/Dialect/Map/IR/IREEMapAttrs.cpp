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
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Map;

//===----------------------------------------------------------------------===//
// IntTuple parsing/printing helpers
//===----------------------------------------------------------------------===//

/// Parse an IntTuple: either a bare integer or `(` IntTuple (`,` IntTuple)*
/// `)`. Uses an iterative shift-reduce approach with an explicit stack to
/// avoid recursion.
static FailureOr<Attribute> parseIntTupleImpl(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();

  SmallVector<SmallVector<Attribute>> stack;
  Attribute result;
  bool startItem = true;

  while (true) {
    if (startItem) {
      if (succeeded(parser.parseOptionalLParen())) {
        stack.push_back({});
      } else {
        int64_t val;
        if (failed(parser.parseInteger(val))) {
          return failure();
        }
        result = makeLeaf(ctx, val);
        startItem = false;
      }
    } else {
      if (stack.empty()) {
        return result;
      }

      stack.back().push_back(result);
      if (succeeded(parser.parseOptionalComma())) {
        startItem = true;
      } else {
        if (failed(parser.parseRParen())) {
          return failure();
        }
        result = makeTuple(ctx, stack.back());
        stack.pop_back();
      }
    }
  }
}

static ParseResult parseIntTuple(AsmParser &parser, Attribute &result) {
  auto parsed = parseIntTupleImpl(parser);
  if (failed(parsed)) {
    return failure();
  }
  result = *parsed;
  return success();
}

/// Print an IntTuple.
static void printIntTuple(AsmPrinter &printer, Attribute attr) {
  if (isLeaf(attr)) {
    printer << getLeafValue(attr);
    return;
  }
  printer << "(";
  llvm::interleaveComma(cast<ArrayAttr>(attr), printer,
                        [&](Attribute elem) { printIntTuple(printer, elem); });
  printer << ")";
}

LogicalResult PackMapAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Attribute shape, Attribute stride) {
  if (!isIntTuple(shape)) {
    return emitError() << "shape must be a valid IntTuple";
  }
  if (!isIntTuple(stride)) {
    return emitError() << "stride must be a valid IntTuple";
  }
  if (!isCongruent(shape, stride)) {
    return emitError()
           << "shape and stride must be congruent (identical tree structure)";
  }

  for (int64_t v : getLeaves(shape)) {
    if (v <= 0) {
      return emitError() << "shape leaf values must be positive, got " << v;
    }
  }
  for (int64_t v : getLeaves(stride)) {
    if (v < 0) {
      return emitError() << "stride leaf values must be non-negative, got "
                         << v;
    }
  }
  // Overflow checking for getSize() / getCosize() is intentionally omitted:
  // those values can overflow int64_t for very large layouts, which is
  // documented as caller responsibility.

  return success();
}

//===----------------------------------------------------------------------===//
// PackMapAttr - property methods
//===----------------------------------------------------------------------===//

int64_t PackMapAttr::getRank() { return Map::getRank(getShape()); }

int64_t PackMapAttr::getDepth() { return Map::getDepth(getShape()); }

int64_t PackMapAttr::getSize() { return Map::getSize(getShape()); }

int64_t PackMapAttr::getCosize() {
  int64_t result = 0;
  for (const auto &leaf : getLeafInfos(getShape(), getStride())) {
    result += (leaf.size - 1) * leaf.stride;
  }
  return result + 1;
}

Attribute PackMapAttr::getShapeMode(int64_t i) {
  return Map::getElement(getShape(), i);
}

Attribute PackMapAttr::getStrideMode(int64_t i) {
  return Map::getElement(getStride(), i);
}

//===----------------------------------------------------------------------===//
// PackMapAttr - evaluation
//===----------------------------------------------------------------------===//

int64_t PackMapAttr::evaluate(ArrayRef<int64_t> coord) {
  SmallVector<int64_t> shapes = getLeaves(getShape());

  SmallVector<int64_t> natCoord;
  if (coord.size() == 1 && shapes.size() > 1) {
    natCoord = idx2crd(coord[0], getShape());
  } else {
    assert(coord.size() == shapes.size() &&
           "coordinate size must match number of leaf shapes");
    natCoord.assign(coord.begin(), coord.end());
  }

  return crd2idx(natCoord, getStride());
}

//===----------------------------------------------------------------------===//
// PackMapAttr - simplification
//===----------------------------------------------------------------------===//

/// Flatten all hierarchy into a single-level tuple of leaves.
/// ((2, 4), 8) : ((16, 1), 4) -> (2, 4, 8) : (16, 1, 4).
PackMapAttr PackMapAttr::flatten() {
  MLIRContext *ctx = getContext();
  return PackMapAttr::get(ctx, Map::flatten(ctx, getShape()),
                          Map::flatten(ctx, getStride()));
}

/// Core merge: scan flat (shapes, strides) right-to-left and merge adjacent
/// contiguous leaves. Returns merged pairs in left-to-right order.
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
static SmallVector<std::pair<int64_t, int64_t>>
coalesceImpl(ArrayRef<int64_t> shapes, ArrayRef<int64_t> strides) {
  assert(shapes.size() == strides.size());
  SmallVector<std::pair<int64_t, int64_t>> result;
  if (shapes.empty()) {
    return result;
  }
  result.push_back({shapes.back(), strides.back()});
  for (int i = static_cast<int>(shapes.size()) - 2; i >= 0; --i) {
    int64_t si = shapes[i], di = strides[i];
    auto &[accShape, accStride] = result.back();
    if (si == 1) {
      continue;
    }
    if (accShape == 1) {
      accShape = si;
      accStride = di;
    } else if (accShape * accStride == di) {
      accShape = si * accShape;
    } else {
      result.push_back({si, di});
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

/// Merge adjacent leaves across all modes (flattens first).
/// Example: (2, 4) : (4, 1) -> (8) : (1).
///
/// Normalizes size-1 leaves to stride 0 to produce a canonical form: any
/// size-1 mode contributes nothing to the index, so its stride is irrelevant.
PackMapAttr PackMapAttr::coalesce() {
  MLIRContext *ctx = getContext();
  SmallVector<Attribute> newShape, newStride;
  for (auto [s, d] :
       coalesceImpl(getLeaves(getShape()), getLeaves(getStride()))) {
    newShape.push_back(makeLeaf(ctx, s));
    newStride.push_back(makeLeaf(ctx, s == 1 ? 0 : d));
  }
  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

/// Coalesce within each top-level mode independently.
/// Unlike coalesce(), leaves are never merged across mode boundaries.
PackMapAttr PackMapAttr::coalesceModes() {
  MLIRContext *ctx = getContext();
  SmallVector<Attribute> newShape, newStride;
  for (int64_t i = 0; i < getRank(); ++i) {
    Attribute mShape = getShapeMode(i), mStride = getStrideMode(i);
    if (isLeaf(mShape)) {
      newShape.push_back(mShape);
      newStride.push_back(getLeafValue(mShape) == 1 ? makeLeaf(ctx, 0)
                                                    : mStride);
      continue;
    }
    auto merged = coalesceImpl(getLeaves(mShape), getLeaves(mStride));
    SmallVector<Attribute> ms, md;
    for (auto [s, d] : merged) {
      ms.push_back(makeLeaf(ctx, s));
      md.push_back(makeLeaf(ctx, s == 1 ? 0 : d));
    }
    newShape.push_back(ms.size() == 1 ? ms[0] : makeTuple(ctx, ms));
    newStride.push_back(md.size() == 1 ? md[0] : makeTuple(ctx, md));
  }
  return PackMapAttr::get(ctx, makeTuple(ctx, newShape),
                          makeTuple(ctx, newStride));
}

//===----------------------------------------------------------------------===//
// PackMapAttr - algebra
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
  SmallVector<int64_t> lhsShapes = getLeaves(lhs.getShape());
  SmallVector<int64_t> lhsStrides = getLeaves(lhs.getStride());

  SmallVector<int64_t> rhsShapes = getLeaves(rhs.getShape());
  SmallVector<int64_t> rhsStrides = getLeaves(rhs.getStride());

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
    int64_t restShape = rhsS;
    int64_t restStride = rhsD;

    for (int j = n - 1; j >= 1; --j) {
      int64_t currShape = lhsShapes[j];
      int64_t currStride = lhsStrides[j];

      assert((currShape % restStride == 0 || restStride % currShape == 0) &&
             "compose: RHS stride must divide LHS shape or vice versa");

      int64_t newShape =
          std::min(std::max(int64_t(1), currShape / restStride), restShape);

      if (newShape != 1) {
        modeShapes.push_back(makeLeaf(ctx, newShape));
        modeStrides.push_back(makeLeaf(ctx, restStride * currStride));
      }

      restShape = restShape / newShape;
      restStride = llvm::divideCeilSigned(restStride, currShape);
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

/// Given a range [0, cotarget) and layout A, find layout B such that
/// logicalProduct(A, B) bijects onto [0, cotarget) -- i.e. A and B together
/// partition the entire range with no gaps and no overlaps.
///
/// The idea: sort A's modes by stride (finest first) and scan left-to-right,
/// tracking `accumulated` = the size of the contiguous index block covered so
/// far. When the next mode has stride d > accumulated, there is a gap
/// [accumulated, d) that A never hits. We fill it with a complement mode
/// (d/accumulated, accumulated) -- stride accumulated steps over the already-
/// covered block, and shape d/accumulated repeats it enough times to reach d.
/// After each mode (s, d), the covered frontier advances to d*s. Finally, a
/// trailing mode covers from the last frontier up to cotarget.
///
/// Example: A = (4):(1), cotarget = 16.
///   A covers {0, 1, 2, 3}. accumulated = 4. Trailing: 16/4 = 4 -> emit (4, 4).
///   Result: (4):(4), which covers {0, 4, 8, 12}.
///   Together A and B partition {0, ..., 15}.
PackMapAttr PackMapAttr::complement(int64_t cotarget) {
  MLIRContext *ctx = getContext();

  auto [filtShape, filtStride] = filterZeros(ctx, getShape(), getStride());
  SmallVector<int64_t> shapes = getLeaves(filtShape);
  SmallVector<int64_t> strides = getLeaves(filtStride);

  SmallVector<std::pair<int64_t, int64_t>> modes;
  for (auto [s, d] : llvm::zip_equal(shapes, strides)) {
    modes.push_back({s, d});
  }
  llvm::sort(modes, [](auto &a, auto &b) { return a.second < b.second; });

  SmallVector<Attribute> compShape, compStride;
  int64_t accumulated = 1;
  for (auto [s, d] : modes) {
    int64_t gap = d / accumulated;
    if (gap > 1) {
      compShape.push_back(makeLeaf(ctx, gap));
      compStride.push_back(makeLeaf(ctx, accumulated));
    }
    accumulated = d * s;
  }

  {
    int64_t remaining = llvm::divideCeilSigned(cotarget, accumulated);
    compShape.push_back(makeLeaf(ctx, remaining));
    compStride.push_back(makeLeaf(ctx, accumulated));
  }

  std::reverse(compShape.begin(), compShape.end());
  std::reverse(compStride.begin(), compStride.end());

  auto result = PackMapAttr::get(ctx, makeTuple(ctx, compShape),
                                 makeTuple(ctx, compStride));
  return result.coalesce();
}

/// Split a layout into two modes: inner (within a tile) and outer (which tile).
///
/// The tiler defines the shape of one tile. The result is a rank-2 layout
/// where mode 0 is the inner view (position within a tile) and mode 1 is the
/// outer view (which tile). Evaluating with (inner_coord, outer_coord) gives
/// the same index as the original layout evaluated at the corresponding
/// flat position.
///
/// Example 1: (32):(1) divided by tiler (4):(1)
///   32 elements split into 8 tiles of 4.
///   Result: ((4), (8)) : ((1), (4))
///   evaluate({2, 3}) = 2*1 + 3*4 = 14 (element 2 of tile 3).
///
/// Example 2: (32):(2) divided by tiler (4):(1)
///   Stride-2 layout (hits even indices) split into tiles of 4.
///   Result: ((4), (8)) : ((2), (8))
///   evaluate({1, 2}) = 1*2 + 2*8 = 18.
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

/// Replicate this layout across multiple copies parameterized by a tiler.
///
/// The result is a rank-2 layout where mode 0 is the original layout (position
/// within one copy) and mode 1 selects which copy using the tiler's coordinate
/// system. The total index space covered is size * tiler.cosize.
///
/// Example 1: (4):(1) product with tiler (8):(1)
///   4 elements replicated 8 times -> 32 total.
///   Result: ((4), (8)) : ((1), (4))
///   evaluate({2, 3}) = 2*1 + 3*4 = 14 (element 2 of copy 3).
///
/// Example 2: (4):(1) product with tiler (4):(2)
///   4 elements replicated with stride-2 tiler -> cosize = (4-1)*2+1 = 7.
///   Result: ((4), (4)) : ((1), (4))
///   evaluate({1, 2}) = 1*1 + 2*4 = 9.
PackMapAttr PackMapAttr::logicalProduct(PackMapAttr tiler) {
  MLIRContext *ctx = getContext();

  int64_t target = getSize() * tiler.getCosize();
  PackMapAttr comp = this->complement(target);
  PackMapAttr mode1 = comp.compose(tiler);

  SmallVector<Attribute> combinedShape = {getShape(), mode1.getShape()};
  SmallVector<Attribute> combinedStride = {getStride(), mode1.getStride()};
  return PackMapAttr::get(ctx, makeTuple(ctx, combinedShape),
                          makeTuple(ctx, combinedStride));
}

PackMapAttr PackMapAttr::filter() {
  MLIRContext *ctx = getContext();
  auto [filtShape, filtStride] = filterZeros(ctx, getShape(), getStride());
  return PackMapAttr::get(ctx, filtShape, filtStride).coalesce();
}

/// Find R such that A(R(x)) = x for all x in [0, injective range).
///
/// R has the same shape as A but with natural row-major strides. This means R
/// unpacks a flat index into digits, and A re-packs them in its own stride
/// order -- the two cancel out and yield the original index.
///
/// If A's smallest stride is > 1 (its output range has a gap at 0), no right
/// inverse exists and the result is the trivial layout (1):(0).
///
/// Example 1: A = (4, 8):(8, 1)  (row-major is an identity)
///   R = (4, 8):(8, 1). A(R(x)) = x.
///
/// Example 2: A = (4, 2):(1, 4)  (column-major 4x2)
///   R = (2, 4):(1, 2).
///   R(5) = 3, A(3) = 5.
PackMapAttr PackMapAttr::rightInverse() {
  MLIRContext *ctx = getContext();
  SmallVector<int64_t> shapes = getLeaves(getShape());
  SmallVector<int64_t> strides = getLeaves(getStride());

  int n = static_cast<int>(shapes.size());
  SmallVector<int64_t> rStrides(n);
  {
    int64_t acc = 1;
    for (int i = n - 1; i >= 0; --i) {
      rStrides[i] = acc;
      acc *= shapes[i];
    }
  }

  SmallVector<std::tuple<int64_t, int64_t, int64_t>> sorted;
  for (int i = 0; i < n; ++i) {
    sorted.push_back({strides[i], shapes[i], rStrides[i]});
  }
  llvm::sort(sorted,
             [](auto &a, auto &b) { return std::get<0>(a) < std::get<0>(b); });

  SmallVector<Attribute> resShapes, resStrides;
  int64_t currentIdx = 1;
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
/// Unlike rightInverse (which requires A's output range to be contiguous),
/// leftInverse works even when A's outputs have gaps. It uses complement to
/// fill the gaps, building a combined layout that covers [0, size) fully,
/// then takes the rightInverse of that. On A's actual outputs, this recovers x.
///
/// Example 1: A = (4):(2)  (maps {0,1,2,3} -> {0,2,4,6})
///   L = (4, 2):(1, 4).
///   A(3) = 6. L(6): unpack 6 in shape (4,2) -> (3,0) -> 3*1 + 0*4 = 3.
///
/// Example 2: A = (8):(2)  (maps {0..7} -> {0,2,4,6,8,10,12,14})
///   L = (8, 2):(1, 8).
///   A(5) = 10. L(10): unpack 10 in shape (8,2) -> (5,0) -> 5*1 + 0*8 = 5.
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
// PackMapAttr - tiled divide and product
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
    for (auto [s, d] : llvm::zip_equal(cast<ArrayAttr>(restShape),
                                       cast<ArrayAttr>(restStride))) {
      newShape.push_back(s);
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
// PackMapAttr - factory
//===----------------------------------------------------------------------===//

/// Create the row-major identity layout: strides are suffix products.
/// shape (M, N, K) -> (M, N, K) : (N*K, K, 1).
PackMapAttr PackMapAttr::makeIdentity(MLIRContext *ctx,
                                      ArrayRef<int64_t> shape) {
  SmallVector<Attribute> leaves;
  for (int64_t s : shape) {
    leaves.push_back(makeLeaf(ctx, s));
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
