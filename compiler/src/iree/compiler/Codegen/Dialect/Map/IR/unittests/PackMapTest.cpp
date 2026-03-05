// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapDialect.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"

namespace mlir::iree_compiler::IREE::Map {
namespace {

class PackMapTest : public ::testing::Test {
protected:
  PackMapTest() {
    DialectRegistry reg;
    reg.insert<IREEMapDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  MLIRContext *getContext() { return &ctx; }

  // Helper: create a flat PackMapAttr from shape and stride vectors.
  PackMapAttr make(ArrayRef<int32_t> shape, ArrayRef<int32_t> stride) {
    MLIRContext *c = getContext();
    SmallVector<Attribute> s, d;
    for (int32_t v : shape) {
      s.push_back(makeLeaf(c, v));
    }
    for (int32_t v : stride) {
      d.push_back(makeLeaf(c, v));
    }
    return PackMapAttr::get(c, makeTuple(c, s), makeTuple(c, d));
  }

private:
  MLIRContext ctx;
};

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, Rank) {
  EXPECT_EQ(make({8}, {1}).getRank(), 1);
  EXPECT_EQ(make({4, 8}, {8, 1}).getRank(), 2);
  EXPECT_EQ(make({2, 3, 4}, {12, 4, 1}).getRank(), 3);
}

TEST_F(PackMapTest, Depth) {
  EXPECT_EQ(make({8}, {1}).getDepth(), 1);
  EXPECT_EQ(make({4, 8}, {8, 1}).getDepth(), 1);
}

TEST_F(PackMapTest, Size) {
  EXPECT_EQ(make({8}, {1}).getSize(), 8);
  EXPECT_EQ(make({4, 8}, {8, 1}).getSize(), 32);
  EXPECT_EQ(make({2, 3, 4}, {12, 4, 1}).getSize(), 24);
}

TEST_F(PackMapTest, Cosize) {
  // (8) : (1) -> cosize = (8-1)*1 + 1 = 8
  EXPECT_EQ(make({8}, {1}).getCosize(), 8);
  // (4, 8) : (8, 1) -> cosize = (4-1)*8 + (8-1)*1 + 1 = 32
  EXPECT_EQ(make({4, 8}, {8, 1}).getCosize(), 32);
  // (4, 8) : (1, 4) -> column-major: (4-1)*1 + (8-1)*4 + 1 = 32
  EXPECT_EQ(make({4, 8}, {1, 4}).getCosize(), 32);
}

//===----------------------------------------------------------------------===//
// Mode access
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, ModeAccess) {
  auto layout = make({4, 8}, {8, 1});
  EXPECT_EQ(getLeafValue(layout.getShapeMode(0)), 4);
  EXPECT_EQ(getLeafValue(layout.getShapeMode(1)), 8);
  EXPECT_EQ(getLeafValue(layout.getStrideMode(0)), 8);
  EXPECT_EQ(getLeafValue(layout.getStrideMode(1)), 1);
}

//===----------------------------------------------------------------------===//
// Evaluation
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, EvaluateFlatIndex) {
  // (4, 8) : (8, 1), identity row-major
  auto layout = make({4, 8}, {8, 1});
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(layout.evaluate({i}), i);
  }
}

TEST_F(PackMapTest, EvaluateMultiDimCoord) {
  // (4, 8) : (8, 1)
  auto layout = make({4, 8}, {8, 1});
  EXPECT_EQ(layout.evaluate({0, 0}), 0);
  EXPECT_EQ(layout.evaluate({0, 7}), 7);
  EXPECT_EQ(layout.evaluate({1, 0}), 8);
  EXPECT_EQ(layout.evaluate({3, 7}), 31);
}

TEST_F(PackMapTest, EvaluateColumnMajor) {
  // (4, 8) : (1, 4) — column-major
  auto layout = make({4, 8}, {1, 4});
  EXPECT_EQ(layout.evaluate({0, 0}), 0);
  EXPECT_EQ(layout.evaluate({1, 0}), 1);
  EXPECT_EQ(layout.evaluate({0, 1}), 4);
  EXPECT_EQ(layout.evaluate({3, 7}), 31);
}

//===----------------------------------------------------------------------===//
// No auto-coalesce (key difference from PackLayout)
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, NoAutoCoalesce) {
  // Create a layout with redundant modes: (2, 4) : (4, 1) is equivalent
  // to (8) : (1) after coalescing, but PackMap should NOT auto-coalesce.
  auto layout = make({2, 4}, {4, 1});
  EXPECT_EQ(layout.getRank(), 2);
  EXPECT_EQ(getLeafValue(layout.getShapeMode(0)), 2);
  EXPECT_EQ(getLeafValue(layout.getShapeMode(1)), 4);
}

//===----------------------------------------------------------------------===//
// Coalesce
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, CoalesceContiguous) {
  // (2, 4) : (4, 1) -> coalesced to (8) : (1)
  auto layout = make({2, 4}, {4, 1});
  auto coalesced = layout.coalesce();
  EXPECT_EQ(coalesced.getRank(), 1);
  EXPECT_EQ(coalesced.getSize(), 8);
  EXPECT_EQ(getLeafValue(coalesced.getShapeMode(0)), 8);
  EXPECT_EQ(getLeafValue(coalesced.getStrideMode(0)), 1);
}

TEST_F(PackMapTest, CoalesceNonContiguous) {
  // (4, 8) : (1, 4) — column-major, not contiguous in lex order
  auto layout = make({4, 8}, {1, 4});
  auto coalesced = layout.coalesce();
  EXPECT_EQ(coalesced.getRank(), 2);
}

TEST_F(PackMapTest, CoalesceRemovesUnitModes) {
  // (1, 8) : (0, 1) -> coalesced to (8) : (1)
  auto layout = make({1, 8}, {0, 1});
  auto coalesced = layout.coalesce();
  EXPECT_EQ(coalesced.getRank(), 1);
  EXPECT_EQ(getLeafValue(coalesced.getShapeMode(0)), 8);
}

//===----------------------------------------------------------------------===//
// Flatten
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, FlattenHierarchical) {
  // Build hierarchical: ((2, 4), 8) : ((16, 1), 4)
  MLIRContext *c = getContext();
  Attribute innerS = makeTuple(c, {makeLeaf(c, 2), makeLeaf(c, 4)});
  Attribute innerD = makeTuple(c, {makeLeaf(c, 16), makeLeaf(c, 1)});
  Attribute outerS = makeLeaf(c, 8);
  Attribute outerD = makeLeaf(c, 4);
  auto layout = PackMapAttr::get(c, makeTuple(c, {innerS, outerS}),
                                 makeTuple(c, {innerD, outerD}));
  auto flat = layout.flatten();
  EXPECT_EQ(flat.getRank(), 3);
  SmallVector<int32_t> shapes = getLeaves(flat.getShape());
  EXPECT_EQ(shapes[0], 2);
  EXPECT_EQ(shapes[1], 4);
  EXPECT_EQ(shapes[2], 8);
}

//===----------------------------------------------------------------------===//
// Compose
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, ComposeIdentity) {
  // A composed with identity should give back A (modulo coalescing of LHS).
  auto a = make({4, 8}, {8, 1});
  auto id = PackMapAttr::makeIdentity(getContext(), {32});
  auto result = a.compose(id);
  // Should produce the same function.
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(result.evaluate({i}), a.evaluate({i}));
  }
}

TEST_F(PackMapTest, ComposeWithBroadcast) {
  // stride-0 in RHS should produce stride-0 in result.
  auto a = make({8}, {1});
  auto rhs = make({4}, {0});
  auto result = a.compose(rhs);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(result.evaluate({i}), 0);
  }
}

TEST_F(PackMapTest, ComposeMultiDigit) {
  // Compose column-major LHS with a stride-2 RHS.
  // LHS (4, 8):(1, 4) maps flat index i -> (i%4)*1 + (i/4)*4.
  // RHS (8):(2) maps index j -> 2*j.
  // Composed: result(j) = LHS(2*j).
  auto lhs = make({4, 8}, {1, 4});
  auto rhs = make({8}, {2});
  auto result = lhs.compose(rhs);
  // Verify functionally: result(j) == lhs(2*j) for all j in [0, 8).
  for (int j = 0; j < 8; ++j) {
    EXPECT_EQ(result.evaluate({j}), lhs.evaluate({2 * j}));
  }
}

TEST_F(PackMapTest, ComposeMultiDigitRowMajor) {
  // Compose row-major 4x8 with a 2x4 row-major layout.
  // LHS = (4, 8):(8, 1), RHS = (2, 4):(4, 1).
  // result(i, j) = LHS(RHS(i, j)) = LHS(4*i + j).
  auto lhs = make({4, 8}, {8, 1});
  auto rhs = make({2, 4}, {4, 1});
  auto result = lhs.compose(rhs);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      int rhsIdx = 4 * i + j;
      EXPECT_EQ(result.evaluate({i, j}), lhs.evaluate({rhsIdx}));
    }
  }
}

//===----------------------------------------------------------------------===//
// Complement
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, ComplementBasic) {
  // (4) : (1) with cotarget=16 -> complement covers indices 4..15
  auto layout = make({4}, {1});
  auto comp = layout.complement(16);
  // The complement should cover the "rest" of the index space.
  EXPECT_EQ(comp.getSize(), 4); // 16 / 4 = 4
}

TEST_F(PackMapTest, ComplementWithGap) {
  // (4) : (2) with cotarget=8.
  // Original covers {0, 2, 4, 6}. Complement fills the interleaving gap.
  auto layout = make({4}, {2});
  auto comp = layout.complement(8);
  EXPECT_EQ(comp.getSize(), 2);
  EXPECT_EQ(comp.getRank(), 1);
  EXPECT_EQ(getLeafValue(comp.getShapeMode(0)), 2);
  EXPECT_EQ(getLeafValue(comp.getStrideMode(0)), 1);
}

//===----------------------------------------------------------------------===//
// LogicalDivide
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, LogicalDivideBasic) {
  // Divide (32):(1) by tiler (4):(1) -> inner=(4):(1), outer=(8):(4).
  auto layout = make({32}, {1});
  auto tiler = make({4}, {1});
  auto result = layout.logicalDivide(tiler);
  EXPECT_EQ(result.getRank(), 2);
  EXPECT_EQ(result.getSize(), 32);
  // Inner mode: tile of 4 with stride 1.
  EXPECT_EQ(getSize(result.getShapeMode(0)), 4);
  EXPECT_EQ(getLeaves(result.getStrideMode(0))[0], 1);
  // Outer mode: 8 tiles with stride 4.
  EXPECT_EQ(getSize(result.getShapeMode(1)), 8);
  EXPECT_EQ(getLeaves(result.getStrideMode(1))[0], 4);
  // Functional: result(inner, outer) = inner + tileSize * outer (since
  // original is identity).
  for (int outer = 0; outer < 8; ++outer) {
    for (int inner = 0; inner < 4; ++inner) {
      EXPECT_EQ(result.evaluate({inner, outer}), inner + 4 * outer);
    }
  }
}

//===----------------------------------------------------------------------===//
// LogicalProduct
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, LogicalProductBasic) {
  // (4):(1) product with (8):(1) -> ((4),(8)):((1),(4)).
  auto layout = make({4}, {1});
  auto tiler = make({8}, {1});
  auto result = layout.logicalProduct(tiler);
  EXPECT_EQ(result.getRank(), 2);
  EXPECT_EQ(result.getSize(), 32);
  // Mode 0 is the original layout.
  EXPECT_EQ(getSize(result.getShapeMode(0)), 4);
  EXPECT_EQ(getLeaves(result.getStrideMode(0))[0], 1);
  // Mode 1 selects which copy (stride = original size).
  EXPECT_EQ(getSize(result.getShapeMode(1)), 8);
  EXPECT_EQ(getLeaves(result.getStrideMode(1))[0], 4);
}

//===----------------------------------------------------------------------===//
// Filter
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, FilterRemovesBroadcast) {
  // (4, 8) : (0, 1) -> filter removes stride-0 mode
  auto layout = make({4, 8}, {0, 1});
  auto filtered = layout.filter();
  EXPECT_EQ(filtered.getSize(), 8);
  EXPECT_EQ(getLeafValue(filtered.getStrideMode(0)), 1);
}

TEST_F(PackMapTest, FilterRemovesUnitSize) {
  // (1, 8) : (5, 1) -> filter removes size-1 mode
  auto layout = make({1, 8}, {5, 1});
  auto filtered = layout.filter();
  EXPECT_EQ(filtered.getSize(), 8);
}

//===----------------------------------------------------------------------===//
// RightInverse
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, RightInverse) {
  // For identity (8) : (1), rightInverse should also be identity-like.
  auto layout = make({8}, {1});
  auto ri = layout.rightInverse();
  // A(R(x)) = x for all x in [0, size)
  for (int i = 0; i < 8; ++i) {
    int32_t intermediate = ri.evaluate({i});
    EXPECT_EQ(layout.evaluate({intermediate}), i);
  }
}

TEST_F(PackMapTest, RightInverseColumnMajor) {
  // (4, 2) : (1, 4) — column-major
  auto layout = make({4, 2}, {1, 4});
  auto ri = layout.rightInverse();
  for (int i = 0; i < 8; ++i) {
    int32_t intermediate = ri.evaluate({i});
    EXPECT_EQ(layout.evaluate({intermediate}), i);
  }
}

TEST_F(PackMapTest, RightInverseNonSurjective) {
  // (4):(2) maps to {0, 2, 4, 6} — injective but stride starts at 2, not 1.
  // rightInverse collects only contiguous strides starting from 1, so the
  // result is trivial (1):(0) since no stride-1 leaf exists.
  auto layout = make({4}, {2});
  auto ri = layout.rightInverse();
  EXPECT_EQ(ri.getSize(), 1);
}

TEST_F(PackMapTest, LeftInverseNonSurjective) {
  // (4):(2) maps to {0, 2, 4, 6}. leftInverse uses complement to fill gaps,
  // then rightInverse of the combined layout. L(A(x)) = x for all x in [0, 4).
  auto layout = make({4}, {2});
  auto li = layout.leftInverse();
  for (int i = 0; i < 4; ++i) {
    int32_t output = layout.evaluate({i});
    EXPECT_EQ(li.evaluate({output}), i);
  }
}

//===----------------------------------------------------------------------===//
// LeftInverse
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, LeftInverse) {
  auto layout = make({8}, {1});
  auto li = layout.leftInverse();
  // L(A(x)) = x for all x in [0, size)
  for (int i = 0; i < 8; ++i) {
    int32_t intermediate = layout.evaluate({i});
    EXPECT_EQ(li.evaluate({intermediate}), i);
  }
}

//===----------------------------------------------------------------------===//
// TiledDivide / TiledProduct
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, TiledDivide) {
  auto layout = make({32}, {1});
  auto tiler = make({4}, {1});
  auto result = layout.tiledDivide(tiler);
  EXPECT_EQ(result.getRank(), 2);
  EXPECT_EQ(result.getSize(), 32);
  EXPECT_EQ(getSize(result.getShapeMode(0)), 4);
  EXPECT_EQ(getSize(result.getShapeMode(1)), 8);
}

TEST_F(PackMapTest, TiledProduct) {
  auto layout = make({4}, {1});
  auto tiler = make({8}, {1});
  auto result = layout.tiledProduct(tiler);
  EXPECT_EQ(result.getRank(), 2);
  EXPECT_EQ(result.getSize(), 32);
  EXPECT_EQ(getSize(result.getShapeMode(0)), 4);
  EXPECT_EQ(getSize(result.getShapeMode(1)), 8);
}

//===----------------------------------------------------------------------===//
// Permute
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, Permute) {
  auto layout = make({4, 8}, {8, 1});
  auto permuted = layout.permute({1, 0});
  EXPECT_EQ(getLeafValue(permuted.getShapeMode(0)), 8);
  EXPECT_EQ(getLeafValue(permuted.getShapeMode(1)), 4);
  EXPECT_EQ(getLeafValue(permuted.getStrideMode(0)), 1);
  EXPECT_EQ(getLeafValue(permuted.getStrideMode(1)), 8);
}

//===----------------------------------------------------------------------===//
// Project
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, Project) {
  auto layout = make({4, 8, 2}, {16, 2, 1});
  auto projected = layout.project({false, true, false});
  EXPECT_EQ(projected.getRank(), 2);
  EXPECT_EQ(getLeafValue(projected.getShapeMode(0)), 4);
  EXPECT_EQ(getLeafValue(projected.getShapeMode(1)), 2);
}

//===----------------------------------------------------------------------===//
// MakeIdentity
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, MakeIdentity) {
  auto id = PackMapAttr::makeIdentity(getContext(), {4, 8});
  EXPECT_EQ(id.getRank(), 2);
  EXPECT_EQ(getLeafValue(id.getShapeMode(0)), 4);
  EXPECT_EQ(getLeafValue(id.getShapeMode(1)), 8);
  EXPECT_EQ(getLeafValue(id.getStrideMode(0)), 8);
  EXPECT_EQ(getLeafValue(id.getStrideMode(1)), 1);
  // Should be row-major identity.
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(id.evaluate({i}), i);
  }
}

TEST_F(PackMapTest, MakeIdentity3D) {
  auto id = PackMapAttr::makeIdentity(getContext(), {2, 3, 4});
  EXPECT_EQ(id.getRank(), 3);
  EXPECT_EQ(getLeafValue(id.getStrideMode(0)), 12); // 3*4
  EXPECT_EQ(getLeafValue(id.getStrideMode(1)), 4);  // 4
  EXPECT_EQ(getLeafValue(id.getStrideMode(2)), 1);  // 1
}

//===----------------------------------------------------------------------===//
// IntTuple: filterLeafInfos and foldLeafInfos
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, FilterLeafInfosZeroStride) {
  // (4, 8) : (0, 1) -> zero-stride leaves: {4, 0, _}
  auto layout = make({4, 8}, {0, 1});
  auto zeroStride =
      filterLeafInfos(layout.getShape(), layout.getStride(),
                      [](const LeafInfo &l) { return l.stride == 0; });
  EXPECT_EQ(zeroStride.size(), 1u);
  EXPECT_EQ(zeroStride[0].size, 4);
}

TEST_F(PackMapTest, FilterLeafInfosNonZeroStride) {
  auto layout = make({4, 8}, {2, 1});
  auto nonZero =
      filterLeafInfos(layout.getShape(), layout.getStride(),
                      [](const LeafInfo &l) { return l.stride > 0; });
  EXPECT_EQ(nonZero.size(), 2u);
}

TEST_F(PackMapTest, FoldLeafInfosProductOfSizes) {
  // Product of sizes for stride > 0 leaves.
  auto layout = make({4, 8}, {2, 1});
  int64_t product = foldLeafInfos(layout.getShape(), layout.getStride(), 1,
                                  [](int64_t acc, const LeafInfo &l) {
                                    return l.stride > 0 ? acc * l.size : acc;
                                  });
  EXPECT_EQ(product, 32); // 4 * 8
}

TEST_F(PackMapTest, FoldLeafInfosWithBroadcast) {
  // (4, 8) : (0, 1) -> product of stride>0 sizes = 8
  auto layout = make({4, 8}, {0, 1});
  int64_t product = foldLeafInfos(layout.getShape(), layout.getStride(), 1,
                                  [](int64_t acc, const LeafInfo &l) {
                                    return l.stride > 0 ? acc * l.size : acc;
                                  });
  EXPECT_EQ(product, 8);
}

} // namespace
} // namespace mlir::iree_compiler::IREE::Map
