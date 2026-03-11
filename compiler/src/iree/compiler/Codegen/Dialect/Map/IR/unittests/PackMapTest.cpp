// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <variant>

#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapAttrs.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IREEMapDialect.h"
#include "iree/compiler/Codegen/Dialect/Map/IR/IntTuple.h"

namespace mlir::iree_compiler::IREE::Map {
namespace {

// Recursive helper for building nested IntTuple literals.
// Leaves are constructed from int64_t; tuples from initializer_list<T>.
// Examples:
//   T{4}              -> leaf(4)
//   T{4, 8}           -> tuple(leaf(4), leaf(8))
//   T{{2, 4}, 8}      -> tuple(tuple(leaf(2), leaf(4)), leaf(8))
//   T{{{2, 4}, 8}, 16} -> depth-3 nesting
struct T {
  std::variant<int64_t, std::vector<T>> val;
  T(int64_t v) : val(v) {}
  T(std::initializer_list<T> children) : val(std::vector<T>(children)) {}

  Attribute toAttr(MLIRContext *c) const {
    if (auto *v = std::get_if<int64_t>(&val)) {
      return makeLeaf(c, *v);
    }
    SmallVector<Attribute> attrs =
        llvm::map_to_vector(std::get<std::vector<T>>(val),
                            [&](const T &child) { return child.toAttr(c); });
    return makeTuple(c, attrs);
  }
};

class PackMapTest : public ::testing::Test {
protected:
  PackMapTest() {
    DialectRegistry reg;
    reg.insert<IREEMapDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  MLIRContext *getContext() { return &ctx; }

  // Create a PackMapAttr from nested shape/stride expressions.
  PackMapAttr make(T shape, T stride) {
    MLIRContext *c = getContext();
    return PackMapAttr::get(c, shape.toAttr(c), stride.toAttr(c));
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
  // Depth 2: ((2, 4), 8) : ((16, 1), 4)
  EXPECT_EQ(make({{2, 4}, 8}, {{16, 1}, 4}).getDepth(), 2);
  // Depth 3: (((2, 4), 8), 16) : (((32, 4), 2), 1)
  EXPECT_EQ(make({{{2, 4}, 8}, 16}, {{{32, 4}, 2}, 1}).getDepth(), 3);
}

TEST_F(PackMapTest, Size) {
  EXPECT_EQ(make({8}, {1}).getSize(), 8);
  EXPECT_EQ(make({4, 8}, {8, 1}).getSize(), 32);
  // Nested: ((2, 3), 4) leaves are (2, 3, 4), size = 24.
  EXPECT_EQ(make({{2, 3}, 4}, {{12, 1}, 4}).getSize(), 24);
}

TEST_F(PackMapTest, Cosize) {
  // (8) : (1) -> cosize = (8-1)*1 + 1 = 8
  EXPECT_EQ(make({8}, {1}).getCosize(), 8);
  // (4, 8) : (8, 1) -> cosize = (4-1)*8 + (8-1)*1 + 1 = 32
  EXPECT_EQ(make({4, 8}, {8, 1}).getCosize(), 32);
  // Nested ((4, 8)) : ((1, 4)) -> cosize = (4-1)*1 + (8-1)*4 + 1 = 32
  EXPECT_EQ(make({{4, 8}}, {{1, 4}}).getCosize(), 32);
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
  // Nested: ((2, 4), 8) : ((16, 1), 4) — leaves (2,4,8) : (16,1,4)
  auto layout = make({{2, 4}, 8}, {{16, 1}, 4});
  EXPECT_EQ(layout.evaluate({0, 0, 0}), 0);
  EXPECT_EQ(layout.evaluate({1, 0, 0}), 16);
  EXPECT_EQ(layout.evaluate({0, 1, 0}), 1);
  EXPECT_EQ(layout.evaluate({0, 0, 1}), 4);
  EXPECT_EQ(layout.evaluate({1, 3, 7}), 47); // 1*16 + 3*1 + 7*4
}

TEST_F(PackMapTest, EvaluateColumnMajor) {
  // (4, 8) : (1, 4) — column-major
  auto layout = make({4, 8}, {1, 4});
  EXPECT_EQ(layout.evaluate({0, 0}), 0);
  EXPECT_EQ(layout.evaluate({1, 0}), 1);
  EXPECT_EQ(layout.evaluate({0, 1}), 4);
  EXPECT_EQ(layout.evaluate({3, 7}), 31);
}

TEST_F(PackMapTest, EvaluateFlatIndexNonIdentity) {
  // Flat index on column-major (4, 8):(1, 4) exercises idx2crd + crd2idx.
  // idx2crd(i, (4,8)) decomposes i row-major; crd2idx then applies col-major
  // strides.
  auto layout = make({4, 8}, {1, 4});
  EXPECT_EQ(layout.evaluate({0}), 0);  // (0,0) -> 0
  EXPECT_EQ(layout.evaluate({1}), 4);  // (0,1) -> 0*1+1*4 = 4
  EXPECT_EQ(layout.evaluate({5}), 20); // (0,5) -> 0*1+5*4 = 20
  EXPECT_EQ(layout.evaluate({8}), 1);  // (1,0) -> 1*1+0*4 = 1
}

TEST_F(PackMapTest, EvaluateStrided) {
  // (4):(3) — stride > 1, cosize (10) > size (4).
  auto layout = make({4}, {3});
  EXPECT_EQ(layout.evaluate({0}), 0);
  EXPECT_EQ(layout.evaluate({1}), 3);
  EXPECT_EQ(layout.evaluate({3}), 9);

  // (4, 8):(2, 1) — outer stride is not shape*inner_stride.
  auto layout2 = make({4, 8}, {2, 1});
  EXPECT_EQ(layout2.evaluate({0, 0}), 0);
  EXPECT_EQ(layout2.evaluate({1, 0}), 2);
  EXPECT_EQ(layout2.evaluate({0, 1}), 1);
  EXPECT_EQ(layout2.evaluate({3, 7}), 13); // 3*2 + 7*1
}

TEST_F(PackMapTest, EvaluateBroadcast) {
  // (4, 8):(0, 1) — stride-0 dim contributes nothing regardless of coord.
  auto layout = make({4, 8}, {0, 1});
  EXPECT_EQ(layout.evaluate({0, 0}), 0);
  EXPECT_EQ(layout.evaluate({3, 5}), 5); // row coord ignored
  EXPECT_EQ(layout.evaluate({0, 7}), 7);
}

//===----------------------------------------------------------------------===//
// Coalesce
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, CoalesceContiguous) {
  // Nested: ((2, 4), 8) : ((32, 8), 1) — leaves (2,4,8):(32,8,1), all
  // contiguous. Merge right-to-left: 8*1=8==8 -> (32,1), 32*1=32==32 -> (64,1).
  auto layout = make({{2, 4}, 8}, {{32, 8}, 1});
  EXPECT_EQ(layout.coalesce(), make({64}, {1}));
}

TEST_F(PackMapTest, CoalesceNonContiguous) {
  // (4, 8) : (1, 4) — column-major, not contiguous in lex order
  auto layout = make({4, 8}, {1, 4});
  auto coalesced = layout.coalesce();
  EXPECT_EQ(coalesced.getRank(), 2);

  // (4, 8) : (16, 1) — holes between groups: stride 16 > 8*1 = 8, so
  // accShape*accStride (8) != 16. Stays at rank 2. cosize=56 > size=32.
  auto holey = make({4, 8}, {16, 1});
  EXPECT_EQ(holey.coalesce(), make({4, 8}, {16, 1}));
}

TEST_F(PackMapTest, CoalesceRemovesUnitModes) {
  // (1, 8) : (0, 1) -> coalesced to (8) : (1)
  auto layout = make({1, 8}, {0, 1});
  EXPECT_EQ(layout.coalesce(), make({8}, {1}));
}

//===----------------------------------------------------------------------===//
// CoalesceModes
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, CoalesceModesMergesWithinMode) {
  // (4, (2, 4)) : (8, (4, 1)) -- mode 0 is a leaf, mode 1 has contiguous
  // sub-leaves and merges internally.
  auto layout = make({4, {2, 4}}, {8, {4, 1}});
  EXPECT_EQ(layout.coalesceModes(), make({4, 8}, {8, 1}));
}

TEST_F(PackMapTest, CoalesceModesVsCoalesce) {
  // coalesce merges across mode boundaries; coalesceModes does not.
  // (4, (2, 4)) : (8, (4, 1)): leaves (4:8, 2:4, 4:1) are all contiguous,
  // so coalesce merges everything to (32):(1), while coalesceModes only
  // merges mode 1 internally, preserving the boundary.
  auto layout = make({4, {2, 4}}, {8, {4, 1}});
  EXPECT_EQ(layout.coalesce(), make({32}, {1}));
  EXPECT_EQ(layout.coalesceModes(), make({4, 8}, {8, 1}));
}

TEST_F(PackMapTest, CoalesceModesNonContiguousWithinMode) {
  // (4, (2, 4)) : (8, (8, 1)) -- mode 1 sub-leaves (2,4):(8,1) are not
  // contiguous (8 != 4*1=4), so mode 1 stays unchanged.
  auto layout = make({4, {2, 4}}, {8, {8, 1}});
  EXPECT_EQ(layout.coalesceModes(), make({4, {2, 4}}, {8, {8, 1}}));
}

TEST_F(PackMapTest, CoalesceModesRemovesUnitModes) {
  // (1, (2, 4)) : (5, (4, 1)) -- mode 0 is size-1, stride normalized to 0;
  // mode 1 merges to (8):(1).
  auto layout = make({1, {2, 4}}, {5, {4, 1}});
  EXPECT_EQ(layout.coalesceModes(), make({1, 8}, {0, 1}));
}

//===----------------------------------------------------------------------===//
// Flatten
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, FlattenHierarchical) {
  // ((2, 4), 8) : ((16, 1), 4)
  auto layout = make({{2, 4}, 8}, {{16, 1}, 4});
  EXPECT_EQ(layout.flatten(), make({2, 4, 8}, {16, 1, 4}));
}

//===----------------------------------------------------------------------===//
// Compose
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, ComposeIdentity) {
  // Nested LHS: ((2, 4), 4) : ((16, 4), 1) composed with identity.
  // Result must equal A for all flat indices.
  auto a = make({{2, 4}, 4}, {{16, 4}, 1});
  auto id = PackMapAttr::makeIdentity(getContext(), {32});
  auto result = a.compose(id);
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
  // Nested ((2, 2)) : ((1, 2)) — leaves (2,2):(1,2), cotarget=16.
  // Strides contiguous (acc 1->2->4), trailing 16/4=4 -> complement (4):(4).
  auto layout = make({{2, 2}}, {{1, 2}});
  EXPECT_EQ(layout.complement(16), make({4}, {4}));
}

TEST_F(PackMapTest, ComplementWithGap) {
  // (4) : (2) with cotarget=8.
  // Original covers {0, 2, 4, 6}. Complement fills the interleaving gap.
  auto layout = make({4}, {2});
  EXPECT_EQ(layout.complement(8), make({2}, {1}));
}

//===----------------------------------------------------------------------===//
// LogicalDivide
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, LogicalDivideBasic) {
  // Nested layout ((4, 8)) : ((8, 1)) — same function as (32):(1).
  // Divide by tiler (4):(1) -> inner=(4):(1), outer=(8):(4).
  auto layout = make({{4, 8}}, {{8, 1}});
  auto tiler = make({4}, {1});
  // Each mode stores its sub-layout's shape as a 1-element tuple.
  auto result = layout.logicalDivide(tiler);
  EXPECT_EQ(result, make({{4}, {8}}, {{1}, {4}}));
  // Functional: result(inner, outer) = inner + tileSize * outer.
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
  // Nested layout ((2, 2)) : ((1, 2)) — same size=4 as (4):(1).
  // Product with (8):(1) -> mode 0 size=4, mode 1 size=8.
  auto layout = make({{2, 2}}, {{1, 2}});
  auto tiler = make({8}, {1});
  // Mode 0 stores original layout's full shape; mode 1 stores
  // comp.compose(tiler).
  EXPECT_EQ(layout.logicalProduct(tiler),
            make({{{2, 2}}, {8}}, {{{1, 2}}, {4}}));
}

//===----------------------------------------------------------------------===//
// Filter
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, FilterRemovesBroadcast) {
  // Nested ((4, 8)) : ((0, 1)) — stride-0 leaf removed, leaves (8):(1).
  auto layout = make({{4, 8}}, {{0, 1}});
  EXPECT_EQ(layout.filter(), make({8}, {1}));
}

TEST_F(PackMapTest, FilterRemovesUnitSize) {
  // (1, 8) : (5, 1) -> filter removes size-1 mode
  auto layout = make({1, 8}, {5, 1});
  EXPECT_EQ(layout.filter(), make({8}, {1}));
}

//===----------------------------------------------------------------------===//
// RightInverse
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, RightInverse) {
  // Nested ((2, 4)) : ((4, 1)) — same function as (8):(1) (row-major).
  auto layout = make({{2, 4}}, {{4, 1}});
  auto ri = layout.rightInverse();
  // A(R(x)) = x for all x in [0, 8)
  for (int i = 0; i < 8; ++i) {
    int64_t intermediate = ri.evaluate({i});
    EXPECT_EQ(layout.evaluate({intermediate}), i);
  }
}

TEST_F(PackMapTest, RightInverseColumnMajor) {
  // (4, 2) : (1, 4) — column-major
  auto layout = make({4, 2}, {1, 4});
  auto ri = layout.rightInverse();
  for (int i = 0; i < 8; ++i) {
    int64_t intermediate = ri.evaluate({i});
    EXPECT_EQ(layout.evaluate({intermediate}), i);
  }
}

TEST_F(PackMapTest, RightInverseNonSurjective) {
  // (4):(2) maps to {0, 2, 4, 6} — injective but stride starts at 2, not 1.
  // rightInverse collects only contiguous strides starting from 1, so the
  // result is trivial (1):(0) since no stride-1 leaf exists.
  auto layout = make({4}, {2});
  EXPECT_EQ(layout.rightInverse(), make({1}, {0}));
}

TEST_F(PackMapTest, LeftInverseNonSurjective) {
  // (4):(2) maps to {0, 2, 4, 6}. leftInverse uses complement to fill gaps,
  // then rightInverse of the combined layout. L(A(x)) = x for all x in [0, 4).
  auto layout = make({4}, {2});
  auto li = layout.leftInverse();
  for (int i = 0; i < 4; ++i) {
    int64_t output = layout.evaluate({i});
    EXPECT_EQ(li.evaluate({output}), i);
  }
}

//===----------------------------------------------------------------------===//
// LeftInverse
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, LeftInverse) {
  // Nested ((2, 4)) : ((4, 1)) — same function as (8):(1).
  auto layout = make({{2, 4}}, {{4, 1}});
  auto li = layout.leftInverse();
  // L(A(x)) = x for all x in [0, 8)
  for (int i = 0; i < 8; ++i) {
    int64_t intermediate = layout.evaluate({i});
    EXPECT_EQ(li.evaluate({intermediate}), i);
  }
}

//===----------------------------------------------------------------------===//
// TiledDivide / TiledProduct
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, TiledDivide) {
  // Nested ((4, 8)) : ((8, 1)) — same function as (32):(1).
  auto layout = make({{4, 8}}, {{8, 1}});
  auto tiler = make({4}, {1});
  // flattenRestModes unwraps mode 1's 1-element tuple into a leaf.
  EXPECT_EQ(layout.tiledDivide(tiler), make({{4}, 8}, {{1}, 4}));
}

TEST_F(PackMapTest, TiledProduct) {
  auto layout = make({4}, {1});
  auto tiler = make({8}, {1});
  EXPECT_EQ(layout.tiledProduct(tiler), make({{4}, 8}, {{1}, 4}));
}

//===----------------------------------------------------------------------===//
// Permute
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, Permute) {
  // mode 0 = (2, 2):(4, 1), mode 1 = 8:8 -> permute({1,0}) swaps them.
  auto layout = make({{2, 2}, 8}, {{4, 1}, 8});
  EXPECT_EQ(layout.permute({1, 0}), make({8, {2, 2}}, {8, {4, 1}}));
}

//===----------------------------------------------------------------------===//
// Project
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, Project) {
  // mode 0 = (2, 2):(4, 1), mode 1 = 8:2, mode 2 = 2:1 -> drop mode 1.
  auto layout = make({{2, 2}, 8, 2}, {{4, 1}, 2, 1});
  EXPECT_EQ(layout.project({false, true, false}),
            make({{2, 2}, 2}, {{4, 1}, 1}));
}

//===----------------------------------------------------------------------===//
// MakeIdentity
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, MakeIdentity) {
  auto id = PackMapAttr::makeIdentity(getContext(), {4, 8});
  EXPECT_EQ(id, make({4, 8}, {8, 1}));
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(id.evaluate({i}), i);
  }
}

TEST_F(PackMapTest, MakeIdentity3D) {
  EXPECT_EQ(PackMapAttr::makeIdentity(getContext(), {2, 3, 4}),
            make({2, 3, 4}, {12, 4, 1}));
}

//===----------------------------------------------------------------------===//
// IntTuple: filterLeafInfos and foldLeafInfos
//===----------------------------------------------------------------------===//

TEST_F(PackMapTest, FilterLeafInfosZeroStride) {
  // (4, 8) : (0, 1) -> zero-stride leaves: LeafInfo{4, 0, 8}
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
