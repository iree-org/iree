// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/QuantizedContraction.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler::GlobalOptimization::detail {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

class QuantizedContractionTest : public ::testing::Test {
protected:
  QuantizedContractionTest() {
    context
        .loadDialect<IREE::LinalgExt::IREELinalgExtDialect, arith::ArithDialect,
                     func::FuncDialect, linalg::LinalgDialect>();
  }

  bool parse(StringRef source) {
    module = parseSourceString<ModuleOp>(source, &context);
    return module && succeeded(verify(*module));
  }

  linalg::GenericOp contraction(StringRef id = "test") {
    linalg::GenericOp result;
    module->walk([&](linalg::GenericOp op) {
      if (auto attr = op->getAttrOfType<StringAttr>("id")) {
        if (attr.getValue() == id) {
          result = op;
        }
      }
    });
    EXPECT_TRUE(result) << "Missing contraction: " << id.str();
    return result;
  }

  AffineMap map(unsigned rank, ArrayRef<unsigned> dims) {
    SmallVector<AffineExpr> exprs;
    for (unsigned dim : dims) {
      exprs.push_back(getAffineDimExpr(dim, &context));
    }
    return AffineMap::get(rank, 0, exprs, &context);
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

TEST_F(QuantizedContractionTest, SymmetricMultiDimensionalReduction) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x4x8xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_THAT(detail->integerReductionDims, ElementsAre(2, 3));
  EXPECT_THAT(detail->floatingReductionDims, IsEmpty());
  EXPECT_EQ(detail->reductionExtent, 32);
  EXPECT_EQ(detail->lhs.inputMap, map(4, {0, 2, 3}));
  EXPECT_EQ(detail->rhs.inputMap, map(4, {2, 3, 1}));
  EXPECT_EQ(detail->lhs.scaleMap, map(4, {}));
  EXPECT_EQ(detail->outputMap, map(4, {0, 1}));
  EXPECT_FALSE(detail->lhs.zeroPointMap);
  EXPECT_FALSE(detail->rhs.zeroPointMap);
  EXPECT_FALSE(detail->needsLhsSum());
  EXPECT_FALSE(detail->needsRhsSum());
}

TEST_F(QuantizedContractionTest, LhsZeroPointNeedsRhsSum) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %az: i8,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #scalar, #identity]}
          ins(%aq, %as, %az : tensor<2x4x8xi8>, f32, i8)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_EQ(detail->needsLhsSum(), false);
  EXPECT_EQ(detail->needsRhsSum(), true);
  EXPECT_EQ(detail->lhs.zeroPointMap, map(4, {}));
  EXPECT_EQ(detail->rhs.zeroPointMap, AffineMap());
}

TEST_F(QuantizedContractionTest, RhsZeroPointNeedsLhsSum) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %bz: i8,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x4x8xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #scalar, #identity]}
          ins(%bq, %bs, %bz : tensor<4x8x3xi8>, f32, i8)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_EQ(detail->needsLhsSum(), true);
  EXPECT_EQ(detail->needsRhsSum(), false);
  EXPECT_EQ(detail->lhs.zeroPointMap, AffineMap());
  EXPECT_EQ(detail->rhs.zeroPointMap, map(4, {}));
}

TEST_F(QuantizedContractionTest, BothZeroPointsNeedBothSums) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %az: i8,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %bz: i8,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #scalar, #identity]}
          ins(%aq, %as, %az : tensor<2x4x8xi8>, f32, i8)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #scalar, #identity]}
          ins(%bq, %bs, %bz : tensor<4x8x3xi8>, f32, i8)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_EQ(detail->needsLhsSum(), true);
  EXPECT_EQ(detail->needsRhsSum(), true);
  EXPECT_EQ(detail->lhs.zeroPointMap, map(4, {}));
  EXPECT_EQ(detail->rhs.zeroPointMap, map(4, {}));
}

TEST_F(QuantizedContractionTest, LhsScaleRetainsFloatingReduction) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: tensor<4xf32>,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, affine_map<(d0, d1, d2) -> (d1)>, #identity]}
          ins(%aq, %as : tensor<2x4x8xi8>, tensor<4xf32>)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_THAT(detail->integerReductionDims, ElementsAre(3));
  EXPECT_THAT(detail->floatingReductionDims, ElementsAre(2));
  EXPECT_EQ(detail->reductionExtent, 8);
}

TEST_F(QuantizedContractionTest, RhsScaleRetainsFloatingReduction) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: tensor<4xf32>,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x4x8xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, affine_map<(d0, d1, d2) -> (d0)>, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, tensor<4xf32>)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_THAT(detail->integerReductionDims, ElementsAre(3));
  EXPECT_THAT(detail->floatingReductionDims, ElementsAre(2));
  EXPECT_EQ(detail->reductionExtent, 8);
}

TEST_F(QuantizedContractionTest, LhsZeroPointRetainsFloatingReduction) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %az: tensor<4xi8>,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, affine_map<(d0, d1, d2) -> (d1)>, #identity]}
          ins(%aq, %as, %az : tensor<2x4x8xi8>, f32, tensor<4xi8>)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_THAT(detail->integerReductionDims, ElementsAre(3));
  EXPECT_THAT(detail->floatingReductionDims, ElementsAre(2));
  EXPECT_EQ(detail->reductionExtent, 8);
}

TEST_F(QuantizedContractionTest, RhsZeroPointRetainsFloatingReduction) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %bz: tensor<4xi8>,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x4x8xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, affine_map<(d0, d1, d2) -> (d0)>, #identity]}
          ins(%bq, %bs, %bz : tensor<4x8x3xi8>, f32, tensor<4xi8>)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_THAT(detail->integerReductionDims, ElementsAre(3));
  EXPECT_THAT(detail->floatingReductionDims, ElementsAre(2));
  EXPECT_EQ(detail->reductionExtent, 8);
}

TEST_F(QuantizedContractionTest, RebasesTransposedDequantizationMaps) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<4x2x8xi8>,
        %as: tensor<4xf32>,
        %az: tensor<4x2xi8>,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d1, d0, d2)>]}
          ins(%aq, %as, %az : tensor<4x2x8xi8>, tensor<4xf32>, tensor<4x2xi8>)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_EQ(detail->lhs.inputMap, map(4, {2, 0, 3}));
  EXPECT_EQ(detail->lhs.scaleMap, map(4, {2}));
  EXPECT_EQ(detail->lhs.zeroPointMap, map(4, {2, 0}));
  EXPECT_THAT(detail->integerReductionDims, ElementsAre(3));
  EXPECT_THAT(detail->floatingReductionDims, ElementsAre(2));
}

TEST_F(QuantizedContractionTest, RequiresAnIntegerReduction) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: tensor<4x8xf32>,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, affine_map<(d0, d1, d2) -> (d1, d2)>, #identity]}
          ins(%aq, %as : tensor<2x4x8xi8>, tensor<4x8xf32>)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  EXPECT_TRUE(failed(getQuantizedContraction(contraction())));
}

TEST_F(QuantizedContractionTest, RejectsDynamicIntegerReduction) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x?x8xf32>
    !b = tensor<?x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x?x8xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<?x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x?x8xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<?x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  EXPECT_TRUE(failed(getQuantizedContraction(contraction())));
}

TEST_F(QuantizedContractionTest, AcceptsDynamicFloatingReduction) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x?x8xf32>
    !b = tensor<?x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x?x8xi8>,
        %as: tensor<?xf32>,
        %ainit: !a,
        %bq: tensor<?x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, affine_map<(d0, d1, d2) -> (d1)>, #identity]}
          ins(%aq, %as : tensor<2x?x8xi8>, tensor<?xf32>)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<?x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_EQ(detail->reductionExtent, 8);
  EXPECT_THAT(detail->integerReductionDims, ElementsAre(3));
  EXPECT_THAT(detail->floatingReductionDims, ElementsAre(2));
}

TEST_F(QuantizedContractionTest, AcceptsSafeProductOfReductionExtents) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x256x511xf32>
    !b = tensor<256x511x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x256x511xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<256x511x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x256x511xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<256x511x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  auto detail = getQuantizedContraction(contraction());
  ASSERT_TRUE(succeeded(detail));
  EXPECT_EQ(detail->reductionExtent, 130816);
}

TEST_F(QuantizedContractionTest, RejectsUnsafeProductOfReductionExtents) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x256x512xf32>
    !b = tensor<256x512x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x256x512xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<256x512x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x256x512xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<256x512x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  // Both extents are individually safe, but their product exceeds 131071.
  EXPECT_TRUE(failed(getQuantizedContraction(contraction())));
}

TEST_F(QuantizedContractionTest, AcceptsCommutedMultiplyAndAdd) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x4x8xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %plain = linalg.generic {
        id = "plain", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      %commuted_mul = linalg.generic {
        id = "commuted_mul", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %y, %x : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      %commuted_add = linalg.generic {
        id = "commuted_add", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %acc, %p : f32
        linalg.yield %sum : f32
      } -> !c
      %commuted_both = linalg.generic {
        id = "commuted_both", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %y, %x : f32
        %sum = arith.addf %acc, %p : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  for (StringRef id :
       {"plain", "commuted_mul", "commuted_add", "commuted_both"}) {
    SCOPED_TRACE(id.str());
    EXPECT_TRUE(succeeded(getQuantizedContraction(contraction(id))));
  }
}

TEST_F(QuantizedContractionTest, RejectsBodyThatIgnoresAnInput) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x4x8xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 0.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %x : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  EXPECT_TRUE(failed(getQuantizedContraction(contraction())));
}

TEST_F(QuantizedContractionTest, RejectsNonzeroInit) {
  ASSERT_TRUE(parse(R"mlir(
    !a = tensor<2x4x8xf32>
    !b = tensor<4x8x3xf32>
    !c = tensor<2x3xf32>
    #identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    #scalar = affine_map<(d0, d1, d2) -> ()>
    #lhs = affine_map<(m, n, g, k) -> (m, g, k)>
    #rhs = affine_map<(m, n, g, k) -> (g, k, n)>
    #out = affine_map<(m, n, g, k) -> (m, n)>
    func.func @test(
        %aq: tensor<2x4x8xi8>,
        %as: f32,
        %ainit: !a,
        %bq: tensor<4x8x3xi8>,
        %bs: f32,
        %binit: !b,
        %cinit: !c) {
      %a = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%aq, %as : tensor<2x4x8xi8>, f32)
          outs(%ainit : !a) -> !a
      %b = iree_linalg_ext.dequantize_affine
          {indexing_maps = [#identity, #scalar, #identity]}
          ins(%bq, %bs : tensor<4x8x3xi8>, f32)
          outs(%binit : !b) -> !b
      %zero = arith.constant 1.0 : f32
      %init = linalg.fill ins(%zero : f32) outs(%cinit : !c) -> !c
      %test = linalg.generic {
        id = "test", indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%a, %b : !a, !b) outs(%init : !c) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %p = arith.mulf %x, %y : f32
        %sum = arith.addf %p, %acc : f32
        linalg.yield %sum : f32
      } -> !c
      return
    }
  )mlir"));
  EXPECT_TRUE(failed(getQuantizedContraction(contraction())));
}

TEST_F(QuantizedContractionTest, ExplicitExtentsCoveredAcrossMaps) {
  EXPECT_EQ(getExplicitExtentDimsMap({map(3, {2, 0}), map(3, {1})}),
            map(3, {}));
}

TEST_F(QuantizedContractionTest, ExplicitExtentsPreserveDomainOrder) {
  EXPECT_EQ(getExplicitExtentDimsMap({map(4, {2}), map(4, {2})}),
            map(4, {0, 1, 3}));
}

TEST_F(QuantizedContractionTest, ExplicitExtentsRequireBareDimensions) {
  AffineExpr d0 = getAffineDimExpr(0, &context);
  AffineExpr d1 = getAffineDimExpr(1, &context);
  AffineMap window = AffineMap::get(2, 0, {2 * d0 + d1}, &context);
  EXPECT_EQ(getExplicitExtentDimsMap({window, map(2, {0})}), map(2, {1}));
  EXPECT_EQ(getExplicitExtentDimsMap({window}), map(2, {0, 1}));
}

TEST_F(QuantizedContractionTest, ExplicitExtentsIgnoreConstants) {
  AffineMap constant =
      AffineMap::get(2, 0, {getAffineConstantExpr(0, &context)}, &context);
  EXPECT_EQ(getExplicitExtentDimsMap({constant, map(2, {})}), map(2, {0, 1}));
  EXPECT_EQ(getExplicitExtentDimsMap({map(0, {})}), map(0, {}));
}

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization::detail
