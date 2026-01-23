// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Utils/Indexing.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace testing;

TEST(GetMapCoefficientTest, unitDim) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  AffineExpr d0 = b.getAffineDimExpr(0);
  ASSERT_EQ(getCoefficient(d0, 0).value(), 1);
  ASSERT_EQ(getCoefficient(d0, 1).value(), 0);
  AffineExpr d1 = b.getAffineDimExpr(1);
  AffineExpr add = d0 + d1;
  ASSERT_EQ(getCoefficient(add, 0).value(), 1);
  ASSERT_EQ(getCoefficient(add, 1).value(), 1);
}

TEST(GetMapCoefficientTest, mul) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr mul1 = d0 * 5;
  ASSERT_EQ(getCoefficient(mul1, 0).value(), 5);
  ASSERT_EQ(getCoefficient(mul1, 1).value(), 0);
  AffineExpr mul2 = 5 * d0;
  ASSERT_EQ(getCoefficient(mul2, 0).value(), 5);
  ASSERT_EQ(getCoefficient(mul2, 1).value(), 0);
  AffineExpr mul3 = -5 * d0;
  ASSERT_EQ(getCoefficient(mul3, 0).value(), -5);
}

TEST(GetMapCoefficientTest, add) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr add1 = d0 + 5;
  ASSERT_EQ(getCoefficient(add1, 0).value(), 1);
  ASSERT_EQ(getCoefficient(add1, 1).value(), 0);
  AffineExpr add2 = d0 * 5 + 2;
  ASSERT_EQ(getCoefficient(add2, 0).value(), 5);
  AffineExpr add3 = d0 * 5 + 2 * d0;
  ASSERT_EQ(getCoefficient(add3, 0).value(), 7);
  AffineExpr d1 = b.getAffineDimExpr(1);
  AffineExpr add4 = 2 * ((5 * d0) + d1);
  ASSERT_EQ(getCoefficient(add4, 0).value(), 10);
  ASSERT_EQ(getCoefficient(add4, 1).value(), 2);
}

TEST(GetMapCoefficientTest, sub) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr sub1 = d0 - 5;
  ASSERT_EQ(getCoefficient(sub1, 0).value(), 1);
  ASSERT_EQ(getCoefficient(sub1, 1), 0);
}

TEST(GetMapCoefficientTest, mod) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr mod1 = d0 % 2;
  ASSERT_FALSE(getCoefficient(mod1, 0).has_value());
  ASSERT_EQ(getCoefficient(mod1, 1).value(), 0);
  AffineExpr mod2 = (d0 % 2) * 5;
  ASSERT_FALSE(getCoefficient(mod2, 0).has_value());
  AffineExpr mod3 = d0 * 5 + d0 % 2;
  ASSERT_FALSE(getCoefficient(mod3, 0).has_value());
  AffineExpr d1 = b.getAffineDimExpr(1);
  AffineExpr mod4 = d0 * 5 + d1 % 2;
  ASSERT_EQ(getCoefficient(mod4, 0).value(), 5);
  ASSERT_FALSE(getCoefficient(mod4, 1).has_value());
}

TEST(GetMapCoefficientTest, ceilDiv) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr div1 = d0.ceilDiv(2);
  ASSERT_FALSE(getCoefficient(div1, 0).has_value());
  ASSERT_EQ(getCoefficient(div1, 1).value(), 0);
  AffineExpr div2 = (d0.ceilDiv(2)) * 5;
  ASSERT_FALSE(getCoefficient(div2, 0).has_value());
  AffineExpr div3 = d0 * 5 + d0.ceilDiv(2);
  ASSERT_FALSE(getCoefficient(div3, 0).has_value());
  AffineExpr d1 = b.getAffineDimExpr(1);
  AffineExpr div4 = d0 * 5 + d1.ceilDiv(2);
  ASSERT_EQ(getCoefficient(div4, 0).value(), 5);
  ASSERT_FALSE(getCoefficient(div4, 1).has_value());
}

TEST(GetMapCoefficientTest, floorDiv) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  AffineExpr d0 = b.getAffineDimExpr(0);
  AffineExpr div1 = d0.floorDiv(2);
  ASSERT_FALSE(getCoefficient(div1, 0).has_value());
  ASSERT_EQ(getCoefficient(div1, 1).value(), 0);
  AffineExpr div2 = (d0.floorDiv(2)) * 5;
  ASSERT_FALSE(getCoefficient(div2, 0).has_value());
  AffineExpr div3 = d0 * 5 + d0.floorDiv(2);
  ASSERT_FALSE(getCoefficient(div3, 0).has_value());
  AffineExpr d1 = b.getAffineDimExpr(1);
  AffineExpr div4 = d0 * 5 + d1.floorDiv(2);
  ASSERT_EQ(getCoefficient(div4, 0).value(), 5);
  ASSERT_FALSE(getCoefficient(div4, 1).has_value());
}
