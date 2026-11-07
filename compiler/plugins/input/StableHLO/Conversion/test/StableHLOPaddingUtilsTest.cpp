// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/StableHLOPaddingUtils.h"

#include <cstdint>
#include <random>

#include <gtest/gtest.h>

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

// Compare the emitted scalar arithmetic against an independent APInt oracle,
// including rejection and the harmless size used after a rejected extent.
static void checkDimension(MLIRContext &context, int64_t size, uint64_t low,
                           uint64_t high, uint64_t interior, bool isUnsigned,
                           unsigned indexWidth) {
  SCOPED_TRACE(::testing::Message()
               << size << ',' << low << ',' << high << ',' << interior << ','
               << isUnsigned << ',' << indexWidth);
  OpBuilder b(&context);
  Location loc = b.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToStart(module->getBody());
  auto constant = [&](uint64_t bits) -> Value {
    return arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(b.getI64Type(), APInt(64, bits)));
  };
  Value dim = arith::ConstantIndexOp::create(b, loc, size);
  PaddedDimension result =
      buildPaddedDimension(b, loc, dim, constant(low), constant(high),
                           constant(interior), isUnsigned, indexWidth);

  auto padding = [isUnsigned](uint64_t bits) {
    APInt value(64, bits);
    return isUnsigned ? value.zext(128) : value.sext(128);
  };
  APInt exact = APInt(128, size) +
                APInt(128, size > 0 ? size - 1 : 0) * APInt(128, interior) +
                padding(low) + padding(high);
  bool negative = exact.isNegative();
  bool fits =
      !negative && exact.ule(APInt::getSignedMaxValue(indexWidth).zext(128));
  bool invalidInterior = !isUnsigned && APInt(64, interior).isNegative();
  if (!invalidInterior) {
    APInt actualSize, actualNegative;
    ASSERT_TRUE(matchPattern(result.size, m_ConstantInt(&actualSize)));
    ASSERT_TRUE(
        matchPattern(result.isNegative, m_ConstantInt(&actualNegative)));
    EXPECT_EQ(actualSize.getZExtValue(), fits ? exact.getZExtValue() : 0);
    EXPECT_EQ(actualNegative.getBoolValue(), negative);
  }
  bool rejected = false;
  module->walk([&rejected](IREE::Util::StatusCheckOkOp check) {
    auto select = check.getStatus().getDefiningOp<arith::SelectOp>();
    ASSERT_TRUE(select);
    APInt condition;
    ASSERT_TRUE(matchPattern(select.getCondition(), m_ConstantInt(&condition)));
    rejected |= condition.isZero();
  });
  EXPECT_EQ(rejected, invalidInterior || (!negative && !fits));
}

TEST(StableHLOPaddingUtilsTest, BoundaryExtents) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, IREE::Util::UtilDialect>();
  for (unsigned indexWidth : {32U, 64U}) {
    checkDimension(context, 3, -9223372036854775804ULL, -9223372036854775806ULL,
                   9223372036854775805ULL, false, indexWidth);
    checkDimension(context, 2, uint64_t{1} << 63, 3, INT64_MAX - 1, false,
                   indexWidth);
    checkDimension(context, 0, 0, 0, UINT64_MAX, true, indexWidth);
    checkDimension(context, 1, 0, 0, UINT64_MAX, true, indexWidth);
    checkDimension(context, 2, 0, 0, UINT64_MAX, true, indexWidth);
    checkDimension(context, INT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, true,
                   indexWidth);
    checkDimension(context, 1, uint64_t{1} << 63, uint64_t{1} << 63, 0, false,
                   indexWidth);
    checkDimension(context, 2, 0, 0, UINT64_MAX, false, indexWidth);
    checkDimension(context, INT32_MAX, 0, 1, 0, false, indexWidth);
  }
}

TEST(StableHLOPaddingUtilsTest, ExtentsAgainstWideOracle) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, IREE::Util::UtilDialect>();
  std::mt19937_64 random(0x504144);
  for (int i = 0; i < 256; ++i) {
    int64_t size = random() & INT64_MAX;
    uint64_t low = random(), high = random(), interior = random();
    if (i % 4 == 0) {
      size %= 32;
      low = static_cast<int64_t>(low % 1024) - 512;
      high = static_cast<int64_t>(high % 1024) - 512;
      interior %= 32;
    }
    for (bool isUnsigned : {false, true}) {
      checkDimension(context, size, low, high, interior, isUnsigned,
                     i % 2 ? 32 : 64);
    }
  }
}

} // namespace
} // namespace mlir::iree_compiler::stablehlo
