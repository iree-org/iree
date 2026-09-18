// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/QuantizationUtils.h"

#include <limits>

#include <gtest/gtest.h>

namespace mlir::iree_compiler::GlobalOptimization {
namespace {

TEST(QuantizationUtilsTest, getStorageMagnitude) {
  EXPECT_EQ(getStorageMagnitude(3, true), 7);
  EXPECT_EQ(getStorageMagnitude(3, false), 4);
  EXPECT_EQ(getStorageMagnitude(8, true), 255);
  EXPECT_EQ(getStorageMagnitude(8, false), 128);

  // i31 is the widest supported storage type; i32 is rejected.
  EXPECT_EQ(getStorageMagnitude(31, true), 2147483647LL);
  EXPECT_EQ(getStorageMagnitude(31, false), 1073741824LL);
  EXPECT_EQ(getStorageMagnitude(32, true), 0);
  EXPECT_EQ(getStorageMagnitude(32, false), 0);

  // Test bit width greater than accumulator width.
  EXPECT_EQ(getStorageMagnitude(33, true), 0);
  EXPECT_EQ(getStorageMagnitude(33, false), 0);

  // Test zero bit width.
  EXPECT_EQ(getStorageMagnitude(0, true), 0);
  EXPECT_EQ(getStorageMagnitude(0, false), 0);
}

TEST(QuantizationUtilsTest, OperandRangesStorageGrid) {
  struct TestCase {
    unsigned bitWidth;
    bool isUnsigned;
    QuantMinMax storage;
  };
  for (const auto &test :
       {TestCase{3, true, {0, 7}}, TestCase{3, false, {-4, 3}},
        TestCase{8, true, {0, 255}}, TestCase{8, false, {-128, 127}},
        TestCase{31, true, {0, 2147483647LL}},
        TestCase{31, false, {-1073741824LL, 1073741823LL}}}) {
    SCOPED_TRACE(test.bitWidth);
    SCOPED_TRACE(test.isUnsigned);
    for (bool isSymmetric : {false, true}) {
      auto ranges = getQuantizedOperandRanges(test.bitWidth, test.isUnsigned,
                                              std::nullopt, isSymmetric);
      ASSERT_TRUE(ranges);
      EXPECT_EQ(ranges->input.min, test.storage.min);
      EXPECT_EQ(ranges->input.max, test.storage.max);
      EXPECT_EQ(ranges->zeroPoint.min, isSymmetric ? 0 : test.storage.min);
      EXPECT_EQ(ranges->zeroPoint.max, isSymmetric ? 0 : test.storage.max);
    }
  }
}

TEST(QuantizationUtilsTest, OperandRangesRestrictedInput) {
  // Restricting inputs does not restrict zero points to the same range.
  for (bool isUnsigned : {false, true}) {
    for (bool isSymmetric : {false, true}) {
      auto ranges = getQuantizedOperandRanges(8, isUnsigned, QuantMinMax{0, 7},
                                              isSymmetric);
      ASSERT_TRUE(ranges);
      EXPECT_EQ(ranges->input.min, 0);
      EXPECT_EQ(ranges->input.max, 7);
      EXPECT_EQ(ranges->zeroPoint.min, isSymmetric || isUnsigned ? 0 : -128);
      EXPECT_EQ(ranges->zeroPoint.max, isSymmetric  ? 0
                                       : isUnsigned ? 255
                                                    : 127);
    }
  }
  auto negative =
      getQuantizedOperandRanges(8, false, QuantMinMax{-8, -1}, false);
  ASSERT_TRUE(negative);
  EXPECT_EQ(negative->input.min, -8);
  EXPECT_EQ(negative->input.max, -1);
  EXPECT_EQ(negative->zeroPoint.min, -128);
  EXPECT_EQ(negative->zeroPoint.max, 127);
}

TEST(QuantizationUtilsTest, OperandRangesUnsupportedWidth) {
  for (unsigned bitWidth : {0, 32, 33, 64}) {
    for (bool isUnsigned : {false, true}) {
      for (bool isSymmetric : {false, true}) {
        EXPECT_FALSE(getQuantizedOperandRanges(bitWidth, isUnsigned,
                                               std::nullopt, isSymmetric));
        // A restricted range does not make unsupported storage legal.
        EXPECT_FALSE(getQuantizedOperandRanges(bitWidth, isUnsigned,
                                               QuantMinMax{0, 7}, isSymmetric));
      }
    }
  }
}

TEST(QuantizationUtilsTest, MaxReductionExtentSymmetric) {
  QuantizedOperandRanges signedI8{{-128, 127}, {0, 0}};
  QuantizedOperandRanges unsignedI8{{0, 255}, {0, 0}};
  // The raw product limits the depth, including -128 * -128 = +16384.
  EXPECT_EQ(getMaxReductionExtent(signedI8, signedI8), 131071);
  EXPECT_EQ(getMaxReductionExtent(unsignedI8, unsignedI8), 33025);
  EXPECT_EQ(getMaxReductionExtent(signedI8, unsignedI8), 65793);
  EXPECT_EQ(getMaxReductionExtent(unsignedI8, signedI8), 65793);
  // A single symmetric i16 product fits; two do not.
  QuantizedOperandRanges signedI16{{-32768, 32767}, {0, 0}};
  EXPECT_EQ(getMaxReductionExtent(signedI16, signedI16), 1);
}

TEST(QuantizationUtilsTest, MaxReductionExtentAsymmetric) {
  QuantizedOperandRanges signedI8{{-128, 127}, {-128, 127}};
  QuantizedOperandRanges unsignedI8{{0, 255}, {0, 255}};
  // The centered signed product is bounded by 255^2, not 256^2.
  EXPECT_EQ(getMaxReductionExtent(signedI8, signedI8), 33025);
  // P+Q limits unsigned depth to INT32_MAX/(2*255^2), even though the
  // centered result is bounded by N*255^2.
  EXPECT_EQ(getMaxReductionExtent(unsignedI8, unsignedI8), 16512);
  // For mixed signs, P+Q-R has range [-97665, 97410] per element. It is
  // larger than both P+Q and the centered product, so it limits the depth.
  EXPECT_EQ(getMaxReductionExtent(signedI8, unsignedI8), 21988);
  EXPECT_EQ(getMaxReductionExtent(unsignedI8, signedI8), 21988);
}

TEST(QuantizationUtilsTest, MaxReductionExtentOneZeroPoint) {
  QuantizedOperandRanges signedSymmetric{{-128, 127}, {0, 0}};
  QuantizedOperandRanges signedAsymmetric{{-128, 127}, {-128, 127}};
  QuantizedOperandRanges unsignedSymmetric{{0, 255}, {0, 0}};
  QuantizedOperandRanges unsignedAsymmetric{{0, 255}, {0, 255}};
  EXPECT_EQ(getMaxReductionExtent(signedSymmetric, signedAsymmetric), 65793);
  EXPECT_EQ(getMaxReductionExtent(signedAsymmetric, signedSymmetric), 65793);
  EXPECT_EQ(getMaxReductionExtent(unsignedSymmetric, unsignedAsymmetric),
            33025);
  EXPECT_EQ(getMaxReductionExtent(unsignedAsymmetric, unsignedSymmetric),
            33025);
}

TEST(QuantizationUtilsTest, MaxReductionExtentRestrictedInput) {
  // Inputs are only three bits, but the zero points still occupy four bits.
  // The cross term and centered result reach 225, exceeding P+Q's 210.
  QuantizedOperandRanges restricted{{0, 7}, {0, 15}};
  EXPECT_EQ(getMaxReductionExtent(restricted, restricted), 9544371);
  QuantizedOperandRanges negative{{-8, -1}, {0, 0}};
  EXPECT_EQ(getMaxReductionExtent(negative, negative), 33554431);
}

TEST(QuantizationUtilsTest, MaxReductionExtentZeroInputs) {
  QuantizedOperandRanges zero{{0, 0}, {0, 0}};
  QuantizedOperandRanges signedI8{{-128, 127}, {0, 0}};
  // Even when products vanish, the separately computed operand sums must fit.
  EXPECT_EQ(getMaxReductionExtent(signedI8, zero), 16777215);
  EXPECT_EQ(getMaxReductionExtent(zero, signedI8), 16777215);
  EXPECT_EQ(getMaxReductionExtent(zero, zero), 0);
  // Nonzero zero points still produce a cross term when both inputs are zero.
  QuantizedOperandRanges zeroInput{{0, 0}, {0, 255}};
  EXPECT_EQ(getMaxReductionExtent(zeroInput, zeroInput), 33025);
}

TEST(QuantizationUtilsTest, MaxReductionExtentI31) {
  QuantizedOperandRanges one{{1, 1}, {0, 0}};
  EXPECT_EQ(getMaxReductionExtent(one, one), 2147483647LL);
  for (bool isUnsigned : {false, true}) {
    for (bool isSymmetric : {false, true}) {
      auto ranges =
          getQuantizedOperandRanges(31, isUnsigned, std::nullopt, isSymmetric);
      ASSERT_TRUE(ranges);
      EXPECT_EQ(getMaxReductionExtent(*ranges, one), 1);
      EXPECT_EQ(getMaxReductionExtent(one, *ranges), 1);
      // Even N=1 is unsafe. For unsigned asymmetric i31, P+Q approaches
      // INT64_MAX; computing its bound must not overflow in the host.
      EXPECT_EQ(getMaxReductionExtent(*ranges, *ranges), 0);
    }
  }
  auto signedRanges = getQuantizedOperandRanges(31, false, std::nullopt, false);
  auto unsignedRanges =
      getQuantizedOperandRanges(31, true, std::nullopt, false);
  ASSERT_TRUE(signedRanges);
  ASSERT_TRUE(unsignedRanges);
  EXPECT_EQ(getMaxReductionExtent(*signedRanges, *unsignedRanges), 0);
  EXPECT_EQ(getMaxReductionExtent(*unsignedRanges, *signedRanges), 0);
}

TEST(QuantizationUtilsTest, BoundedReductionExtentAtLimit) {
  EXPECT_EQ(getBoundedReductionExtent({131071}, 131071), 131071);
  EXPECT_EQ(getBoundedReductionExtent({131072}, 131071), std::nullopt);
  EXPECT_EQ(getBoundedReductionExtent({3, 5, 7}, 105), 105);
  EXPECT_EQ(getBoundedReductionExtent({3, 5, 7}, 104), std::nullopt);
  EXPECT_EQ(getBoundedReductionExtent({7, 3, 5}, 105), 105);
  EXPECT_EQ(getBoundedReductionExtent({1, 1}, 1), 1);
}

TEST(QuantizationUtilsTest,
     BoundedReductionExtentRequiresPositiveStaticExtents) {
  EXPECT_EQ(getBoundedReductionExtent({}, 100), std::nullopt);
  EXPECT_EQ(getBoundedReductionExtent({0}, 100), std::nullopt);
  EXPECT_EQ(getBoundedReductionExtent({2, -1}, 100), std::nullopt);
  EXPECT_EQ(getBoundedReductionExtent({1}, 0), std::nullopt);
  EXPECT_EQ(getBoundedReductionExtent({1}, -1), std::nullopt);
}

TEST(QuantizationUtilsTest, BoundedReductionExtentAvoidsOverflow) {
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(getBoundedReductionExtent({max}, max), max);
  EXPECT_EQ(getBoundedReductionExtent({max, 2}, max), std::nullopt);
  EXPECT_EQ(getBoundedReductionExtent({2, max}, max), std::nullopt);
  EXPECT_EQ(getBoundedReductionExtent({4294967296LL, 4294967296LL}, max),
            std::nullopt);
}

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
