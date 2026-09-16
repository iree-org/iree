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

TEST(QuantizationUtilsTest, getDifferenceMagnitude) {
  EXPECT_EQ(getDifferenceMagnitude(3, true, std::nullopt, false), 7 + 7);
  EXPECT_EQ(getDifferenceMagnitude(3, true, std::nullopt, true), 7 + 0);
  EXPECT_EQ(getDifferenceMagnitude(3, false, std::nullopt, false), 4 + 4);
  EXPECT_EQ(getDifferenceMagnitude(3, false, std::nullopt, true), 4 + 0);
}

TEST(QuantizationUtilsTest, DifferenceMagnitudeRestrictedRange) {
  // The input range fits both signed and unsigned i4. Zero points still use
  // the full storage grid, independently of the input range.
  EXPECT_EQ(getDifferenceMagnitude(4, true, QuantMinMax{0, 7}, false), 7 + 15);
  EXPECT_EQ(getDifferenceMagnitude(4, true, QuantMinMax{0, 7}, true), 7);
  EXPECT_EQ(getDifferenceMagnitude(4, false, QuantMinMax{0, 7}, false), 7 + 8);
  EXPECT_EQ(getDifferenceMagnitude(4, false, QuantMinMax{0, 7}, true), 7);

  // A four-bit range carried in signed i8 is bounded by its negative endpoint.
  EXPECT_EQ(getDifferenceMagnitude(8, false, QuantMinMax{-8, 7}, false),
            8 + 128);
  EXPECT_EQ(getDifferenceMagnitude(8, false, QuantMinMax{-8, 7}, true), 8);
  EXPECT_EQ(getDifferenceMagnitude(8, false, QuantMinMax{-8, -1}, true), 8);
}

TEST(QuantizationUtilsTest, DifferenceMagnitudeFullRange) {
  EXPECT_EQ(getDifferenceMagnitude(8, false, QuantMinMax{-128, 127}, false),
            128 + 128);
  EXPECT_EQ(getDifferenceMagnitude(8, false, QuantMinMax{-128, 127}, true),
            128);
  // The unsigned bound intentionally overestimates |input - zero_point|.
  EXPECT_EQ(getDifferenceMagnitude(8, true, QuantMinMax{0, 255}, false),
            255 + 255);
  EXPECT_EQ(getDifferenceMagnitude(8, true, QuantMinMax{0, 255}, true), 255);
}

TEST(QuantizationUtilsTest, DifferenceMagnitudeI31) {
  // The bound itself can exceed i32 even though the storage type is supported.
  // Keep the calculation in i64 for the subsequent reduction-extent check.
  EXPECT_EQ(getDifferenceMagnitude(31, false, std::nullopt, true),
            1073741824LL);
  EXPECT_EQ(getDifferenceMagnitude(31, false, std::nullopt, false),
            2147483648LL);
  EXPECT_EQ(getDifferenceMagnitude(31, true, std::nullopt, true), 2147483647LL);
  EXPECT_EQ(getDifferenceMagnitude(31, true, std::nullopt, false),
            4294967294LL);
  EXPECT_EQ(getDifferenceMagnitude(31, false, QuantMinMax{-8, 7}, false),
            1073741832LL);
  EXPECT_EQ(getDifferenceMagnitude(31, true, QuantMinMax{0, 7}, false),
            2147483654LL);
}

TEST(QuantizationUtilsTest, DifferenceMagnitudeUnsupportedWidth) {
  for (bool isUnsigned : {false, true}) {
    for (bool isSymmetric : {false, true}) {
      EXPECT_EQ(
          getDifferenceMagnitude(0, isUnsigned, std::nullopt, isSymmetric), 0);
      EXPECT_EQ(
          getDifferenceMagnitude(32, isUnsigned, std::nullopt, isSymmetric), 0);
      // A restricted range does not make i32 storage supported.
      EXPECT_EQ(getDifferenceMagnitude(32, isUnsigned, QuantMinMax{0, 7},
                                       isSymmetric),
                0);
    }
  }
}

TEST(QuantizationUtilsTest, MaxReductionExtent) {
  EXPECT_EQ(getMaxReductionExtent(128, 128), 131071);
  EXPECT_EQ(getMaxReductionExtent(256, 256), 32767);
  EXPECT_EQ(getMaxReductionExtent(128, 256), 65535);
  EXPECT_EQ(getMaxReductionExtent(256, 128), 65535);
  EXPECT_EQ(getMaxReductionExtent(1, 1), 2147483647LL);
  // A single symmetric i16 product fits; two do not.
  EXPECT_EQ(getMaxReductionExtent(32768, 32768), 1);
}

TEST(QuantizationUtilsTest, MaxReductionExtentRejectsUnboundedOrUnsafeInputs) {
  EXPECT_EQ(getMaxReductionExtent(0, 128), 0);
  EXPECT_EQ(getMaxReductionExtent(128, 0), 0);
  EXPECT_EQ(getMaxReductionExtent(-1, 128), 0);
  EXPECT_EQ(getMaxReductionExtent(128, -1), 0);
  EXPECT_EQ(getMaxReductionExtent(65536, 65536), 0);
  EXPECT_EQ(getMaxReductionExtent(2147483648LL, 1), 0);
  // Asymmetric unsigned i31 magnitudes have a product larger than INT64_MAX.
  EXPECT_EQ(getMaxReductionExtent(4294967294LL, 4294967294LL), 0);
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
