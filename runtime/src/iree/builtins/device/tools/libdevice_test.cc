// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/builtins/device/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

static constexpr uint16_t kF16ExponentMask = 0x7C00;
static constexpr uint16_t kMantissaMask = 0x03FF;

static uint16_t f16BitsIsNaN(uint16_t bits) {
  return ((bits & kF16ExponentMask) == kF16ExponentMask) &&
         (bits & kMantissaMask);
}

static uint16_t f16BitsIsDenormalOrZero(uint16_t bits) {
  return !(bits & kF16ExponentMask);
}

TEST(LibDeviceTest, iree_h2f_ieee) {
  // Iterate over all f16 values as u16. Needs a wider type for loop condition.
  for (uint32_t f16Bits = 0; f16Bits <= 0xffff; ++f16Bits) {
    float f32 = iree_h2f_ieee(f16Bits);
    if (f16BitsIsNaN(f16Bits)) {
      EXPECT_TRUE(std::isnan(f32));
    } else if (f16Bits == 0) {
      EXPECT_EQ(f32, 0.f);
    } else if (f16BitsIsDenormalOrZero(f16Bits)) {
      EXPECT_LE(std::abs(f32), 6.1e-5f);
    } else {
      EXPECT_EQ(f32, iree_math_f16_to_f32(f16Bits));
    }
  }
}

TEST(LibDeviceTest, iree_f2h_ieee) {
  auto testcase = [](uint32_t f32Bits) {
    float f32 = 0.f;
    memcpy(&f32, &f32Bits, sizeof f32);
    uint16_t f16Bits = iree_f2h_ieee(f32);
    if (std::isnan(f32)) {
      EXPECT_TRUE(f16BitsIsNaN(f16Bits));
    } else if (f32 == 0.f) {
      EXPECT_EQ(f16Bits, std::signbit(f32) ? 0x8000 : 0);
    } else if (std::abs(f32) < 6.1e-5f) {
      EXPECT_TRUE(f16BitsIsDenormalOrZero(f16Bits));
    } else {
      EXPECT_EQ(f16Bits, iree_math_f32_to_f16(f32));
    }
  };
  // Testing all 2^32 float32 values is too much. We test two slices of that
  // space.
  //
  // Test all 2^12 float32 values that have only their top 12 bits potentially
  // set. That covers all combination of sign x exponent x the top 3 bits of
  // mantissa. The bottom 20 mantissa bits stay zero, so this lacks coverage
  // of rounding behavior.
  for (uint32_t f32Top12Bits = 0; f32Top12Bits <= 0xfff; ++f32Top12Bits) {
    testcase(f32Top12Bits << 20);
  }
  // For a few select exponent values, test all 2^12 float32 values whose
  // *mantissa* bits have only their top 12 bits potentially set.
  // Since float16 has only 10 bits of mantissa, that covers all float16
  // mantissas plus 2 additional bits of float32 mantissa past the truncation.
  // Having 2 extra bits should be exactly what is relevant to testing rounding
  // behavior including tie breaks to "nearest even".
  for (uint32_t f32MantissaTop12Bits = 0; f32MantissaTop12Bits <= 0xfff;
       ++f32MantissaTop12Bits) {
    // A few select exponent values.
    for (uint32_t f32ExponentBits :
         {0 /*denormal*/, 1 /*minimum normal*/, 127 /*neutral*/,
          254 /*maximum finite*/, 255 /*infinite*/}) {
      for (uint32_t f32SignBit : {0, 1}) {
        testcase((f32SignBit << 31) | (f32ExponentBits << 23) |
                 (f32MantissaTop12Bits << 11));
      }
    }
  }
}
