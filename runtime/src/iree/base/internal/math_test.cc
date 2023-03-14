// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"

#include <cfloat>

#include "iree/testing/gtest.h"

namespace {

//==============================================================================
// Bitwise rotation (aka circular shifts)
//==============================================================================

TEST(BitwiseRotationTest, ROTL64) {
  EXPECT_EQ(0ull, iree_math_rotl_u64(0ull, 0u));
  EXPECT_EQ(0ull, iree_math_rotl_u64(0ull, 0u));
  EXPECT_EQ(1ull, iree_math_rotl_u64(1ull, 0u));
  EXPECT_EQ(1ull, iree_math_rotl_u64(1ull, 0u));

  EXPECT_EQ(2ull, iree_math_rotl_u64(1ull, 1u));
  EXPECT_EQ(2ull, iree_math_rotl_u64(1ull, 1u));
  EXPECT_EQ(UINT64_MAX, iree_math_rotl_u64(UINT64_MAX, 63u));
  EXPECT_EQ(UINT64_MAX, iree_math_rotl_u64(UINT64_MAX, 64u));
}

TEST(BitwiseRotationTest, ROTR64) {
  EXPECT_EQ(0ull, iree_math_rotr_u64(0ull, 0u));
  EXPECT_EQ(0ull, iree_math_rotr_u64(0ull, 0u));
  EXPECT_EQ(1ull, iree_math_rotr_u64(1ull, 0u));
  EXPECT_EQ(1ull, iree_math_rotr_u64(1ull, 0u));

  EXPECT_EQ(1ull, iree_math_rotr_u64(2ull, 1u));
  EXPECT_EQ(0x8000000000000000ull, iree_math_rotr_u64(2ull, 2u));
  EXPECT_EQ(0x8000000000000000ull, iree_math_rotr_u64(1ull, 1u));
  EXPECT_EQ(0x4000000000000000ull, iree_math_rotr_u64(1ull, 2u));
}

//==============================================================================
// Bit scanning/counting
//==============================================================================

TEST(BitwiseScansTest, CLZ32) {
  EXPECT_EQ(32, iree_math_count_leading_zeros_u32(uint32_t{}));
  EXPECT_EQ(0, iree_math_count_leading_zeros_u32(~uint32_t{}));
  for (int index = 0; index < 32; index++) {
    uint32_t x = 1u << index;
    const int cnt = 31 - index;
    ASSERT_EQ(cnt, iree_math_count_leading_zeros_u32(x)) << index;
    ASSERT_EQ(cnt, iree_math_count_leading_zeros_u32(x + x - 1)) << index;
  }
}

TEST(BitwiseScansTest, CLZ64) {
  EXPECT_EQ(64, iree_math_count_leading_zeros_u64(uint64_t{}));
  EXPECT_EQ(0, iree_math_count_leading_zeros_u64(~uint64_t{}));
  for (int index = 0; index < 64; index++) {
    uint64_t x = 1ull << index;
    const int cnt = 63 - index;
    ASSERT_EQ(cnt, iree_math_count_leading_zeros_u64(x)) << index;
    ASSERT_EQ(cnt, iree_math_count_leading_zeros_u64(x + x - 1)) << index;
  }
}

TEST(BitwiseScansTest, CTZ32) {
  EXPECT_EQ(0, iree_math_count_trailing_zeros_u32(~uint32_t{}));
  for (int index = 0; index < 32; index++) {
    uint32_t x = static_cast<uint32_t>(1) << index;
    const int cnt = index;
    ASSERT_EQ(cnt, iree_math_count_trailing_zeros_u32(x)) << index;
    ASSERT_EQ(cnt, iree_math_count_trailing_zeros_u32(~(x - 1))) << index;
  }
}

TEST(BitwiseScansTest, CTZ64) {
  // iree_math_count_trailing_zeros_u32
  EXPECT_EQ(0, iree_math_count_trailing_zeros_u64(~uint64_t{}));
  for (int index = 0; index < 64; index++) {
    uint64_t x = static_cast<uint64_t>(1) << index;
    const int cnt = index;
    ASSERT_EQ(cnt, iree_math_count_trailing_zeros_u64(x)) << index;
    ASSERT_EQ(cnt, iree_math_count_trailing_zeros_u64(~(x - 1))) << index;
  }
}

//==============================================================================
// Population count
//==============================================================================

TEST(PopulationCountTest, Ones32) {
  EXPECT_EQ(0, iree_math_count_ones_u32(0u));
  EXPECT_EQ(1, iree_math_count_ones_u32(1u));
  EXPECT_EQ(29, iree_math_count_ones_u32(-15u));
  EXPECT_EQ(5, iree_math_count_ones_u32(341u));
  EXPECT_EQ(32, iree_math_count_ones_u32(UINT32_MAX));
  EXPECT_EQ(31, iree_math_count_ones_u32(UINT32_MAX - 1));
}

TEST(PopulationCountTest, Ones64) {
  EXPECT_EQ(0, iree_math_count_ones_u64(0ull));
  EXPECT_EQ(1, iree_math_count_ones_u64(1ull));
  EXPECT_EQ(61, iree_math_count_ones_u64(-15ull));
  EXPECT_EQ(5, iree_math_count_ones_u64(341ull));
  EXPECT_EQ(64, iree_math_count_ones_u64(UINT64_MAX));
  EXPECT_EQ(63, iree_math_count_ones_u64(UINT64_MAX - 1ull));
}

//==============================================================================
// Rounding and alignment
//==============================================================================

TEST(RoundingTest, UpToNextPow232) {
  constexpr uint32_t kUint16Max = UINT16_MAX;
  constexpr uint32_t kUint32Max = UINT32_MAX;
  EXPECT_EQ(0u, iree_math_round_up_to_pow2_u32(0u));
  EXPECT_EQ(1u, iree_math_round_up_to_pow2_u32(1u));
  EXPECT_EQ(2u, iree_math_round_up_to_pow2_u32(2u));
  EXPECT_EQ(4u, iree_math_round_up_to_pow2_u32(3u));
  EXPECT_EQ(8u, iree_math_round_up_to_pow2_u32(8u));
  EXPECT_EQ(16u, iree_math_round_up_to_pow2_u32(9u));
  EXPECT_EQ(kUint16Max + 1u, iree_math_round_up_to_pow2_u32(kUint16Max - 1u));
  EXPECT_EQ(kUint16Max + 1u, iree_math_round_up_to_pow2_u32(kUint16Max));
  EXPECT_EQ(kUint16Max + 1u, iree_math_round_up_to_pow2_u32(kUint16Max + 1u));
  EXPECT_EQ(131072u, iree_math_round_up_to_pow2_u32(kUint16Max + 2u));
  EXPECT_EQ(262144u, iree_math_round_up_to_pow2_u32(262144u - 1u));
  EXPECT_EQ(0x80000000u, iree_math_round_up_to_pow2_u32(0x7FFFFFFFu));
  EXPECT_EQ(0x80000000u, iree_math_round_up_to_pow2_u32(0x80000000u));

  // NOTE: wrap to 0.
  EXPECT_EQ(0u, iree_math_round_up_to_pow2_u32(0x80000001u));
  EXPECT_EQ(0u, iree_math_round_up_to_pow2_u32(kUint32Max - 1u));
  EXPECT_EQ(0u, iree_math_round_up_to_pow2_u32(kUint32Max));
}

TEST(RoundingTest, UpToNextPow264) {
  constexpr uint64_t kUint16Max = UINT16_MAX;
  constexpr uint64_t kUint64Max = UINT64_MAX;
  EXPECT_EQ(0ull, iree_math_round_up_to_pow2_u64(0ull));
  EXPECT_EQ(1ull, iree_math_round_up_to_pow2_u64(1ull));
  EXPECT_EQ(2ull, iree_math_round_up_to_pow2_u64(2ull));
  EXPECT_EQ(4ull, iree_math_round_up_to_pow2_u64(3ull));
  EXPECT_EQ(8ull, iree_math_round_up_to_pow2_u64(8ull));
  EXPECT_EQ(16ull, iree_math_round_up_to_pow2_u64(9ull));
  EXPECT_EQ(kUint16Max + 1ull,
            iree_math_round_up_to_pow2_u64(kUint16Max - 1ull));
  EXPECT_EQ(kUint16Max + 1ull, iree_math_round_up_to_pow2_u64(kUint16Max));
  EXPECT_EQ(kUint16Max + 1ull,
            iree_math_round_up_to_pow2_u64(kUint16Max + 1ull));
  EXPECT_EQ(131072ull, iree_math_round_up_to_pow2_u64(kUint16Max + 2ull));
  EXPECT_EQ(0x100000000ull, iree_math_round_up_to_pow2_u64(0xFFFFFFFEull));
  EXPECT_EQ(0x100000000ull, iree_math_round_up_to_pow2_u64(0xFFFFFFFFull));
  EXPECT_EQ(0x80000000ull, iree_math_round_up_to_pow2_u64(0x7FFFFFFFull));
  EXPECT_EQ(0x80000000ull, iree_math_round_up_to_pow2_u64(0x80000000ull));
  EXPECT_EQ(0x100000000ull, iree_math_round_up_to_pow2_u64(0x80000001ull));

  // NOTE: wrap to 0.
  EXPECT_EQ(0ull, iree_math_round_up_to_pow2_u64(0x8000000000000001ull));
  EXPECT_EQ(0ull, iree_math_round_up_to_pow2_u64(kUint64Max - 1ull));
  EXPECT_EQ(0ull, iree_math_round_up_to_pow2_u64(kUint64Max));
}

//==============================================================================
// FP16 support
//==============================================================================

TEST(F16ConversionTest, F32ToF16) {
  // Within range, normal truncation.
  EXPECT_EQ(0x3400, iree_math_f32_to_f16(0.25f));
  EXPECT_EQ(0xd646, iree_math_f32_to_f16(-100.375f));
  // Overflow
  EXPECT_EQ(0x7C00, iree_math_f32_to_f16(FLT_MAX));
  EXPECT_EQ(0xFC00, iree_math_f32_to_f16(-FLT_MAX));
  // Underflow
  EXPECT_EQ(0, iree_math_f32_to_f16(FLT_MIN));
  EXPECT_EQ(0x8000, iree_math_f32_to_f16(-FLT_MIN));
}

TEST(F16ConversionTest, F32ToF16ToF32) {
  constexpr float kF16Max = 65504.f;
  constexpr float kF16Min = 0.0000610351563f;
  // Within range, should just recover.
  EXPECT_EQ(0.25f, iree_math_f16_to_f32(iree_math_f32_to_f16(0.25f)));
  EXPECT_EQ(-100.375f, iree_math_f16_to_f32(iree_math_f32_to_f16(-100.375f)));
  EXPECT_EQ(kF16Max, iree_math_f16_to_f32(iree_math_f32_to_f16(kF16Max)));
  EXPECT_EQ(kF16Min, iree_math_f16_to_f32(iree_math_f32_to_f16(kF16Min)));
  // Overflow
  EXPECT_EQ(INFINITY, iree_math_f16_to_f32(iree_math_f32_to_f16(FLT_MAX)));
  EXPECT_EQ(-INFINITY, iree_math_f16_to_f32(iree_math_f32_to_f16(-FLT_MAX)));
  EXPECT_GT(kF16Max + 1.f,
            iree_math_f16_to_f32(iree_math_f32_to_f16(kF16Max + 1.f)));
  // Underflow
  EXPECT_EQ(0.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(FLT_MIN)));
  EXPECT_EQ(0.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(-FLT_MIN)));
  EXPECT_EQ(0.0f,
            iree_math_f16_to_f32(iree_math_f32_to_f16(kF16Min - kF16Min / 2)));
  // Inf and Nan
  EXPECT_EQ(INFINITY, iree_math_f16_to_f32(iree_math_f32_to_f16(INFINITY)));
  EXPECT_EQ(-INFINITY, iree_math_f16_to_f32(iree_math_f32_to_f16(-INFINITY)));
  // Check that the result is a Nan with nan != nan.
  float nan = iree_math_f16_to_f32(iree_math_f32_to_f16(NAN));
  EXPECT_NE(nan, nan);
}

}  // namespace
