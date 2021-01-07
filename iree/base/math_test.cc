// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/base/math.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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
// Pseudo-random number generators (PRNGs): **NOT CRYPTOGRAPHICALLY SECURE*
//==============================================================================
// NOTE: we leave the real testing to the authors; this just ensures we aren't
// `return 4;`ing it or ignoring the seed.

TEST(PRNG, SplitMix64) {
  iree_prng_splitmix64_state_t state;

  iree_prng_splitmix64_initialize(/*seed=*/0ull, &state);
  EXPECT_EQ(16294208416658607535ull, iree_prng_splitmix64_next(&state));
  EXPECT_EQ(7960286522194355700ull, iree_prng_splitmix64_next(&state));

  iree_prng_splitmix64_initialize(/*seed=*/1ull, &state);
  EXPECT_EQ(10451216379200822465ull, iree_prng_splitmix64_next(&state));
  EXPECT_EQ(13757245211066428519ull, iree_prng_splitmix64_next(&state));

  iree_prng_splitmix64_initialize(/*seed=*/UINT64_MAX, &state);
  EXPECT_EQ(16490336266968443936ull, iree_prng_splitmix64_next(&state));
  EXPECT_EQ(16834447057089888969ull, iree_prng_splitmix64_next(&state));
}

TEST(PRNG, Xoroshiro128) {
  iree_prng_xoroshiro128_state_t state;

  iree_prng_xoroshiro128_initialize(/*seed=*/0ull, &state);
  EXPECT_EQ(5807750865143411619ull,
            iree_prng_xoroshiro128plus_next_uint60(&state));
  EXPECT_TRUE(iree_prng_xoroshiro128plus_next_bool(&state));
  EXPECT_EQ(218u, iree_prng_xoroshiro128plus_next_uint8(&state));
  EXPECT_EQ(1647201753u, iree_prng_xoroshiro128plus_next_uint32(&state));
  EXPECT_EQ(7260361800523965311ull,
            iree_prng_xoroshiro128starstar_next_uint64(&state));

  iree_prng_xoroshiro128_initialize(/*seed=*/1ull, &state);
  EXPECT_EQ(5761717516557699368ull,
            iree_prng_xoroshiro128plus_next_uint60(&state));
  EXPECT_TRUE(iree_prng_xoroshiro128plus_next_bool(&state));
  EXPECT_EQ(103u, iree_prng_xoroshiro128plus_next_uint8(&state));
  EXPECT_EQ(2242241045u, iree_prng_xoroshiro128plus_next_uint32(&state));
  EXPECT_EQ(661144386810419178ull,
            iree_prng_xoroshiro128starstar_next_uint64(&state));

  iree_prng_xoroshiro128_initialize(/*seed=*/UINT64_MAX, &state);
  EXPECT_EQ(14878039250348781289ull,
            iree_prng_xoroshiro128plus_next_uint60(&state));
  EXPECT_FALSE(iree_prng_xoroshiro128plus_next_bool(&state));
  EXPECT_EQ(137u, iree_prng_xoroshiro128plus_next_uint8(&state));
  EXPECT_EQ(2111322015u, iree_prng_xoroshiro128plus_next_uint32(&state));
  EXPECT_EQ(138107609852220106ull,
            iree_prng_xoroshiro128starstar_next_uint64(&state));
}

TEST(PRNG, MiniLcg128) {
  iree_prng_minilcg128_state_t state;

  iree_prng_minilcg128_initialize(/*seed=*/0ull, &state);
  EXPECT_EQ(111u, iree_prng_minilcg128_next_uint8(&state));
  for (int i = 0; i < 100; ++i) {
    iree_prng_minilcg128_next_uint8(&state);
  }
  EXPECT_EQ(212u, iree_prng_minilcg128_next_uint8(&state));

  iree_prng_minilcg128_initialize(/*seed=*/1ull, &state);
  EXPECT_EQ(198u, iree_prng_minilcg128_next_uint8(&state));
  for (int i = 0; i < 100; ++i) {
    iree_prng_minilcg128_next_uint8(&state);
  }
  EXPECT_EQ(135u, iree_prng_minilcg128_next_uint8(&state));

  iree_prng_minilcg128_initialize(/*seed=*/UINT64_MAX, &state);
  EXPECT_EQ(12u, iree_prng_minilcg128_next_uint8(&state));
  for (int i = 0; i < 100; ++i) {
    iree_prng_minilcg128_next_uint8(&state);
  }
  EXPECT_EQ(229u, iree_prng_minilcg128_next_uint8(&state));
}

}  // namespace
