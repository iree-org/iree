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

//==============================================================================
// Pseudo-random number generators (PRNGs): **NOT CRYPTOGRAPHICALLY SECURE*
//==============================================================================
// NOTE: we leave the real testing to the authors; this just ensures we aren't
// `return 4;`ing it or ignoring the seed.

#include "iree/base/internal/prng.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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
  EXPECT_EQ(21u, iree_prng_minilcg128_next_uint8(&state));
  for (int i = 0; i < 100; ++i) {
    iree_prng_minilcg128_next_uint8(&state);
  }
  EXPECT_EQ(18u, iree_prng_minilcg128_next_uint8(&state));

  iree_prng_minilcg128_initialize(/*seed=*/1ull, &state);
  EXPECT_EQ(20u, iree_prng_minilcg128_next_uint8(&state));
  for (int i = 0; i < 100; ++i) {
    iree_prng_minilcg128_next_uint8(&state);
  }
  EXPECT_EQ(13u, iree_prng_minilcg128_next_uint8(&state));

  iree_prng_minilcg128_initialize(/*seed=*/UINT64_MAX, &state);
  EXPECT_EQ(234u, iree_prng_minilcg128_next_uint8(&state));
  for (int i = 0; i < 100; ++i) {
    iree_prng_minilcg128_next_uint8(&state);
  }
  EXPECT_EQ(59u, iree_prng_minilcg128_next_uint8(&state));
}

}  // namespace
