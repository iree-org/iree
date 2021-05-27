// Copyright 2019 Google LLC
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
//
// Pseudo-random number generators (PRNGs): **NOT CRYPTOGRAPHICALLY SECURE*
//
// Only use these tiny little PRNGs to introduce a bit of randomnessish behavior
// to things like balancing and backoff algorithms.
//
//==============================================================================

#ifndef IREE_BASE_INTERNAL_PRNG_H_
#define IREE_BASE_INTERNAL_PRNG_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/internal/math.h"
#include "iree/base/target_platform.h"

#if defined(IREE_ARCH_ARM_64)
#include <arm_neon.h>
#endif  // IREE_ARCH_ARM_64

//==============================================================================
// Pseudo-random number generators (PRNGs): **NOT CRYPTOGRAPHICALLY SECURE*
//==============================================================================

// A fixed-increment version of Java 8's SplittableRandom generator
// See http://dx.doi.org/10.1145/2714064.2660195 and
// http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html
//
// SplitMix64 as recommended for use with xoroshiro by the authors:
// http://prng.di.unimi.it/splitmix64.c
// http://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
typedef uint64_t iree_prng_splitmix64_state_t;

// Initializes a SplitMix64 PRNG state vector; |out_state| is overwritten.
// |seed| may be any 64-bit value.
static inline void iree_prng_splitmix64_initialize(
    uint64_t seed, iree_prng_splitmix64_state_t* out_state) {
  *out_state = seed;
}

// Steps a SplitMix64 PRNG state vector and yields a value for use.
static inline uint64_t iree_prng_splitmix64_next(
    iree_prng_splitmix64_state_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// A small **pseudorandom** number generator (named after the operations used).
// http://prng.di.unimi.it/
typedef struct {
  uint64_t value[2];
} iree_prng_xoroshiro128_state_t;

// Initializes a xoroshiro128+ PRNG state vector; |out_state| is overwritten.
// |seed| may be any 64-bit value.
static inline void iree_prng_xoroshiro128_initialize(
    uint64_t seed, iree_prng_xoroshiro128_state_t* out_state) {
  // The authors recommend using SplitMix64 to go from a single int seed
  // into the two state values we need. It's critical that we don't use a
  // xoroshiro128 for this as seeding a PRNG with the results of itself is...
  // unsound.
  iree_prng_splitmix64_state_t init_state;
  iree_prng_splitmix64_initialize(seed, &init_state);
  out_state->value[0] = iree_prng_splitmix64_next(&seed);
  out_state->value[1] = iree_prng_splitmix64_next(&seed);

  // A state of 0 will never produce anything but zeros so ensure that doesn't
  // happen; of course, after running splitmix that should be closer to the
  // side of never than not.
  if (!out_state->value[0] && !out_state->value[1]) {
    out_state->value[0] = 1;
  }
}

// Steps a xoroshiro128 state vector and yields a value for use.
// xoroshiro128+ variant: produces a single value with 32-bit bits of entropy.
// This is the fastest variant but the lower 4 bits of the returned value may
// not be sufficiently well-distributed. This is fine if the usage requires
// fewer than 60 bits such as when sampling bools or array indices.
// Note also that this works great for floating-point numbers where only 23 or
// 53 bits are required to populate a mantissa and an additional step can be
// used to generate the sign/exponent when required.
//
//   footprint: 128-bits
//      period: 2^128 - 1
//  ns/64-bits: 0.72
// cycles/byte: 0.29
//
// http://prng.di.unimi.it/xoroshiro128plus.c
static inline uint64_t iree_prng_xoroshiro128plus_next_uint60(
    iree_prng_xoroshiro128_state_t* state) {
  uint64_t s0 = state->value[0];
  uint64_t s1 = state->value[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  state->value[0] = iree_math_rotl_u64(s0, 24) ^ s1 ^ (s1 << 16);  // a, b
  state->value[1] = iree_math_rotl_u64(s1, 37);                    // c
  return result;
}

// Steps a xoroshiro128 state vector and yields a single boolean value for use.
// See iree_prng_xoroshiro128plus_next_uint60 for details.
static inline bool iree_prng_xoroshiro128plus_next_bool(
    iree_prng_xoroshiro128_state_t* state) {
  return (bool)(iree_prng_xoroshiro128plus_next_uint60(state) >> (64 - 1));
}

// Steps a xoroshiro128 state vector and yields a single uint8_t value for use.
// See iree_prng_xoroshiro128plus_next_uint60 for details.
static inline uint8_t iree_prng_xoroshiro128plus_next_uint8(
    iree_prng_xoroshiro128_state_t* state) {
  return (uint8_t)(iree_prng_xoroshiro128plus_next_uint60(state) >> (64 - 8));
}

// Steps a xoroshiro128 state vector and yields a single uint32_t value for use.
// See iree_prng_xoroshiro128plus_next_uint60 for details.
static inline uint32_t iree_prng_xoroshiro128plus_next_uint32(
    iree_prng_xoroshiro128_state_t* state) {
  return (uint32_t)(iree_prng_xoroshiro128plus_next_uint60(state) >> (64 - 32));
}

// Steps a xoroshiro128 state vector and yields a value for use.
// xoroshiro128** variant: produces a single value with 32-bit bits of entropy.
// Prefer this to xoroshiro128+ when good distribution over the integer range
// is required; see xoroshiro128+ for details of its issues.
//
//   footprint: 128-bits
//      period: 2^128 - 1
//  ns/64-bits: 0.93
// cycles/byte: 0.42
//
// http://prng.di.unimi.it/xoroshiro128starstar.c
static inline uint64_t iree_prng_xoroshiro128starstar_next_uint64(
    iree_prng_xoroshiro128_state_t* state) {
  uint64_t s0 = state->value[0];
  uint64_t s1 = state->value[1];
  const uint64_t result = iree_math_rotl_u64(s0 * 5, 7) * 9;
  s1 ^= s0;
  state->value[0] = iree_math_rotl_u64(s0, 24) ^ s1 ^ (s1 << 16);  // a, b
  state->value[1] = iree_math_rotl_u64(s1, 37);                    // c
  return result;
}

// MiniLcg by @bjacob: A shot at the cheapest possible PRNG on ARM NEON
// https://gist.github.com/bjacob/7d635b91acd02559d73a6d159fe9cfbe
// I have no idea what the entropy characteristics of it are but it's really
// fast and in a lot of places that's all we need. For example, whatever number
// we generate when doing worker thread selection is going to get AND'ed with
// some other bitmasks by the caller -- and once you do that to a random number
// you've pretty much admitted it's ok to not be so strong and may as well
// capitalize on it!
typedef iree_alignas(iree_max_align_t) struct {
  uint8_t value[16];  // first to ensure alignment
  int8_t remaining;   // number of remaining valid values in the state
} iree_prng_minilcg128_state_t;

#define IREE_PRNG_MINILCG_INIT_MUL_CONSTANT 13
#define IREE_PRNG_MINILCG_INIT_ADD_CONSTANT 47
#define IREE_PRNG_MINILCG_NEXT_MUL_CONSTANT 37
#define IREE_PRNG_MINILCG_NEXT_ADD_CONSTANT 47

// Initializes a MiniLcg PRNG state vector; |out_state| is overwritten.
// |seed| may be any 8-bit value.
static inline void iree_prng_minilcg128_initialize(
    uint64_t seed, iree_prng_minilcg128_state_t* out_state) {
  uint8_t value = (seed ^ 11400714819323198485ull) & 0xFF;
  for (size_t i = 0; i < 16; ++i) {
    out_state->value[i] = value;
    value = value * IREE_PRNG_MINILCG_INIT_MUL_CONSTANT +
            IREE_PRNG_MINILCG_INIT_ADD_CONSTANT;
  }
  out_state->remaining = 16;
}

static inline uint8_t iree_prng_minilcg128_next_uint8(
    iree_prng_minilcg128_state_t* state) {
  if (IREE_UNLIKELY(--state->remaining < 0)) {
#if defined(IREE_ARCH_ARM_64)
    uint8x16_t kmul = vdupq_n_u8(IREE_PRNG_MINILCG_NEXT_MUL_CONSTANT);
    uint8x16_t kadd = vdupq_n_u8(IREE_PRNG_MINILCG_NEXT_ADD_CONSTANT);
    vst1q_u8(state->value, vmlaq_u8(kadd, kmul, vld1q_u8(state->value)));
#else
    for (size_t i = 0; i < 16; ++i) {
      state->value[i] = state->value[i] * IREE_PRNG_MINILCG_NEXT_MUL_CONSTANT +
                        IREE_PRNG_MINILCG_NEXT_ADD_CONSTANT;
    }
#endif  // IREE_ARCH_ARM_64
    state->remaining = 15;
  }
  return state->value[16 - state->remaining - 1];
}

#endif  // IREE_BASE_INTERNAL_PRNG_H_
