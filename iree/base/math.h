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

#ifndef IREE_BASE_MATH_H_
#define IREE_BASE_MATH_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/target_platform.h"

// Haswell or later, gcc compile time option: -mlzcnt
#if defined(__LZCNT__)
#include <x86intrin.h>
#endif

// Clang on Windows has __builtin_clzll; otherwise we need to use the
// windows intrinsic functions.
#if defined(IREE_COMPILER_MSVC)
#include <intrin.h>
#if defined(IREE_ARCH_ARM_64) || defined(IREE_ARCH_X86_64)
#pragma intrinsic(_BitScanReverse64)
#pragma intrinsic(_BitScanForward64)
#endif
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#endif  // IREE_COMPILER_MSVC

//==============================================================================
// Bitwise rotation (aka circular shifts)
//==============================================================================

// Unsigned rotate-left a 64-bit integer.
// https://en.cppreference.com/w/cpp/numeric/rotl
//
//
// NOTE: this exact form is confirmed to be recognized by the compilers we care
// about; do not modify: https://godbolt.org/z/xzof9d
static inline uint64_t iree_math_rotl_u64(const uint64_t n, uint32_t c) {
  if (!c) return n;
  const uint32_t mask = 8 * sizeof(n) - 1;
  c &= mask;
  return (n << c) | (n >> (64 - c));
}

// Unsigned rotate-right a 64-bit integer.
// https://en.cppreference.com/w/cpp/numeric/rotr
//
// NOTE: this exact form is confirmed to be recognized by the compilers we care
// about **except MSVC**; do not modify: https://godbolt.org/z/xzof9d
static inline uint64_t iree_math_rotr_u64(const uint64_t n, uint32_t c) {
  if (!c) return n;
  const uint32_t mask = 8 * sizeof(n) - 1;
  c &= mask;
  return (n >> c) | (n << ((-c) & mask));
}

//==============================================================================
// Bit scanning/counting
//==============================================================================

static inline int iree_math_count_leading_zeros_u32(const uint32_t n) {
#if defined(IREE_COMPILER_MSVC)
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse(&result, n)) {
    return (int)(31 - result);
  }
  return 32;
#elif defined(IREE_COMPILER_GCC_COMPAT)
#if defined(__LCZNT__)
  // NOTE: LZCNT is a risky instruction; it is not supported on architectures
  // before Haswell, yet it is encoded as 'rep bsr', which typically ignores
  // invalid rep prefixes, and interprets it as the 'bsr' instruction, which
  // returns the index of the value rather than the count, resulting in
  // incorrect code.
  return (int)__lzcnt32(n);
#endif  // defined(__LCZNT__)

  // Handle 0 as a special case because __builtin_clz(0) is undefined.
  if (n == 0) return 32;
  // Use __builtin_clz, which uses the following instructions:
  //  x86: bsr
  //  ARM64: clz
  //  PPC: cntlzd
  return (int)__builtin_clz(n);
#else
#error No clz for this arch.
#endif  // IREE_COMPILER_MSVC / IREE_COMPILER_GCC_COMPAT
}

static inline int iree_math_count_leading_zeros_u64(uint64_t n) {
#if defined(IREE_COMPILER_MSVC) && \
    (defined(IREE_ARCH_ARM_64) || defined(IREE_ARCH_X86_64))
  // MSVC does not have __buitin_clzll. Use _BitScanReverse64.
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse64(&result, n)) {
    return (int)(63 - result);
  }
  return 64;
#elif defined(IREE_COMPILER_MSVC)
  // MSVC does not have __buitin_clzll. Compose two calls to _BitScanReverse
  unsigned long result = 0;  // NOLINT(runtime/int)
  if ((n >> 32) && _BitScanReverse(&result, n >> 32)) {
    return (int)(31 - result);
  }
  if (_BitScanReverse(&result, n)) {
    return (int)(63 - result);
  }
  return 64;
#elif defined(IREE_COMPILER_GCC_COMPAT)
#if defined(__LCZNT__)
  // NOTE: LZCNT is a risky instruction; it is not supported on architectures
  // before Haswell, yet it is encoded as 'rep bsr', which typically ignores
  // invalid rep prefixes, and interprets it as the 'bsr' instruction, which
  // returns the index of the value rather than the count, resulting in
  // incorrect code.
  return __lzcnt64(n);
#elif defined(__aarch64__) || defined(__powerpc64__)
  // Empirically verified that __builtin_clzll(0) works as expected.
  return (int)__builtin_clzll(n);
#endif
  // Handle 0 as a special case because __builtin_clzll(0) is undefined.
  if (!n) return 64;
  // Use __builtin_clzll, which uses the following instructions:
  //    x86: bsr
  //    PPC: cntlzd
  //   WASM: i32.clz
  // RISC-V: __clzsi2 in GCC, splat out in clang
  return (int)__builtin_clzll(n);
#else
#error No clz for this arch.
#endif  // IREE_COMPILER_MSVC / IREE_COMPILER_GCC_COMPAT
}

static inline int iree_math_count_trailing_zeros_u32(uint32_t n) {
#if defined(IREE_COMPILER_MSVC)
  unsigned long result = 0;  // NOLINT(runtime/int)
  _BitScanForward(&result, n);
  return (int)result;
#elif defined(IREE_COMPILER_GCC_COMPAT)
  return (int)__builtin_ctz(n);
#else
  int c = 31;
  n &= ~n + 1;
  if (n & 0x0000FFFFu) c -= 16;
  if (n & 0x00FF00FFu) c -= 8;
  if (n & 0x0F0F0F0Fu) c -= 4;
  if (n & 0x33333333u) c -= 2;
  if (n & 0x55555555u) c -= 1;
  return c;
#endif  // IREE_COMPILER_MSVC / IREE_COMPILER_GCC_COMPAT
}

static inline int iree_math_count_trailing_zeros_u64(uint64_t n) {
#if defined(IREE_COMPILER_MSVC) && defined(IREE_PTR_SIZE_64)
  unsigned long result = 0;  // NOLINT(runtime/int)
  _BitScanForward64(&result, n);
  return (int)result;
#elif defined(IREE_COMPILER_MSVC) && defined(IREE_PTR_SIZE_32)
  unsigned long result = 0;  // NOLINT(runtime/int)
  if ((uint32_t)(n) == 0) {
    _BitScanForward(&result, n >> 32);
    return result + 32;
  }
  _BitScanForward(&result, n);
  return (int)result;
#elif defined(IREE_COMPILER_GCC_COMPAT)
  // Use __builtin_clzll, which uses the following instructions:
  //    x86: bsr
  //    PPC: cntlzd
  //   WASM: i64.clz
  // RISC-V: __clzdi2 in GCC, splat out in clang
  return __builtin_ctzll(n);
#else
  int c = 63;
  n &= ~n + 1;
  if (n & 0x00000000FFFFFFFFull) c -= 32;
  if (n & 0x0000FFFF0000FFFFull) c -= 16;
  if (n & 0x00FF00FF00FF00FFull) c -= 8;
  if (n & 0x0F0F0F0F0F0F0F0Full) c -= 4;
  if (n & 0x3333333333333333ull) c -= 2;
  if (n & 0x5555555555555555ull) c -= 1;
  return c;
#endif  // IREE_COMPILER_MSVC / IREE_COMPILER_GCC_COMPAT
}

//==============================================================================
// Population count
//==============================================================================

// Returns the number of 1 bits in a 32 bit value.
static inline int iree_math_count_ones_u32(uint32_t n) {
  n -= ((n >> 1) & 0x55555555u);
  n = ((n >> 2) & 0x33333333u) + (n & 0x33333333u);
  return (int)((((n + (n >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
}

// Returns the number of 1 bits in a 64 bit value.
static inline int iree_math_count_ones_u64(uint64_t n) {
  return iree_math_count_ones_u32(n >> 32) +
         iree_math_count_ones_u32(n & 0xFFFFFFFFu);
}

//==============================================================================
// Rounding and alignment
//==============================================================================
// There are certain platforms - mostly those with poorer quality compilers or
// more restricted instruction sets - where we want to avoid the clz path as
// it is emulated and instead we use some bit-twiddling hacks. On other
// platforms it's the opposite - they may emulate clz but doing so saves
// dozens of bytes that otherwise would have been the shift/or tree.
//
// Which to choose is entirely determined by fiddling on godbolt for the
// target platform: https://godbolt.org/z/h4vPzo

// Rounds up the value to the nearest power of 2 (if not already a power of 2).
// For 32-bit numbers this only supports values <= 2^31; higher will wrap.
static inline uint32_t iree_math_round_up_to_pow2_u32(uint32_t n) {
#if 0    // golf required; can be bloated
  const uint32_t i = (n != 1);
  return (1 + i) << ((iree_math_count_leading_zeros_u32(n - i) ^ 31));
#elif 0  // golf required; can be bloated
  return n == 1 ? 1u : 2u << ((iree_math_count_leading_zeros_u32(n - 1) ^ 31));
#else
  // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
#endif  // 1
}

// Rounds up the value to the nearest power of 2 (if not already a power of 2).
// For 64-bit numbers this only supports values <= 2^63; higher will wrap.
static inline uint64_t iree_math_round_up_to_pow2_u64(uint64_t n) {
#if 0    // golf required; can be bloated
  const uint64_t i = (n != 1);
  return (1 + i) << ((iree_math_count_leading_zeros_u64(n - i) ^ 63));
#elif 0  // golf required; can be bloated
  return n == 1 ? 1ull
                : 2ull << ((iree_math_count_leading_zeros_u64(n - 1) ^ 63));
#else
  // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
#endif  // 1
}

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

#endif  // IREE_BASE_MATH_H_
