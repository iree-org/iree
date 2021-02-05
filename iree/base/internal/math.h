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

#ifndef IREE_BASE_INTERNAL_MATH_H_
#define IREE_BASE_INTERNAL_MATH_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/alignment.h"
#include "iree/base/api.h"
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

#define iree_shr(value, shamt) \
  (((shamt) < sizeof(value) * 8) ? ((value) >> (shamt)) : 0)

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

#endif  // IREE_BASE_INTERNAL_MATH_H_
