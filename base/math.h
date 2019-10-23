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

#include <cstdint>

#include "absl/base/attributes.h"

// Haswell or later, gcc compile time option: -mlzcnt
#if defined(__LZCNT__)
#include <x86intrin.h>
#endif

// Clang on Windows has __builtin_clzll; otherwise we need to use the
// windows intrinsic functions.
#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_X64)
#pragma intrinsic(_BitScanReverse64)
#pragma intrinsic(_BitScanForward64)
#endif
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#endif

#if defined(_MSC_VER)
// We can achieve something similar to attribute((always_inline)) with MSVC by
// using the __forceinline keyword, however this is not perfect. MSVC is
// much less aggressive about inlining, and even with the __forceinline keyword.
#define IREE_FORCEINLINE __forceinline
#else
// Use default attribute inline.
#define IREE_FORCEINLINE inline ABSL_ATTRIBUTE_ALWAYS_INLINE
#endif

namespace iree {

IREE_FORCEINLINE int CountLeadingZeros32(uint32_t n) {
#if defined(_MSC_VER)
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse(&result, n)) {
    return 31 - result;
  }
  return 32;
#elif defined(__GNUC__)
  // Use __builtin_clz, which uses the following instructions:
  //  x86: bsr
  //  ARM64: clz
  //  PPC: cntlzd
  static_assert(sizeof(int) == sizeof(n),
                "__builtin_clz does not take 32-bit arg");

#if defined(__LCZNT__)
  // NOTE: LZCNT is a risky instruction; it is not supported on architectures
  // before Haswell, yet it is encoded as 'rep bsr', which typically ignores
  // invalid rep prefixes, and interprets it as the 'bsr' instruction, which
  // returns the index of the value rather than the count, resulting in
  // incorrect code.
  return __lzcnt32(n);
#endif  // defined(__LCZNT__)

  // Handle 0 as a special case because __builtin_clz(0) is undefined.
  if (n == 0) {
    return 32;
  }
  return __builtin_clz(n);
#else
#error No clz for this arch.
#endif
}

IREE_FORCEINLINE int CountLeadingZeros64(uint64_t n) {
#if defined(_MSC_VER) && defined(_M_X64)
  // MSVC does not have __buitin_clzll. Use _BitScanReverse64.
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse64(&result, n)) {
    return 63 - result;
  }
  return 64;
#elif defined(_MSC_VER)
  // MSVC does not have __buitin_clzll. Compose two calls to _BitScanReverse
  unsigned long result = 0;  // NOLINT(runtime/int)
  if ((n >> 32) && _BitScanReverse(&result, n >> 32)) {
    return 31 - result;
  }
  if (_BitScanReverse(&result, n)) {
    return 63 - result;
  }
  return 64;
#elif defined(__GNUC__)
  // Use __builtin_clzll, which uses the following instructions:
  //  x86: bsr
  //  ARM64: clz
  //  PPC: cntlzd
  static_assert(sizeof(unsigned long long) == sizeof(n),  // NOLINT(runtime/int)
                "__builtin_clzll does not take 64-bit arg");

#if defined(__LCZNT__)
  // NOTE: LZCNT is a risky instruction; it is not supported on architectures
  // before Haswell, yet it is encoded as 'rep bsr', which typically ignores
  // invalid rep prefixes, and interprets it as the 'bsr' instruction, which
  // returns the index of the value rather than the count, resulting in
  // incorrect code.
  return __lzcnt64(n);
#elif defined(__aarch64__) || defined(__powerpc64__)
  // Empirically verified that __builtin_clzll(0) works as expected.
  return __builtin_clzll(n);
#endif

  // Handle 0 as a special case because __builtin_clzll(0) is undefined.
  if (n == 0) {
    return 64;
  }
  return __builtin_clzll(n);
#else
#error No clz for this arch.
#endif
}

IREE_FORCEINLINE int CountTrailingZerosNonZero32(uint32_t n) {
#if defined(_MSC_VER)
  unsigned long result = 0;  // NOLINT(runtime/int)
  _BitScanForward(&result, n);
  return result;
#elif defined(__GNUC__)
  static_assert(sizeof(int) == sizeof(n),
                "__builtin_ctz does not take 32-bit arg");
  return __builtin_ctz(n);
#else
  int c = 31;
  n &= ~n + 1;
  if (n & 0x0000FFFF) c -= 16;
  if (n & 0x00FF00FF) c -= 8;
  if (n & 0x0F0F0F0F) c -= 4;
  if (n & 0x33333333) c -= 2;
  if (n & 0x55555555) c -= 1;
  return c;
#endif
}

IREE_FORCEINLINE int CountTrailingZerosNonZero64(uint64_t n) {
#if defined(_MSC_VER) && defined(_M_X64)
  unsigned long result = 0;  // NOLINT(runtime/int)
  _BitScanForward64(&result, n);
  return result;
#elif defined(_MSC_VER)
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (static_cast<uint32_t>(n) == 0) {
    _BitScanForward(&result, n >> 32);
    return result + 32;
  }
  _BitScanForward(&result, n);
  return result;
#elif defined(__GNUC__)
  static_assert(sizeof(unsigned long long) == sizeof(n),  // NOLINT(runtime/int)
                "__builtin_ctzll does not take 64-bit arg");
  return __builtin_ctzll(n);
#else
  int c = 63;
  n &= ~n + 1;
  if (n & 0x00000000FFFFFFFF) c -= 32;
  if (n & 0x0000FFFF0000FFFF) c -= 16;
  if (n & 0x00FF00FF00FF00FF) c -= 8;
  if (n & 0x0F0F0F0F0F0F0F0F) c -= 4;
  if (n & 0x3333333333333333) c -= 2;
  if (n & 0x5555555555555555) c -= 1;
  return c;
#endif
}

template <typename T>
IREE_FORCEINLINE int TrailingZeros(T x) {
  return sizeof(T) == 8 ? CountTrailingZerosNonZero64(static_cast<uint64_t>(x))
                        : CountTrailingZerosNonZero32(static_cast<uint32_t>(x));
}

template <typename T>
IREE_FORCEINLINE int LeadingZeros(T x) {
  return sizeof(T) == 8 ? CountLeadingZeros64(static_cast<uint64_t>(x))
                        : CountLeadingZeros32(static_cast<uint32_t>(x));
}

// Rounds up the value to the nearest power of 2 (if not already a power of 2).
IREE_FORCEINLINE int RoundUpToNearestPow2(int n) {
  return n ? ~0u >> LeadingZeros(n) : 1;
}

}  // namespace iree

#endif  // IREE_BASE_MATH_H_
