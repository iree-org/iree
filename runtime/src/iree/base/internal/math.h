// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_MATH_H_
#define IREE_BASE_INTERNAL_MATH_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"

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
  const uint32_t mask = 8 * sizeof(n) - 1;
  c &= mask;
  if (!c) return n;
  return (n << c) | (n >> (64 - c));
}

// Unsigned rotate-right a 64-bit integer.
// https://en.cppreference.com/w/cpp/numeric/rotr
//
// NOTE: this exact form is confirmed to be recognized by the compilers we care
// about **except MSVC**; do not modify: https://godbolt.org/z/xzof9d
static inline uint64_t iree_math_rotr_u64(const uint64_t n, uint32_t c) {
  const uint32_t mask = 8 * sizeof(n) - 1;
  c &= mask;
  if (!c) return n;
  return (n >> c) | (n << ((-c) & mask));
}

//==============================================================================
// Bit scanning/counting
//==============================================================================

static inline int iree_math_count_leading_zeros_u32(const uint32_t n) {
#if defined(IREE_COMPILER_MSVC_COMPAT)
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
#if defined(IREE_COMPILER_MSVC_COMPAT) && \
    (defined(IREE_ARCH_ARM_64) || defined(IREE_ARCH_X86_64))
  // MSVC does not have __buitin_clzll. Use _BitScanReverse64.
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse64(&result, n)) {
    return (int)(63 - result);
  }
  return 64;
#elif defined(IREE_COMPILER_MSVC_COMPAT)
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
#if defined(IREE_COMPILER_MSVC_COMPAT)
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
#if defined(IREE_COMPILER_MSVC_COMPAT) && defined(IREE_PTR_SIZE_64)
  unsigned long result = 0;  // NOLINT(runtime/int)
  _BitScanForward64(&result, n);
  return (int)result;
#elif defined(IREE_COMPILER_MSVC_COMPAT) && defined(IREE_PTR_SIZE_32)
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
// FP16, BFloat16 and FP8 support
//==============================================================================

// NOTE: We used to have code here using built-in _Float16 type support.
// It worked well (https://godbolt.org/z/3a6WM39M1) until it didn't for
// some people (#14549). It's not worth the hassle, this is only used
// in slow generic fallbacks or test code, and we weren't able to use
// a builtin for bf16 anyway.

// Define some helper constants for working with a floating-point format with
// the given number of {exponent,mantissa} bits.
#define IREE_MATH_FP_FORMAT_CONSTANTS(prefix, ebits, mbits)                  \
  const int prefix##exp_bits IREE_ATTRIBUTE_UNUSED = ebits;                  \
  const int prefix##mantissa_bits IREE_ATTRIBUTE_UNUSED = mbits;             \
  const int prefix##sign_shift IREE_ATTRIBUTE_UNUSED = ebits + mbits;        \
  const int prefix##exp_shift IREE_ATTRIBUTE_UNUSED = prefix##mantissa_bits; \
  const int prefix##sign_mask IREE_ATTRIBUTE_UNUSED = 1u                     \
                                                      << prefix##sign_shift; \
  const int prefix##mantissa_mask IREE_ATTRIBUTE_UNUSED =                    \
      (1u << prefix##exp_shift) - 1;                                         \
  const int prefix##exp_mask IREE_ATTRIBUTE_UNUSED =                         \
      (1u << prefix##sign_shift) - (1u << prefix##exp_shift);                \
  const int prefix##exp_bias IREE_ATTRIBUTE_UNUSED =                         \
      (1u << (prefix##exp_bits - 1)) - 1;

// Generic conversion from any less-than-32-bit floating-point format to f32.
// The `src` value is typed as a uint32_t for genericity but occupies only the
// bottom (1 + exp_bits + mantissa_bits) bits. The upper bits of `src` are
// unused.
static inline float iree_math_make_f32_from_bits(uint32_t src, int exp_bits,
                                                 int mantissa_bits,
                                                 bool have_infinity) {
  IREE_MATH_FP_FORMAT_CONSTANTS(src_, exp_bits, mantissa_bits)
  IREE_MATH_FP_FORMAT_CONSTANTS(f32_, 8, 23)
  const uint32_t src_sign = src & src_sign_mask;
  const uint32_t f32_sign = src_sign << (f32_sign_shift - src_sign_shift);
  const uint32_t src_exp = src & src_exp_mask;
  const uint32_t src_mantissa = src & src_mantissa_mask;
  uint32_t f32_exp = 0;
  uint32_t f32_mantissa = 0;
  if (src_exp == src_exp_mask) {
    // No infinities => more large finite values.
    if (!have_infinity && src_mantissa != src_mantissa_mask) {
      float sign = (src & src_sign_mask) ? -1.0f : 1.0f;
      return sign * 2 * (1u << src_exp_bits) *
             ((1u << src_mantissa_bits) + src_mantissa);
    }
    // NaN or Inf case.
    f32_exp = f32_exp_mask;
    if (src_mantissa) {
      // NaN. Generate a quiet NaN.
      f32_mantissa = f32_mantissa_mask;
    } else {
      // Inf. Leave zero mantissa.
    }
  } else if (src_exp == 0) {
    // Zero or subnormal. Generate zero. Leave zero mantissa.
  } else {
    // Normal finite value.
    int arithmetic_src_exp = src_exp >> src_exp_shift;
    int arithmetic_f32_exp = arithmetic_src_exp + (1 << (f32_exp_bits - 1)) -
                             (1 << (src_exp_bits - 1));
    f32_exp = arithmetic_f32_exp << f32_exp_shift;
    f32_mantissa = src_mantissa << (f32_mantissa_bits - src_mantissa_bits);
  }
  const uint32_t u32_value = f32_sign | f32_exp | f32_mantissa;
  float f32_value;
  memcpy(&f32_value, &u32_value, sizeof f32_value);
  return f32_value;
}

// Generic conversion from f32 to any less-than-32-bit floating-point format,
// rounding to nearest-even. The return value is typed as a uint32_t for
// genericity but occupies only the bottom (1 + exp_bits + mantissa_bits) bits.
// The upper bits of the return value are unused.
static inline uint32_t iree_math_truncate_f32_to_bits_rounding_to_nearest_even(
    float value, int exp_bits, int mantissa_bits, bool have_infinity) {
  IREE_MATH_FP_FORMAT_CONSTANTS(dst_, exp_bits, mantissa_bits)
  IREE_MATH_FP_FORMAT_CONSTANTS(f32_, 8, 23)
  uint32_t u32_value;
  memcpy(&u32_value, &value, sizeof value);
  const uint32_t f32_sign = u32_value & f32_sign_mask;
  const uint32_t dst_sign = f32_sign >> (f32_sign_shift - dst_sign_shift);
  const uint32_t f32_exp = u32_value & f32_exp_mask;
  const uint32_t f32_mantissa = u32_value & f32_mantissa_mask;
  uint32_t dst_exp = 0;
  uint32_t dst_mantissa = 0;
  if (f32_exp >= f32_exp_mask) {
    // NaN or Inf case.
    dst_exp = dst_exp_mask;
    if (f32_mantissa || !have_infinity) {
      // NaN. Generate a quiet NaN.
      dst_mantissa = dst_mantissa_mask;
    } else {
      // Inf. Leave zero mantissa.
    }
  } else if (f32_exp == 0) {
    // Zero or subnormal. Generate zero. Leave zero mantissa.
  } else {
    // Normal finite value.
    int arithmetic_exp = (f32_exp >> f32_exp_shift) - f32_exp_bias;
    // Test if the exponent is too large for the destination type. If
    // the destination type does not have infinities, that frees up the
    // max exponent value for additional finite values.
    if (arithmetic_exp > (1 << (dst_exp_bits - 1)) - have_infinity) {
      // Overflow. Generate Inf. Leave zero mantissa.
      dst_exp = dst_exp_mask;
      if (!have_infinity) {
        // Generate NaN.
        dst_mantissa = dst_mantissa_mask;
      }
    } else if (arithmetic_exp < -(1 << (dst_exp_bits - 1))) {
      // Underflow. Generate zero. Leave zero mantissa.
      dst_exp = 0;
    } else {
      // Normal case.
      // Implement round-to-nearest-even, by adding a bias before truncating.
      int even_bit = 1u << (f32_mantissa_bits - dst_mantissa_bits);
      int odd_bit = even_bit >> 1;
      uint32_t biased_f32_mantissa =
          f32_mantissa +
          ((f32_mantissa & even_bit) ? (odd_bit) : (odd_bit - 1));
      // Adding the bias may cause an exponent increment.
      if (biased_f32_mantissa > f32_mantissa_mask) {
        // Note: software implementations that try to be fast tend to get this
        // conditional increment of exp and zeroing of mantissa for free by
        // simplying incrementing the whole uint32 encoding of the float value,
        // so that the mantissa overflows into the exponent bits.
        // This results in magical-looking code like in the following links.
        // We'd rather not care too much about performance of this function;
        // we should only care about fp16 performance on fp16 hardware, and
        // then, we should use hardware instructions.
        // https://github.com/pytorch/pytorch/blob/e1502c0cdbfd17548c612f25d5a65b1e4b86224d/c10/util/BFloat16.h#L76
        // https://gitlab.com/libeigen/eigen/-/blob/21cd3fe20990a5ac1d683806f605110962aac3f1/Eigen/src/Core/arch/Default/BFloat16.h#L565
        biased_f32_mantissa = 0;
        ++arithmetic_exp;
      }
      // In the !have_infinity case, arithmetic_exp might have been the top
      // value already, so incrementing it may have overflown it.
      if (!have_infinity && arithmetic_exp > (1 << (dst_exp_bits - 1))) {
        dst_exp = dst_exp_mask;
        dst_mantissa = dst_mantissa_mask;
      } else {
        // The exponent increment in the above if() branch may cause overflow.
        // This is exercised by converting 65520.0f from f32 to f16. No special
        // handling is needed for this case: the above if() branch already set
        // biased_f32_mantissa=0, so we will be generating a 0 mantissa, as
        // needed for infinite values.
        dst_exp = (arithmetic_exp + dst_exp_bias) << dst_exp_shift;
        dst_mantissa =
            biased_f32_mantissa >> (f32_mantissa_bits - dst_mantissa_bits);
      }
    }
  }
  uint32_t dst_value = dst_sign | dst_exp | dst_mantissa;
  return dst_value;
}

#define IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(NAME, INT_TYPE, EXP_BITS,     \
                                          MANTISSA_BITS, HAVE_INFINITY) \
  /* Converts a to a 32-bit C `float`. */                               \
  static inline float iree_math_##NAME##_to_f32(INT_TYPE src) {         \
    return iree_math_make_f32_from_bits(src, EXP_BITS, MANTISSA_BITS,   \
                                        HAVE_INFINITY);                 \
  }                                                                     \
  /* Truncates a 32-bit C `float`, rounding to nearest even. */         \
  static inline INT_TYPE iree_math_f32_to_##NAME(float value) {         \
    return iree_math_truncate_f32_to_bits_rounding_to_nearest_even(     \
        value, EXP_BITS, MANTISSA_BITS, HAVE_INFINITY);                 \
  }                                                                     \
  /* Round-trip f32->f32 rounding via the narrow float type */          \
  static inline float iree_math_round_to_nearest_##NAME(float value) {  \
    return iree_math_##NAME##_to_f32(iree_math_f32_to_##NAME(value));   \
  }

// IEEE half-precision a.k.a. float16,
// https://en.wikipedia.org/wiki/Half-precision_floating-point_format
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f16, uint16_t, 5, 10, /*have_infinity=*/true)

// Bfloat16, https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(bf16, uint16_t, 8, 7, /*have_infinity=*/true)

// F8E5M2 type, https://arxiv.org/abs/2209.05433
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f8e5m2, uint8_t, 5, 2, /*have_infinity=*/true)

// F8E4M3 type, https://arxiv.org/abs/2209.05433.
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f8e4m3, uint8_t, 4, 3,
                                  /*have_infinity=*/false)

#endif  // IREE_BASE_INTERNAL_MATH_H_
