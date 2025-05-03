// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_MATH_H_
#define IREE_BASE_INTERNAL_MATH_H_

#include <math.h>
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
// Floating point types conversion support.
//==============================================================================

// NOTE: We used to have code here using built-in _Float16 type support.
// It worked well (https://godbolt.org/z/3a6WM39M1) until it didn't for
// some people (#14549). It's not worth the hassle, this is only used
// in slow generic fallbacks or test code, and we weren't able to use
// a builtin for bf16 anyway.

// Define some helper constants for working with a floating-point format with
// the given number of {exponent,mantissa} bits.
#define IREE_MATH_FP_FORMAT_CONSTANTS(prefix, ebits, mbits, bias_tweak)      \
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
      bias_tweak + (1u << (prefix##exp_bits - 1)) - 1;

// Generic conversion from any less-than-32-bit floating-point format to f32.
// The `src` value is typed as a uint32_t for genericity but occupies only the
// bottom (1 + exp_bits + mantissa_bits) bits. The upper bits of `src` are
// unused.
static inline float iree_math_make_f32_from_bits(uint32_t src, int exp_bits,
                                                 int mantissa_bits,
                                                 bool have_infinity,
                                                 bool have_nan, int bias_tweak,
                                                 bool nan_as_neg_zero) {
  IREE_MATH_FP_FORMAT_CONSTANTS(src_, exp_bits, mantissa_bits, bias_tweak)
  const float float_sign = (src & src_sign_mask) ? -1.f : 1.f;
  const uint32_t src_exp = src & src_exp_mask;
  const uint32_t src_mantissa = src & src_mantissa_mask;
  if (src_exp == src_exp_mask) {
    // Top exponent value normally means infinity or NaN.
    if (have_infinity) {
      // NaN or Inf case.
      if (have_nan && src_mantissa) {
        return NAN;
      } else {
        return float_sign * INFINITY;
      }
    } else if (have_nan) {
      // No infinities => more large finite values, unless this is a NaN.
      if (src_mantissa == src_mantissa_mask && !nan_as_neg_zero) {
        return NAN;
      }
    }
  } else if (have_nan && nan_as_neg_zero && src == src_sign_mask) {
    // Case of small FP types using the negative-0 encoding for NaN.
    return NAN;
  } else if (src_exp == 0) {
    // Denormals. In that case, the exponent is interpreted as 1 instead of the
    // encoded 0, and the result is proportional to src_mantissa instead of
    // having an implied leading 1.
    return float_sign *
           ldexpf(src_mantissa, 1 - src_exp_bias - src_mantissa_bits);
  }
  // Normal value.
  return float_sign *
         ldexpf(src_mantissa + (1 << src_mantissa_bits),
                (src_exp >> src_exp_shift) - src_exp_bias - src_mantissa_bits);
}

// Generic conversion from f32 to any less-than-32-bit floating-point format,
// rounding to nearest-even. The return value is typed as a uint32_t for
// genericity but occupies only the bottom (1 + exp_bits + mantissa_bits) bits.
// The upper bits of the return value are unused.
static inline uint32_t iree_math_truncate_f32_to_bits_rounding_to_nearest_even(
    float value, int exp_bits, int mantissa_bits, bool have_infinity,
    bool have_nan, int bias_tweak, bool nan_as_neg_zero) {
  IREE_MATH_FP_FORMAT_CONSTANTS(dst_, exp_bits, mantissa_bits, bias_tweak)
  IREE_MATH_FP_FORMAT_CONSTANTS(f32_, 8, 23, 0)
  uint32_t u32_value;
  memcpy(&u32_value, &value, sizeof value);
  const uint32_t f32_sign = u32_value & f32_sign_mask;
  uint32_t dst_sign = f32_sign >> (f32_sign_shift - dst_sign_shift);
  const uint32_t f32_exp = u32_value & f32_exp_mask;
  const uint32_t f32_mantissa = u32_value & f32_mantissa_mask;
  uint32_t dst_exp = 0;
  uint32_t dst_mantissa = 0;
  // Flags that we set when we determine that we need to generate a NaN / an Inf
  // deferring to handlers are the end of this function.
  bool convert_nan = false;
  bool convert_inf = false;
  if (f32_exp >= f32_exp_mask) {
    // NaN or Inf case.
    dst_exp = dst_exp_mask;
    if (f32_mantissa) {
      convert_nan = true;
    } else {
      convert_inf = true;
    }
  } else if (f32_exp == 0) {
    // Zero or subnormal.
    if (dst_exp_bits == f32_exp_bits) {
      // When the destination type still has as many exponent bits, denormals
      // can remain nonzero. This happens only with the bf16 type.
      // Just divide the mantissa (rounding shift).
      int shift_amount = f32_mantissa_bits - dst_mantissa_bits;
      uint32_t rounding_term = 1 << (shift_amount - 1);
      dst_mantissa = (f32_mantissa + rounding_term) >> shift_amount;
    }
    // The destination type has fewer exponent bits, so f32 subnormal values
    // become exactly zero. Leave the mantissa zero.
  } else {
    // Normal finite value.
    int arithmetic_exp = (f32_exp >> f32_exp_shift) - f32_exp_bias;
    // Test if the exponent is too large for the destination type. If
    // the destination type does not have infinities, that frees up the
    // max exponent value for additional finite values.
    if (arithmetic_exp > (1 << (dst_exp_bits - 1)) - have_infinity) {
      // Overflow.
      convert_inf = true;
    } else if (arithmetic_exp + dst_exp_bias <= 0) {
      // Underflow. Generate a subnormal or zero.
      dst_exp = 0;
      // Arithmetic exponent of destination type subnormals.
      int dst_arithmetic_exp = 1 - dst_exp_bias;
      // The exponent has to be clamped to 0 when the value
      // (arithmetic_exp + dst_exp_bias) is negative. This has to be compensated
      // by right-shifting the subnormal mantissa.
      int shift_amount = f32_mantissa_bits - dst_mantissa_bits -
                         arithmetic_exp + dst_arithmetic_exp;
      if (shift_amount < 0 || shift_amount > f32_mantissa_bits) {
        dst_mantissa = 0;
      } else {
        // Source f32 value is normal so has an implied 1... leading bit.
        int effective_f32_mantissa = (1 << f32_mantissa_bits) + f32_mantissa;
        // Add this term to achieve rounding to nearest instead of truncation
        // towards zero.
        int rounding_term = 1 << (shift_amount - 1);
        // Finally compute the destination mantissa as a rounded right shift.
        dst_mantissa = (effective_f32_mantissa + rounding_term) >> shift_amount;
      }
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
      // The exponent increment in the above if() branch may cause overflow.
      // This is exercised by converting 65520.0f from f32 to f16. When the
      // destination type has infinities, no special handling is needed for this
      // case: the above if() branch already set biased_f32_mantissa=0, so we
      // will be generating a 0 mantissa, as needed for infinite values. The one
      // case where special handling is needed is when the destination type has
      // no infinities and we need to generate NaN.
      dst_exp = (arithmetic_exp + dst_exp_bias) << dst_exp_shift;
      dst_mantissa =
          biased_f32_mantissa >> (f32_mantissa_bits - dst_mantissa_bits);
      if (!have_infinity && dst_exp > dst_exp_mask) {
        convert_nan = true;
      }
    }
  }

  // Handler for converting Inf values. Needs to be before handler for Nan as it
  // may fall through to either when the destination type does not have Inf.
  if (convert_inf) {
    if (have_infinity) {
      return dst_sign | dst_exp_mask;
    } else if (have_nan) {
      convert_nan = true;
    } else {
      // Generate the max finite value.
      // As we are here in the case where there is no NaN, the max finite value
      // is encoded with all mantissa bits set.
      return dst_sign | dst_exp_mask | dst_mantissa_mask;
    }
  }

  // Handler for converting NaN values.
  if (convert_nan) {
    if (!have_nan) {
      // When the destination type has no NaN encoding, conversion of NaN is
      // implementation-defined. We choose to convert NaN to +0.0.
      return 0;
    } else if (nan_as_neg_zero) {
      return dst_sign_mask;
    } else {
      return dst_sign | dst_exp_mask | dst_mantissa_mask;
    }
  }

  // Normal case.
  if (nan_as_neg_zero && dst_exp == 0 && dst_mantissa == 0) {
    // Negative zero needs to be rounded to positive zero to avoid
    // accidentally producing NaN when negative-zero is the NaN encoding.
    return 0;
  } else {
    return dst_sign | dst_exp | dst_mantissa;
  }
}

#define IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(                                   \
    NAME, INT_TYPE, EXP_BITS, MANTISSA_BITS, HAVE_INFINITY, HAVE_NAN,        \
    BIAS_TWEAK, NAN_AS_NEG_ZERO)                                             \
  /* Converts a to a 32-bit C `float`. */                                    \
  static inline float iree_math_##NAME##_to_f32(INT_TYPE src) {              \
    return iree_math_make_f32_from_bits(src, EXP_BITS, MANTISSA_BITS,        \
                                        HAVE_INFINITY, HAVE_NAN, BIAS_TWEAK, \
                                        NAN_AS_NEG_ZERO);                    \
  }                                                                          \
  /* Truncates a 32-bit C `float`, rounding to nearest even. */              \
  static inline INT_TYPE iree_math_f32_to_##NAME(float value) {              \
    return iree_math_truncate_f32_to_bits_rounding_to_nearest_even(          \
        value, EXP_BITS, MANTISSA_BITS, HAVE_INFINITY, HAVE_NAN, BIAS_TWEAK, \
        NAN_AS_NEG_ZERO);                                                    \
  }                                                                          \
  /* Round-trip f32->f32 rounding via the narrow float type */               \
  static inline float iree_math_round_to_nearest_##NAME(float value) {       \
    return iree_math_##NAME##_to_f32(iree_math_f32_to_##NAME(value));        \
  }

// IEEE half-precision a.k.a. float16,
// https://en.wikipedia.org/wiki/Half-precision_floating-point_format
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f16, uint16_t, 5, 10, /*have_infinity=*/true,
                                  /*have_nan=*/true,
                                  /*bias_tweak=*/0, /*nan_as_neg_zero=*/false)

// Bfloat16, https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(bf16, uint16_t, 8, 7, /*have_infinity=*/true,
                                  /*have_nan=*/true,
                                  /*bias_tweak=*/0, /*nan_as_neg_zero=*/false)

// F8E5M2 type, https://arxiv.org/abs/2209.05433
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f8e5m2, uint8_t, 5, 2, /*have_infinity=*/true,
                                  /*have_nan=*/true,
                                  /*bias_tweak=*/0, /*nan_as_neg_zero=*/false)

// F8E4M3FN type, https://arxiv.org/abs/2209.05433. The paper doesn't use the FN
// suffix, but APFloat and MLIR do to indicate that the float is Finite and has
// one NaN (or maybe just that it's FiNite, can't recall).
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f8e4m3fn, uint8_t, 4, 3,
                                  /*have_infinity=*/false, /*have_nan=*/true,
                                  /*bias_tweak=*/0,
                                  /*nan_as_neg_zero=*/false)

// F8E5M2FNUZ type, found in some AMD GPUs (MI300), called "BF8" there.
// Quoting LLVM's APFloat.h:
//   8-bit floating point number mostly following IEEE-754 conventions
//   and bit layout S1E5M2 described in https://arxiv.org/abs/2206.02915,
//   with expanded range and with no infinity or signed zero.
//   NaN is represented as negative zero. (FN -> Finite, UZ -> unsigned zero).
//   This format's exponent bias is 16, instead of the 15 (2 ** (5 - 1) - 1)
//   that IEEE precedent would imply.
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f8e5m2fnuz, uint8_t, 5, 2,
                                  /*have_infinity=*/false, /*have_nan=*/true,
                                  /*bias_tweak=*/1,
                                  /*nan_as_neg_zero=*/true)

// F8E4M3FNUZ type, found in some AMD GPUs (MI300), called "FP8" there.
//   Quoting LLVM's APFloat.h:
//   8-bit floating point number mostly following IEEE-754 conventions
//   and bit layout S1E4M3 described in https://arxiv.org/abs/2206.02915,
//   with expanded range and with no infinity or signed zero.
//   NaN is represented as negative zero. (FN -> Finite, UZ -> unsigned zero).
//   This format's exponent bias is 8, instead of the 7 (2 ** (4 - 1) - 1)
//   that IEEE precedent would imply.
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f8e4m3fnuz, uint8_t, 4, 3,
                                  /*have_infinity=*/false, /*have_nan=*/true,
                                  /*bias_tweak=*/1,
                                  /*nan_as_neg_zero=*/true)

// F6E3M2FN type. Quoting LLVM's APFloat.h:
//   6-bit floating point number with bit layout S1E3M2. Unlike IEEE-754
//   types, there are no infinity or NaN values. The format is detailed in
//   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f6e3m2fn, uint8_t, 3, 2,
                                  /*have_infinity=*/false, /*have_nan=*/false,
                                  /*bias_tweak=*/0,
                                  /*nan_as_neg_zero=*/false)

// F6E2M3FN type. Quoting LLVM's APFloat.h:
//   6-bit floating point number with bit layout S1E2M3. Unlike IEEE-754
//   types, there are no infinity or NaN values. The format is detailed in
//   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f6e2m3fn, uint8_t, 2, 3,
                                  /*have_infinity=*/false, /*have_nan=*/false,
                                  /*bias_tweak=*/0,
                                  /*nan_as_neg_zero=*/false)

// F4E2M1FN type. Quoting LLVM's APFloat.h:
//   4-bit floating point number with bit layout S1E2M1. Unlike IEEE-754
//   types, there are no infinity or NaN values. The format is detailed in
//   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
IREE_MATH_MAKE_FLOAT_TYPE_HELPERS(f4e2m1fn, uint8_t, 2, 1,
                                  /*have_infinity=*/false, /*have_nan=*/false,
                                  /*bias_tweak=*/0,
                                  /*nan_as_neg_zero=*/false)

// The scale type E8M0FNU is unique in multiple ways: no mantissa, no sign, and
// no zero. Retrofitting it into the above shared conversion code would be
// tricky and not worth it, so here are stand-alone conversion routines:
static inline float iree_math_f8e8m0fnu_to_f32(uint8_t src) {
  if (src == 0xFF) {
    return NAN;
  } else {
    return ldexpf(1.0f, src - 127);
  }
}
static inline uint8_t iree_math_f32_to_f8e8m0fnu(float value) {
  if (!isfinite(value)) {
    return 0xFF;
  }
  if (value <= 0.f) {
    return 0;
  }
  int exp = 0;
  // Normalized is in the interval [0.5, 1.0).
  float normalized = frexpf(value, &exp);
  // If the normalized value is closer to 0.5 than to 1.0, decrement the
  // exponent.
  int rounded = exp - (normalized < 0.75f);
  int biased = rounded + 127;
  // The clamping below is to 0xFF, mapping to NaN any value that is larger than
  // the max finite value.
  return biased < 0 ? 0 : biased > 0xFF ? 0xFF : biased;
}

#endif  // IREE_BASE_INTERNAL_MATH_H_
