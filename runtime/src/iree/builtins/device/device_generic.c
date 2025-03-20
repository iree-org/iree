// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "device.h"

#if !defined(IREE_DEVICE_STANDALONE)
int libdevice_platform_example_flag = LIBDEVICE_PLATFORM_EXAMPLE_FLAG;
#endif  // IREE_DEVICE_STANDALONE

IREE_DEVICE_EXPORT float iree_h2f_ieee(short param) {
  unsigned short expHalf16 = param & 0x7C00;
  int exp1 = (int)expHalf16;
  unsigned short mantissa16 = param & 0x03FF;
  int mantissa1 = (int)mantissa16;
  int sign = (int)(param & 0x8000);
  sign = sign << 16;

  // nan or inf
  if (expHalf16 == 0x7C00) {
    // nan
    if (mantissa16 > 0) {
      union {
        int i;
        float f;
      } res = {
          .i = 0x7FC00000 | sign,
      };
      return res.f;
    }
    // inf
    union {
      int i;
      float f;
    } res = {
        .i = 0x7F800000 | sign,
    };
    return res.f;
  }
  if (expHalf16 != 0) {
    exp1 += ((127 - 15) << 10);  // exponents converted to float32 bias
    union {
      int i;
      float f;
    } res = {
        .i = ((exp1 | mantissa1) << 13) | sign,
    };
    return res.f;
  }

  int xmm1 = exp1 > (1 << 10) ? exp1 : (1 << 10);
  xmm1 = (xmm1 << 13);
  xmm1 += ((127 - 15 - 10) << 23);  // add the bias difference to xmm1
  xmm1 = xmm1 | sign;               // Combine with the sign mask

  union {
    int i;
    float f;
  } res = {
      .i = xmm1,
  };
  return mantissa1 * res.f;
}

IREE_DEVICE_EXPORT short iree_f2h_ieee(float param) {
  // Some constants about the f32 and f16 types.
  const int f32_mantissa_bits = 23;
  const int f32_exp_bias = 127;
  const uint32_t f32_sign_mask = 0x80000000u;
  const uint32_t f32_exp_mask = 0x7F800000u;
  const uint32_t f32_mantissa_mask = 0x007FFFFFu;
  const int f16_mantissa_bits = 10;
  const int f16_exp_bits = 5;
  const int f16_exp_bias = 15;
  const uint16_t f16_exp_mask = 0x7C00u;
  const uint16_t f16_mantissa_mask = 0x03FFu;

  // Bitcast float param to uint32.
  union {
    unsigned int u;
    float f;
  } param_bits = {
      .f = param,
  };
  uint32_t u32_value = param_bits.u;

  // Split the f32 sign/exponent/mantissa components.
  const uint32_t f32_sign = u32_value & f32_sign_mask;
  const uint32_t f32_exp = u32_value & f32_exp_mask;
  const uint32_t f32_mantissa = u32_value & f32_mantissa_mask;
  // Initialize the f16 sign/exponent/mantissa components.
  uint32_t f16_sign = f32_sign >> 16;
  uint32_t f16_exp = 0;
  uint32_t f16_mantissa = 0;

  if (f32_exp >= f32_exp_mask) {
    // NaN or Inf case.
    f16_exp = f16_exp_mask;
    if (f32_mantissa) {
      // NaN. Generate a quiet NaN.
      return f16_sign | f16_exp_mask | f16_mantissa_mask;
    } else {
      // Inf. Leave zero mantissa.
    }
  } else if (f32_exp == 0) {
    // Zero or subnormal. Generate zero. Leave zero mantissa.
  } else {
    // Normal finite value.
    int arithmetic_exp = (f32_exp >> f32_mantissa_bits) - f32_exp_bias;
    // Test if the exponent is too large for the destination type. If
    // the destination type does not have infinities, that frees up the
    // max exponent value for additional finite values.
    if (arithmetic_exp >= 1 << (f16_exp_bits - 1)) {
      // Overflow. Generate Inf. Leave zero mantissa.
      f16_exp = f16_exp_mask;
    } else if (arithmetic_exp + f16_exp_bias <= 0) {
      // Underflow. Generate zero. Leave zero mantissa.
      f16_exp = 0;
    } else {
      // Normal case.
      // Implement round-to-nearest-even, by adding a bias before truncating.
      int even_bit = 1u << (f32_mantissa_bits - f16_mantissa_bits);
      int odd_bit = even_bit >> 1;
      uint32_t biased_f32_mantissa =
          f32_mantissa +
          ((f32_mantissa & even_bit) ? (odd_bit) : (odd_bit - 1));
      // Adding the bias may cause an exponent increment.
      if (biased_f32_mantissa > f32_mantissa_mask) {
        biased_f32_mantissa = 0;
        ++arithmetic_exp;
      }
      // The exponent increment in the above if() branch may cause overflow.
      // This is exercised by converting 65520.0f from f32 to f16.
      f16_exp = (arithmetic_exp + f16_exp_bias) << f16_mantissa_bits;
      f16_mantissa =
          biased_f32_mantissa >> (f32_mantissa_bits - f16_mantissa_bits);
    }
  }

  return f16_sign | f16_exp | f16_mantissa;
}

#if defined(IREE_DEVICE_STANDALONE)

IREE_DEVICE_EXPORT float __gnu_h2f_ieee(short param) {
  return iree_h2f_ieee(param);
}

IREE_DEVICE_EXPORT float __extendhfsf2(float param) {
  return iree_h2f_ieee(*((short *)&param));
}

IREE_DEVICE_EXPORT short __gnu_f2h_ieee(float param) {
  return iree_f2h_ieee(param);
}

IREE_DEVICE_EXPORT float __truncsfhf2(float param) {
  short ret = iree_f2h_ieee(param);
  return *((float *)&ret);
}

#endif  // IREE_DEVICE_STANDALONE
