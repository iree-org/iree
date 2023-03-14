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
      int res = (0x7FC00000 | sign);
      float fres = *((float *)(&res));
      return fres;
    }
    // inf
    int res = (0x7F800000 | sign);
    float fres = *((float *)(&res));
    return fres;
  }
  if (expHalf16 != 0) {
    exp1 += ((127 - 15) << 10);  // exponents converted to float32 bias
    int res = (exp1 | mantissa1);
    res = res << 13;
    res = (res | sign);
    float fres = *((float *)(&res));
    return fres;
  }

  int xmm1 = exp1 > (1 << 10) ? exp1 : (1 << 10);
  xmm1 = (xmm1 << 13);
  xmm1 += ((127 - 15 - 10) << 23);  // add the bias difference to xmm1
  xmm1 = xmm1 | sign;               // Combine with the sign mask

  float res = (float)mantissa1;  // Convert mantissa to float
  res *= *((float *)(&xmm1));

  return res;
}

IREE_DEVICE_EXPORT short iree_f2h_ieee(float param) {
  unsigned int param_bit = *((unsigned int *)(&param));
  int sign = param_bit >> 31;
  int mantissa = param_bit & 0x007FFFFF;
  int exp = ((param_bit & 0x7F800000) >> 23) + 15 - 127;
  short res;
  if (exp > 0 && exp < 30) {
    // use rte rounding mode, round the significand, combine sign, exponent and
    // significand into a short.
    res = (sign << 15) | (exp << 10) | ((mantissa + 0x00001000) >> 13);
  } else if (param_bit == 0) {
    res = 0;
  } else {
    if (exp <= 0) {
      if (exp < -10) {
        // value is less than min half float point
        res = 0;
      } else {
        // normalized single, magnitude is less than min normal half float
        // point.
        mantissa = (mantissa | 0x00800000) >> (1 - exp);
        // round to nearest
        if ((mantissa & 0x00001000) > 0) {
          mantissa = mantissa + 0x00002000;
        }
        // combine sign & mantissa (exp is zero to get denormalized number)
        res = (sign << 15) | (mantissa >> 13);
      }
    } else if (exp == (255 - 127 + 15)) {
      if (mantissa == 0) {
        // input float is infinity, return infinity half
        res = (sign << 15) | 0x7C00;
      } else {
        // input float is NaN, return half NaN
        res = (sign << 15) | 0x7C00 | (mantissa >> 13);
      }
    } else {
      // exp > 0, normalized single, round to nearest
      if ((mantissa & 0x00001000) > 0) {
        mantissa = mantissa + 0x00002000;
        if ((mantissa & 0x00800000) > 0) {
          mantissa = 0;
          exp = exp + 1;
        }
      }
      if (exp > 30) {
        // exponent overflow - return infinity half
        res = (sign << 15) | 0x7C00;
      } else {
        // combine sign, exp and mantissa into normalized half
        res = (sign << 15) | (exp << 10) | (mantissa >> 13);
      }
    }
  }
  return res;
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
