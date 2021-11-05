// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBMMT4D_MMT4D_H_
#define LIBMMT4D_MMT4D_H_

#include <stdint.h>

//===----------------------------------------------------------------------===//
// Attributes and metadata
//===----------------------------------------------------------------------===//

// Tagged on functions that are part of the public API.
#ifdef __cplusplus
#define MMT4D_EXPORT extern "C"
#else
#define MMT4D_EXPORT
#endif  // __cplusplus

// `restrict` keyword, not supported by some older compilers.
// We define our own macro in case dependencies use `restrict` differently.
#if defined(_MSC_VER) && _MSC_VER >= 1900
#define MMT4D_RESTRICT __restrict
#elif defined(_MSC_VER)
#define MMT4D_RESTRICT
#elif defined(__cplusplus)
#define MMT4D_RESTRICT __restrict__
#else
#define MMT4D_RESTRICT restrict
#endif  // _MSC_VER

//===----------------------------------------------------------------------===//
// Target-specific queries
//===----------------------------------------------------------------------===//
// These are substituted with values from the compiler and must not be specified
// here in C before we generate the IR.

// Do not use: here as an example. Remove once we have any other flag.
extern int libmmt4d_platform_example_flag;

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

MMT4D_EXPORT void mmt4d_8x4x8_i8i8i32(int k_size, const int8_t* lhs,
                                      const int8_t* rhs,
                                      int32_t* MMT4D_RESTRICT dst);

//===----------------------------------------------------------------------===//
// Utilities for templating
//===----------------------------------------------------------------------===//

#define MMT4D_CAT_I(a, b) a##b
#define MMT4D_CAT2(a, b) MMT4D_CAT_I(a, b)
#define MMT4D_CAT3(a, b, c) MMT4D_CAT2(MMT4D_CAT2(a, b), c)
#define MMT4D_CAT4(a, b, c, d) MMT4D_CAT2(MMT4D_CAT3(a, b, c), d)

#define MMT4D_SHAPE_LITERAL(M0, K0, N0) M0##x##K0##x##N0

#define MMT4D_TYPE_int8_t i8
#define MMT4D_TYPE_int16_t i16
#define MMT4D_TYPE_int32_t i32
#define MMT4D_TYPE_int64_t i64
#define MMT4D_TYPE_float f32
#define MMT4D_TYPE_double f64
#define MMT4D_TYPE_LITERAL(lhs_type, rhs_type, dst_type)   \
  MMT4D_CAT3(MMT4D_TYPE_##lhs_type, MMT4D_TYPE_##rhs_type, \
             MMT4D_TYPE_##dst_type)

#define MMT4D_FUNC_NAME(M0, K0, N0, lhs_t, rhs_t, dst_t) \
  MMT4D_CAT4(mmt4d_, MMT4D_SHAPE_LITERAL(M0, K0, N0), _, \
             MMT4D_TYPE_LITERAL(lhs_t, rhs_t, dst_t))

#define MMT4D_GENERIC(M0, K0, N0, lhs_t, rhs_t, dst_t)                \
  MMT4D_EXPORT void MMT4D_FUNC_NAME(M0, K0, N0, lhs_t, rhs_t, dst_t)( \
      int k_size, const lhs_t* lhs, const rhs_t* rhs,                 \
      dst_t* MMT4D_RESTRICT dst) {                                    \
    for (int k = 0; k < k_size; k += K0) {                            \
      for (int m0 = 0; m0 < M0; m0++) {                               \
        for (int n0 = 0; n0 < N0; n0++) {                             \
          dst_t a = 0;                                                \
          for (int k0 = 0; k0 < K0; k0++) {                           \
            a += lhs[m0 * K0 + k0] * rhs[n0 * K0 + k0];               \
          }                                                           \
          dst[m0 * N0 + n0] += a;                                     \
        }                                                             \
      }                                                               \
      lhs += M0 * K0;                                                 \
      rhs += N0 * K0;                                                 \
    }                                                                 \
  }

#endif  // LIBMMT4D_MMT4D_H_
