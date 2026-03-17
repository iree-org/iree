// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Freestanding <float.h> for wasm32 (IEEE 754).

#ifndef IREE_WASM_LIBC_FLOAT_H_
#define IREE_WASM_LIBC_FLOAT_H_

#define FLT_RADIX 2
#define FLT_ROUNDS 1  // Round to nearest.

#define FLT_EVAL_METHOD 0  // Each type evaluated to itself.

#define DECIMAL_DIG 21

// float: IEEE 754 binary32.
#define FLT_MANT_DIG 24
#define FLT_DIG 6
#define FLT_MIN_EXP (-125)
#define FLT_MIN_10_EXP (-37)
#define FLT_MAX_EXP 128
#define FLT_MAX_10_EXP 38
#define FLT_MIN 1.17549435e-38F
#define FLT_MAX 3.40282347e+38F
#define FLT_EPSILON 1.19209290e-07F
#define FLT_TRUE_MIN 1.40129846e-45F
#define FLT_DECIMAL_DIG 9
#define FLT_HAS_SUBNORM 1

// double: IEEE 754 binary64.
#define DBL_MANT_DIG 53
#define DBL_DIG 15
#define DBL_MIN_EXP (-1021)
#define DBL_MIN_10_EXP (-307)
#define DBL_MAX_EXP 1024
#define DBL_MAX_10_EXP 308
#define DBL_MIN 2.2250738585072014e-308
#define DBL_MAX 1.7976931348623157e+308
#define DBL_EPSILON 2.2204460492503131e-16
#define DBL_TRUE_MIN 5.0e-324
#define DBL_DECIMAL_DIG 17
#define DBL_HAS_SUBNORM 1

// long double: on wasm32, long double is IEEE 754 binary128.
#define LDBL_MANT_DIG 113
#define LDBL_DIG 33
#define LDBL_MIN_EXP (-16381)
#define LDBL_MIN_10_EXP (-4931)
#define LDBL_MAX_EXP 16384
#define LDBL_MAX_10_EXP 4932
#define LDBL_MIN 3.36210314311209350626267781732175260e-4932L
#define LDBL_MAX 1.18973149535723176502126385303097021e+4932L
#define LDBL_EPSILON 1.92592994438723585305597794258492732e-34L
#define LDBL_TRUE_MIN 6.47517511943802511092443895822764655e-4966L
#define LDBL_DECIMAL_DIG 36
#define LDBL_HAS_SUBNORM 1

#endif  // IREE_WASM_LIBC_FLOAT_H_
