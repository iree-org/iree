// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <math.h> for wasm32.
// Implementations are provided by wasm builtins where available,
// otherwise by musl libm bitcode linked at build time.

#ifndef IREE_WASM_LIBC_MATH_H_
#define IREE_WASM_LIBC_MATH_H_

#define HUGE_VAL __builtin_huge_val()
#define HUGE_VALF __builtin_huge_valf()
#define INFINITY __builtin_inff()
#define NAN __builtin_nanf("")

#define FP_NAN 0
#define FP_INFINITE 1
#define FP_ZERO 2
#define FP_SUBNORMAL 3
#define FP_NORMAL 4

#define fpclassify(x)                                                         \
  __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, \
                       (x))
#define isfinite(x) __builtin_isfinite(x)
#define isinf(x) __builtin_isinf(x)
#define isnan(x) __builtin_isnan(x)
#define isnormal(x) __builtin_isnormal(x)
#define signbit(x) __builtin_signbit(x)

// Wasm has native instructions for these (via clang builtins).
// clang lowers them directly to wasm opcodes.
double fabs(double x);
float fabsf(float x);
double sqrt(double x);
float sqrtf(float x);
double ceil(double x);
float ceilf(float x);
double floor(double x);
float floorf(float x);
double trunc(double x);
float truncf(float x);
double nearbyint(double x);
float nearbyintf(float x);
double rint(double x);
float rintf(float x);
double fmin(double x, double y);
float fminf(float x, float y);
double fmax(double x, double y);
float fmaxf(float x, float y);
double copysign(double x, double y);
float copysignf(float x, float y);

// Software implementations (from musl libm or our own).
double sin(double x);
float sinf(float x);
double cos(double x);
float cosf(float x);
double tan(double x);
float tanf(float x);
double asin(double x);
float asinf(float x);
double acos(double x);
float acosf(float x);
double atan(double x);
float atanf(float x);
double atan2(double y, double x);
float atan2f(float y, float x);
double exp(double x);
float expf(float x);
double exp2(double x);
float exp2f(float x);
double log(double x);
float logf(float x);
double log2(double x);
float log2f(float x);
double log10(double x);
float log10f(float x);
double pow(double base, double exponent);
float powf(float base, float exponent);
double fmod(double x, double y);
float fmodf(float x, float y);
double remainder(double x, double y);
float remainderf(float x, float y);
double ldexp(double x, int exponent);
float ldexpf(float x, int exponent);
double frexp(double x, int* exponent);
float frexpf(float x, int* exponent);
double modf(double x, double* integer_part);
float modff(float x, float* integer_part);
double round(double x);
float roundf(float x);
long lround(double x);
long lroundf(float x);
long long llround(double x);
long long llroundf(float x);
long lrint(double x);
long lrintf(float x);
long long llrint(double x);
long long llrintf(float x);
double cbrt(double x);
float cbrtf(float x);
double hypot(double x, double y);
float hypotf(float x, float y);
double sinh(double x);
float sinhf(float x);
double cosh(double x);
float coshf(float x);
double tanh(double x);
float tanhf(float x);
double erf(double x);
float erff(float x);
double erfc(double x);
float erfcf(float x);
double tgamma(double x);
float tgammaf(float x);
double lgamma(double x);
float lgammaf(float x);
double expm1(double x);
float expm1f(float x);
double log1p(double x);
float log1pf(float x);
double scalbn(double x, int n);
float scalbnf(float x, int n);
int ilogb(double x);
int ilogbf(float x);
double logb(double x);
float logbf(float x);
double nextafter(double x, double y);
float nextafterf(float x, float y);
double fdim(double x, double y);
float fdimf(float x, float y);
double fma(double x, double y, double z);
float fmaf(float x, float y, float z);

#endif  // IREE_WASM_LIBC_MATH_H_
