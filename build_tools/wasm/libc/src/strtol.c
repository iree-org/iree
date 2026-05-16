// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Integer and floating-point string-to-number conversion for wasm32.
//
// strtol/strtoul/strtoll/strtoull: derived from musl libc (MIT license).
// These are self-contained with no locale dependency — always C locale.
//
// strtod/strtof: simplified implementation sufficient for IREE's use case
// (configuration parsing, flag values). Handles decimal and basic scientific
// notation. Does not attempt the full IEEE 754 correctly-rounded conversion
// that a production strtod requires — that would need either the Eisel-Lemire
// algorithm or musl's __floatscan multi-precision fallback (~1000 lines).
// IREE only parses floating-point values from its own output (e.g., benchmark
// results, debug flags), so exact round-trip is not a concern.

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

//===----------------------------------------------------------------------===//
// Integer conversion (strtol family)
//===----------------------------------------------------------------------===//

// Core unsigned magnitude conversion with overflow detection.
// Returns the absolute value of the parsed number. The sign is returned
// separately via |*out_negative| so that callers (strtol/strtoll) can
// correctly distinguish positive overflow from the valid negative minimum
// (e.g., LONG_MIN = -(LONG_MAX+1)).
static unsigned long long strtox(const char* string, char** end, int base,
                                 unsigned long long limit, int* out_negative) {
  const char* p = string;

  // Skip leading whitespace.
  while (isspace((unsigned char)*p)) ++p;

  // Optional sign.
  int negative = 0;
  if (*p == '-') {
    negative = 1;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  *out_negative = negative;

  // Base detection.
  if ((base == 0 || base == 16) && p[0] == '0' &&
      (p[1] == 'x' || p[1] == 'X')) {
    base = 16;
    p += 2;
  } else if ((base == 0 || base == 2) && p[0] == '0' &&
             (p[1] == 'b' || p[1] == 'B')) {
    base = 2;
    p += 2;
  } else if (base == 0) {
    base = (p[0] == '0') ? 8 : 10;
  }

  const char* digits_start = p;
  unsigned long long result = 0;
  int overflow = 0;

  // Cutoff values for overflow detection.
  unsigned long long cutoff = limit / (unsigned long long)base;
  unsigned int cutlim = (unsigned int)(limit % (unsigned long long)base);

  for (;; ++p) {
    unsigned int digit;
    if (isdigit((unsigned char)*p)) {
      digit = (unsigned int)(*p - '0');
    } else if (isalpha((unsigned char)*p)) {
      digit = (unsigned int)(((*p | 0x20) - 'a') + 10);
    } else {
      break;
    }
    if (digit >= (unsigned int)base) break;

    if (result > cutoff || (result == cutoff && digit > cutlim)) {
      overflow = 1;
    } else {
      result = result * (unsigned long long)base + digit;
    }
  }

  // No digits consumed — set end to original string.
  if (p == digits_start) {
    if (end) *end = (char*)string;
    return 0;
  }

  if (end) *end = (char*)p;

  if (overflow) {
    errno = ERANGE;
    return limit;
  }

  return result;
}

long strtol(const char* string, char** end, int base) {
  int negative = 0;
  unsigned long long magnitude =
      strtox(string, end, base, (unsigned long long)LONG_MAX + 1, &negative);
  if (negative) {
    if (magnitude > (unsigned long long)LONG_MAX + 1) {
      errno = ERANGE;
      return LONG_MIN;
    }
    return -(long)magnitude;
  }
  if (magnitude > (unsigned long long)LONG_MAX) {
    errno = ERANGE;
    return LONG_MAX;
  }
  return (long)magnitude;
}

unsigned long strtoul(const char* string, char** end, int base) {
  int negative = 0;
  unsigned long long magnitude =
      strtox(string, end, base, ULONG_MAX, &negative);
  // C standard: strtoul("-X") wraps to ULONG_MAX - X + 1.
  if (negative) return (unsigned long)(0 - magnitude);
  return (unsigned long)magnitude;
}

long long strtoll(const char* string, char** end, int base) {
  int negative = 0;
  unsigned long long magnitude =
      strtox(string, end, base, (unsigned long long)LLONG_MAX + 1, &negative);
  if (negative) {
    if (magnitude > (unsigned long long)LLONG_MAX + 1) {
      errno = ERANGE;
      return LLONG_MIN;
    }
    return -(long long)magnitude;
  }
  if (magnitude > (unsigned long long)LLONG_MAX) {
    errno = ERANGE;
    return LLONG_MAX;
  }
  return (long long)magnitude;
}

unsigned long long strtoull(const char* string, char** end, int base) {
  int negative = 0;
  unsigned long long magnitude =
      strtox(string, end, base, ULLONG_MAX, &negative);
  if (negative) return 0 - magnitude;
  return magnitude;
}

int atoi(const char* string) { return (int)strtol(string, NULL, 10); }
long atol(const char* string) { return strtol(string, NULL, 10); }
long long atoll(const char* string) { return strtoll(string, NULL, 10); }

//===----------------------------------------------------------------------===//
// Floating-point conversion (strtod family)
//===----------------------------------------------------------------------===//

// Simplified strtod: handles [+-]digits[.digits][eE[+-]digits], infinity, nan.
// This does NOT provide correctly-rounded IEEE 754 conversion for all inputs.
// It uses double-precision arithmetic throughout, which is exact for mantissas
// up to 2^53 and common exponents. Edge cases (very long digit strings,
// subnormal boundaries) may differ by 1 ULP from a fully conformant strtod.
static double strtod_impl(const char* string, char** end) {
  const char* p = string;

  while (isspace((unsigned char)*p)) ++p;

  int negative = 0;
  if (*p == '-') {
    negative = 1;
    ++p;
  } else if (*p == '+') {
    ++p;
  }

  // Handle infinity and NaN.
  if ((p[0] == 'i' || p[0] == 'I') && (p[1] == 'n' || p[1] == 'N') &&
      (p[2] == 'f' || p[2] == 'F')) {
    p += 3;
    // Skip optional "inity".
    if ((p[0] == 'i' || p[0] == 'I') && (p[1] == 'n' || p[1] == 'N') &&
        (p[2] == 'i' || p[2] == 'I') && (p[3] == 't' || p[3] == 'T') &&
        (p[4] == 'y' || p[4] == 'Y')) {
      p += 5;
    }
    if (end) *end = (char*)p;
    return negative ? -INFINITY : INFINITY;
  }

  if ((p[0] == 'n' || p[0] == 'N') && (p[1] == 'a' || p[1] == 'A') &&
      (p[2] == 'n' || p[2] == 'N')) {
    p += 3;
    // Skip optional parenthesized payload.
    if (*p == '(') {
      const char* q = p + 1;
      while (*q && *q != ')') ++q;
      if (*q == ')') p = q + 1;
    }
    if (end) *end = (char*)p;
    return NAN;
  }

  // Parse hexadecimal float: 0x prefix.
  if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
    p += 2;
    double result = 0.0;
    int has_digits = 0;

    // Integer part.
    while (isxdigit((unsigned char)*p)) {
      unsigned int digit;
      if (*p >= '0' && *p <= '9')
        digit = (unsigned int)(*p - '0');
      else
        digit = (unsigned int)((*p | 0x20) - 'a' + 10);
      result = result * 16.0 + (double)digit;
      has_digits = 1;
      ++p;
    }

    // Fractional part.
    if (*p == '.') {
      ++p;
      double place = 1.0 / 16.0;
      while (isxdigit((unsigned char)*p)) {
        unsigned int digit;
        if (*p >= '0' && *p <= '9')
          digit = (unsigned int)(*p - '0');
        else
          digit = (unsigned int)((*p | 0x20) - 'a' + 10);
        result += (double)digit * place;
        place /= 16.0;
        has_digits = 1;
        ++p;
      }
    }

    if (!has_digits) {
      if (end) *end = (char*)string;
      return 0.0;
    }

    // Binary exponent.
    if (*p == 'p' || *p == 'P') {
      ++p;
      int exp_negative = 0;
      if (*p == '-') {
        exp_negative = 1;
        ++p;
      } else if (*p == '+') {
        ++p;
      }
      int exponent = 0;
      while (isdigit((unsigned char)*p)) {
        exponent = exponent * 10 + (*p - '0');
        if (exponent > 2000) {
          // Clamp to avoid signed int overflow. Any binary exponent this
          // large will produce infinity or zero after ldexp.
          exponent = 2000;
        }
        ++p;
      }
      if (exp_negative) exponent = -exponent;
      result = ldexp(result, exponent);
    }

    if (end) *end = (char*)p;
    return negative ? -result : result;
  }

  // Parse decimal float.
  double result = 0.0;
  int has_digits = 0;

  // Integer part.
  while (isdigit((unsigned char)*p)) {
    result = result * 10.0 + (double)(*p - '0');
    has_digits = 1;
    ++p;
  }

  // Fractional part.
  if (*p == '.') {
    ++p;
    double place = 0.1;
    while (isdigit((unsigned char)*p)) {
      result += (double)(*p - '0') * place;
      place *= 0.1;
      has_digits = 1;
      ++p;
    }
  }

  if (!has_digits) {
    if (end) *end = (char*)string;
    return 0.0;
  }

  // Exponent.
  if (*p == 'e' || *p == 'E') {
    ++p;
    int exp_negative = 0;
    if (*p == '-') {
      exp_negative = 1;
      ++p;
    } else if (*p == '+') {
      ++p;
    }
    int exponent = 0;
    while (isdigit((unsigned char)*p)) {
      exponent = exponent * 10 + (*p - '0');
      if (exponent > 400) {
        // Clamp to avoid unbounded pow() computation. Any exponent this
        // large will produce infinity or zero after the multiply.
        exponent = 400;
      }
      ++p;
    }
    if (exp_negative) exponent = -exponent;

    // Apply exponent via power of 10. For common exponents [-22, +22]
    // this is exact (all powers of 10 up to 10^22 are representable in
    // double precision). For larger exponents, precision loss may occur.
    if (exponent != 0) {
      result *= pow(10.0, (double)exponent);
    }
  }

  if (end) *end = (char*)p;
  result = negative ? -result : result;

  // Detect overflow/underflow and set errno per C standard.
  if (has_digits && result != 0.0 && !isinf(result)) {
    // Normal finite result — no error.
  } else if (has_digits && isinf(result)) {
    errno = ERANGE;
  } else if (has_digits && result == 0.0) {
    // Could be underflow if there were significant digits. For IREE's config
    // parsing use case, true underflow is rare enough that we skip the
    // has-significant-digits check and just don't set errno for zero results.
  }
  return result;
}

double strtod(const char* string, char** end) {
  return strtod_impl(string, end);
}

float strtof(const char* string, char** end) {
  return (float)strtod_impl(string, end);
}

long double strtold(const char* string, char** end) {
  // wasm32 long double is the same as double (64-bit).
  return (long double)strtod_impl(string, end);
}

double atof(const char* string) { return strtod(string, NULL); }
