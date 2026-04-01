// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE's platform-independent printf implementation.
//
// Replaces the vendored eyalroz/printf with a focused implementation covering
// exactly the specifiers IREE uses. Every line is ours, every path is tested,
// every bounds check precedes its dereference.
//
// Supported specifiers:
//   %d %i %u %o %x %X  (integers, with hh/h/l/ll/z/t/j length modifiers)
//   %s %c %p            (string, character, pointer)
//   %f %F %e %E %g %G   (floating-point, double only)
//   %%                   (literal percent)
//
// Supported flags: - + 0 # (space)
// Supported width/precision: literal and dynamic (*)
//
// NOT supported (by design):
//   %n (writeback — security hazard)
//   %a %A (hex float — 0 uses in IREE)
//   %ls %lc (wide strings/chars — 0 uses in IREE)
//   %L (long double — 0 uses in IREE)
//   MSVC %I64d style — 0 uses in IREE
//
// Returns the number of characters that would have been written (excluding NUL)
// if the output were unbounded, or -1 on format error (unknown specifier).

#include "iree/base/printf.h"

#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Portable math (no libm dependency)
//===----------------------------------------------------------------------===//
// Provides fma, fmod, floor, and modf equivalents without requiring -lm.
// Uses hardware FMA when the compilation target has it (single instruction,
// no function call); falls back to the Dekker two-product algorithm otherwise.

#ifdef _MSC_VER
#include <math.h>  // fma() — part of CRT on Windows, no separate libm.
#endif

// Hardware FMA detection: on these targets, __builtin_fma compiles to a single
// instruction with no libm dependency.
#if defined(__FMA__) || defined(__aarch64__) || defined(__ARM_FEATURE_FMA) || \
    (defined(__riscv) && defined(__riscv_flen) && __riscv_flen >= 64)
#define IREE_PRINTF_HAS_HARDWARE_FMA 1
#else
#define IREE_PRINTF_HAS_HARDWARE_FMA 0
#endif

// Compute the exact rounding error of a*b: returns e such that
// a*b = product + e exactly (where product was computed as a*b by the FPU).
// On hardware-FMA targets, this is a single fused multiply-add instruction.
// On other targets, uses the Dekker-Veltkamp two-product algorithm, which
// computes the same result using only basic arithmetic. The Dekker path is
// safe from unintended FP contraction because it only activates on targets
// without FMA hardware (where the compiler cannot contract multiplications
// into FMA instructions).
static inline double iree_printf_mul_error(double a, double b, double product) {
#if IREE_PRINTF_HAS_HARDWARE_FMA
  return __builtin_fma(a, b, -product);
#elif defined(_MSC_VER)
  // MSVC: fma is in the CRT, no separate libm.
  return fma(a, b, -product);
#else
  // Dekker-Veltkamp two-product: split each operand into high/low halves
  // (each with <= 26 mantissa bits), then compute the exact product error
  // from the four partial products.
  const double VELTKAMP_SPLIT = 134217729.0;  // 2^27 + 1
  double a_big = VELTKAMP_SPLIT * a;
  double a_hi = a_big - (a_big - a);
  double a_lo = a - a_hi;
  double b_big = VELTKAMP_SPLIT * b;
  double b_hi = b_big - (b_big - b);
  double b_lo = b - b_hi;
  return ((a_hi * b_hi - product) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
#endif  // FMA
}

// Portable floor: returns the largest integer <= x.
// Only called on positive, finite values in our formatting paths.
static inline double iree_printf_floor(double x) {
  // Doubles >= 2^52 in magnitude are always integers (the significand cannot
  // represent a fractional part at that scale).
  if (x >= 4503599627370496.0) return x;
  if (x <= -4503599627370496.0) return x;
  double truncated = (double)(int64_t)x;
  // C truncation rounds toward zero; floor rounds toward -infinity.
  if (truncated > x) truncated -= 1.0;
  return truncated;
}

// Portable modf: split value into integral and fractional parts.
// Only called on positive, finite values in our formatting paths.
static inline double iree_printf_modf(double value, double* integral_part) {
  if (value >= 4503599627370496.0 || value <= -4503599627370496.0) {
    *integral_part = value;
    return 0.0;
  }
  double truncated = (double)(int64_t)value;
  *integral_part = truncated;
  return value - truncated;
}

// Check if an integer-valued double is odd. Used for banker's rounding.
// Doubles >= 2^53 in magnitude always represent even integers (the lowest
// bit of the significand corresponds to 2 or more at that scale).
static inline bool iree_printf_is_odd(double x) {
  if (x >= 9007199254740992.0 || x <= -9007199254740992.0) return false;
  return ((int64_t)x) & 1;
}

//===----------------------------------------------------------------------===//
// Output abstraction
//===----------------------------------------------------------------------===//
// Handles three output modes through a single interface:
//   1. Buffer mode: writes to a sized char buffer with truncation.
//   2. Callback mode: calls a per-character callback for each output character.
//   3. Dry-run mode: counts characters without writing (NULL buffer, no
//      callback).
// Position is always tracked, even past the buffer capacity, so the return
// value correctly reports the total formatted length.

typedef struct {
  char* buffer;
  size_t capacity;
  size_t position;
  iree_printf_callback_t callback;
  void* callback_data;
} iree_printf_output_t;

static inline void iree_printf_output_char(iree_printf_output_t* out, char c) {
  if (out->callback) {
    out->callback(c, out->callback_data);
  } else if (out->buffer && out->position < out->capacity) {
    out->buffer[out->position] = c;
  }
  out->position++;
}

// Emit |count| copies of |c|.
static void iree_printf_output_fill(iree_printf_output_t* out, char c,
                                    size_t count) {
  for (size_t i = 0; i < count; i++) {
    iree_printf_output_char(out, c);
  }
}

// Emit |length| bytes from |str|. The caller must guarantee |str| points to at
// least |length| readable bytes.
static void iree_printf_output_string(iree_printf_output_t* out,
                                      const char* str, size_t length) {
  for (size_t i = 0; i < length; i++) {
    iree_printf_output_char(out, str[i]);
  }
}

//===----------------------------------------------------------------------===//
// Format specifier parsing
//===----------------------------------------------------------------------===//

// Flags that modify formatting behavior.
#define IREE_PRINTF_FLAG_LEFT (1u << 0)   // '-': left-justify within width.
#define IREE_PRINTF_FLAG_PLUS (1u << 1)   // '+': show sign for positive.
#define IREE_PRINTF_FLAG_SPACE (1u << 2)  // ' ': space before positive.
#define IREE_PRINTF_FLAG_ZERO (1u << 3)   // '0': zero-pad within width.
#define IREE_PRINTF_FLAG_HASH (1u << 4)   // '#': alternate form (0x, etc).

// Length modifier encoding.
typedef enum {
  IREE_PRINTF_LENGTH_NONE = 0,
  IREE_PRINTF_LENGTH_HH,  // char / unsigned char (promoted to int in va_arg).
  IREE_PRINTF_LENGTH_H,   // short / unsigned short (promoted to int).
  IREE_PRINTF_LENGTH_L,   // long / unsigned long.
  IREE_PRINTF_LENGTH_LL,  // long long / unsigned long long.
  IREE_PRINTF_LENGTH_Z,   // size_t / ssize_t.
  IREE_PRINTF_LENGTH_T,   // ptrdiff_t.
  IREE_PRINTF_LENGTH_J,   // intmax_t / uintmax_t.
} iree_printf_length_t;

// Parsed format specifier. Extracted from the format string between '%' and the
// conversion character.
typedef struct {
  uint32_t flags;
  int width;
  int precision;
  bool has_precision;
  iree_printf_length_t length;
  char specifier;
} iree_printf_spec_t;

// Parse flags from the format string. Returns pointer past the last flag.
static const char* iree_printf_parse_flags(const char* format,
                                           uint32_t* out_flags) {
  uint32_t flags = 0;
  for (;;) {
    switch (*format) {
      case '-':
        flags |= IREE_PRINTF_FLAG_LEFT;
        break;
      case '+':
        flags |= IREE_PRINTF_FLAG_PLUS;
        break;
      case ' ':
        flags |= IREE_PRINTF_FLAG_SPACE;
        break;
      case '0':
        flags |= IREE_PRINTF_FLAG_ZERO;
        break;
      case '#':
        flags |= IREE_PRINTF_FLAG_HASH;
        break;
      default:
        *out_flags = flags;
        return format;
    }
    format++;
  }
}

// Parse an unsigned decimal integer from the format string.
// Returns pointer past the last digit. |out_value| is unchanged if no digits.
// Values are clamped to avoid signed integer overflow (undefined behavior).
// The clamp limit is generous enough for any legitimate width/precision while
// preventing UB from malicious or malformed format strings.
#define IREE_PRINTF_MAX_WIDTH_PRECISION 10000
static const char* iree_printf_parse_uint(const char* format, int* out_value) {
  if (*format < '0' || *format > '9') return format;
  int value = 0;
  while (*format >= '0' && *format <= '9') {
    if (value <= IREE_PRINTF_MAX_WIDTH_PRECISION) {
      value = value * 10 + (*format - '0');
    }
    format++;
  }
  if (value > IREE_PRINTF_MAX_WIDTH_PRECISION) {
    value = IREE_PRINTF_MAX_WIDTH_PRECISION;
  }
  *out_value = value;
  return format;
}

// Parse a complete format specifier starting after the '%'.
// Returns pointer past the specifier character, or NULL on error.
static const char* iree_printf_parse_spec(const char* format, va_list* args,
                                          iree_printf_spec_t* out_spec) {
  memset(out_spec, 0, sizeof(*out_spec));
  out_spec->precision = -1;  // Sentinel: no precision specified.

  // Flags.
  format = iree_printf_parse_flags(format, &out_spec->flags);

  // Width: literal or '*' (read from args).
  if (*format == '*') {
    out_spec->width = va_arg(*args, int);
    if (out_spec->width < 0) {
      // Negative width means left-justify with the absolute value.
      // Cast to unsigned before negation to handle INT_MIN without UB.
      out_spec->flags |= IREE_PRINTF_FLAG_LEFT;
      out_spec->width = (int)(-(unsigned int)out_spec->width);
      if (out_spec->width < 0) out_spec->width = 0;  // INT_MIN edge case.
    }
    if (out_spec->width > IREE_PRINTF_MAX_WIDTH_PRECISION) {
      out_spec->width = IREE_PRINTF_MAX_WIDTH_PRECISION;
    }
    format++;
  } else {
    format = iree_printf_parse_uint(format, &out_spec->width);
  }

  // Precision: '.' followed by literal or '*'.
  if (*format == '.') {
    format++;
    out_spec->has_precision = true;
    out_spec->precision = 0;  // Default precision after '.' is 0.
    if (*format == '*') {
      out_spec->precision = va_arg(*args, int);
      if (out_spec->precision < 0) {
        // Negative precision is treated as if precision were omitted.
        out_spec->has_precision = false;
        out_spec->precision = -1;
      } else if (out_spec->precision > IREE_PRINTF_MAX_WIDTH_PRECISION) {
        out_spec->precision = IREE_PRINTF_MAX_WIDTH_PRECISION;
      }
      format++;
    } else {
      format = iree_printf_parse_uint(format, &out_spec->precision);
    }
  }

  // Length modifier.
  switch (*format) {
    case 'h':
      format++;
      if (*format == 'h') {
        out_spec->length = IREE_PRINTF_LENGTH_HH;
        format++;
      } else {
        out_spec->length = IREE_PRINTF_LENGTH_H;
      }
      break;
    case 'l':
      format++;
      if (*format == 'l') {
        out_spec->length = IREE_PRINTF_LENGTH_LL;
        format++;
      } else {
        out_spec->length = IREE_PRINTF_LENGTH_L;
      }
      break;
    case 'z':
      out_spec->length = IREE_PRINTF_LENGTH_Z;
      format++;
      break;
    case 't':
      out_spec->length = IREE_PRINTF_LENGTH_T;
      format++;
      break;
    case 'j':
      out_spec->length = IREE_PRINTF_LENGTH_J;
      format++;
      break;
    default:
      break;
  }

  // Specifier character.
  out_spec->specifier = *format;
  if (*format == '\0') return NULL;  // Truncated format string.
  return format + 1;
}

//===----------------------------------------------------------------------===//
// Integer formatting
//===----------------------------------------------------------------------===//
// Handles %d, %i, %u, %o, %x, %X with all length modifiers and flags.
//
// Approach: extract the value as uint64_t (with sign handling for signed
// specifiers), convert digits into a reverse buffer, then emit with
// padding/prefix/sign.
//
// INT64_MIN correctness: for signed specifiers, the value is cast to unsigned
// BEFORE negation. This avoids undefined behavior from negating the most
// negative value (-(-2^63) overflows signed int64_t, but works as unsigned).

// Maximum digits: 64 binary digits, or 22 octal digits for UINT64_MAX.
// We also need space for the prefix (0x) and sign, but those are handled
// separately — this buffer is just for the digit characters.
#define IREE_PRINTF_INT_BUFFER_SIZE 64

static void iree_printf_format_integer(iree_printf_output_t* out,
                                       const iree_printf_spec_t* spec,
                                       va_list* args) {
  // Determine base and digit case from specifier.
  unsigned int base = 10;
  const char* digits = "0123456789abcdef";
  switch (spec->specifier) {
    case 'o':
      base = 8;
      digits = "0123456789abcdef";
      break;
    case 'x':
      base = 16;
      digits = "0123456789abcdef";
      break;
    case 'X':
      base = 16;
      digits = "0123456789ABCDEF";
      break;
    default:  // d, i, u
      base = 10;
      digits = "0123456789abcdef";
      break;
  }

  bool is_signed = (spec->specifier == 'd' || spec->specifier == 'i');

  // Extract the value from va_args as the correct type. For signed specifiers,
  // track the sign separately and work with the absolute value as unsigned.
  bool is_negative = false;
  uint64_t value = 0;
  if (is_signed) {
    int64_t signed_value = 0;
    switch (spec->length) {
      case IREE_PRINTF_LENGTH_HH:
        // char promoted to int.
        signed_value = (signed char)va_arg(*args, int);
        break;
      case IREE_PRINTF_LENGTH_H:
        // short promoted to int.
        signed_value = (short)va_arg(*args, int);
        break;
      case IREE_PRINTF_LENGTH_L:
        signed_value = va_arg(*args, long);
        break;
      case IREE_PRINTF_LENGTH_LL:
        signed_value = va_arg(*args, long long);
        break;
      case IREE_PRINTF_LENGTH_Z:
        // The 'z' modifier with a signed specifier (%zd) expects ssize_t,
        // which is POSIX-only. Use ptrdiff_t (C99) as a portable equivalent —
        // both are the signed counterpart of size_t on all IREE targets.
        signed_value = va_arg(*args, ptrdiff_t);
        break;
      case IREE_PRINTF_LENGTH_T:
        signed_value = va_arg(*args, ptrdiff_t);
        break;
      case IREE_PRINTF_LENGTH_J:
        signed_value = va_arg(*args, intmax_t);
        break;
      default:
        signed_value = va_arg(*args, int);
        break;
    }
    if (signed_value < 0) {
      is_negative = true;
      // Cast to unsigned BEFORE negation to handle INT64_MIN.
      value = (uint64_t)(-(uint64_t)signed_value);
    } else {
      value = (uint64_t)signed_value;
    }
  } else {
    switch (spec->length) {
      case IREE_PRINTF_LENGTH_HH:
        value = (unsigned char)va_arg(*args, unsigned int);
        break;
      case IREE_PRINTF_LENGTH_H:
        value = (unsigned short)va_arg(*args, unsigned int);
        break;
      case IREE_PRINTF_LENGTH_L:
        value = va_arg(*args, unsigned long);
        break;
      case IREE_PRINTF_LENGTH_LL:
        value = va_arg(*args, unsigned long long);
        break;
      case IREE_PRINTF_LENGTH_Z:
        value = va_arg(*args, size_t);
        break;
      case IREE_PRINTF_LENGTH_T:
        // ptrdiff_t treated as unsigned for %u/%x/%o.
        value = (uint64_t)va_arg(*args, ptrdiff_t);
        break;
      case IREE_PRINTF_LENGTH_J:
        value = va_arg(*args, uintmax_t);
        break;
      default:
        value = va_arg(*args, unsigned int);
        break;
    }
  }

  // Convert value to digit characters in reverse order.
  // Track whether the original value was zero — this affects '#' prefix logic
  // (the C standard says '#' does not add 0x/0 prefix for zero).
  bool is_zero = (value == 0);
  char digit_buffer[IREE_PRINTF_INT_BUFFER_SIZE];
  int digit_count = 0;
  if (value == 0) {
    // Special case: zero. If precision is explicitly 0, no digits are emitted —
    // EXCEPT for octal with '#' flag, where C99 §7.19.6.1p6 requires:
    // "if the value and precision are both 0, a single 0 is printed."
    if (spec->has_precision && spec->precision == 0) {
      if ((spec->flags & IREE_PRINTF_FLAG_HASH) && base == 8) {
        digit_buffer[digit_count++] = '0';
      }
    } else {
      digit_buffer[digit_count++] = '0';
    }
  } else {
    while (value > 0) {
      digit_buffer[digit_count++] = digits[value % base];
      value /= base;
    }
  }

  // Determine the precision-padded digit count (precision specifies minimum
  // number of digits for integers).
  int precision = spec->has_precision ? spec->precision : 1;
  int padded_digit_count = digit_count > precision ? digit_count : precision;

  // Determine the prefix: sign character for signed specifiers, '0x'/'0X' for
  // hex with '#' flag, or combinations thereof.
  const char* prefix = "";
  int prefix_length = 0;

  if ((spec->flags & IREE_PRINTF_FLAG_HASH) && base == 16 && !is_zero) {
    // Hex with '#' flag: "0x"/"0X" prefix.
    // C standard: '#' has no effect on zero value.
    // Note: %x/%X are unsigned specifiers, so no sign prefix is needed.
    prefix = (spec->specifier == 'X') ? "0X" : "0x";
    prefix_length = 2;
  } else if ((spec->flags & IREE_PRINTF_FLAG_HASH) && base == 8 && !is_zero) {
    // Octal '#': ensure there's a leading zero. If precision already provides
    // one, no additional zero is needed. The value==0 case is handled earlier
    // in the digit emission (a single '0' is always emitted for #o with 0).
    // Note: %o is an unsigned specifier, so no sign prefix is needed.
    if (padded_digit_count <= digit_count) {
      padded_digit_count = digit_count + 1;
    }
  } else {
    // No '#' hex/octal prefix — just sign.
    if (is_negative) {
      prefix = "-";
      prefix_length = 1;
    } else if (is_signed && (spec->flags & IREE_PRINTF_FLAG_PLUS)) {
      prefix = "+";
      prefix_length = 1;
    } else if (is_signed && (spec->flags & IREE_PRINTF_FLAG_SPACE)) {
      prefix = " ";
      prefix_length = 1;
    }
  }

  // Total content width: prefix + precision-padded digits.
  int content_width = prefix_length + padded_digit_count;

  // Determine padding. Zero-padding is only used when:
  //   - '0' flag is set
  //   - '-' flag is NOT set (left-justify overrides zero-pad)
  //   - No precision is specified (precision overrides zero-pad for integers)
  bool use_zero_pad = (spec->flags & IREE_PRINTF_FLAG_ZERO) &&
                      !(spec->flags & IREE_PRINTF_FLAG_LEFT) &&
                      !spec->has_precision;

  int padding = 0;
  if (spec->width > content_width) {
    padding = spec->width - content_width;
  }

  if (use_zero_pad) {
    // Zero padding goes between the prefix and digits.
    iree_printf_output_string(out, prefix, prefix_length);
    iree_printf_output_fill(out, '0', padding);
  } else if (!(spec->flags & IREE_PRINTF_FLAG_LEFT)) {
    // Right-justify: space padding before prefix.
    iree_printf_output_fill(out, ' ', padding);
    iree_printf_output_string(out, prefix, prefix_length);
  } else {
    // Left-justify: prefix first, padding at the end.
    iree_printf_output_string(out, prefix, prefix_length);
  }

  // Precision zero-padding (minimum digits).
  int precision_zeros = padded_digit_count - digit_count;
  iree_printf_output_fill(out, '0', precision_zeros);

  // Digits in forward order (they're stored in reverse).
  for (int i = digit_count - 1; i >= 0; i--) {
    iree_printf_output_char(out, digit_buffer[i]);
  }

  // Left-justify trailing spaces.
  if (spec->flags & IREE_PRINTF_FLAG_LEFT) {
    iree_printf_output_fill(out, ' ', padding);
  }
}

//===----------------------------------------------------------------------===//
// String, character, and pointer formatting
//===----------------------------------------------------------------------===//

static void iree_printf_format_string(iree_printf_output_t* out,
                                      const iree_printf_spec_t* spec,
                                      va_list* args) {
  const char* str = va_arg(*args, const char*);
  if (!str) str = "(null)";

  // Determine the output length. Precision limits the number of bytes read
  // from the string — the bounds check MUST precede the dereference.
  size_t length = 0;
  if (spec->has_precision) {
    // Read at most |precision| bytes. Check the bound BEFORE dereferencing.
    size_t max_length = (size_t)spec->precision;
    while (length < max_length && str[length] != '\0') {
      length++;
    }
  } else {
    length = strlen(str);
  }

  // Padding.
  int padding = 0;
  if (spec->width > 0 && (size_t)spec->width > length) {
    padding = spec->width - (int)length;
  }

  if (!(spec->flags & IREE_PRINTF_FLAG_LEFT)) {
    iree_printf_output_fill(out, ' ', padding);
  }

  iree_printf_output_string(out, str, length);

  if (spec->flags & IREE_PRINTF_FLAG_LEFT) {
    iree_printf_output_fill(out, ' ', padding);
  }
}

static void iree_printf_format_char(iree_printf_output_t* out,
                                    const iree_printf_spec_t* spec,
                                    va_list* args) {
  char c = (char)va_arg(*args, int);

  int padding = 0;
  if (spec->width > 1) {
    padding = spec->width - 1;
  }

  if (!(spec->flags & IREE_PRINTF_FLAG_LEFT)) {
    iree_printf_output_fill(out, ' ', padding);
  }

  iree_printf_output_char(out, c);

  if (spec->flags & IREE_PRINTF_FLAG_LEFT) {
    iree_printf_output_fill(out, ' ', padding);
  }
}

static void iree_printf_format_pointer(iree_printf_output_t* out,
                                       const iree_printf_spec_t* spec,
                                       va_list* args) {
  uintptr_t ptr = (uintptr_t)va_arg(*args, void*);

  // Format as "0x" followed by lowercase hex digits. No leading zeros beyond
  // what the value requires (matching glibc behavior).
  // For NULL, we still print "0x0".
  char digit_buffer[IREE_PRINTF_INT_BUFFER_SIZE];
  int digit_count = 0;
  if (ptr == 0) {
    digit_buffer[digit_count++] = '0';
  } else {
    while (ptr > 0) {
      digit_buffer[digit_count++] = "0123456789abcdef"[ptr & 0xF];
      ptr >>= 4;
    }
  }

  int content_width = 2 + digit_count;  // "0x" + digits.
  int padding = 0;
  if (spec->width > content_width) {
    padding = spec->width - content_width;
  }

  if (!(spec->flags & IREE_PRINTF_FLAG_LEFT)) {
    iree_printf_output_fill(out, ' ', padding);
  }

  iree_printf_output_string(out, "0x", 2);
  for (int i = digit_count - 1; i >= 0; i--) {
    iree_printf_output_char(out, digit_buffer[i]);
  }

  if (spec->flags & IREE_PRINTF_FLAG_LEFT) {
    iree_printf_output_fill(out, ' ', padding);
  }
}

//===----------------------------------------------------------------------===//
// Floating-point formatting
//===----------------------------------------------------------------------===//
// Handles %f, %F, %e, %E, %g, %G for double values.
//
// Precision: we do NOT need shortest-round-trip representation (Grisu/Ryu).
// We need correctness to the displayed precision with correct rounding.
//
// Approach: IEEE 754 bit extraction for sign/special values, then arithmetic
// decomposition into integral + fractional parts using int64_t digit
// extraction. This is accurate to 17 significant digits (full double
// precision). Precision is capped at 17; beyond that we pad with zeros.

// Maximum precision we compute accurately. Beyond this, we emit zeros.
// 17 significant digits covers the full precision of IEEE 754 double.
#define IREE_PRINTF_MAX_FLOAT_PRECISION 17

// Maximum effective precision for the fixed-point fractional digit path. For
// small fractional values (e.g., 0.00000001) the effective precision is
// extended beyond IREE_PRINTF_MAX_FLOAT_PRECISION because the leading zeros
// don't consume significant digits. Capped at 19 to keep 10^n within uint64_t.
#define IREE_PRINTF_MAX_FIXED_FRACTION_DIGITS 19

// Maximum digits in the integral part of a double. DBL_MAX is ~1.8e308,
// so 309 digits maximum. We use a buffer large enough for this plus some
// margin for the decimal point, sign, and exponent notation.
#define IREE_PRINTF_FLOAT_BUFFER_SIZE 330

// IEEE 754 double-precision bit layout.
typedef union {
  double value;
  uint64_t bits;
} iree_printf_double_bits_t;

static inline bool iree_printf_double_is_negative(double value) {
  iree_printf_double_bits_t u = {0};
  u.value = value;
  return (u.bits >> 63) != 0;
}

static inline bool iree_printf_double_is_nan(double value) {
  iree_printf_double_bits_t u = {0};
  u.value = value;
  uint64_t exponent = (u.bits >> 52) & 0x7FF;
  uint64_t mantissa = u.bits & 0x000FFFFFFFFFFFFFull;
  return exponent == 0x7FF && mantissa != 0;
}

static inline bool iree_printf_double_is_inf(double value) {
  iree_printf_double_bits_t u = {0};
  u.value = value;
  uint64_t exponent = (u.bits >> 52) & 0x7FF;
  uint64_t mantissa = u.bits & 0x000FFFFFFFFFFFFFull;
  return exponent == 0x7FF && mantissa == 0;
}

// Compute floor(log10(value)) for positive, finite, non-zero values.
// Uses the base-2 exponent from IEEE 754 bits and a correction factor.
// Accuracy: exact for most values; may be off by 1, which the caller corrects.
//
// Subnormals require special handling: the biased exponent is 0 for ALL
// subnormals (spanning ~5e-324 to ~2.2e-308), so the normal approximation
// collapses them all to the same estimate. For subnormals, we find the
// highest set bit in the mantissa to compute the effective binary exponent.
static int iree_printf_log10_approx(double value) {
  iree_printf_double_bits_t u = {0};
  u.value = value;
  int exponent_biased = (int)((u.bits >> 52) & 0x7FF);
  int effective_log2 = 0;

  if (exponent_biased == 0) {
    // Subnormal: value = 2^(-1022) * (mantissa / 2^52).
    // Effective binary exponent = -1074 + position_of_highest_set_bit.
    uint64_t mantissa = u.bits & 0x000FFFFFFFFFFFFFull;
    if (mantissa == 0) return 0;  // ±0, shouldn't reach here.
    // Count leading zeros in the 52-bit mantissa field.
    // Start from bit 51 (MSB of mantissa) and count down.
    int highest_bit = 0;
    for (int bit = 51; bit >= 0; bit--) {
      if (mantissa & (1ull << bit)) {
        highest_bit = bit;
        break;
      }
    }
    effective_log2 = -1074 + highest_bit;
  } else {
    effective_log2 = exponent_biased - 1023;
  }

  // log10(2) ≈ 0.30103. Use integer arithmetic: log10 ≈ log2 * 301 / 1000.
  // For negative values, C integer division truncates toward zero, which gives
  // us ceil(log10) instead of floor(log10). Correct by subtracting 1 when the
  // product is negative and not evenly divisible.
  int product = effective_log2 * 301;
  int log10_estimate = product / 1000;
  if (product < 0 && product % 1000 != 0) {
    log10_estimate--;
  }
  return log10_estimate;
}

// Powers of 10 as doubles, for exponents 0..22. Beyond 22, 10^n is not exactly
// representable as a double, but we only need this for normalization where
// small errors are acceptable (they affect the last displayed digit at most).
static const double kPowersOf10[] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
};

static double iree_printf_pow10(int n) {
  // Keep the table lookup behind a simple checked unsigned magnitude. GCC 11 at
  // -O3 can otherwise warn with a false-positive -Warray-bounds after inlining
  // and constprop when it reasons about wrapped unsigned values.
  unsigned int abs_n = n < 0 ? -(unsigned int)n : (unsigned int)n;
  if (abs_n <= 22) {
    return n < 0 ? 1.0 / kPowersOf10[abs_n] : kPowersOf10[abs_n];
  }

  // For large exponents, use iterative multiplication/division.
  // This loses precision in the last few digits but is fine for our use case
  // (debug formatting at precision <= 17).
  double result = 1.0;
  if (n > 0) {
    double base = 10.0;
    int exp = n;
    while (exp > 0) {
      if (exp & 1) result *= base;
      base *= base;
      exp >>= 1;
    }
  } else {
    double base = 10.0;
    int exp = -n;
    while (exp > 0) {
      if (exp & 1) result /= base;
      base *= base;
      exp >>= 1;
    }
  }
  return result;
}

// Compute value * 10^n with FMA-corrected error tracking (double-double).
// Stores the high part in *out_result and returns the low part (error term),
// such that *out_result + return_value = value * 10^n to extended precision.
//
// For n <= 22, pow10(n) is exactly representable as a double, so a single FMA
// gives the exact error of the multiplication.
//
// For 22 < n <= 44, the multiplication is split into two exact steps:
//   value * 10^n = (value * 10^22) * 10^(n-22)
// Both 10^22 and 10^(n-22) are exactly representable (both <= 10^22). FMA
// error terms from each step are combined for double-double precision.
//
// For n > 44, falls back to single-step with inexact pow10 (best-effort).
static double iree_printf_mul_pow10_error(double value, int n,
                                          double* out_result) {
  if (n <= 22) {
    double scale = iree_printf_pow10(n);
    *out_result = value * scale;
    return iree_printf_mul_error(value, scale, *out_result);
  } else if (n <= 44) {
    // Split 10^n = 10^22 * 10^(n-22) where both factors are exact doubles.
    double scale_first = iree_printf_pow10(22);
    double scale_second = iree_printf_pow10(n - 22);
    // Step 1: value * 10^22 with error tracking.
    double partial = value * scale_first;
    double error_first = iree_printf_mul_error(value, scale_first, partial);
    // Step 2: partial * 10^(n-22) with error tracking.
    *out_result = partial * scale_second;
    double error_second =
        iree_printf_mul_error(partial, scale_second, *out_result);
    // Combined error: the second step's error plus the first step's error
    // propagated through the second multiplication.
    return error_second + error_first * scale_second;
  } else {
    // n > 44: pow10(n) is not exactly representable, so error correction
    // handles only the multiplication rounding, not the scale factor.
    // Best-effort.
    double scale = iree_printf_pow10(n);
    *out_result = value * scale;
    return iree_printf_mul_error(value, scale, *out_result);
  }
}

// Format a non-negative, finite double in %f style (fixed-point notation).
// Writes computed digits into |buffer| and returns the number of characters
// written. |buffer| must have at least IREE_PRINTF_FLOAT_BUFFER_SIZE bytes.
//
// Trailing fractional zeros beyond the computational precision limit (17
// significant digits) are NOT written to the buffer. Instead, their count is
// stored in |*out_trailing_zeros|. The caller must emit these zeros directly
// to the output after the buffer contents. This prevents stack buffer overflow
// when precision is large (e.g., %.400f would write 400 bytes of zeros into a
// 330-byte buffer without this separation).
static int iree_printf_format_fixed(char* buffer, double value, int precision,
                                    bool force_decimal_point,
                                    int* out_trailing_zeros) {
  *out_trailing_zeros = 0;
  int position = 0;

  // Split into integral and fractional parts.
  double integral_part = 0.0;
  double fractional_part = iree_printf_modf(value, &integral_part);

  // Clamp precision to what we can compute accurately. The base limit is 17
  // significant digits (full double precision), but for fractional parts with
  // leading zeros (e.g., 0.0005...), the precision in "digits after decimal
  // point" includes leading zeros that don't consume significant digits. In
  // those cases, higher effective_precision is safe because the scaled result
  // (frac * 10^n) is still small enough to fit exactly in double (< 2^53).
  int effective_precision = precision;
  if (effective_precision > IREE_PRINTF_MAX_FLOAT_PRECISION) {
    effective_precision = IREE_PRINTF_MAX_FLOAT_PRECISION;
    // Extend precision for small fractional parts where the scaled result
    // still fits in double precision (< 2^53 ~= 9e15). Also cap at 19 to
    // prevent 10^effective_precision from overflowing uint64_t.
    if (fractional_part > 0.0) {
      while (effective_precision < precision &&
             effective_precision < IREE_PRINTF_MAX_FIXED_FRACTION_DIGITS) {
        double next_scale = iree_printf_pow10(effective_precision + 1);
        if (fractional_part * next_scale >= 9.0e15) break;
        effective_precision++;
      }
    }
  }

  // Convert fractional part to integer digits by multiplying by 10^precision.
  // This gives us the fractional digits as an integer that we can extract.
  double frac_scale = iree_printf_pow10(effective_precision);
  double frac_scaled = fractional_part * frac_scale;
  uint64_t frac_max = (uint64_t)frac_scale;

  // Round with banker's rounding (round-half-to-even). When the fractional
  // remainder is exactly 0.5, round to the nearest even digit. The "digit"
  // that matters is the last displayed digit: the lowest fractional digit if
  // precision > 0, or the ones digit of the integral part if precision == 0.
  //
  // Correct the multiplication rounding error: the product frac * 10^n may
  // round to a value with remainder exactly 0.5 when the true remainder is
  // near but not equal to 0.5.
  double mul_error =
      iree_printf_mul_error(fractional_part, frac_scale, frac_scaled);
  uint64_t frac_digits = (uint64_t)frac_scaled;
  double remainder = (frac_scaled - (double)frac_digits) + mul_error;
  if (remainder > 0.5) {
    frac_digits++;
  } else if (remainder == 0.5) {
    // Exactly half: determine which digit to check for evenness.
    if (effective_precision > 0) {
      // The last fractional digit determines rounding.
      if (frac_digits & 1) frac_digits++;
    } else {
      // Precision is 0: the ones digit of the integral part determines
      // rounding. frac_digits is 0 here (precision 0 means no frac digits).
      uint64_t int_ones = (uint64_t)integral_part % 10;
      if (int_ones & 1) frac_digits++;  // Will carry into integral_part.
    }
  }

  // Check for carry from rounding: if frac_digits == 10^precision, we need to
  // increment the integral part.
  if (frac_digits >= frac_max) {
    frac_digits = 0;
    integral_part += 1.0;
  }

  // Format the integral part.
  if (integral_part == 0.0) {
    buffer[position++] = '0';
  } else if (integral_part < 18446744073709551616.0) {  // < 2^64
    // Value fits in uint64_t — extract digits directly for full precision.
    char int_digits[IREE_PRINTF_FLOAT_BUFFER_SIZE];
    int int_count = 0;
    uint64_t int_value = (uint64_t)integral_part;
    while (int_value > 0) {
      int_digits[int_count++] = '0' + (char)(int_value % 10);
      int_value /= 10;
    }
    for (int i = int_count - 1; i >= 0; i--) {
      buffer[position++] = int_digits[i];
    }
  } else {
    // Very large float (> UINT64_MAX, ~1.8e19). IEEE 754 double has at most
    // 17 significant digits; beyond that the digits are determined by the
    // binary-to-decimal conversion, not the original value. We extract the
    // significant digits via normalization and pad the rest with zeros —
    // this matches what libc printf does.
    int exponent = iree_printf_log10_approx(integral_part);
    double normalized = integral_part / iree_printf_pow10(exponent);
    // Correct approximation.
    if (normalized >= 10.0) {
      normalized /= 10.0;
      exponent++;
    } else if (normalized < 1.0) {
      normalized *= 10.0;
      exponent--;
    }

    // Extract up to 17 significant digits from the normalized value.
    int sig_digits = IREE_PRINTF_MAX_FLOAT_PRECISION;
    if (sig_digits > exponent + 1) sig_digits = exponent + 1;
    // Scale to get all significant digits as an integer.
    double scaled = normalized * iree_printf_pow10(sig_digits - 1);
    uint64_t sig_value = (uint64_t)(scaled + 0.5);

    // Extract digits in reverse.
    char sig_buffer[IREE_PRINTF_MAX_FLOAT_PRECISION + 1];
    int sig_count = 0;
    while (sig_value > 0) {
      sig_buffer[sig_count++] = '0' + (char)(sig_value % 10);
      sig_value /= 10;
    }
    // Emit significant digits in forward order.
    for (int i = sig_count - 1; i >= 0; i--) {
      buffer[position++] = sig_buffer[i];
    }
    // Pad remaining digits with zeros.
    int zero_pad_count = (exponent + 1) - sig_count;
    for (int i = 0; i < zero_pad_count; i++) {
      buffer[position++] = '0';
    }
  }

  // Decimal point and fractional digits.
  if (precision > 0 || force_decimal_point) {
    buffer[position++] = '.';
  }
  if (precision > 0) {
    // Format fractional digits with leading zeros. The extended precision path
    // can push effective_precision up to IREE_PRINTF_MAX_FIXED_FRACTION_DIGITS
    // for small fractional values, so the buffer must accommodate that.
    char frac_buffer[IREE_PRINTF_MAX_FIXED_FRACTION_DIGITS + 1];
    int frac_count = 0;
    uint64_t frac_remaining = frac_digits;
    for (int i = 0; i < effective_precision; i++) {
      frac_buffer[frac_count++] = '0' + (char)(frac_remaining % 10);
      frac_remaining /= 10;
    }
    // Emit fractional digits in forward order (they're stored in reverse).
    for (int i = frac_count - 1; i >= 0; i--) {
      buffer[position++] = frac_buffer[i];
    }
    // Track trailing zeros needed beyond our computational limit. These are
    // NOT written to the buffer — the caller emits them directly to the
    // output to avoid overflowing the fixed-size stack buffer.
    if (precision > effective_precision) {
      *out_trailing_zeros = precision - effective_precision;
    }
  }

  return position;
}

// Format a non-negative, finite double in %e style (exponential notation).
// Writes into |buffer| and returns the number of characters written.
//
// Trailing fractional zeros beyond the computational precision limit are NOT
// written to the buffer; their count is stored in |*out_trailing_zeros|.
// The caller must emit: buffer[0..mantissa_end], trailing zeros,
// buffer[mantissa_end..end] (the exponent suffix). The split point is stored
// in |*out_exponent_offset|.
//
// Approach: compute all significant digits at once as a single integer via a
// single division, rather than normalizing to [1,10) and routing through the
// fixed-point formatter. The normalize → modf → multiply chain amplifies
// representation error at rounding boundaries (e.g., 575537150/1e8 produces
// 5.75537149999... instead of 5.7553715, flipping a banker's rounding
// decision). By dividing by 10^(exponent - precision) instead, we keep the
// full precision of the original value in a single operation.
static int iree_printf_format_exponential(
    char* buffer, double value, int precision, bool force_decimal_point,
    bool uppercase, int* out_trailing_zeros, int* out_exponent_offset) {
  *out_trailing_zeros = 0;
  int position = 0;

  if (value == 0.0) {
    // Zero: 0.000...e+00.
    buffer[position++] = '0';
    if (precision > 0 || force_decimal_point) {
      buffer[position++] = '.';
      // Write zeros up to our buffer limit; track the rest as trailing.
      int zeros_to_write = precision;
      if (zeros_to_write > IREE_PRINTF_MAX_FLOAT_PRECISION) {
        *out_trailing_zeros = zeros_to_write - IREE_PRINTF_MAX_FLOAT_PRECISION;
        zeros_to_write = IREE_PRINTF_MAX_FLOAT_PRECISION;
      }
      for (int i = 0; i < zeros_to_write; i++) {
        buffer[position++] = '0';
      }
    }
    *out_exponent_offset = position;
    buffer[position++] = uppercase ? 'E' : 'e';
    buffer[position++] = '+';
    buffer[position++] = '0';
    buffer[position++] = '0';
    return position;
  }

  // Compute the base-10 exponent.
  int exponent = iree_printf_log10_approx(value);

  // We need (precision + 1) significant digits total: 1 before the decimal
  // point and |precision| after. Clamp to our computational limit.
  int sig_count = precision + 1;
  int effective_sig_count = sig_count;
  if (effective_sig_count > IREE_PRINTF_MAX_FLOAT_PRECISION) {
    effective_sig_count = IREE_PRINTF_MAX_FLOAT_PRECISION;
  }

  // Scale the value so that its significant digits are an integer with
  // exactly |effective_sig_count| digits: scaled ≈ value * 10^n where n is
  // chosen to shift the significant digits above the decimal point.
  //
  // For normal doubles (exponents roughly -290 to +290), we do this in a
  // single division/multiplication by 10^|scale_exponent|, which preserves
  // full precision. For extreme exponents (subnormals, huge values near
  // DBL_MAX), the scale factor itself would overflow double range, so we
  // normalize in steps: first bring the value to [1, 10), then scale the
  // normalized value by 10^(sig_count - 1).
  uint64_t sig_max = (uint64_t)iree_printf_pow10(effective_sig_count);
  uint64_t sig_min = sig_max / 10;
  int scale_exponent = exponent - effective_sig_count + 1;
  double scaled = 0.0;

  if (scale_exponent > 290 || scale_exponent < -290) {
    // Extreme exponent: normalize to [1, 10) in steps of at most 10^22
    // (the largest power of 10 exactly representable as a double), then
    // extract digits from the normalized value. This is slightly less
    // precise than the single-step approach but doubles only have ~15.9
    // decimal digits of precision anyway — the last 1-2 digits of a
    // subnormal or extreme value are noise regardless.
    double normalized = value;
    int remaining = exponent;
    if (remaining > 0) {
      while (remaining > 22) {
        normalized /= 1e22;
        remaining -= 22;
      }
      normalized /= iree_printf_pow10(remaining);
    } else if (remaining < 0) {
      remaining = -remaining;
      while (remaining > 22) {
        normalized *= 1e22;
        remaining -= 22;
      }
      normalized *= iree_printf_pow10(remaining);
    }
    // Correct for approximation errors.
    while (normalized >= 10.0) {
      normalized /= 10.0;
      exponent++;
    }
    while (normalized < 1.0 && normalized > 0.0) {
      normalized *= 10.0;
      exponent--;
    }
    // Now normalized is in [1, 10). Scale to get sig_count digits.
    scaled = normalized * iree_printf_pow10(effective_sig_count - 1);
  } else {
    // Normal case: single-step scaling preserves full precision.
    if (scale_exponent >= 0) {
      scaled = value / iree_printf_pow10(scale_exponent);
    } else {
      scaled = value * iree_printf_pow10(-scale_exponent);
    }
  }

  // Round to nearest integer with banker's rounding (round-half-to-even).
  //
  // The naive approach (look at `scaled - floor(scaled)`) is unreliable
  // because the division `value / 10^k` introduces up to 0.5 ULP error,
  // which can turn a 0.4999... remainder into 0.5000..., flipping the
  // rounding direction. This affects values larger than ~2^50 where the
  // double division's rounding error is significant at the unit level.
  //
  // For the normal-range case (where we divided by 10^k), compute the exact
  // remainder via double-double arithmetic: verify the truncated quotient
  // with a double-double product, correct if the division rounded up, then
  // use the double-double residual for the rounding decision.
  uint64_t sig_integer = 0;
  if (scale_exponent > 0 && scale_exponent <= 22) {
    // Division path: scaled = value / 10^k where 10^k is exactly representable.
    //
    // For the truncated quotient (floor), use direct division and then verify
    // with a double-double product. IEEE 754 correctly-rounded division can
    // round the quotient up past an integer boundary when the true fractional
    // part is close to 1.0 (specifically, when frac > 1.0 - 0.5*ULP at the
    // quotient's magnitude). When this happens, (uint64_t)(value/divisor)
    // gives Q+1 instead of Q. We detect and correct this by computing
    // sig_integer * divisor as a double-double and checking if it exceeds
    // value. After correction, the double-double residual gives the exact
    // remainder for the rounding decision.
    double divisor = iree_printf_pow10(scale_exponent);
    double quotient = value / divisor;
    sig_integer = (uint64_t)quotient;
    // Verify the floor: double-double product sig_integer * divisor. If the
    // product exceeds value, the division rounded up past the true floor.
    double product_high = (double)sig_integer * divisor;
    double product_low =
        iree_printf_mul_error((double)sig_integer, divisor, product_high);
    double residual = (value - product_high) - product_low;
    if (residual < 0) {
      // Division rounded up: true floor is sig_integer - 1. Adjust the
      // remainder algebraically rather than recomputing via double-double
      // (recomputing would require (double)sig_integer, which loses precision
      // when sig_integer > 2^53).
      sig_integer--;
      residual += divisor;
    }
    double half = divisor * 0.5;
    if (residual > half) {
      sig_integer++;
    } else if (residual == half) {
      if (sig_integer & 1) sig_integer++;  // Banker's rounding.
    }
  } else if (scale_exponent < 0 && scale_exponent >= -290) {
    // Multiplication path: scaled = value * 10^|n|. The multiplication can
    // introduce rounding error from two sources: (1) the multiplication itself
    // (up to 0.5 ULP), and (2) the pow10 scale factor being inexact for n > 22.
    // Use iree_printf_mul_pow10_error to get a double-double result that
    // corrects both sources (splitting into exact factors for 22 < n <= 44).
    double mul_result = 0.0;
    double mul_error =
        iree_printf_mul_pow10_error(value, -scale_exponent, &mul_result);
    sig_integer = (uint64_t)mul_result;
    double sig_remainder = (mul_result - (double)sig_integer) + mul_error;
    // The combined FMA error from the two-step multiplication can push the
    // true remainder outside [0, 1). A negative remainder means the true
    // product is below sig_integer; a remainder >= 1.0 means it is at least
    // 1 above sig_integer. Normalize into [0, 1) before rounding.
    while (sig_remainder >= 1.0) {
      sig_integer++;
      sig_remainder -= 1.0;
    }
    while (sig_remainder < 0.0) {
      sig_integer--;
      sig_remainder += 1.0;
    }
    if (sig_remainder > 0.5) {
      sig_integer++;
    } else if (sig_remainder == 0.5) {
      if (sig_integer & 1) sig_integer++;
    }
  } else {
    // Extreme exponents (multi-step normalization path): the value has been
    // normalized through multiple divisions/multiplications, so FMA-based
    // error correction isn't applicable. Direct rounding is sufficient since
    // extreme-exponent values have inherently limited precision.
    sig_integer = (uint64_t)scaled;
    double sig_remainder = scaled - (double)sig_integer;
    if (sig_remainder > 0.5) {
      sig_integer++;
    } else if (sig_remainder == 0.5) {
      if (sig_integer & 1) sig_integer++;
    }
  }

  // Handle exponent off by 1 in either direction. The log10 approximation can
  // be off by 1, producing too many or too few digits. In either case,
  // recompute with the corrected exponent so that rounding happens at the
  // correct precision.
  //
  // This also handles carry from rounding (9999999.5 → 10000000): recomputing
  // with exponent+1 gives the same correct result as truncation would, but
  // without losing the rounding decision.
  //
  // For the recomputation we always use the single-step approach. The
  // correction is at most ±1, so the new scale_exponent is within 1 of the
  // old one — if the old one was in range, so is the new one. If we used the
  // multi-step path (extreme exponent), the correction brings us back to the
  // multi-step regime. To keep things simple, use the step-based normalization
  // for recomputation when the new scale_exponent is extreme.
  if (sig_integer >= sig_max || (sig_integer < sig_min && sig_integer > 0)) {
    if (sig_integer >= sig_max) {
      exponent++;
    } else {
      exponent--;
    }
    scale_exponent = exponent - effective_sig_count + 1;
    if (scale_exponent > 290 || scale_exponent < -290) {
      // Extreme: re-normalize in steps.
      double normalized = value;
      int remaining = exponent;
      if (remaining > 0) {
        while (remaining > 22) {
          normalized /= 1e22;
          remaining -= 22;
        }
        normalized /= iree_printf_pow10(remaining);
      } else if (remaining < 0) {
        remaining = -remaining;
        while (remaining > 22) {
          normalized *= 1e22;
          remaining -= 22;
        }
        normalized *= iree_printf_pow10(remaining);
      }
      while (normalized >= 10.0) {
        normalized /= 10.0;
        exponent++;
      }
      while (normalized < 1.0 && normalized > 0.0) {
        normalized *= 10.0;
        exponent--;
      }
      scaled = normalized * iree_printf_pow10(effective_sig_count - 1);
    } else {
      if (scale_exponent >= 0) {
        scaled = value / iree_printf_pow10(scale_exponent);
      } else {
        scaled = value * iree_printf_pow10(-scale_exponent);
      }
    }
    sig_integer = (uint64_t)scaled;
    // Use precise rounding for the recomputed value (same logic as above).
    if (scale_exponent > 0 && scale_exponent <= 22) {
      // Division path: direct quotient + double-double remainder rounding.
      double divisor = iree_printf_pow10(scale_exponent);
      double quotient = value / divisor;
      sig_integer = (uint64_t)quotient;
      double product_high = (double)sig_integer * divisor;
      double product_low =
          iree_printf_mul_error((double)sig_integer, divisor, product_high);
      double residual = (value - product_high) - product_low;
      if (residual < 0) {
        // Same algebraic correction as the main division path above.
        sig_integer--;
        residual += divisor;
      }
      double half = divisor * 0.5;
      if (residual > half) {
        sig_integer++;
      } else if (residual == half) {
        if (sig_integer & 1) sig_integer++;
      }
    } else if (scale_exponent < 0 && scale_exponent >= -290) {
      // Multiplication path recomputation: use two-step FMA (same as above).
      double mul_result = 0.0;
      double mul_error =
          iree_printf_mul_pow10_error(value, -scale_exponent, &mul_result);
      sig_integer = (uint64_t)mul_result;
      double sig_remainder = (mul_result - (double)sig_integer) + mul_error;
      while (sig_remainder >= 1.0) {
        sig_integer++;
        sig_remainder -= 1.0;
      }
      while (sig_remainder < 0.0) {
        sig_integer--;
        sig_remainder += 1.0;
      }
      if (sig_remainder > 0.5) {
        sig_integer++;
      } else if (sig_remainder == 0.5) {
        if (sig_integer & 1) sig_integer++;
      }
    } else {
      double sig_remainder = scaled - (double)sig_integer;
      if (sig_remainder > 0.5) {
        sig_integer++;
      } else if (sig_remainder == 0.5) {
        if (sig_integer & 1) sig_integer++;
      }
    }
    // After correction, carry is still possible (value at exact power-of-10
    // boundary). Use a loop rather than a single check: while a single carry
    // (sig_integer == sig_max) is the expected case, a loop is defensive
    // against hypothetical multi-digit overshoots from compounding FMA errors.
    while (sig_integer >= sig_max) {
      sig_integer /= 10;
      exponent++;
    }
  }

  // Extract digits from the integer in reverse order.
  char sig_digits[IREE_PRINTF_MAX_FLOAT_PRECISION + 1];
  int digit_count = 0;
  uint64_t temp = sig_integer;
  while (temp > 0) {
    sig_digits[digit_count++] = '0' + (char)(temp % 10);
    temp /= 10;
  }
  // Pad to effective_sig_count if needed (e.g., value was 1.000000).
  while (digit_count < effective_sig_count) {
    sig_digits[digit_count++] = '0';
  }

  // Emit the leading digit (most significant).
  buffer[position++] = sig_digits[digit_count - 1];

  // Decimal point and fractional digits.
  if (precision > 0 || force_decimal_point) {
    buffer[position++] = '.';
  }
  // Emit computed significant digits after the decimal point.
  int frac_digits_emitted = 0;
  for (int i = digit_count - 2; i >= 0 && frac_digits_emitted < precision;
       i--, frac_digits_emitted++) {
    buffer[position++] = sig_digits[i];
  }
  // Track trailing zeros needed beyond our computational limit. These are NOT
  // written to the buffer — the caller inserts them between the mantissa and
  // the exponent suffix to avoid overflowing the fixed-size stack buffer.
  if (precision > frac_digits_emitted) {
    *out_trailing_zeros = precision - frac_digits_emitted;
  }

  // Record where the exponent suffix starts so the caller can insert trailing
  // zeros before it.
  *out_exponent_offset = position;

  // Exponent: e+dd (at least 2 digits, 3 for exponents >= 100).
  buffer[position++] = uppercase ? 'E' : 'e';
  if (exponent >= 0) {
    buffer[position++] = '+';
  } else {
    buffer[position++] = '-';
    exponent = -exponent;
  }
  if (exponent >= 100) {
    buffer[position++] = '0' + (char)(exponent / 100);
    buffer[position++] = '0' + (char)((exponent / 10) % 10);
    buffer[position++] = '0' + (char)(exponent % 10);
  } else {
    buffer[position++] = '0' + (char)(exponent / 10);
    buffer[position++] = '0' + (char)(exponent % 10);
  }

  return position;
}

// Entry point for floating-point formatting. Handles sign, special values
// (NaN, Inf), and dispatches to the appropriate formatter.
static void iree_printf_format_float(iree_printf_output_t* out,
                                     const iree_printf_spec_t* spec,
                                     va_list* args) {
  double value = va_arg(*args, double);

  bool uppercase = (spec->specifier == 'F' || spec->specifier == 'E' ||
                    spec->specifier == 'G');

  // Handle sign.
  bool is_negative = iree_printf_double_is_negative(value);
  const char* sign_prefix = "";
  int sign_length = 0;
  if (is_negative) {
    sign_prefix = "-";
    sign_length = 1;
    value = -value;  // Work with absolute value.
  } else if (spec->flags & IREE_PRINTF_FLAG_PLUS) {
    sign_prefix = "+";
    sign_length = 1;
  } else if (spec->flags & IREE_PRINTF_FLAG_SPACE) {
    sign_prefix = " ";
    sign_length = 1;
  }

  // Handle special values: NaN and Inf.
  if (iree_printf_double_is_nan(value)) {
    const char* text = uppercase ? "NAN" : "nan";
    int content_width = sign_length + 3;
    int padding = 0;
    if (spec->width > content_width) padding = spec->width - content_width;

    if (!(spec->flags & IREE_PRINTF_FLAG_LEFT)) {
      iree_printf_output_fill(out, ' ', padding);
    }
    iree_printf_output_string(out, sign_prefix, sign_length);
    iree_printf_output_string(out, text, 3);
    if (spec->flags & IREE_PRINTF_FLAG_LEFT) {
      iree_printf_output_fill(out, ' ', padding);
    }
    return;
  }

  if (iree_printf_double_is_inf(value)) {
    const char* text = uppercase ? "INF" : "inf";
    int content_width = sign_length + 3;
    int padding = 0;
    if (spec->width > content_width) padding = spec->width - content_width;

    if (!(spec->flags & IREE_PRINTF_FLAG_LEFT)) {
      iree_printf_output_fill(out, ' ', padding);
    }
    iree_printf_output_string(out, sign_prefix, sign_length);
    iree_printf_output_string(out, text, 3);
    if (spec->flags & IREE_PRINTF_FLAG_LEFT) {
      iree_printf_output_fill(out, ' ', padding);
    }
    return;
  }

  // Default precision is 6.
  int precision = spec->has_precision ? spec->precision : 6;
  bool force_decimal_point = (spec->flags & IREE_PRINTF_FLAG_HASH) != 0;

  // Format into a temporary buffer. The formatters may report trailing zeros
  // that should NOT be written to the buffer (to avoid stack overflow with
  // large precisions). These are emitted directly to the output.
  char buffer[IREE_PRINTF_FLOAT_BUFFER_SIZE];
  int length = 0;
  int trailing_zeros = 0;
  // For %e: the buffer contains [mantissa][exponent_suffix]. Trailing zeros
  // must be inserted between the mantissa and exponent parts.
  int exponent_offset = 0;
  bool has_exponent_suffix = false;
  char specifier = spec->specifier;
  // Normalize case for dispatch.
  if (specifier == 'F') specifier = 'f';
  if (specifier == 'E') specifier = 'e';
  if (specifier == 'G') specifier = 'g';

  if (specifier == 'f') {
    length = iree_printf_format_fixed(buffer, value, precision,
                                      force_decimal_point, &trailing_zeros);
  } else if (specifier == 'e') {
    length = iree_printf_format_exponential(buffer, value, precision,
                                            force_decimal_point, uppercase,
                                            &trailing_zeros, &exponent_offset);
    has_exponent_suffix = true;
  } else {
    // %g: choose between %f and %e based on the exponent.
    // Use %e if exponent < -4 or exponent >= precision.
    // For %g, precision means "significant digits", not "digits after decimal".
    int sig_precision = precision;
    if (sig_precision == 0) sig_precision = 1;  // %g with precision 0 is 1.

    int exponent = 0;
    if (value != 0.0) {
      exponent = iree_printf_log10_approx(value);
      // Correct approximation.
      double check = value / iree_printf_pow10(exponent);
      if (check >= 10.0) {
        exponent++;
      } else if (check < 1.0) {
        exponent--;
      }
      // The C standard says the %g routing decision uses the exponent *after*
      // rounding to sig_precision significant digits. If rounding carries over
      // (e.g., 9.5 rounded to 1 sig digit becomes 10), the exponent increases.
      // Use multiplication (not division) with error correction to get the
      // exact scaled significand, then check if rounding carries.
      int mul_exp = sig_precision - exponent - 1;
      if (mul_exp >= 0 && mul_exp <= 22) {
        double scale = iree_printf_pow10(mul_exp);
        double scaled = value * scale;
        double mul_err = iree_printf_mul_error(value, scale, scaled);
        double int_part = iree_printf_floor(scaled);
        double frac = (scaled - int_part) + mul_err;
        if (frac < 0) {
          int_part -= 1.0;
          frac += 1.0;
        }
        bool rounds_up =
            frac > 0.5 || (frac == 0.5 && iree_printf_is_odd(int_part));
        if (rounds_up && (int_part + 1.0) >= iree_printf_pow10(sig_precision)) {
          exponent++;
        }
      } else if (mul_exp < 0 && mul_exp >= -22) {
        // Division path: value * 10^(-|mul_exp|) would lose precision, so
        // divide instead. Division by exact pow10 (|mul_exp| <= 22) is safe.
        // Use double-double product to get exact remainder (avoids precision
        // loss when int_part > 2^53).
        double divisor = iree_printf_pow10(-mul_exp);
        double quotient = value / divisor;
        double int_part = iree_printf_floor(quotient);
        double product_high = int_part * divisor;
        double product_low =
            iree_printf_mul_error(int_part, divisor, product_high);
        double remainder = (value - product_high) - product_low;
        if (remainder < 0) {
          int_part -= 1.0;
          remainder += divisor;
        }
        double half = divisor * 0.5;
        bool rounds_up = remainder > half ||
                         (remainder == half && iree_printf_is_odd(int_part));
        if (rounds_up && (int_part + 1.0) >= iree_printf_pow10(sig_precision)) {
          exponent++;
        }
      }
    }

    if (exponent < -4 || exponent >= sig_precision) {
      // Use exponential notation. Precision for %e is sig_precision - 1
      // (significant digits minus the one before the decimal point).
      length = iree_printf_format_exponential(
          buffer, value, sig_precision - 1, force_decimal_point, uppercase,
          &trailing_zeros, &exponent_offset);
      has_exponent_suffix = true;
    } else {
      // Use fixed notation. Precision for %f is sig_precision - exponent - 1
      // (digits after the decimal point to get the right significant digits).
      int f_precision = sig_precision - exponent - 1;
      if (f_precision < 0) f_precision = 0;
      length = iree_printf_format_fixed(buffer, value, f_precision,
                                        force_decimal_point, &trailing_zeros);
    }

    // Strip trailing zeros after the decimal point (unless '#' flag).
    // Any virtual trailing zeros (not in the buffer) are also stripped.
    if (!force_decimal_point) {
      trailing_zeros = 0;
      // Find the decimal point and exponent marker positions.
      int decimal_position = -1;
      int exponent_position = -1;
      for (int i = 0; i < length; i++) {
        if (buffer[i] == '.') decimal_position = i;
        if (buffer[i] == 'e' || buffer[i] == 'E') {
          exponent_position = i;
          break;
        }
      }
      if (decimal_position >= 0) {
        int strip_end = (exponent_position >= 0) ? exponent_position : length;
        while (strip_end > decimal_position + 1 &&
               buffer[strip_end - 1] == '0') {
          strip_end--;
        }
        // Also strip the decimal point if all fractional digits were removed.
        if (strip_end == decimal_position + 1) {
          strip_end = decimal_position;
        }
        if (exponent_position >= 0) {
          // Move the exponent part down.
          int exponent_part_length = length - exponent_position;
          memmove(buffer + strip_end, buffer + exponent_position,
                  exponent_part_length);
          length = strip_end + exponent_part_length;
          exponent_offset = strip_end;
        } else {
          length = strip_end;
        }
      }
    }
  }

  // Now emit with sign and padding.
  // Total content width includes the buffer contents plus any trailing zeros
  // that aren't in the buffer.
  int content_width = sign_length + length + trailing_zeros;
  int padding = 0;
  if (spec->width > content_width) {
    padding = spec->width - content_width;
  }

  bool use_zero_pad = (spec->flags & IREE_PRINTF_FLAG_ZERO) &&
                      !(spec->flags & IREE_PRINTF_FLAG_LEFT);

  if (use_zero_pad) {
    iree_printf_output_string(out, sign_prefix, sign_length);
    iree_printf_output_fill(out, '0', padding);
  } else if (!(spec->flags & IREE_PRINTF_FLAG_LEFT)) {
    iree_printf_output_fill(out, ' ', padding);
    iree_printf_output_string(out, sign_prefix, sign_length);
  } else {
    iree_printf_output_string(out, sign_prefix, sign_length);
  }

  // Emit the buffer contents with trailing zeros in the right position.
  // For %e/%E: trailing zeros go between the mantissa and the exponent suffix.
  // For %f/%F: trailing zeros go at the end.
  if (has_exponent_suffix && trailing_zeros > 0) {
    iree_printf_output_string(out, buffer, exponent_offset);
    iree_printf_output_fill(out, '0', trailing_zeros);
    iree_printf_output_string(out, buffer + exponent_offset,
                              length - exponent_offset);
  } else {
    iree_printf_output_string(out, buffer, length);
    iree_printf_output_fill(out, '0', trailing_zeros);
  }

  if (spec->flags & IREE_PRINTF_FLAG_LEFT) {
    iree_printf_output_fill(out, ' ', padding);
  }
}

//===----------------------------------------------------------------------===//
// Main format dispatcher
//===----------------------------------------------------------------------===//

// Format a string according to |format| with arguments from |args|.
// Writes output through |out|. Returns the total number of characters that
// would be written (excluding NUL), or -1 on format error.
static int iree_printf_format(iree_printf_output_t* out, const char* format,
                              va_list args) {
  // We need a mutable copy of the va_list because the spec parser consumes
  // arguments (for '*' width/precision).
  va_list args_copy;
  va_copy(args_copy, args);

  const char* p = format;
  while (*p) {
    if (*p != '%') {
      // Literal character — emit directly.
      iree_printf_output_char(out, *p);
      p++;
      continue;
    }

    p++;  // Skip '%'.

    // '%%' → literal '%'.
    if (*p == '%') {
      iree_printf_output_char(out, '%');
      p++;
      continue;
    }

    // Parse the format specifier.
    iree_printf_spec_t spec = {0};
    const char* next = iree_printf_parse_spec(p, &args_copy, &spec);
    if (!next) {
      // Truncated or malformed format string.
      va_end(args_copy);
      return -1;
    }
    p = next;

    // Dispatch to the appropriate formatter.
    switch (spec.specifier) {
      case 'd':
      case 'i':
      case 'u':
      case 'o':
      case 'x':
      case 'X':
        iree_printf_format_integer(out, &spec, &args_copy);
        break;
      case 's':
        iree_printf_format_string(out, &spec, &args_copy);
        break;
      case 'c':
        iree_printf_format_char(out, &spec, &args_copy);
        break;
      case 'p':
        iree_printf_format_pointer(out, &spec, &args_copy);
        break;
      case 'f':
      case 'F':
      case 'e':
      case 'E':
      case 'g':
      case 'G':
        iree_printf_format_float(out, &spec, &args_copy);
        break;
      default:
        // Unknown specifier. Fail loudly — do not silently skip.
        va_end(args_copy);
        return -1;
    }
  }

  va_end(args_copy);
  // Guard against size_t→int overflow. Formatted output exceeding INT_MAX bytes
  // is not representable as a return value; return -1 to signal the error
  // rather than wrapping to a bogus (possibly negative) count that callers
  // might use to size subsequent allocations.
  if (out->position > (size_t)INT_MAX) return -1;
  return (int)out->position;
}

//===----------------------------------------------------------------------===//
// Public API implementation
//===----------------------------------------------------------------------===//

int iree_vsnprintf(char* buffer, size_t count, const char* format,
                   va_list varargs) {
  iree_printf_output_t out = {
      .buffer = buffer,
      .capacity = count > 0 ? count - 1 : 0,  // Reserve 1 byte for NUL.
      .position = 0,
      .callback = NULL,
      .callback_data = NULL,
  };

  int result = iree_printf_format(&out, format, varargs);

  // NUL-terminate the buffer. If buffer is NULL (dry-run), skip.
  if (buffer) {
    if (count > 0) {
      size_t nul_position = out.position < count - 1 ? out.position : count - 1;
      buffer[nul_position] = '\0';
    }
  }

  return result;
}

int iree_snprintf(char* buffer, size_t count, const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  int result = iree_vsnprintf(buffer, count, format, varargs);
  va_end(varargs);
  return result;
}

int iree_vfctprintf(iree_printf_callback_t callback, void* user_data,
                    const char* format, va_list varargs) {
  iree_printf_output_t out = {
      .buffer = NULL,
      .capacity = 0,
      .position = 0,
      .callback = callback,
      .callback_data = user_data,
  };

  return iree_printf_format(&out, format, varargs);
}

int iree_fctprintf(iree_printf_callback_t callback, void* user_data,
                   const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  int result = iree_vfctprintf(callback, user_data, format, varargs);
  va_end(varargs);
  return result;
}
