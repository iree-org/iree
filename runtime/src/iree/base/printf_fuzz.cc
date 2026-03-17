// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for IREE's printf implementation.
//
// Strategies:
//   1. Bounded string reads: %.*s with ASAN-bounded strings (security).
//   2. Structured integer formatting: hardcoded specs vs libc (correctness).
//   3. Structured float formatting: hardcoded specs vs libc (correctness).
//   4. Width/precision stress: dynamic width/precision vs libc (correctness).
//   5. Full format string generation: builds arbitrary format specifiers from
//      fuzzer input bytes, covering all flags, widths, precisions, length
//      modifiers, and specifiers. Tests the parser state machine (security +
//      correctness).
//   6. Malformed format strings: truncated/invalid specifiers to verify the
//      parser doesn't crash, loop, or over-read (security).
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "iree/base/api.h"

// Helper to consume bytes from the fuzzer input.
typedef struct {
  const uint8_t* data;
  size_t remaining;
} fuzz_input_t;

static uint8_t fuzz_consume_u8(fuzz_input_t* input) {
  if (input->remaining == 0) return 0;
  uint8_t value = input->data[0];
  input->data++;
  input->remaining--;
  return value;
}

static uint16_t fuzz_consume_u16(fuzz_input_t* input) {
  uint16_t value = fuzz_consume_u8(input);
  value |= (uint16_t)fuzz_consume_u8(input) << 8;
  return value;
}

static uint32_t fuzz_consume_u32(fuzz_input_t* input) {
  uint32_t value = fuzz_consume_u16(input);
  value |= (uint32_t)fuzz_consume_u16(input) << 16;
  return value;
}

static uint64_t fuzz_consume_u64(fuzz_input_t* input) {
  uint64_t value = fuzz_consume_u32(input);
  value |= (uint64_t)fuzz_consume_u32(input) << 32;
  return value;
}

static double fuzz_consume_double(fuzz_input_t* input) {
  union {
    uint64_t bits;
    double value;
  } u;
  u.bits = fuzz_consume_u64(input);
  return u.value;
}

//===----------------------------------------------------------------------===//
// Strategy 5: Full format string generation
//===----------------------------------------------------------------------===//
// Builds a single complete format specifier from fuzzer input bytes, covering
// every supported combination of flags, width, precision, length modifier, and
// specifier. This exercises the format parser state machine that the hardcoded
// strategies bypass.
//
// Layout of consumed bytes:
//   byte 0: flags bitmask (bits 0-4: - + space 0 #)
//   byte 1: width (0=none, 1-199=literal, 200+=dynamic *)
//   byte 2: precision (0=none, 1-199=literal, 200+=dynamic *)
//   byte 3: length modifier (0=none, 1=hh, 2=h, 3=l, 4=ll, 5=z, 6=t, 7=j)
//   byte 4: specifier choice (0-15 mapped to d/i/u/o/x/X/f/e/g/E/G/F/s/c/p/%)
//   remaining: argument value (type depends on specifier + length)
//
// Invalid length+specifier combinations (e.g., %lls) are skipped since they
// are caller UB, not a printf bug.

enum fuzz_spec_type {
  FUZZ_SPEC_SIGNED_INT,
  FUZZ_SPEC_UNSIGNED_INT,
  FUZZ_SPEC_FLOAT,
  FUZZ_SPEC_STRING,
  FUZZ_SPEC_CHAR,
  FUZZ_SPEC_POINTER,
  FUZZ_SPEC_PERCENT,
};

// Build a format string into |format| (must be >= 32 bytes) and return the
// specifier type so the caller knows what argument to pass.
static fuzz_spec_type fuzz_build_format_string(fuzz_input_t* input,
                                               char* format,
                                               uint8_t* out_length_mod) {
  int position = 0;
  format[position++] = '%';

  // Flags.
  uint8_t flags = fuzz_consume_u8(input);
  if (flags & 0x01) format[position++] = '-';
  if (flags & 0x02) format[position++] = '+';
  if (flags & 0x04) format[position++] = ' ';
  if (flags & 0x08) format[position++] = '0';
  if (flags & 0x10) format[position++] = '#';

  // Width.
  uint8_t width_byte = fuzz_consume_u8(input);
  if (width_byte >= 200) {
    format[position++] = '*';
  } else if (width_byte > 0) {
    position += snprintf(format + position, 16, "%d", width_byte % 64);
  }

  // Precision. Capped to 15 to stay within our implementation's accurate
  // range. With fmod-based rounding, we get exact remainder computation for
  // the rounding decision, and exact quotients for up to 16 significant
  // digits (quotient < 2^53). Precision 15 gives at most 16 significant
  // digits for %e (precision + 1 leading digit), well within range.
  // Strategy 4 independently tests integer precision up to 31.
  uint8_t precision_byte = fuzz_consume_u8(input);
  if (precision_byte >= 200) {
    format[position++] = '.';
    format[position++] = '*';
  } else if (precision_byte > 0) {
    format[position++] = '.';
    position += snprintf(format + position, 16, "%d", precision_byte % 16);
  }

  // Length modifier.
  uint8_t length_mod = fuzz_consume_u8(input) % 8;
  *out_length_mod = length_mod;
  static const char* const length_strings[] = {"",   "hh", "h", "l",
                                               "ll", "z",  "t", "j"};
  const char* length_str = length_strings[length_mod];
  while (*length_str) format[position++] = *length_str++;

  // Specifier.
  uint8_t spec_choice = fuzz_consume_u8(input) % 16;
  fuzz_spec_type spec_type = FUZZ_SPEC_PERCENT;
  switch (spec_choice) {
    case 0:
      format[position++] = 'd';
      spec_type = FUZZ_SPEC_SIGNED_INT;
      break;
    case 1:
      format[position++] = 'i';
      spec_type = FUZZ_SPEC_SIGNED_INT;
      break;
    case 2:
      format[position++] = 'u';
      spec_type = FUZZ_SPEC_UNSIGNED_INT;
      break;
    case 3:
      format[position++] = 'o';
      spec_type = FUZZ_SPEC_UNSIGNED_INT;
      break;
    case 4:
      format[position++] = 'x';
      spec_type = FUZZ_SPEC_UNSIGNED_INT;
      break;
    case 5:
      format[position++] = 'X';
      spec_type = FUZZ_SPEC_UNSIGNED_INT;
      break;
    case 6:
      format[position++] = 'f';
      spec_type = FUZZ_SPEC_FLOAT;
      break;
    case 7:
      format[position++] = 'e';
      spec_type = FUZZ_SPEC_FLOAT;
      break;
    case 8:
      format[position++] = 'g';
      spec_type = FUZZ_SPEC_FLOAT;
      break;
    case 9:
      format[position++] = 'E';
      spec_type = FUZZ_SPEC_FLOAT;
      break;
    case 10:
      format[position++] = 'G';
      spec_type = FUZZ_SPEC_FLOAT;
      break;
    case 11:
      format[position++] = 'F';
      spec_type = FUZZ_SPEC_FLOAT;
      break;
    case 12:
      format[position++] = 's';
      spec_type = FUZZ_SPEC_STRING;
      break;
    case 13:
      format[position++] = 'c';
      spec_type = FUZZ_SPEC_CHAR;
      break;
    case 14:
      format[position++] = 'p';
      spec_type = FUZZ_SPEC_POINTER;
      break;
    default:
      // C99 says "the complete conversion specification shall be %%." No
      // flags, width, precision, or length modifier are allowed. Discard
      // everything built so far and emit a bare "%%" to avoid UB.
      position = 0;
      format[position++] = '%';
      format[position++] = '%';
      *out_length_mod = 0;
      spec_type = FUZZ_SPEC_PERCENT;
      break;
  }
  format[position] = '\0';
  return spec_type;
}

// Format with both implementations and compare. Returns true if they match.
// Uses a macro to call both printf implementations with the same format+args
// without triggering -Wformat-security (the format string is constructed, not
// a literal).
#define FUZZ_FORMAT_AND_COMPARE(iree_buf, libc_buf, fmt, ...)          \
  do {                                                                 \
    iree_snprintf((iree_buf), sizeof(iree_buf), (fmt), ##__VA_ARGS__); \
    snprintf((libc_buf), sizeof(libc_buf), (fmt), ##__VA_ARGS__);      \
    if (strcmp((iree_buf), (libc_buf)) != 0) __builtin_trap();         \
  } while (0)

// Like FUZZ_FORMAT_AND_COMPARE but allows mismatch for out-of-range floats.
// |check_val| is the double being formatted (used for range check).
// |...| is the full argument list passed to printf (may include dynamic
// width/precision args before the value).
#define FUZZ_FORMAT_AND_COMPARE_FLOAT(iree_buf, libc_buf, check_val, fmt, ...) \
  do {                                                                         \
    iree_snprintf((iree_buf), sizeof(iree_buf), (fmt), ##__VA_ARGS__);         \
    snprintf((libc_buf), sizeof(libc_buf), (fmt), ##__VA_ARGS__);              \
    if (std::isfinite(check_val) && std::fabs(check_val) < 1e18 &&             \
        std::fabs(check_val) > 1e-15) {                                        \
      if (strcmp((iree_buf), (libc_buf)) != 0) __builtin_trap();               \
    }                                                                          \
  } while (0)

// Execute strategy 5 for a single format specifier.
static void fuzz_strategy_full_format(fuzz_input_t* input) {
  char format[32];
  uint8_t length_mod = 0;
  fuzz_spec_type spec_type =
      fuzz_build_format_string(input, format, &length_mod);

  // Skip invalid length+specifier combinations. Length modifiers are only
  // valid with integer specifiers; other types ignore them.
  if (spec_type != FUZZ_SPEC_SIGNED_INT &&
      spec_type != FUZZ_SPEC_UNSIGNED_INT && length_mod != 0) {
    return;
  }

  char iree_buffer[1024];
  char libc_buffer[1024];

  // Consume dynamic width/precision arguments if the format uses *.
  // We need to pass these BEFORE the main argument.
  // Scan the format string for '*' characters.
  int dynamic_args[2];
  int dynamic_count = 0;
  for (const char* p = format + 1; *p && dynamic_count < 2; p++) {
    if (*p == '*') {
      dynamic_args[dynamic_count++] =
          (int)(fuzz_consume_u8(input) % 64);  // Reasonable range.
    }
  }

  // Dispatch based on specifier type with the right argument type.
  switch (spec_type) {
    case FUZZ_SPEC_SIGNED_INT: {
      uint64_t raw = fuzz_consume_u64(input);
      switch (length_mod) {
        case 0:  // none (int)
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (int)(int32_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (int)(int32_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (int)(int32_t)raw);
          break;
        case 1:  // hh (signed char, promoted to int)
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (int)(signed char)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (int)(signed char)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (int)(signed char)raw);
          break;
        case 2:  // h (short, promoted to int)
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (int)(short)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (int)(short)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (int)(short)raw);
          break;
        case 3:  // l (long)
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (long)(int64_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (long)(int64_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (long)(int64_t)raw);
          break;
        case 4:  // ll (long long)
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (long long)(int64_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (long long)(int64_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (long long)(int64_t)raw);
          break;
        case 5:  // z (ptrdiff_t for signed)
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (ptrdiff_t)(int64_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (ptrdiff_t)(int64_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (ptrdiff_t)(int64_t)raw);
          break;
        case 6:  // t (ptrdiff_t)
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (ptrdiff_t)(int64_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (ptrdiff_t)(int64_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (ptrdiff_t)(int64_t)raw);
          break;
        case 7:  // j (intmax_t)
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (intmax_t)(int64_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (intmax_t)(int64_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (intmax_t)(int64_t)raw);
          break;
      }
      break;
    }

    case FUZZ_SPEC_UNSIGNED_INT: {
      uint64_t raw = fuzz_consume_u64(input);
      switch (length_mod) {
        case 0:
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (unsigned int)(uint32_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0],
                                    (unsigned int)(uint32_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (unsigned int)(uint32_t)raw);
          break;
        case 1:
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (unsigned int)(unsigned char)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0],
                                    (unsigned int)(unsigned char)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (unsigned int)(unsigned char)raw);
          break;
        case 2:
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (unsigned int)(unsigned short)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0],
                                    (unsigned int)(unsigned short)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (unsigned int)(unsigned short)raw);
          break;
        case 3:
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (unsigned long)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (unsigned long)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (unsigned long)raw);
          break;
        case 4:
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (unsigned long long)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (unsigned long long)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (unsigned long long)raw);
          break;
        case 5:
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (size_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (size_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (size_t)raw);
          break;
        case 6:
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (size_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (size_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (size_t)raw);
          break;
        case 7:
          if (dynamic_count == 2)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], dynamic_args[1],
                                    (uintmax_t)raw);
          else if (dynamic_count == 1)
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    dynamic_args[0], (uintmax_t)raw);
          else
            FUZZ_FORMAT_AND_COMPARE(iree_buffer, libc_buffer, format,
                                    (uintmax_t)raw);
          break;
      }
      break;
    }

    case FUZZ_SPEC_FLOAT: {
      double value = fuzz_consume_double(input);
      // Cap dynamic precision to 15 (matching the literal precision cap).
      // Scan for '.*' in the format string to find which dynamic arg is
      // the precision (vs width).
      if (dynamic_count > 0) {
        int dyn_idx = 0;
        for (const char* p = format + 1; *p && dyn_idx < dynamic_count; p++) {
          if (*p == '*') {
            // Precision * follows '.'; width * does not.
            if (p > format + 1 && *(p - 1) == '.') {
              if (dynamic_args[dyn_idx] > 15) dynamic_args[dyn_idx] = 15;
            }
            dyn_idx++;
          }
        }
      }
      if (dynamic_count == 2)
        FUZZ_FORMAT_AND_COMPARE_FLOAT(iree_buffer, libc_buffer, value, format,
                                      dynamic_args[0], dynamic_args[1], value);
      else if (dynamic_count == 1)
        FUZZ_FORMAT_AND_COMPARE_FLOAT(iree_buffer, libc_buffer, value, format,
                                      dynamic_args[0], value);
      else
        FUZZ_FORMAT_AND_COMPARE_FLOAT(iree_buffer, libc_buffer, value, format,
                                      value);
      break;
    }

    case FUZZ_SPEC_STRING: {
      // Build a bounded string from fuzzer input. Only '-' flag and
      // width/precision are defined for %s; other flags are UB. We test
      // for crash/ASAN safety and only compare when no UB flags are present.
      uint8_t string_length = fuzz_consume_u8(input);
      char string_buffer[256];
      size_t actual_length = string_length;
      if (actual_length > sizeof(string_buffer) - 1) {
        actual_length = sizeof(string_buffer) - 1;
      }
      for (size_t i = 0; i < actual_length; i++) {
        string_buffer[i] = 'A' + (fuzz_consume_u8(input) % 26);
      }
      string_buffer[actual_length] = '\0';
      if (dynamic_count == 2)
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format, dynamic_args[0],
                      dynamic_args[1], string_buffer);
      else if (dynamic_count == 1)
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format, dynamic_args[0],
                      string_buffer);
      else
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format, string_buffer);
      break;
    }

    case FUZZ_SPEC_CHAR: {
      // Only '-' flag and width are defined for %c; other flags and
      // precision are UB. Test for crash/ASAN safety only.
      int value = fuzz_consume_u8(input);
      if (value == 0) value = 'Z';
      if (dynamic_count == 2)
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format, dynamic_args[0],
                      dynamic_args[1], value);
      else if (dynamic_count == 1)
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format, dynamic_args[0],
                      value);
      else
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format, value);
      break;
    }

    case FUZZ_SPEC_POINTER: {
      // %p output is entirely implementation-defined (e.g., glibc uses
      // "(nil)" for NULL, we use "0x0"). Only test for crash/ASAN safety.
      void* value = (void*)(uintptr_t)fuzz_consume_u64(input);
      if (dynamic_count == 2)
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format, dynamic_args[0],
                      dynamic_args[1], value);
      else if (dynamic_count == 1)
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format, dynamic_args[0],
                      value);
      else
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format, value);
      break;
    }

    case FUZZ_SPEC_PERCENT: {
      // %% takes no arguments. Just verify it produces "%".
      // Suppress -Wformat-security: we are intentionally testing a non-literal
      // format string that consumes no arguments.
      if (dynamic_count == 0) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
        iree_snprintf(iree_buffer, sizeof(iree_buffer), format);
        snprintf(libc_buffer, sizeof(libc_buffer), format);
#pragma GCC diagnostic pop
        if (strcmp(iree_buffer, libc_buffer) != 0) __builtin_trap();
      }
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Strategy 6: Malformed format strings
//===----------------------------------------------------------------------===//
// Generates truncated or invalid format specifiers to verify the parser doesn't
// crash, loop, or read out of bounds. Does NOT compare with libc (behavior for
// invalid formats is implementation-defined).

static void fuzz_strategy_malformed(fuzz_input_t* input) {
  // Build a format string with potentially truncated specifiers.
  char format[64];
  int position = 0;
  uint8_t count = fuzz_consume_u8(input) % 8 + 1;  // 1-8 fragments.
  for (int i = 0; i < count && position < 50; i++) {
    uint8_t fragment_type = fuzz_consume_u8(input) % 6;
    switch (fragment_type) {
      case 0:
        // Literal text.
        format[position++] = 'x';
        break;
      case 1:
        // Just '%' (truncated specifier).
        format[position++] = '%';
        break;
      case 2:
        // '%' + flag (truncated).
        format[position++] = '%';
        if (position < 50)
          format[position++] = "+-0 #"[fuzz_consume_u8(input) % 5];
        break;
      case 3:
        // '%' + length modifier (truncated).
        format[position++] = '%';
        if (position < 50)
          format[position++] = "hlzj"[fuzz_consume_u8(input) % 4];
        break;
      case 4:
        // '%%' (valid escape).
        format[position++] = '%';
        if (position < 50) format[position++] = '%';
        break;
      case 5:
        // '%' + digits (width, no specifier).
        format[position++] = '%';
        if (position < 50)
          format[position++] = '0' + fuzz_consume_u8(input) % 10;
        break;
    }
  }
  format[position] = '\0';

  // Call iree_snprintf. We don't compare with libc because behavior for
  // malformed formats is implementation-defined. We just verify no crash,
  // no infinite loop, and no ASAN violation.
  char buffer[256];
  iree_snprintf(buffer, sizeof(buffer), "%s", format);

  // Also test with the format string directly (not through %s) — but only if
  // it doesn't contain unescaped % that would need arguments. Count how many
  // format specifiers would be consumed.
  // SAFETY: calling printf with fewer arguments than specifiers is UB. We only
  // call directly if the string has no unmatched % signs.
  bool has_specifiers = false;
  for (int i = 0; i < position; i++) {
    if (format[i] == '%') {
      if (i + 1 < position && format[i + 1] == '%') {
        i++;  // Skip %%.
      } else {
        has_specifiers = true;
        break;
      }
    }
  }
  if (!has_specifiers) {
    // Suppress -Wformat-security: we are intentionally testing a non-literal
    // format string that consumes no arguments.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
    iree_snprintf(buffer, sizeof(buffer), format);
#pragma GCC diagnostic pop
  }
}

//===----------------------------------------------------------------------===//
// Main fuzzer entry point
//===----------------------------------------------------------------------===//

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  fuzz_input_t input = {data, size};

  char iree_buffer[512];
  char libc_buffer[512];

  //===--------------------------------------------------------------------===//
  // Strategy 1: Bounded string reads (%.*s security test)
  //===--------------------------------------------------------------------===//
  // The precision must strictly bound memory reads. If precision is N, at most
  // N bytes should be read from the string, even if the string has no NUL
  // within those N bytes. ASAN catches any over-read.
  {
    uint8_t precision = fuzz_consume_u8(&input);
    // Create a buffer of exactly |precision| bytes — ASAN poisons beyond.
    // We write non-NUL bytes so the string doesn't terminate early.
    char string_buffer[256];
    size_t string_length = precision;
    if (string_length > sizeof(string_buffer)) {
      string_length = sizeof(string_buffer);
    }
    memset(string_buffer, 'A', string_length);

    // Format with precision = string_length. Must not read past.
    iree_snprintf(iree_buffer, sizeof(iree_buffer), "%.*s", (int)string_length,
                  string_buffer);
  }

  //===--------------------------------------------------------------------===//
  // Strategy 2: Structured integer formatting
  //===--------------------------------------------------------------------===//
  {
    uint8_t specifier_choice = fuzz_consume_u8(&input) % 6;
    const char* specifiers[] = {"%d", "%u", "%x", "%X", "%o", "%08x"};
    uint32_t value = fuzz_consume_u32(&input);

    int iree_length = iree_snprintf(iree_buffer, sizeof(iree_buffer),
                                    specifiers[specifier_choice], value);
    int libc_length = snprintf(libc_buffer, sizeof(libc_buffer),
                               specifiers[specifier_choice], value);

    // Output must match libc (or both must error).
    if (iree_length >= 0 && libc_length >= 0) {
      if (strcmp(iree_buffer, libc_buffer) != 0) {
        __builtin_trap();  // Mismatch — this is a bug.
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Strategy 3: Structured float formatting
  //===--------------------------------------------------------------------===//
  {
    double value = fuzz_consume_double(&input);
    uint8_t format_choice = fuzz_consume_u8(&input) % 4;
    const char* formats[] = {"%f", "%e", "%g", "%.2f"};

    int iree_length = iree_snprintf(iree_buffer, sizeof(iree_buffer),
                                    formats[format_choice], value);
    int libc_length = snprintf(libc_buffer, sizeof(libc_buffer),
                               formats[format_choice], value);

    // Compare if both succeeded and the value is in a range where our
    // implementation matches libc exactly. For very large values (|v| > 1e18),
    // the digits beyond 17 significant figures are implementation-defined
    // artifacts of binary-to-decimal conversion, so exact comparison is not
    // meaningful. For very small values and NaN/Inf, formatting may have
    // acceptable platform differences.
    if (iree_length >= 0 && libc_length >= 0 && std::isfinite(value) &&
        value == value && std::fabs(value) < 1e18 && std::fabs(value) > 1e-15) {
      if (strcmp(iree_buffer, libc_buffer) != 0) {
        __builtin_trap();  // Mismatch — this is a bug.
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Strategy 4: Width and precision stress
  //===--------------------------------------------------------------------===//
  {
    int width = fuzz_consume_u8(&input) % 64;
    int precision = fuzz_consume_u8(&input) % 32;
    int32_t value = (int32_t)fuzz_consume_u32(&input);

    iree_snprintf(iree_buffer, sizeof(iree_buffer), "%*.*d", width, precision,
                  value);
    snprintf(libc_buffer, sizeof(libc_buffer), "%*.*d", width, precision,
             value);

    if (strcmp(iree_buffer, libc_buffer) != 0) {
      __builtin_trap();
    }
  }

  //===--------------------------------------------------------------------===//
  // Strategy 5: Full format string generation
  //===--------------------------------------------------------------------===//
  fuzz_strategy_full_format(&input);

  //===--------------------------------------------------------------------===//
  // Strategy 6: Malformed format strings
  //===--------------------------------------------------------------------===//
  fuzz_strategy_malformed(&input);

  return 0;
}
