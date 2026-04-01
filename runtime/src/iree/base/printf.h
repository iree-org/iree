// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE's platform-independent printf implementation.
//
// All runtime code should use these functions instead of libc printf/snprintf/
// vsnprintf to ensure consistent formatting across all platforms and targets
// (GCC/Clang, Linux/macOS/Windows/WASM/embedded).
//
// Supported format specifiers:
//
//   Integers:  %d %i %u %o %x %X
//     Length modifiers: hh h l ll z t j
//     Flags: - + 0 # (space)
//     Width and precision (literal or * from args)
//
//   Strings:   %s (with precision for bounded reads: %.*s)
//   Character: %c
//   Pointer:   %p (formatted as 0x followed by lowercase hex digits)
//   Float:     %f %F %e %E %g %G (double only; default precision 6)
//   Literal:   %%
//
// NOT supported (by design — zero uses in IREE, reduces attack surface):
//   %n (writeback — security hazard)
//   %a %A (hex float)
//   %ls %lc (wide strings/chars)
//   %L (long double)
//   MSVC-style %I64d
//
// Length modifier reference:
//   (none)  int / unsigned int
//   hh      signed char / unsigned char (promoted to int in va_arg)
//   h       short / unsigned short (promoted to int in va_arg)
//   l       long / unsigned long
//   ll      long long / unsigned long long
//   z       size_t / ssize_t  (use for iree_host_size_t, raw size_t)
//   t       ptrdiff_t
//   j       intmax_t / uintmax_t
//
// For fixed-width integer types (int64_t, uint64_t, etc.), use the standard
// PRI macros from <inttypes.h> (PRId64, PRIu64, PRIx64, etc.) — the
// underlying type of int64_t varies by platform (long on LP64, long long on
// LLP64/Windows), and the PRI macros expand to the correct length modifier.

#ifndef IREE_BASE_PRINTF_H_
#define IREE_BASE_PRINTF_H_

#include <stdarg.h>
#include <stddef.h>

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Callback type for iree_vfctprintf/iree_fctprintf.
// Called once per output character during formatting.
typedef void (*iree_printf_callback_t)(char character, void* user_data);

// Formats a string into |buffer| of |count| bytes using a printf-style
// |format| string and variable arguments. Returns the number of characters
// that would have been written (excluding NUL) if the buffer were large enough,
// or -1 on format error (unknown specifier, truncated format string).
// If |buffer| is NULL, no characters are written and the return value is the
// number of characters that would be written (dry-run for size measurement).
IREE_PRINTF_ATTRIBUTE(3, 4)
int iree_snprintf(char* buffer, size_t count, const char* format, ...);

// Like iree_snprintf but takes a va_list.
int iree_vsnprintf(char* buffer, size_t count, const char* format,
                   va_list varargs);

// Formats a string using a printf-style |format| and sends each character to
// |callback| with the provided |user_data|. The callback does NOT receive a NUL
// terminator. Returns the number of characters for which the callback was
// invoked, or -1 on format error.
IREE_PRINTF_ATTRIBUTE(3, 4)
int iree_fctprintf(iree_printf_callback_t callback, void* user_data,
                   const char* format, ...);

// Like iree_fctprintf but takes a va_list.
int iree_vfctprintf(iree_printf_callback_t callback, void* user_data,
                    const char* format, va_list varargs);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_PRINTF_H_
