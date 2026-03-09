// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Platform-independent printf implementation wrapping eyalroz/printf.
// All runtime code should use these functions instead of libc snprintf/
// vsnprintf to ensure consistent formatting across all platforms.

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
// that would have been written (excluding NUL) if the buffer were large enough.
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
// invoked.
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
