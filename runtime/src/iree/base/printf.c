// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/printf.h"

#include "printf/printf.h"

// eyalroz/printf requires consumers to provide a putchar_ implementation for
// its printf_/vprintf_ functions (stdout output). IREE only uses snprintf_/
// vsnprintf_/vfctprintf so this is never called — but the library compiles the
// functions unconditionally, producing a linker reference we must satisfy.
void putchar_(char c) { (void)c; }

int iree_snprintf(char* buffer, size_t count, const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  int result = vsnprintf_(buffer, count, format, varargs);
  va_end(varargs);
  return result;
}

int iree_vsnprintf(char* buffer, size_t count, const char* format,
                   va_list varargs) {
  return vsnprintf_(buffer, count, format, varargs);
}

int iree_fctprintf(iree_printf_callback_t callback, void* user_data,
                   const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  int result = vfctprintf(callback, user_data, format, varargs);
  va_end(varargs);
  return result;
}

int iree_vfctprintf(iree_printf_callback_t callback, void* user_data,
                    const char* format, va_list varargs) {
  return vfctprintf(callback, user_data, format, varargs);
}
