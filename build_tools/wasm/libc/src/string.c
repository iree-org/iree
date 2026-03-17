// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// String and memory functions for wasm32.
//
// memcpy, memmove, and memset are NOT implemented here — clang lowers them
// directly to wasm bulk-memory instructions (memory.copy, memory.fill) when
// compiled with -mbulk-memory. Those instructions are single wasm opcodes
// that the engine implements natively. Providing C implementations would
// prevent the compiler from using the builtins.
//
// The functions below are the ones that have no wasm instruction equivalent
// and require software implementations.

#include <errno.h>
#include <stdint.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Memory comparison and search
//===----------------------------------------------------------------------===//

int memcmp(const void* left, const void* right, size_t count) {
  const unsigned char* a = (const unsigned char*)left;
  const unsigned char* b = (const unsigned char*)right;
  for (size_t i = 0; i < count; i++) {
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return 0;
}

void* memchr(const void* pointer, int value, size_t count) {
  const unsigned char* p = (const unsigned char*)pointer;
  unsigned char v = (unsigned char)value;
  for (size_t i = 0; i < count; i++) {
    if (p[i] == v) return (void*)(p + i);
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// String length
//===----------------------------------------------------------------------===//

size_t strlen(const char* string) {
  const char* p = string;
  while (*p) ++p;
  return (size_t)(p - string);
}

size_t strnlen(const char* string, size_t max_length) {
  const char* p = string;
  while (max_length-- && *p) ++p;
  return (size_t)(p - string);
}

//===----------------------------------------------------------------------===//
// String comparison
//===----------------------------------------------------------------------===//

int strcmp(const char* left, const char* right) {
  const unsigned char* a = (const unsigned char*)left;
  const unsigned char* b = (const unsigned char*)right;
  while (*a && *a == *b) {
    ++a;
    ++b;
  }
  return (int)*a - (int)*b;
}

int strncmp(const char* left, const char* right, size_t count) {
  const unsigned char* a = (const unsigned char*)left;
  const unsigned char* b = (const unsigned char*)right;
  while (count && *a && *a == *b) {
    ++a;
    ++b;
    --count;
  }
  if (count == 0) return 0;
  return (int)*a - (int)*b;
}

//===----------------------------------------------------------------------===//
// String search
//===----------------------------------------------------------------------===//

char* strchr(const char* string, int character) {
  char c = (char)character;
  while (*string) {
    if (*string == c) return (char*)string;
    ++string;
  }
  // strchr finds the null terminator when c == '\0'.
  return c == '\0' ? (char*)string : NULL;
}

char* strrchr(const char* string, int character) {
  char c = (char)character;
  const char* last = NULL;
  while (*string) {
    if (*string == c) last = string;
    ++string;
  }
  if (c == '\0') return (char*)string;
  return (char*)last;
}

char* strstr(const char* haystack, const char* needle) {
  if (!*needle) return (char*)haystack;

  for (; *haystack; ++haystack) {
    const char* h = haystack;
    const char* n = needle;
    while (*h && *n && *h == *n) {
      ++h;
      ++n;
    }
    if (!*n) return (char*)haystack;
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// String copying
//===----------------------------------------------------------------------===//

char* strcpy(char* dest, const char* src) {
  char* d = dest;
  while ((*d++ = *src++)) {
  }
  return dest;
}

char* strncpy(char* dest, const char* src, size_t count) {
  char* d = dest;
  while (count && (*d = *src)) {
    ++d;
    ++src;
    --count;
  }
  // Pad remaining bytes with null.
  while (count--) *d++ = '\0';
  return dest;
}

char* strcat(char* dest, const char* src) {
  char* d = dest;
  while (*d) ++d;
  while ((*d++ = *src++)) {
  }
  return dest;
}

char* strncat(char* dest, const char* src, size_t count) {
  char* d = dest;
  while (*d) ++d;
  while (count-- && (*d = *src)) {
    ++d;
    ++src;
  }
  *d = '\0';
  return dest;
}

//===----------------------------------------------------------------------===//
// Error string
//===----------------------------------------------------------------------===//

// Minimal strerror — returns the error name rather than a human-readable
// message. IREE uses iree_status_code_string() for its own error reporting;
// this is only needed for code that calls strerror directly.
char* strerror(int error_number) {
  switch (error_number) {
    case 0:
      return (char*)"Success";
    case EPERM:
      return (char*)"EPERM";
    case ENOENT:
      return (char*)"ENOENT";
    case EIO:
      return (char*)"EIO";
    case EBADF:
      return (char*)"EBADF";
    case ENOMEM:
      return (char*)"ENOMEM";
    case EACCES:
      return (char*)"EACCES";
    case EEXIST:
      return (char*)"EEXIST";
    case EINVAL:
      return (char*)"EINVAL";
    case ENOSYS:
      return (char*)"ENOSYS";
    case ERANGE:
      return (char*)"ERANGE";
    case EOVERFLOW:
      return (char*)"EOVERFLOW";
    case ENOTSUP:
      return (char*)"ENOTSUP";
    case ETIMEDOUT:
      return (char*)"ETIMEDOUT";
    case ECANCELED:
      return (char*)"ECANCELED";
    default:
      return (char*)"Unknown error";
  }
}
