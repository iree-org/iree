// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <string.h> for wasm32.

#ifndef IREE_WASM_LIBC_STRING_H_
#define IREE_WASM_LIBC_STRING_H_

#include <stddef.h>

// Copying (memcpy/memset/memmove are wasm builtins via -mbulk-memory).
void* memcpy(void* dest, const void* src, size_t count);
void* memmove(void* dest, const void* src, size_t count);
void* memset(void* dest, int value, size_t count);

// Comparison.
int memcmp(const void* left, const void* right, size_t count);

// Search.
void* memchr(const void* pointer, int value, size_t count);

// String operations.
size_t strlen(const char* string);
size_t strnlen(const char* string, size_t max_length);
int strcmp(const char* left, const char* right);
int strncmp(const char* left, const char* right, size_t count);
char* strchr(const char* string, int character);
char* strrchr(const char* string, int character);
char* strstr(const char* haystack, const char* needle);

// String copying.
char* strcpy(char* dest, const char* src);
char* strncpy(char* dest, const char* src, size_t count);
char* strcat(char* dest, const char* src);
char* strncat(char* dest, const char* src, size_t count);

// Error string.
char* strerror(int error_number);

#endif  // IREE_WASM_LIBC_STRING_H_
