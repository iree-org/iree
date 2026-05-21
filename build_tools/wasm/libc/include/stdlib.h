// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <stdlib.h> for wasm32.

#ifndef IREE_WASM_LIBC_STDLIB_H_
#define IREE_WASM_LIBC_STDLIB_H_

#include <stddef.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

// Memory allocation (backed by dlmalloc over memory.grow).
void* malloc(size_t size);
void* calloc(size_t count, size_t size);
void* realloc(void* pointer, size_t size);
void free(void* pointer);
void* aligned_alloc(size_t alignment, size_t size);

// Program termination.
_Noreturn void abort(void);
_Noreturn void exit(int status);

// Integer conversion.
int atoi(const char* string);
long atol(const char* string);
long long atoll(const char* string);
long strtol(const char* string, char** end, int base);
unsigned long strtoul(const char* string, char** end, int base);
long long strtoll(const char* string, char** end, int base);
unsigned long long strtoull(const char* string, char** end, int base);

// Floating-point conversion.
double atof(const char* string);
double strtod(const char* string, char** end);
float strtof(const char* string, char** end);
long double strtold(const char* string, char** end);

// Integer arithmetic.
int abs(int value);
long labs(long value);
long long llabs(long long value);

typedef struct {
  int quot;
  int rem;
} div_t;
typedef struct {
  long quot;
  long rem;
} ldiv_t;
typedef struct {
  long long quot;
  long long rem;
} lldiv_t;

div_t div(int numerator, int denominator);
ldiv_t ldiv(long numerator, long denominator);
lldiv_t lldiv(long long numerator, long long denominator);

// Pseudo-random numbers (seeded per-thread, not cryptographic).
#define RAND_MAX 0x7fffffff
int rand(void);
void srand(unsigned int seed);

// Environment.
char* getenv(const char* name);

// Temporary files.
int mkstemp(char* template_string);

#endif  // IREE_WASM_LIBC_STDLIB_H_
