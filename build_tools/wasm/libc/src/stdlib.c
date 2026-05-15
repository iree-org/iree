// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Miscellaneous <stdlib.h> functions for wasm32.
//
// Memory allocation (malloc/free) is in dlmalloc.c.
// String-to-number conversion (strtol/strtod) is in strtol.c.
// Program termination (abort/exit) is in abort.c.

#include <stdint.h>
#include <stdlib.h>

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

int abs(int value) { return value < 0 ? -value : value; }
long labs(long value) { return value < 0 ? -value : value; }
long long llabs(long long value) { return value < 0 ? -value : value; }

div_t div(int numerator, int denominator) {
  div_t result = {numerator / denominator, numerator % denominator};
  return result;
}

ldiv_t ldiv(long numerator, long denominator) {
  ldiv_t result = {numerator / denominator, numerator % denominator};
  return result;
}

lldiv_t lldiv(long long numerator, long long denominator) {
  lldiv_t result = {numerator / denominator, numerator % denominator};
  return result;
}

//===----------------------------------------------------------------------===//
// Pseudo-random numbers
//===----------------------------------------------------------------------===//

// Simple LCG — not cryptographic, not for security use. IREE has its own
// CSPRNG (iree_csprng_fill) backed by crypto.getRandomValues for real
// randomness. This is only for code that calls rand() directly (e.g., some
// test utilities).
static unsigned int rand_state = 1;

int rand(void) {
  rand_state = rand_state * 1103515245u + 12345u;
  return (int)((rand_state >> 16) & 0x7fff);
}

void srand(unsigned int seed) { rand_state = seed; }

//===----------------------------------------------------------------------===//
// Environment
//===----------------------------------------------------------------------===//

// No environment variables in freestanding wasm.
char* getenv(const char* name) {
  (void)name;
  return NULL;
}

// No temporary files in freestanding wasm.
int mkstemp(char* template_string) {
  (void)template_string;
  return -1;
}
