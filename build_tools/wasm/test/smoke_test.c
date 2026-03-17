// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Smoke test for the wasm32 toolchain. Verifies that clang can compile C to
// wasm32 with our libc headers and wasm-ld can link it.

#include <stdint.h>

#ifndef __wasm__
#error "Expected __wasm__ to be defined"
#endif

#ifndef IREE_PLATFORM_WEB
#error "Expected IREE_PLATFORM_WEB to be defined by toolchain"
#endif

// Exported so the linker doesn't strip it and the host can call it.
__attribute__((export_name("add"))) int32_t add(int32_t a, int32_t b) {
  return a + b;
}
