// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <wchar.h> for wasm32 (minimal).

#ifndef IREE_WASM_LIBC_WCHAR_H_
#define IREE_WASM_LIBC_WCHAR_H_

#include <stddef.h>

// WCHAR_MIN/MAX for 32-bit signed wchar_t (from -fno-short-wchar).
#define WCHAR_MIN (-2147483647 - 1)
#define WCHAR_MAX 2147483647

// Multi-byte conversion state (opaque — no real locale on wasm).
typedef struct {
  unsigned __opaque[2];
} mbstate_t;

#endif  // IREE_WASM_LIBC_WCHAR_H_
