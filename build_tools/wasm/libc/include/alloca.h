// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <alloca.h> for wasm32.

#ifndef IREE_WASM_LIBC_ALLOCA_H_
#define IREE_WASM_LIBC_ALLOCA_H_

#define alloca(size) __builtin_alloca(size)

#endif  // IREE_WASM_LIBC_ALLOCA_H_
