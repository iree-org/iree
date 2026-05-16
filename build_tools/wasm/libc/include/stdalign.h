// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Freestanding <stdalign.h> for wasm32.

#ifndef IREE_WASM_LIBC_STDALIGN_H_
#define IREE_WASM_LIBC_STDALIGN_H_

// C11 alignas/alignof are keywords in C23; macros for older standards.
#if !defined(__cplusplus)
#define alignas _Alignas
#define alignof _Alignof
#endif
#define __alignas_is_defined 1
#define __alignof_is_defined 1

#endif  // IREE_WASM_LIBC_STDALIGN_H_
