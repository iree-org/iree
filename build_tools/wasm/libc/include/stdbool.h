// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Freestanding <stdbool.h> for wasm32.

#ifndef IREE_WASM_LIBC_STDBOOL_H_
#define IREE_WASM_LIBC_STDBOOL_H_

// C23 makes bool/true/false keywords; older standards need these macros.
#if !defined(__cplusplus) && !defined(__bool_true_false_are_defined)
#if __STDC_VERSION__ < 202311L
#define bool _Bool
#define true 1
#define false 0
#endif
#define __bool_true_false_are_defined 1
#endif

#endif  // IREE_WASM_LIBC_STDBOOL_H_
