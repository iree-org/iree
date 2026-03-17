// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <sys/types.h> for wasm32.

#ifndef IREE_WASM_LIBC_SYS_TYPES_H_
#define IREE_WASM_LIBC_SYS_TYPES_H_

#include <stddef.h>
#include <stdint.h>

typedef int32_t ssize_t;
typedef int32_t off_t;
typedef int32_t pid_t;
typedef uint32_t mode_t;
typedef uint32_t dev_t;
typedef uint32_t ino_t;
typedef uint32_t nlink_t;
typedef uint32_t uid_t;
typedef uint32_t gid_t;

#endif  // IREE_WASM_LIBC_SYS_TYPES_H_
