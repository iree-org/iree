// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_TSAN_H_
#define IREE_BASE_INTERNAL_TSAN_H_

#include "iree/base/target_platform.h"

// Fasilitates compilation with thread sanitizer support.

#ifdef IREE_SANITIZER_THREAD

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
void __tsan_acquire(void *addr);
void __tsan_release(void *addr);
#ifdef __cplusplus
}
#endif  // __cplusplus

#define IREE_TSAN_ACQUIRE(addr) __tsan_acquire(addr)
#define IREE_TSAN_RELEASE(addr) __tsan_release(addr)

#else  // IREE_SANITIZER_THREAD

#define IREE_TSAN_ACQUIRE(addr) (void)((addr))
#define IREE_TSAN_RELEASE(addr) (void)((addr))

#endif  // IREE_SANITIZER_THREAD

#ifdef IREE_COMPILER_GCC_COMPAT

#define IREE_DISABLE_COMPILER_TSAN_ERRORS()           \
  _Pragma("GCC diagnostic push")                      \
      _Pragma("GCC diagnostic ignored \"-Wpragmas\"") \
          _Pragma("GCC diagnostic ignored \"-Wtsan\"")
#define IREE_RESTORE_COMPILER_TSAN_ERRORS() _Pragma("GCC diagnostic pop")

#else  // IREE_COMPILER_GCC_COMPAT

#define IREE_DISABLE_COMPILER_TSAN_ERRORS()
#define IREE_RESTORE_COMPILER_TSAN_ERRORS()

#endif  // IREE_COMPILER_GCC_COMPAT

#endif  // IREE_BASE_INTERNAL_TSAN_H_
