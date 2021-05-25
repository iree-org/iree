// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if defined(_WIN32)
#define IREE_SYM_EXPORT __declspec(dllexport)
#else
#define IREE_SYM_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

int IREE_SYM_EXPORT times_two(int value) { return value * 2; }

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
