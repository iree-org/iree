// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_TIME_H_
#define IREE_BASE_INTERNAL_TIME_H_

#include "iree/base/time.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Implementation of iree_time_now(), available to avoid circular dependencies.
iree_time_t iree_time_now_impl(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_TIME_H_
