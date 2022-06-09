// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLS_UTILS_DEVICE_UTIL_H_
#define IREE_TOOLS_UTILS_DEVICE_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns a driver registry initialized with all linked driver registered.
iree_hal_driver_registry_t* iree_hal_available_driver_registry(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLS_UTILS_DEVICE_UTIL_H_
