// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Client-side proxy executable cache for remote HAL devices.
//
// The cache proxy sends EXECUTABLE_UPLOAD control channel RPCs to the
// server and returns remote executable proxies wrapping the server's
// resolved resource_ids.

#ifndef IREE_HAL_REMOTE_CLIENT_EXECUTABLE_CACHE_H_
#define IREE_HAL_REMOTE_CLIENT_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_client_device_t iree_hal_remote_client_device_t;

// Creates a remote executable cache proxy for the given device.
iree_status_t iree_hal_remote_client_executable_cache_create(
    iree_hal_remote_client_device_t* device, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_EXECUTABLE_CACHE_H_
