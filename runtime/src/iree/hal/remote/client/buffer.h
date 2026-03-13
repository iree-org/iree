// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_BUFFER_H_
#define IREE_HAL_REMOTE_CLIENT_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_client_device_t iree_hal_remote_client_device_t;

// Remote buffer proxy. Wraps the HAL buffer base with a server-assigned
// resource_id. All immutable properties (memory_type, allowed_access,
// allowed_usage, allocation_size) are cached in the base struct and served
// locally — no round-trip for property queries.
typedef struct iree_hal_remote_client_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;

  // Back-pointer to the owning device for release notifications.
  iree_hal_remote_client_device_t* device;

  // Server-assigned resource ID.
  iree_hal_remote_resource_id_t resource_id;
} iree_hal_remote_client_buffer_t;

// Creates a buffer proxy wrapping a server-assigned resource.
iree_status_t iree_hal_remote_client_buffer_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns the resource_id from a remote client buffer proxy.
// The buffer must have been allocated by a remote client allocator.
static inline iree_hal_remote_resource_id_t
iree_hal_remote_client_buffer_resource_id(iree_hal_buffer_t* buffer) {
  return ((iree_hal_remote_client_buffer_t*)buffer)->resource_id;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BUFFER_H_
