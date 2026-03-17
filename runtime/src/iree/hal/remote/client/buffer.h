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

  // Active mapping state. Set during map_range, cleared on unmap_range.
  // Used by flush_range to locate the staging data (the vtable's
  // flush_range only receives offset+length, not the mapping struct).
  uint8_t* active_mapping_data;
  iree_device_size_t active_mapping_offset;
  iree_device_size_t active_mapping_length;
} iree_hal_remote_client_buffer_t;

// Creates a buffer proxy wrapping a server-assigned resource.
iree_status_t iree_hal_remote_client_buffer_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns the resource_id from a remote client buffer proxy.
// Handles subspan buffers by traversing to the root allocation.
static inline iree_hal_remote_resource_id_t
iree_hal_remote_client_buffer_resource_id(iree_hal_buffer_t* buffer) {
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
  if (allocated) buffer = allocated;
  return ((iree_hal_remote_client_buffer_t*)buffer)->resource_id;
}

// Updates the resource_id on a buffer proxy. Used to resolve provisional
// IDs to canonical server-assigned IDs when the ADVANCE frame arrives.
// Must be called before the corresponding semaphore is signaled (so the
// application sees the resolved ID when it wakes from the wait).
static inline void iree_hal_remote_client_buffer_set_resource_id(
    iree_hal_buffer_t* buffer, iree_hal_remote_resource_id_t resolved_id) {
  ((iree_hal_remote_client_buffer_t*)buffer)->resource_id = resolved_id;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BUFFER_H_
