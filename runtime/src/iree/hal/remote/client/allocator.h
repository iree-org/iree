// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_ALLOCATOR_H_
#define IREE_HAL_REMOTE_CLIENT_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_client_device_t iree_hal_remote_client_device_t;

// Remote client allocator. Proxies allocation requests to the server via
// control channel RPCs. Buffer objects are lightweight proxies that cache
// immutable properties locally and carry a server-assigned resource_id for
// use in queue operations.
//
// The allocator holds a non-retained back-pointer to the device (the device
// outlives the allocator since the device retains it via device_allocator).
typedef struct iree_hal_remote_client_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Back-pointer to the owning device for control RPC calls.
  iree_hal_remote_client_device_t* device;

  // Cached heap descriptions from BUFFER_QUERY_HEAPS response.
  // NULL until the first successful query_memory_heaps call.
  iree_hal_allocator_memory_heap_t* heaps;
  iree_host_size_t heap_count;

  iree_string_view_t identifier;
  char identifier_storage[];
} iree_hal_remote_client_allocator_t;

// Creates a remote client allocator bound to the given device.
// The device pointer is borrowed (not retained) — the device must outlive
// the allocator.
iree_status_t iree_hal_remote_client_allocator_create(
    iree_hal_remote_client_device_t* device, iree_string_view_t identifier,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_ALLOCATOR_H_
