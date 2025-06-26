// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_external_buffer_t
//===----------------------------------------------------------------------===//

// Wraps an external device-accessible |device_ptr| allocation in an
// iree_hal_buffer_t. The |release_callback| will be issued upon release of the
// last buffer reference.
iree_status_t iree_hal_amdgpu_external_buffer_wrap(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    uint64_t device_ptr, iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_transient_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_transient_buffer_t {
  iree_hal_buffer_t base;  // must be at 0

  // Device-side allocation handle in a memory pool accessible to all agents.
  // This may reside in host local memory.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_allocation_handle_t* handle;

  // Release callback that handles deallocation.
  iree_hal_buffer_release_callback_t release_callback;
} iree_hal_amdgpu_transient_buffer_t;

// Initializes a transient buffer in-place with a 0 ref count.
// The owning pool must increment the ref count to 1 before returning the
// buffer to users.
void iree_hal_amdgpu_transient_buffer_initialize(
    iree_hal_buffer_placement_t placement,
    iree_hal_amdgpu_device_allocation_handle_t* handle,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_amdgpu_transient_buffer_t* out_buffer);

// Deinitializes a transient buffer in-place assuming it has a 0 ref count.
void iree_hal_amdgpu_transient_buffer_deinitialize(
    iree_hal_amdgpu_transient_buffer_t* buffer);

// Resets |buffer| to the given parameters as if it had just been allocated.
void iree_hal_amdgpu_transient_buffer_reset(
    iree_hal_amdgpu_transient_buffer_t* buffer, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_offset,
    iree_device_size_t byte_length);

//===----------------------------------------------------------------------===//
// Buffer Resolution
//===----------------------------------------------------------------------===//

// Resolves a HAL buffer to a device-side type and pointer/handle.
// Returns success if the buffer is of a type that can be used toll-free on any
// device but does not verify the memory referenced is accessible to any
// particular device.
iree_status_t iree_hal_amdgpu_resolve_buffer(
    iree_hal_buffer_t* buffer, iree_hal_amdgpu_device_buffer_type_t* out_type,
    uint64_t* out_bits);

// Resolves a HAL buffer that is required to be a transient buffer allocated via
// iree_hal_device_queue_alloca. Fails if the buffer is any other type.
iree_status_t iree_hal_amdgpu_resolve_transient_buffer(
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_device_allocation_handle_t** out_handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_BUFFER_H_
