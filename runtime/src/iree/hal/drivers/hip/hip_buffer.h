// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_BUFFER_H_
#define IREE_HAL_DRIVERS_HIP_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/hip_headers.h"

typedef enum iree_hal_hip_buffer_type_e {
  // Device local buffer; allocated with hipMalloc/hipMallocManaged, freed
  // with hipFree.
  IREE_HAL_HIP_BUFFER_TYPE_DEVICE = 0,
  // Host local buffer; allocated with hipHostMalloc, freed with hipHostFree.
  IREE_HAL_HIP_BUFFER_TYPE_HOST,
  // Host local buffer; registered with hipHostRegister, freed with
  // hipHostUnregister.
  IREE_HAL_HIP_BUFFER_TYPE_HOST_REGISTERED,
  // Device local buffer, allocated with hipMallocFromPoolAsync, freed with
  // hipFree/hipFreeAsync.
  IREE_HAL_HIP_BUFFER_TYPE_ASYNC,
  // Externally registered buffer whose providence is unknown.
  // Must be freed by the user.
  IREE_HAL_HIP_BUFFER_TYPE_EXTERNAL,
  // Wrapper of a device local buffer, allocated with
  // hipMalloc/hipMallocManaged, freed with hipFree.
  IREE_HAL_HIP_BUFFER_TYPE_WRAPPER,
} iree_hal_hip_buffer_type_t;

// Wraps a HIP allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_hip_buffer_wrap(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_hip_buffer_type_t buffer_type, hipDeviceptr_t device_ptr,
    void* host_ptr, iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns the underlying HIP buffer type.
iree_hal_hip_buffer_type_t iree_hal_hip_buffer_type(
    const iree_hal_buffer_t* buffer);

// Returns the HIP base pointer for the given |buffer|.
// This is the entire allocated_buffer and must be offset by the buffer
// byte_offset and byte_length when used.
hipDeviceptr_t iree_hal_hip_buffer_device_pointer(iree_hal_buffer_t* buffer);

// Sets the HIP base pointer for the given |buffer|.
// This is the entire allocated_buffer and must be offset by the buffer
// byte_offset and byte_length when used.
void iree_hal_hip_buffer_set_device_pointer(iree_hal_buffer_t* buffer,
                                            hipDeviceptr_t pointer);

// Marks the buffer as having an intentionally empty allocation.
void iree_hal_hip_buffer_set_allocation_empty(iree_hal_buffer_t* buffer);

// Returns the HIP host pointer for the given |buffer|, if available.
void* iree_hal_hip_buffer_host_pointer(const iree_hal_buffer_t* buffer);

// Drops the release callback so that when the buffer is destroyed no callback
// will be made. This is not thread safe but all callers are expected to be
// holding an allocation and the earliest the buffer could be destroyed is after
// this call returns and the caller has released its reference.
void iree_hal_hip_buffer_drop_release_callback(iree_hal_buffer_t* buffer);

// Sets a HIP buffer to the given |base_buffer|, if the given |base_buffer|
// is a hip buffer that wraps around another HIP buffer.
void iree_hal_hip_buffer_set_wrapped_buffer(iree_hal_buffer_t* base_buffer,
                                            iree_hal_buffer_t* wrapped_buffer);

#endif  // IREE_HAL_DRIVERS_HIP_BUFFER_H_
