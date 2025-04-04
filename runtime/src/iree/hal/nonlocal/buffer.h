// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_NONLOCAL_BUFFER_H_
#define IREE_HAL_NONLOCAL_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "nl_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_nl_buffer_type_e {
  // Device local buffer; allocated with nl_mem_alloc, freed
  // with nl_mem_free.
  IREE_HAL_NL_BUFFER_TYPE_DEVICE = 0,
  // Host local buffer; allocated with nl_mem_host_alloc, freed with nl_mem_freeHost.
  IREE_HAL_NL_BUFFER_TYPE_HOST,
  // Externally registered buffer whose providence is unknown.
  // Must be freed by the user.
  IREE_HAL_NL_BUFFER_TYPE_EXTERNAL,
} iree_hal_nl_buffer_type_t;

// Wraps a NL allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_nl_buffer_wrap(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_nl_buffer_type_t buffer_type, nl_mem_device_ptr_t device_ptr,
    void* host_ptr, iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns the underlying NL buffer type of the given |buffer|.
iree_hal_nl_buffer_type_t iree_hal_nl_buffer_type(
    const iree_hal_buffer_t* buffer);

// Returns the NL base device pointer for the given |buffer|.
//
// Note that this is the entire allocated_buffer and must be offset by the
// buffer byte_offset and byte_length when used.
nl_mem_device_ptr_t iree_hal_nl_buffer_device_pointer(
    const iree_hal_buffer_t* buffer);

// Returns the NL host pointer for the given |buffer|, if available.
void* iree_hal_nl_buffer_host_pointer(const iree_hal_buffer_t* buffer);

// Drops the release callback so that when the buffer is destroyed no callback
// will be made. This is not thread safe but all callers are expected to be
// holding an allocation and the earliest the buffer could be destroyed is after
// this call returns and the caller has released its reference.
void iree_hal_nl_buffer_drop_release_callback(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_NONLOCAL_BUFFER_H_
