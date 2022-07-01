// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_BUFFER_H_
#define IREE_HAL_LEVEL_ZERO_BUFFER_H_

#include "experimental/level_zero/level_zero_headers.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef void* iree_hal_level_zero_device_ptr_t;

// Wraps a Level Zero allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_level_zero_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_level_zero_device_ptr_t device_ptr, void* host_ptr,
    iree_hal_buffer_t** out_buffer);

// Returns the Level Zero base pointer for the given |buffer|.
// This is the entire allocated_buffer and must be offset by the buffer
// byte_offset and byte_length when used.
iree_hal_level_zero_device_ptr_t iree_hal_level_zero_buffer_device_pointer(
    iree_hal_buffer_t* buffer);

// Returns the Level Zero host pointer for the given |buffer|, if available.
void* iree_hal_level_zero_buffer_host_pointer(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_BUFFER_H_
