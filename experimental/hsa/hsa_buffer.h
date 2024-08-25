// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HSA_BUFFER_H_
#define IREE_EXPERIMENTAL_HSA_BUFFER_H_

#include "experimental/hsa/hsa_headers.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef void* hsa_device_pointer_t;

typedef enum iree_hal_hsa_buffer_type_e {
  // Device local buffer
  IREE_HAL_HSA_BUFFER_TYPE_DEVICE = 0,
  // Host local buffer
  IREE_HAL_HSA_BUFFER_TYPE_HOST,
  // Host local buffer.
  IREE_HAL_HSA_BUFFER_TYPE_HOST_REGISTERED,
  // Device local buffer.
  IREE_HAL_HSA_BUFFER_TYPE_ASYNC,
  // Externally registered buffer whose providence is unknown.
  // Must be freed by the user.
  IREE_HAL_HSA_BUFFER_TYPE_EXTERNAL,
  // Kernel arguments buffer
  IREE_HAL_HSA_BUFFER_TYPE_KERNEL_ARG,

} iree_hal_hsa_buffer_type_t;

// Wraps a HSA allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_hsa_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_hsa_buffer_type_t buffer_type, hsa_device_pointer_t device_ptr,
    void* host_ptr, iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns the underlying HSA buffer type.
iree_hal_hsa_buffer_type_t iree_hal_hsa_buffer_type(
    const iree_hal_buffer_t* buffer);

// Returns the HSA base pointer for the given |buffer|.
// This is the entire allocated_buffer and must be offset by the buffer
// byte_offset and byte_length when used.
hsa_device_pointer_t iree_hal_hsa_buffer_device_pointer(
    const iree_hal_buffer_t* buffer);

hsa_device_pointer_t iree_hal_hsa_buffer_device_pointer(
    const iree_hal_buffer_t* buffer);

// Returns the HSA host pointer for the given |buffer|, if available.
void* iree_hal_hsa_buffer_host_pointer(const iree_hal_buffer_t* buffer);

// Drops the release callback so that when the buffer is destroyed no callback
// will be made. This is not thread safe but all callers are expected to be
// holding an allocation and the earliest the buffer could be destroyed is after
// this call returns and the caller has released its reference.
void iree_hal_hsa_buffer_drop_release_callback(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HSA_BUFFER_H_
