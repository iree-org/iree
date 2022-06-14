// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_HEAP_IMPL_H_
#define IREE_HAL_BUFFER_HEAP_IMPL_H_

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Private utilities for working with heap buffers
//===----------------------------------------------------------------------===//

// Shared heap allocator statistics; owned by a heap allocator.
// Access to the base statistics must be guarded by |mutex|.
typedef struct iree_hal_heap_allocator_statistics_t {
  iree_slim_mutex_t mutex;
  iree_hal_allocator_statistics_t base;
} iree_hal_heap_allocator_statistics_t;

// Allocates a new heap buffer from the specified |data_allocator|.
// |host_allocator| is used for the iree_hal_buffer_t metadata. If both
// |data_allocator| and |host_allocator| are the same the buffer will be created
// as a flat slab. |out_buffer| must be released by the caller.
iree_status_t iree_hal_heap_buffer_create(
    iree_hal_allocator_t* allocator,
    iree_hal_heap_allocator_statistics_t* statistics,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_const_byte_span_t initial_data, iree_allocator_t data_allocator,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Wraps an existing host allocation in a buffer.
// When the buffer is destroyed the provided |release_callback| will be called.
//
// The buffer must be aligned to at least IREE_HAL_HEAP_BUFFER_ALIGNMENT and if
// it is not the call will fail with IREE_STATUS_OUT_OF_RANGE.
//
// |out_buffer| must be released by the caller. |data| must be kept live for the
// lifetime of the wrapping buffer.
iree_status_t iree_hal_heap_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_byte_span_t data, iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_HEAP_IMPL_H_
