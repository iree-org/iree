// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_UTIL_BUFFER_TABLE_H_
#define IREE_EXPERIMENTAL_STREAMING_UTIL_BUFFER_TABLE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef uint64_t iree_hal_streaming_deviceptr_t;
typedef uint64_t iree_hal_streaming_any_ptr_t;  // Host or device pointer.

//===----------------------------------------------------------------------===//
// iree_hal_streaming_buffer_table_t
//===----------------------------------------------------------------------===//

// Buffer table for mapping host or device pointers to stream buffers.
// The table owns the buffers it contains and manages their lifetime.
typedef struct iree_hal_streaming_buffer_table_t
    iree_hal_streaming_buffer_table_t;

// Allocates a new buffer table.
// The table starts empty and grows as needed.
iree_status_t iree_hal_streaming_buffer_table_allocate(
    iree_allocator_t host_allocator,
    iree_hal_streaming_buffer_table_t** out_table);

// Frees a buffer table and releases all registered buffers.
// Safe to call with NULL.
void iree_hal_streaming_buffer_table_free(
    iree_hal_streaming_buffer_table_t* table);

// Inserts a buffer into the table.
// The table takes ownership of the buffer and will release it when removed
// or when the table is freed.
// The buffer's device_ptr and optional host_ptr are registered for lookup.
// Returns an error if the buffer's pointers are already registered.
iree_status_t iree_hal_streaming_buffer_table_insert(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_buffer_t* buffer);

// Removes a buffer from the table by host or device pointer.
// The pointer may be anywhere within the buffer's range.
// The buffer is released as part of removal.
// Returns an error if no buffer contains the pointer.
iree_status_t iree_hal_streaming_buffer_table_remove(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_any_ptr_t any_ptr);

// Looks up a buffer by host or device pointer.
// The pointer may be anywhere within the buffer's range.
// Returns a borrowed reference to the buffer (does not transfer ownership).
// Returns an error if no buffer contains the pointer.
iree_status_t iree_hal_streaming_buffer_table_lookup(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_any_ptr_t any_ptr,
    iree_hal_streaming_buffer_t** out_buffer);

// Looks up a buffer that contains the specified address range.
// The pointer may be a host or device pointer.
// Returns a borrowed reference to the buffer (does not transfer ownership).
// Returns an error if no buffer contains the entire range [any_ptr,
// any_ptr + size).
iree_status_t iree_hal_streaming_buffer_table_lookup_range(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_any_ptr_t any_ptr, iree_device_size_t size,
    iree_hal_streaming_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_STREAMING_UTIL_BUFFER_TABLE_H_
