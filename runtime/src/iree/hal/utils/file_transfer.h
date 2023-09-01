// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_FILE_TRANSFER_H_
#define IREE_HAL_UTILS_FILE_TRANSFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Generic file transfer IO implementation
//===----------------------------------------------------------------------===//

#define IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT 0
#define IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT 0

// Options for file-based transfer operations.
typedef struct iree_hal_file_transfer_options_t {
  // Loop to use for asynchronous host operations. If inline then the transfer
  // will run synchronously with the caller.
  iree_loop_t loop;
  // Total number of staging buffer chunks to allocate.
  // Setting to >1 will allow for overlapped staging and transfer at the cost
  // of additional staging buffer memory consumption.
  // IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT can be used to have the
  // implementation select a chunk size based on whether the device can benefit
  // from overlapping staging.
  iree_device_size_t chunk_count;
  // Maximum size of chunks in bytes. The size may be adjusted to meet alignment
  // requirements of the implementation.
  // IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT can be used to have the
  // implementation select a chunk size based on the size of the transfer.
  iree_device_size_t chunk_size;
} iree_hal_file_transfer_options_t;

// EXPERIMENTAL: eventually we'll focus this only on emulating support where
// otherwise unavailable. For now no HAL targets support files and all use this.
//
// Performs a streaming read of |source_file| into |target_buffer| using
// host-based staging buffers. This implementation may require staging buffers
// in which case |options.chunk_size| specifies the maximum size in bytes of
// each chunk and |options.chunk_count| specifies how many chunks will be
// allocated at once.
//
// The provided |options.loop| is used for any asynchronous host operations
// performed as part of the transfer.
//
// WARNING: this only works with memory files as created via
// iree_hal_memory_file_wrap.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_read_streaming(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags,
    iree_hal_file_transfer_options_t options);

// EXPERIMENTAL: eventually we'll focus this only on emulating support where
// otherwise unavailable. For now no HAL targets support files and all use this.
//
// Performs a streaming write of |source_buffer| into |target_file| using
// host-based staging buffers. This implementation may require staging buffers
// in which case |options.chunk_size| specifies the maximum size in bytes of
// each chunk and |options.chunk_count| specifies how many chunks will be
// allocated at once.
//
// The provided |options.loop| is used for any asynchronous host operations
// performed as part of the transfer.
//
// WARNING: this only works with memory files as created via
// iree_hal_memory_file_wrap.
IREE_API_EXPORT iree_status_t iree_hal_device_queue_write_streaming(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags,
    iree_hal_file_transfer_options_t options);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_FILE_TRANSFER_H_
