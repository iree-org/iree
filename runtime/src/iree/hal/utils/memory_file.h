// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_MEMORY_FILE_H_
#define IREE_HAL_UTILS_MEMORY_FILE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_memory_file_t
//===----------------------------------------------------------------------===//

// Creates a file handle backed by |contents| without copying the data.
// |release_callback| will be called when the file is destroyed.
// If the memory can be imported into a usable staging buffer |device_allocator|
// will be used to do so.
IREE_API_EXPORT iree_status_t iree_hal_memory_file_wrap(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file);

//===----------------------------------------------------------------------===//
// EXPERIMENTAL: synchronous file read/write API
//===----------------------------------------------------------------------===//
// This is incomplete and may not appear like this on the iree_hal_file_t
// vtable; this does work for memory files though.

// Returns the memory access allowed to the file.
// This may be more strict than the original file handle backing the resource
// if for example we want to prevent particular users from mutating the file.
IREE_API_EXPORT iree_hal_memory_access_t
iree_hal_file_allowed_access(iree_hal_file_t* file);

// Returns the total accessible range of the file.
// This may be a portion of the original file backing this handle.
IREE_API_EXPORT uint64_t iree_hal_file_length(iree_hal_file_t* file);

// Returns an optional device-accessible storage buffer representing the file.
// Available if the implementation is able to perform import/address-space
// mapping/etc such that device-side transfers can directly access the resources
// as if they were a normal device buffer.
IREE_API_EXPORT iree_hal_buffer_t* iree_hal_file_storage_buffer(
    iree_hal_file_t* file);

// TODO(benvanik): truncate/extend? (both can be tricky with async)

// Synchronously reads a segment of |file| into |buffer|.
// Blocks the caller until completed. Buffers are always host mappable.
IREE_API_EXPORT iree_status_t iree_hal_file_read(
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length);

// Synchronously writes a segment of |buffer| into |file|.
// Blocks the caller until completed. Buffers are always host mappable.
IREE_API_EXPORT iree_status_t iree_hal_file_write(
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_MEMORY_FILE_H_
