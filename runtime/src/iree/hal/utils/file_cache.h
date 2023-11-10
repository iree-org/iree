// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_FILE_CACHE_H_
#define IREE_HAL_UTILS_FILE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An in-memory cache of file handles to devices and queues that have an open
// HAL file reference. A single file cache may be shared across multiple devices
// and/or multiple queues within individual devices.
//
// Thread-safe: multiple threads can access the cache concurrently.
typedef struct iree_hal_file_cache_t iree_hal_file_cache_t;

// Creates a new empty file cache.
IREE_API_EXPORT iree_status_t iree_hal_file_cache_create(
    iree_allocator_t host_allocator, iree_hal_file_cache_t** out_file_cache);

// Retains the given |file_cache| for the caller.
IREE_API_EXPORT void iree_hal_file_cache_retain(
    iree_hal_file_cache_t* file_cache);

// Releases the given |file_cache| from the caller.
IREE_API_EXPORT void iree_hal_file_cache_release(
    iree_hal_file_cache_t* file_cache);

// Drops all cached file references.
// Note that resources may not be returned to the system immediately as others
// may still retain them. Avoid trimming in such cases as it can easily lead
// to multiple open files pointing at the same underlying resources.
IREE_API_EXPORT void iree_hal_file_cache_trim(
    iree_hal_file_cache_t* file_cache);

// Looks up the file |handle| for use on |device| with any of the queues
// specified with |queue_affinity| and returns it retained in |out_file|.
// If the file has not been used on the device yet it will be imported and
// cached until the cache is trimmed.
IREE_API_EXPORT iree_status_t iree_hal_file_cache_lookup(
    iree_hal_file_cache_t* file_cache, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_hal_external_file_flags_t flags,
    iree_hal_file_t** out_file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_FILE_CACHE_H_
