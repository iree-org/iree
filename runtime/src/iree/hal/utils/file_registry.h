// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_FILE_REGISTRY_H_
#define IREE_HAL_UTILS_FILE_REGISTRY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a file backed by |handle| using a common host implementation.
// Supported file handle types are determined based on compile configuration.
//
// Some implementations - such as for IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION -
// will try to import the backing storage directly into a usable staging buffer
// using |device_allocator| and available |queue_affinity|. Otherwise the
// file is allowed to be used with any device or queue as it is host-only.
IREE_API_EXPORT iree_status_t iree_hal_file_from_handle(
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_FILE_REGISTRY_H_
