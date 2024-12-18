// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_FD_FILE_H_
#define IREE_HAL_UTILS_FD_FILE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_fd_file_t
//===----------------------------------------------------------------------===//

// Creates a file backed by |handle| on disk.
// Only supports file handles of IREE_IO_FILE_HANDLE_TYPE_FD.
// File handles are stateless and each host file opened from one may see
// different versions of the file depending on the platform and file type.
IREE_API_EXPORT iree_status_t iree_hal_fd_file_from_handle(
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_FD_FILE_H_
