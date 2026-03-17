// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal file wrapper for FD-type file handles on wasm.
//
// The standard iree_hal_fd_file_t uses pread/pwrite (POSIX I/O), which is
// unavailable on wasm. On the WebGPU bridge, "fd" is a JS file object table
// index — an integer that the JS side resolves to an ArrayBuffer (or other
// readable/writable object). This wrapper stores the fd and file length
// without any POSIX dependencies.
//
// storage_buffer() returns NULL (no host pointer or device buffer). The
// queue_read/write implementations check for this and fall through to the
// FD-specific bridge imports (queue_write_buffer_from_file, etc.).

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_FD_FILE_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_FD_FILE_H_

#include "iree/base/api.h"
#include "iree/hal/file.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a file wrapper for an FD-type file handle.
// |handle| is retained for the lifetime of the file.
// |length| is the total file size in bytes — the caller must provide this
// because fstat is unavailable on wasm.
iree_status_t iree_hal_webgpu_fd_file_from_handle(
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    uint64_t length, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file);

// Returns the fd (JS file object table index) from a WebGPU FD file.
// The caller must ensure |file| was created by
// iree_hal_webgpu_fd_file_from_handle.
int iree_hal_webgpu_fd_file_fd(iree_hal_file_t* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_FD_FILE_H_
