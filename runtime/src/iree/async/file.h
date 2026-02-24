// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_FILE_H_
#define IREE_ASYNC_FILE_H_

#include "iree/async/primitive.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;

//===----------------------------------------------------------------------===//
// File
//===----------------------------------------------------------------------===//

// A proactor-managed async file handle.
// Created via iree_async_file_import() (importing an existing handle)
// or as the result of an async open operation.
//
// An iree_async_file_t wraps a platform file descriptor/handle and is bound to
// a specific proactor for async I/O operations. Files are created by importing
// an existing platform handle via iree_async_file_import(), or
// obtained as the result of an async open operation.
//
// All file I/O uses positioned semantics (pread/pwrite): each operation
// specifies its own offset, eliminating shared-position races between
// concurrent operations on the same file.
typedef struct iree_async_file_t {
  iree_atomic_ref_count_t ref_count;

  // The proactor this file is bound to. All operations on this file must be
  // submitted to this proactor. Not retained (proactor outlives files).
  iree_async_proactor_t* proactor;

  // Underlying platform handle.
  iree_async_primitive_t primitive;

  // io_uring fixed file index for reduced syscall overhead (-1 if not
  // registered). Backend-specific optimization; ignored on other platforms.
  int32_t fixed_file_index;

  IREE_TRACE(char debug_path[256];)
} iree_async_file_t;

// Imports an existing platform handle as a proactor-managed file.
//
// Use this to bring externally-opened files under proactor management for
// async read/write operations.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Ownership:
//   The proactor takes logical ownership of the handle and will close it when
//   the file is released or a close operation completes. The caller must not
//   close the handle after a successful import.
//
// Platform handles:
//   - POSIX: int fd (file descriptor from open())
//   - Windows: HANDLE (from CreateFile)
//
// Returns:
//   IREE_STATUS_OK: File imported successfully.
//   IREE_STATUS_INVALID_ARGUMENT: Invalid handle.
IREE_API_EXPORT iree_status_t iree_async_file_import(
    iree_async_proactor_t* proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file);

// Increments the reference count.
IREE_API_EXPORT void iree_async_file_retain(iree_async_file_t* file);

// Decrements the reference count and destroys if it reaches zero.
IREE_API_EXPORT void iree_async_file_release(iree_async_file_t* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_FILE_H_
