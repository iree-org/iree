// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_FILE_HANDLE_H_
#define IREE_IO_FILE_HANDLE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/io/stream.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and enums
//===----------------------------------------------------------------------===//

// Bits defining which operations are allowed on a file.
enum iree_io_file_access_bits_t {
  // Allows operations that read from the file.
  IREE_IO_FILE_ACCESS_READ = 1u << 0,
  // Allows operations that write to the file.
  IREE_IO_FILE_ACCESS_WRITE = 1u << 1,
};
typedef uint32_t iree_io_file_access_t;

//===----------------------------------------------------------------------===//
// iree_io_file_handle_primitive_t
//===----------------------------------------------------------------------===//

// Defines the type of a platform file handle primitive.
// Each type may only be usable in a subset of implementations and platforms and
// may even vary based on the runtime device properties or file instance.
//
// See the notes on each type for requirements; compatibility often requires
// the handle to check and trying to import/export is the most reliable way to
// check for support.
typedef enum iree_io_file_handle_type_e {
  // A fixed-size range of host memory.
  // The handle creator is responsible for ensuring the memory remains live for
  // as long as the file handle referencing it.
  IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION = 0u,

  // TODO(benvanik): file descriptor, FILE*, HANDLE, etc.
} iree_io_file_handle_type_t;

// A platform handle to a file primitive.
// Unowned/unreferenced on its own but may be embedded in instances that manage
// lifetime such as iree_io_file_handle_t.
typedef union iree_io_file_handle_primitive_value_t {
  // IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION
  iree_byte_span_t host_allocation;
} iree_io_file_handle_primitive_value_t;

// A (type, value) pair describing a system file primitive handle.
typedef struct iree_io_file_handle_primitive_t {
  iree_io_file_handle_type_t type;
  iree_io_file_handle_primitive_value_t value;
} iree_io_file_handle_primitive_t;

//===----------------------------------------------------------------------===//
// iree_io_file_handle_t
//===----------------------------------------------------------------------===//

typedef void(IREE_API_PTR* iree_io_file_handle_release_fn_t)(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive);

// A callback issued when a file is released.
typedef struct {
  // Callback function pointer.
  iree_io_file_handle_release_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_io_file_handle_release_callback_t;

// Returns a no-op file release callback that implies that no cleanup is
// required.
static inline iree_io_file_handle_release_callback_t
iree_io_file_handle_release_callback_null(void) {
  iree_io_file_handle_release_callback_t callback = {NULL, NULL};
  return callback;
}

// A reference-counted file handle wrapping a platform primitive.
// Handles are stateless (unlike file descriptors) and only used to reference
// files.
typedef struct iree_io_file_handle_t iree_io_file_handle_t;

// Wraps a platform file primitive |handle_primitive| in a reference-counted
// file handle. |allowed_access| declares which operations are allowed on the
// handle and may be more restrictive than the underlying platform primitive.
// The optional provided |release_callback| will be issued when the last
// reference to the handle is released.
IREE_API_EXPORT iree_status_t iree_io_file_handle_wrap(
    iree_io_file_access_t allowed_access,
    iree_io_file_handle_primitive_t handle_primitive,
    iree_io_file_handle_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle);

// Wraps a |host_allocation| in a reference-counted file handle.
// |allowed_access| declares which operations are allowed on the handle and may
// be more restrictive than the underlying memory protection.
// The optional provided |release_callback| will be issued when the last
// reference to the handle is released.
IREE_API_EXPORT iree_status_t iree_io_file_handle_wrap_host_allocation(
    iree_io_file_access_t allowed_access, iree_byte_span_t host_allocation,
    iree_io_file_handle_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_io_file_handle_t** out_handle);

// Retains the file |handle| for the caller.
IREE_API_EXPORT void iree_io_file_handle_retain(iree_io_file_handle_t* handle);

// Releases the file |handle| and releases the underlying platform primitive if
// the caller held the last owning reference.
IREE_API_EXPORT void iree_io_file_handle_release(iree_io_file_handle_t* handle);

// Returns the allowed access operations on the file |handle|.
// As handles can specify more restrictive access than the underlying platform
// primitive users of the handles must always verify accesses using this value.
IREE_API_EXPORT iree_io_file_access_t
iree_io_file_handle_access(const iree_io_file_handle_t* handle);

// Returns the underlying platform file |handle| primitive value.
// Only valid for as long as the file handle is retained. Closing or otherwise
// invalidating the platform primitive while the handle still has uses will
// result in undefined behavior.
IREE_API_EXPORT iree_io_file_handle_primitive_t
iree_io_file_handle_primitive(const iree_io_file_handle_t* handle);

// Returns the type of the file |handle|.
static inline iree_io_file_handle_type_t iree_io_file_handle_type(
    const iree_io_file_handle_t* handle) {
  return iree_io_file_handle_primitive(handle).type;
}

// Returns the underlying platform file |handle| primitive value.
// Only valid for as long as the file handle is retained. Closing or otherwise
// invalidating the platform primitive while the handle still has uses will
// result in undefined behavior.
static inline iree_io_file_handle_primitive_value_t iree_io_file_handle_value(
    const iree_io_file_handle_t* handle) {
  return iree_io_file_handle_primitive(handle).value;
}

// Flushes pending writes of |handle| to its backing storage.
IREE_API_EXPORT iree_status_t
iree_io_file_handle_flush(iree_io_file_handle_t* handle);

//===----------------------------------------------------------------------===//
// iree_io_stream_t utilities
//===----------------------------------------------------------------------===//

// TODO(benvanik): remove/rework iree_io_stream_open so that it doesn't pull in
// any implementations by putting callbacks on the file handle constructors.

// Opens a stream from the given |file_handle| at the absolute |file_offset|.
// The returned stream will retain the file until it is released.
IREE_API_EXPORT iree_status_t iree_io_stream_open(
    iree_io_stream_mode_t mode, iree_io_file_handle_t* file_handle,
    uint64_t file_offset, iree_allocator_t host_allocator,
    iree_io_stream_t** out_stream);

// Writes up to |length| bytes of |source_file_handle| starting at offset
// |source_file_offset| to the target |stream|. |host_allocator| may be used
// for transient allocations required during file I/O.
IREE_API_EXPORT iree_status_t iree_io_stream_write_file(
    iree_io_stream_t* stream, iree_io_file_handle_t* source_file_handle,
    uint64_t source_file_offset, iree_io_stream_pos_t length,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_FILE_HANDLE_H_
