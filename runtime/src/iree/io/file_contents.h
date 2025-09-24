// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_FILE_CONTENTS_H_
#define IREE_IO_FILE_CONTENTS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_io_file_contents_t
//===----------------------------------------------------------------------===//

// Non-reference counted wrapper for loaded file contents.
// Provided for ease of use when simply loading files into memory for
// consumption exclusively as host-addressable pointers.
//
// Contents structures returned from one of the iree_io_file_contents_* APIs
// must be freed with iree_io_file_contents_free.
typedef struct iree_io_file_contents_t {
  // Allocator used for the buffer.
  iree_allocator_t allocator;
  // Allocated buffer memory containing the file contents.
  union {
    iree_byte_span_t buffer;
    iree_const_byte_span_t const_buffer;
  };
  // Mapping handle, if the file contents are mapped.
  iree_io_file_mapping_t* mapping;
} iree_io_file_contents_t;

// Frees the file contents using the original buffer allocator.
IREE_API_EXPORT void iree_io_file_contents_free(
    iree_io_file_contents_t* contents);

// Returns an allocator that deallocates the file |contents|.
// This can be passed to functions that require a deallocation mechanism.
//
// Example:
//   void consumer_fn(iree_byte_span_t data, iree_allocator_t data_allocator);
//   iree_io_file_contents_t* contents = ...;
//   consumer_fn(contents->buffer, iree_io_file_contents_deallocator(contents));
IREE_API_EXPORT iree_allocator_t
iree_io_file_contents_deallocator(iree_io_file_contents_t* contents);

// Reads the contents of stdin until EOF into memory.
// The contents will specify up until EOF and the allocation will have a
// trailing NUL to allow use as a C-string (assuming the contents themselves
// don't contain NUL).
//
// Returns the contents of the file in |out_contents|.
// |host_allocator| is used to allocate the memory and the caller must use
// iree_io_file_contents_free to release the memory.
IREE_API_EXPORT iree_status_t iree_io_file_contents_read_stdin(
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents);

// Reads file contents into memory as a clone.
// The underlying file may change or be deleted after the read completes.
//
// Returns the cloned contents of the file in |out_contents|.
// |host_allocator| is used to allocate the memory and the caller must use
// iree_io_file_contents_free to release the memory.
IREE_API_EXPORT iree_status_t iree_io_file_contents_read(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_io_file_contents_t** out_contents);

// Maps file contents into memory for the specified access.
// The file must remain valid for the lifetime of the mapping. Changes in file
// size will not be respected.
//
// Returns the mapped contents of the file in |out_contents|.
// |host_allocator| is used to allocate the memory and the caller must use
// iree_io_file_contents_free to release the memory.
IREE_API_EXPORT iree_status_t iree_io_file_contents_map(
    iree_string_view_t path, iree_io_file_access_t access,
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents);

// Synchronously writes a byte buffer into a file.
// |host_allocator| may be used for transient memory during the write operation.
// Existing contents are overwritten.
IREE_API_EXPORT iree_status_t iree_io_file_contents_write(
    iree_string_view_t path, iree_const_byte_span_t contents,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_FILE_CONTENTS_H_
