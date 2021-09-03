// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_FILE_IO_H_
#define IREE_BASE_INTERNAL_FILE_IO_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Checks if a file exists at the provided |path|.
//
// Returns an OK status if the file definitely exists. An OK status does not
// indicate that attempts to read or write the file will succeed.
// Returns IREE_STATUS_NOT_FOUND if the file does not exist.
iree_status_t iree_file_exists(const char* path);

// Synchronously reads a file's contents into memory.
//
// Returns the contents of the file in |out_contents|.
// |allocator| is used to allocate the memory and the caller must use the same
// allocator when freeing it.
iree_status_t iree_file_read_contents(const char* path,
                                      iree_allocator_t allocator,
                                      iree_byte_span_t* out_contents);

// Synchronously writes a byte buffer into a file.
// Existing contents are overwritten.
iree_status_t iree_file_write_contents(const char* path,
                                       iree_const_byte_span_t content);

// Reads the contents of stdin until EOF into memory.
// The contents will specify up until EOF and the allocation will have a
// trailing NUL to allow use as a C-string (assuming the contents themselves
// don't contain NUL).
//
// Returns the contents of the file in |out_contents|.
// |allocator| is used to allocate the memory and the caller must use the same
// allocator when freeing it.
iree_status_t iree_stdin_read_contents(iree_allocator_t allocator,
                                       iree_byte_span_t* out_contents);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_FILE_IO_H_
