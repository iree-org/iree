// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_STDIO_STREAM_H_
#define IREE_IO_STDIO_STREAM_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/io/stream.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_io_stdio_stream_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): rework this to handle optional features like EXISTS and such.
// fopen support is not very well specced and some impls don't support all modes
// and we'll have to emulate.
//
// Roughly aligns to the fopen modes:
//  READ: open existing file for reading.
//  WRITE: open existing file for writing.
//  READ|WRITE: open existing file for reading/writing.
//  WRITE|DISCARD: open new file (discarding existing) for writing.
//  WRITE|APPEND: open existing file for writing, start at end.
//  READ|WRITE|DISCARD: open new file (discarding existing) for reading/writing.
//  READ|WRITE|APPEND: open existing file for reading/writing, start at end.
enum iree_io_stdio_stream_mode_bits_t {
  IREE_IO_STDIO_STREAM_MODE_DISCARD = 1u << 0,
  IREE_IO_STDIO_STREAM_MODE_READ = 1u << 1,
  IREE_IO_STDIO_STREAM_MODE_WRITE = 1u << 2,
  IREE_IO_STDIO_STREAM_MODE_APPEND = 1u << 3,
};
typedef uint32_t iree_io_stdio_stream_mode_t;

// Wraps an existing stdio |handle|. If |owns_handle| is true then the
// file will be closed when the stream is destroyed.
IREE_API_EXPORT iree_status_t iree_io_stdio_stream_wrap(
    iree_io_stream_mode_t mode, FILE* handle, bool owns_handle,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream);

// Opens a file at |path| using fopen with the mode determined by |mode|.
IREE_API_EXPORT iree_status_t iree_io_stdio_stream_open(
    iree_io_stdio_stream_mode_t mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_STDIO_STREAM_H_
