// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_MEMORY_STREAM_H_
#define IREE_IO_MEMORY_STREAM_H_

#include "iree/base/api.h"
#include "iree/io/stream.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_io_memory_stream_t
//===----------------------------------------------------------------------===//

typedef void(IREE_API_PTR* iree_io_memory_stream_release_fn_t)(
    void* user_data, iree_io_stream_t* stream);

// A callback issued when a memory stream is released.
typedef struct {
  // Callback function pointer.
  iree_io_memory_stream_release_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_io_memory_stream_release_callback_t;

// Returns a no-op file release callback that implies that no cleanup is
// required.
static inline iree_io_memory_stream_release_callback_t
iree_io_memory_stream_release_callback_null(void) {
  iree_io_memory_stream_release_callback_t callback = {NULL, NULL};
  return callback;
}

// Wraps a fixed-size host memory allocation |contents| in a stream.
// |release_callback| can be used to receive a callback when the stream is
// destroyed and the reference to the contents is no longer required.
IREE_API_EXPORT iree_status_t iree_io_memory_stream_wrap(
    iree_io_stream_mode_t mode, iree_byte_span_t contents,
    iree_io_memory_stream_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_MEMORY_STREAM_H_
