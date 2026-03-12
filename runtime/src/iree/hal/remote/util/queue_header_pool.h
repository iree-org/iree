// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared utility for creating the small buffer pool used by queue channel
// frame_sender to encode queue frame headers and frontier pairs.

#ifndef IREE_HAL_REMOTE_UTIL_QUEUE_HEADER_POOL_H_
#define IREE_HAL_REMOTE_UTIL_QUEUE_HEADER_POOL_H_

#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default buffer count and size for queue channel header pools.
// 32 buffers at 256 bytes each handles queue frame headers (16 bytes) plus
// frontier pairs with up to ~7 entries each.
#define IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_COUNT 32
#define IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_SIZE 256

// Creates a buffer pool for queue channel frame_sender header encoding.
// The pool is backed by a single contiguous allocation with a self-cleaning
// region that frees itself when the pool is destroyed.
iree_status_t iree_hal_remote_create_queue_header_pool(
    iree_host_size_t buffer_count, iree_host_size_t buffer_size,
    iree_allocator_t host_allocator, iree_async_buffer_pool_t** out_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_UTIL_QUEUE_HEADER_POOL_H_
