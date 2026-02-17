// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-size pool of completion entries for passing results from workers
// to the poll thread without hot-path allocation.

#ifndef IREE_ASYNC_UTIL_COMPLETION_POOL_H_
#define IREE_ASYNC_UTIL_COMPLETION_POOL_H_

#include "iree/async/operation.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A completion entry carrying (operation, status, flags) through the queue.
typedef struct iree_async_posix_completion_t {
  iree_atomic_slist_entry_t slist_entry;
  iree_async_operation_t* operation;
  iree_status_t status;
  iree_async_completion_flags_t flags;
} iree_async_posix_completion_t;

// Fixed-size pool of completion entries with lock-free freelist.
typedef struct iree_async_posix_completion_pool_t {
  iree_async_posix_completion_t* entries;  // Contiguous allocation.
  iree_host_size_t capacity;
  iree_atomic_slist_t free_list;
  iree_allocator_t allocator;
} iree_async_posix_completion_pool_t;

// Initializes a completion pool using externally-provided entry storage.
// |entries| must point to |capacity| entries of storage that outlives the pool.
// The pool does NOT free the entries on deinitialize (caller manages storage).
// Use this when embedding pool entries in a larger allocation.
void iree_async_posix_completion_pool_initialize_with_storage(
    iree_host_size_t capacity, iree_async_posix_completion_t* entries,
    iree_async_posix_completion_pool_t* out_pool);

// Deinitializes the pool. Frees storage only if it was internally allocated.
// All entries must have been released back to the pool.
void iree_async_posix_completion_pool_deinitialize(
    iree_async_posix_completion_pool_t* pool);

// Acquires a completion entry from the pool.
// Returns NULL if the pool is exhausted (caller should handle backpressure).
iree_async_posix_completion_t* iree_async_posix_completion_pool_acquire(
    iree_async_posix_completion_pool_t* pool);

// Releases a completion entry back to the pool.
void iree_async_posix_completion_pool_release(
    iree_async_posix_completion_pool_t* pool,
    iree_async_posix_completion_t* completion);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_COMPLETION_POOL_H_
