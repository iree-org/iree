// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-size pool of ready operation entries for passing work from the poll
// thread to workers without hot-path allocation.

#ifndef IREE_ASYNC_UTIL_READY_POOL_H_
#define IREE_ASYNC_UTIL_READY_POOL_H_

#include "iree/async/operation.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A ready operation entry carrying (operation, poll_events) to workers.
typedef struct iree_async_posix_ready_op_t {
  iree_atomic_slist_entry_t slist_entry;
  iree_async_operation_t* operation;
  iree_async_poll_events_t poll_events;  // IREE_ASYNC_POLL_EVENT_IN, etc.
} iree_async_posix_ready_op_t;

// Fixed-size pool of ready operation entries with lock-free freelist.
typedef struct iree_async_posix_ready_pool_t {
  iree_async_posix_ready_op_t* entries;
  iree_host_size_t capacity;
  iree_atomic_slist_t free_list;
  iree_allocator_t allocator;
} iree_async_posix_ready_pool_t;

// Initializes a ready pool using externally-provided entry storage.
// |entries| must point to |capacity| entries of storage that outlives the pool.
// The pool does NOT free the entries on deinitialize (caller manages storage).
// Use this when embedding pool entries in a larger allocation.
void iree_async_posix_ready_pool_initialize_with_storage(
    iree_host_size_t capacity, iree_async_posix_ready_op_t* entries,
    iree_async_posix_ready_pool_t* out_pool);

// Deinitializes the pool. Frees storage only if it was internally allocated.
void iree_async_posix_ready_pool_deinitialize(
    iree_async_posix_ready_pool_t* pool);

// Acquires a ready entry from the pool.
// Returns NULL if the pool is exhausted.
iree_async_posix_ready_op_t* iree_async_posix_ready_pool_acquire(
    iree_async_posix_ready_pool_t* pool);

// Releases a ready entry back to the pool.
void iree_async_posix_ready_pool_release(iree_async_posix_ready_pool_t* pool,
                                         iree_async_posix_ready_op_t* ready_op);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_READY_POOL_H_
