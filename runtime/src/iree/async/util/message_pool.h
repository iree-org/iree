// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pre-allocated pool for cross-proactor message passing.
//
// Provides a bounded, lock-free MPSC queue of message entries. Multiple sender
// threads can concurrently send messages via iree_async_message_pool_send(),
// while a single consumer (the poll thread) flushes and processes them via
// iree_async_message_pool_flush() + iree_async_message_pool_release().
//
// Entries are pre-allocated as trailing data in the proactor's allocation
// (via IREE_STRUCT_LAYOUT). No per-message heap allocation occurs at runtime.
// When the pool is exhausted, send returns IREE_STATUS_RESOURCE_EXHAUSTED,
// providing consistent backpressure behavior across all proactor backends.

#ifndef IREE_ASYNC_UTIL_MESSAGE_POOL_H_
#define IREE_ASYNC_UTIL_MESSAGE_POOL_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default message pool capacity if not specified in proactor options.
#define IREE_ASYNC_MESSAGE_POOL_DEFAULT_CAPACITY 256

// A single message entry in the pool. Each entry carries a 64-bit payload
// and is linked into either the free list or the pending list via its
// slist_entry.
typedef struct iree_async_message_pool_entry_t {
  // Intrusive lock-free singly-linked list node. Must be first for
  // iree_atomic_slist compatibility.
  iree_atomic_slist_entry_t slist_entry;
  // Arbitrary 64-bit message payload set by the sender and read by the
  // receiver's callback during flush.
  uint64_t message_data;
} iree_async_message_pool_entry_t;

// Fixed-size pool of message entries with lock-free send and single-consumer
// flush. The free list uses an atomic slist for thread-safe send from any
// thread. The pending list accumulates sent messages for batch flush by the
// poll thread.
typedef struct iree_async_message_pool_t {
  // Entries available for new sends. Senders pop from here.
  iree_atomic_slist_t free_list;
  // Messages awaiting delivery. Senders push here after filling an entry.
  iree_atomic_slist_t pending_list;
  // Total number of entries in the pool (for diagnostics, not for logic).
  iree_host_size_t capacity;
} iree_async_message_pool_t;

// Initializes a message pool using externally-provided entry storage.
// |entries| must point to |capacity| contiguous entries that outlive the pool.
// All entries are pushed onto the free list, ready for use by send.
void iree_async_message_pool_initialize(
    iree_host_size_t capacity, iree_async_message_pool_entry_t* entries,
    iree_async_message_pool_t* out_pool);

// Deinitializes the pool. All entries must have been flushed and released
// back to the pool before calling this.
void iree_async_message_pool_deinitialize(iree_async_message_pool_t* pool);

// Sends a message by acquiring an entry from the free list, writing
// |message_data| into it, and pushing it onto the pending list.
//
// Thread-safe: may be called from any thread concurrently. The caller must
// wake the target proactor's poll thread after a successful send so that
// the message is delivered promptly.
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the pool has no free entries.
iree_status_t iree_async_message_pool_send(iree_async_message_pool_t* pool,
                                           uint64_t message_data);

// Flushes all pending messages from the pool in approximate FIFO order.
// Returns the head of the flushed entry list, or NULL if no messages were
// pending. The caller owns the returned list and must release each entry
// back to the pool via iree_async_message_pool_release() after processing.
//
// Single-consumer: must be called from the poll thread only.
iree_async_message_pool_entry_t* iree_async_message_pool_flush(
    iree_async_message_pool_t* pool);

// Returns the next entry in a flushed list, or NULL at the end.
static inline iree_async_message_pool_entry_t*
iree_async_message_pool_entry_next(iree_async_message_pool_entry_t* entry) {
  iree_atomic_slist_entry_t* next = entry->slist_entry.next;
  if (!next) return NULL;
  return (iree_async_message_pool_entry_t*)next;
}

// Releases an entry back to the pool's free list after the caller has
// finished processing it. The entry must have come from a prior flush.
void iree_async_message_pool_release(iree_async_message_pool_t* pool,
                                     iree_async_message_pool_entry_t* entry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_MESSAGE_POOL_H_
