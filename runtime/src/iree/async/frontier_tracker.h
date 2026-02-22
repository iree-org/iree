// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_FRONTIER_TRACKER_H_
#define IREE_ASYNC_FRONTIER_TRACKER_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_semaphore_t iree_async_semaphore_t;

//===----------------------------------------------------------------------===//
// Axis table
//===----------------------------------------------------------------------===//

// An entry in the axis table mapping an axis to its current epoch and optional
// semaphore. The semaphore field enables bridging: when the axis advances, the
// corresponding semaphore is signaled, connecting the frontier system to the
// proactor's semaphore-based wait infrastructure.
//
// For local axes (our own GPUs), the semaphore is the device's timeline
// semaphore — the same one the HAL driver signals on completion.
//
// For remote axes (other machines' GPUs), the semaphore is a proxy: a
// software semaphore that the network layer signals when it receives a
// frontier update from the remote machine.
typedef struct iree_async_axis_table_entry_t {
  // The full 64-bit axis identifier.
  iree_async_axis_t axis;

  // Current epoch for this axis. Updated atomically (acquire/release) when
  // the axis advances. Reads on the hot path use acquire semantics.
  iree_atomic_int64_t current_epoch;

  // Optional semaphore backing this axis. If non-NULL, advancing the epoch
  // also signals this semaphore. This enables proactor wait operations to
  // trigger on axis advancement without polling.
  // Not retained by the table (caller manages lifetime).
  iree_async_semaphore_t* semaphore;
} iree_async_axis_table_entry_t;

// A fixed-capacity table mapping axes to their current state.
//
// The axis table is the bridge between the frontier system and individual
// semaphores. When decoding a frontier from the wire, the receiver looks up
// each axis in the table to find the corresponding semaphore to wait on.
//
// Hot-path access pattern:
//   entry = &table->entries[wire_index];
//   semaphore = entry->semaphore;
//   // No locks, no hash lookups — just array indexing.
//
// Thread safety:
//   - Entries are added only during session setup (single-threaded).
//   - current_epoch is updated atomically during steady-state.
//   - The table itself (entries pointer, count) is immutable after setup.
typedef struct iree_async_axis_table_t {
  iree_async_axis_table_entry_t* entries;
  uint32_t count;
  uint32_t capacity;
} iree_async_axis_table_t;

// Initializes an axis table with the given pre-allocated storage.
// |entries| must point to an array of |capacity| entries. The table starts
// empty (count = 0).
static inline void iree_async_axis_table_initialize(
    iree_async_axis_table_t* table, iree_async_axis_table_entry_t* entries,
    uint32_t capacity) {
  table->entries = entries;
  table->count = 0;
  table->capacity = capacity;
}

// Adds an axis to the table. Must be called during setup (not thread-safe
// with concurrent reads). Returns the index of the new entry, or -1 if the
// table is full.
static inline int32_t iree_async_axis_table_add(
    iree_async_axis_table_t* table, iree_async_axis_t axis,
    iree_async_semaphore_t* semaphore) {
  if (table->count >= table->capacity) return -1;
  uint32_t index = table->count++;
  table->entries[index].axis = axis;
  iree_atomic_store(&table->entries[index].current_epoch, 0,
                    iree_memory_order_release);
  table->entries[index].semaphore = semaphore;
  return (int32_t)index;
}

// Finds the table index for a given axis. Returns -1 if not found.
// O(n) scan — acceptable because tables are small (≤64 entries typical)
// and this is only used during setup or on control-path operations.
static inline int32_t iree_async_axis_table_find(
    const iree_async_axis_table_t* table, iree_async_axis_t axis) {
  for (uint32_t i = 0; i < table->count; ++i) {
    if (table->entries[i].axis == axis) return (int32_t)i;
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// Frontier waiter
//===----------------------------------------------------------------------===//

// Callback invoked when a frontier wait is satisfied (or fails).
// |status| is OK if all entries reached their target epochs, or an error if
// any axis failed (e.g., device lost → semaphore failure propagates here).
typedef void (*iree_async_frontier_waiter_fn_t)(void* user_data,
                                                iree_status_t status);

// Caller-owned storage for a pending frontier wait. Linked into the tracker's
// waiter list while active. The caller must not modify or free this storage
// until the callback fires.
//
// Usage:
//   iree_async_frontier_waiter_t waiter;
//   iree_async_frontier_tracker_wait(tracker, &frontier, on_ready, ctx,
//   &waiter);
//   // ... waiter is now pending. callback fires when frontier is satisfied.
//   // After callback fires (or after cancel returns), waiter storage is yours
//   again.
typedef struct iree_async_frontier_waiter_t {
  // Intrusive linked list (tracker-internal).
  struct iree_async_frontier_waiter_t* next;

  // The frontier being waited on. Must remain valid while the waiter is
  // pending. The tracker reads entries during advance() to check satisfaction.
  const iree_async_frontier_t* frontier;

  // Callback and context.
  iree_async_frontier_waiter_fn_t callback;
  void* user_data;
} iree_async_frontier_waiter_t;

//===----------------------------------------------------------------------===//
// Frontier tracker
//===----------------------------------------------------------------------===//

// Tracks axis progress and dispatches frontier waiters when satisfied.
//
// The frontier tracker is the runtime component that makes frontiers
// actionable. It answers two questions:
//   1. "Has this frontier been reached yet?" (is_satisfied / wait)
//   2. "An axis just advanced — who should be woken up?" (advance)
//
// ## How it connects to the rest of the system
//
//   GPU completes dispatch
//     → HAL driver signals semaphore (epoch N)
//     → Semaphore timepoint fires
//     → iree_async_frontier_tracker_advance(tracker, gpu_axis, N)
//     → Tracker checks all pending waiters
//     → Waiters whose frontiers are now fully satisfied get their callbacks
//     → Callbacks submit next operations to the proactor
//
//   Network receives frontier update from remote machine
//     → Protocol decoder reads {axis, epoch} pairs
//     → iree_async_frontier_tracker_advance(tracker, remote_axis, epoch)
//     → Same flow as above — local waiters on remote axes get woken
//
// ## Thread safety
//
//   advance():     Thread-safe. Multiple threads can advance different axes
//                  concurrently. Same axis from multiple threads is safe
//                  (monotonic — max wins, lower values are no-ops).
//
//   wait():        Thread-safe. Can be called from any thread.
//
//   cancel_wait(): Thread-safe. Blocks if the callback is currently in-flight
//                  on another thread (same guarantee as semaphore timepoints).
//
// The internal waiters list is protected by a slim mutex. The advance() hot
// path does an atomic epoch compare first and only takes the lock if waiters
// might be satisfied (avoiding lock contention in the common case where axes
// advance without pending waiters).
//
// ## Lifecycle
//
// Per-session object. Created when a session is established, destroyed when the
// session ends. The axis table is populated during session setup (topology
// exchange) and is immutable during steady-state operation.
//
//   iree_async_frontier_tracker_initialize(tracker, table_entries, capacity,
//   ...);
//   // ... populate axis table during session setup ...
//   // ... steady-state: advance() and wait() from multiple threads ...
//   iree_async_frontier_tracker_deinitialize(tracker);
typedef struct iree_async_frontier_tracker_t {
  // Current epoch for each known axis. Populated during session setup.
  iree_async_axis_table_t axis_table;

  // Pending frontier waiters. Protected by |waiters_mutex|.
  iree_slim_mutex_t waiters_mutex;
  iree_async_frontier_waiter_t* waiters_head;

  // Per-axis failure status. Parallel array with axis_table.entries (same
  // capacity). iree_ok_status() means healthy; non-OK means the axis has
  // permanently failed. Stored statuses are owned by the tracker and freed
  // on deinitialize.
  iree_status_t* axis_failure_statuses;

  // Allocator for internal bookkeeping (not for waiter storage — that's
  // caller-owned).
  iree_allocator_t allocator;
} iree_async_frontier_tracker_t;

// Initializes a frontier tracker with pre-allocated axis table storage.
// |axis_table_entries| must point to an array of |axis_table_capacity| entries
// that remains valid for the lifetime of the tracker.
iree_status_t iree_async_frontier_tracker_initialize(
    iree_async_frontier_tracker_t* tracker,
    iree_async_axis_table_entry_t* axis_table_entries,
    uint32_t axis_table_capacity, iree_allocator_t allocator);

// Deinitializes a frontier tracker. All pending waiters are cancelled (their
// callbacks fire with IREE_STATUS_CANCELLED). The tracker must not be used
// after this call.
void iree_async_frontier_tracker_deinitialize(
    iree_async_frontier_tracker_t* tracker);

// Advances an axis to a new epoch value. Thread-safe.
//
// If |epoch| is greater than the axis's current epoch, the axis is updated
// and any waiters whose frontiers are now fully satisfied are dispatched
// (callbacks fire from within this call, on the calling thread).
//
// If |epoch| is less than or equal to the current epoch, this is a no-op
// (monotonic timelines never go backwards).
//
// If the axis has an associated semaphore, the semaphore is also signaled
// to |epoch| (bridging frontier advancement to the proactor's semaphore wait
// infrastructure).
//
// Returns the number of waiters that were dispatched.
iree_host_size_t iree_async_frontier_tracker_advance(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    uint64_t epoch);

// Registers a waiter for a frontier to be satisfied. Thread-safe.
//
// The |callback| fires (from an advance() call) when every entry in |frontier|
// has its axis at or beyond the target epoch. If the frontier is already
// satisfied at the time of this call, the callback fires immediately (before
// this function returns).
//
// |waiter| is caller-owned storage that must remain valid until the callback
// fires or the wait is cancelled. |frontier| must also remain valid for the
// same duration.
//
// Returns IREE_STATUS_NOT_FOUND if any axis in the frontier is not in the
// tracker's axis table (programming error — all axes must be registered during
// session setup).
iree_status_t iree_async_frontier_tracker_wait(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier,
    iree_async_frontier_waiter_fn_t callback, void* user_data,
    iree_async_frontier_waiter_t* waiter);

// Cancels a pending wait. After this function returns, the callback will not
// fire (or has already fired and completed). If the callback is currently
// executing on another thread, this function blocks until it returns.
//
// It is safe to call this even if the waiter has already been dispatched
// (it becomes a no-op).
void iree_async_frontier_tracker_cancel_wait(
    iree_async_frontier_tracker_t* tracker,
    iree_async_frontier_waiter_t* waiter);

// Fails an axis permanently. All pending waiters that reference this axis are
// dispatched with |status|. Future waits on this axis will fail immediately.
// Takes ownership of |status|.
//
// This propagates device-lost and similar fatal errors through the frontier
// system: a failed GPU queue axis causes all dependent operations to fail,
// regardless of which machine they're on.
void iree_async_frontier_tracker_fail_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_FRONTIER_TRACKER_H_
