// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier_tracker.h"

#include "iree/async/semaphore.h"
#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// Axis table
//===----------------------------------------------------------------------===//
//
// The axis table is a fixed-capacity array mapping each registered axis to its
// current epoch, optional bridge semaphore, and permanent failure status. It
// is an implementation detail of the frontier tracker: the tracker owns the
// storage (as a flexible array member), populates it during setup via
// iree_async_frontier_tracker_register_axis, and queries it during advance/
// wait/fail/retire. Nothing outside this file should touch it directly — the
// public API lives on iree_async_frontier_tracker_t.

// An entry in the axis table mapping an axis to its current epoch and optional
// semaphore. The semaphore field enables bridging: when the axis advances, the
// corresponding semaphore is signaled, connecting the frontier system to the
// proactor's semaphore-based wait infrastructure.
//
// For directly-owned axes, the semaphore is the participant's local timeline
// semaphore signaled when that participant advances.
//
// For remotely-owned axes, the semaphore is a proxy signaled by the transport
// or protocol layer when it receives a frontier update from the remote owner.
typedef struct iree_async_axis_table_entry_t {
  // The full 64-bit axis identifier.
  iree_async_axis_t axis;

  // Current epoch for this axis. Updated atomically (acquire/release) when
  // the axis advances. Reads on the hot path use acquire semantics.
  iree_atomic_int64_t current_epoch;

  // Optional semaphore backing this axis. If non-NULL, advancing the epoch
  // also signals this semaphore. This enables proactor wait operations to
  // trigger on axis advancement without polling.
  //
  // This is a borrowed bridge pointer. Participants must quiesce all
  // advancement on the axis and retire the axis before destroying a borrowed
  // semaphore or any owner state it references.
  iree_async_semaphore_t* semaphore;

  // Permanent failure status for this axis. OK means the axis is healthy or
  // retired after satisfying all relevant work. Non-OK statuses are owned by
  // the table entry and freed when the tracker is destroyed.
  iree_status_t failure_status;
} iree_async_axis_table_entry_t;

// A fixed-capacity table mapping axes to their current state.
//
// Hot-path access pattern:
//   entry = &table->entries[wire_index];
//   semaphore = entry->semaphore;
//   // No locks, no hash lookups — just array indexing.
//
// Thread safety:
//   - Entries are added only during setup (single-threaded).
//   - current_epoch is updated atomically during steady-state.
//   - The table itself (entries pointer, count) is immutable after setup.
typedef struct iree_async_axis_table_t {
  // Storage for all registered axis entries. The table does not own this
  // pointer; the enclosing tracker holds the underlying FAM storage.
  iree_async_axis_table_entry_t* entries;

  // Number of currently registered axis entries.
  uint32_t count;

  // Maximum number of axis entries the table storage can hold.
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
  table->entries[index].failure_status = iree_ok_status();
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
// Tracker storage
//===----------------------------------------------------------------------===//

struct iree_async_frontier_tracker_t {
  // Reference count controlling the tracker lifetime.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for the tracker slab allocation.
  iree_allocator_t allocator;

  // Generation value embedded into locally constructed axes to avoid ABA across
  // execution contexts that reuse the same machine/domain/ordinal values.
  uint8_t session_epoch;

  // Local machine ordinal embedded into locally constructed axes.
  uint8_t machine_index;

  // Axis-to-progress table populated during setup and queried during waits.
  iree_async_axis_table_t axis_table;

  // Mutex protecting the intrusive waiter list and per-axis failure states.
  iree_slim_mutex_t waiters_mutex;

  // Head of the intrusive list of pending frontier waiters, stored as an
  // atomic pointer. All writes and most reads happen under |waiters_mutex| —
  // the atomic type exists solely so that iree_async_frontier_tracker_advance
  // can perform a lock-free "any waiters?" check outside the mutex without
  // racing against insertions under the C memory model (and without TSAN
  // flagging it). Acquire load on the lock-free read pairs with release stores
  // from wait()/cancel_wait()/etc. under the mutex.
  iree_atomic_intptr_t waiters_head;

  // Flexible array member containing tracker-owned axis table entries.
  iree_async_axis_table_entry_t axis_table_entries[];
};

// Typed load for the atomic waiters_head. Callers outside the mutex must use
// iree_memory_order_acquire to pair with release stores; callers under the
// mutex can use iree_memory_order_relaxed (mutual exclusion gives ordering).
static inline iree_async_frontier_waiter_t*
iree_async_frontier_tracker_load_waiters_head(
    const iree_async_frontier_tracker_t* tracker, iree_memory_order_t order) {
  return (iree_async_frontier_waiter_t*)iree_atomic_load(&tracker->waiters_head,
                                                         order);
}

// Typed store for the atomic waiters_head. All stores happen under the
// waiters_mutex; iree_memory_order_release is used so that the lock-free
// acquire load in advance() observes a well-defined value chain.
static inline void iree_async_frontier_tracker_store_waiters_head(
    iree_async_frontier_tracker_t* tracker, iree_async_frontier_waiter_t* value,
    iree_memory_order_t order) {
  iree_atomic_store(&tracker->waiters_head, (intptr_t)value, order);
}

// Typed exchange for the atomic waiters_head. Returns the old value and
// atomically stores |value|. Used by destroy to take ownership of the entire
// waiter list in a single atomic step.
static inline iree_async_frontier_waiter_t*
iree_async_frontier_tracker_exchange_waiters_head(
    iree_async_frontier_tracker_t* tracker, iree_async_frontier_waiter_t* value,
    iree_memory_order_t order) {
  return (iree_async_frontier_waiter_t*)iree_atomic_exchange(
      &tracker->waiters_head, (intptr_t)value, order);
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

// Returns the index of |axis| in |frontier|->entries, or -1 if not found.
static int32_t iree_async_frontier_find_axis(
    const iree_async_frontier_t* frontier, iree_async_axis_t axis) {
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    if (frontier->entries[i].axis == axis) return (int32_t)i;
  }
  return -1;
}

// Result of checking a waiter's satisfaction state.
typedef enum iree_async_waiter_check_result_e {
  IREE_ASYNC_WAITER_CHECK_PENDING = 0,    // Not yet satisfied.
  IREE_ASYNC_WAITER_CHECK_SATISFIED = 1,  // All entries satisfied.
  IREE_ASYNC_WAITER_CHECK_FAILED = 2,     // At least one axis failed.
} iree_async_waiter_check_result_t;

// Checks whether a waiter is satisfied or failed.
// If FAILED, |out_failure_status| is set to the cloned failure status (caller
// owns). Must be called under the waiters_mutex.
static iree_async_waiter_check_result_t
iree_async_frontier_tracker_check_waiter(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_waiter_t* waiter,
    iree_status_t* out_failure_status) {
  const iree_async_frontier_t* frontier = waiter->frontier;
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    int32_t index = iree_async_axis_table_find(&tracker->axis_table,
                                               frontier->entries[i].axis);
    if (index < 0) {
      // Axis not in table — should not happen if wait() validated correctly.
      // Treat as unsatisfied (will never satisfy → stuck waiter).
      return IREE_ASYNC_WAITER_CHECK_PENDING;
    }

    // Check for axis failure.
    iree_status_t failure = tracker->axis_table.entries[index].failure_status;
    if (!iree_status_is_ok(failure)) {
      *out_failure_status = iree_status_clone(failure);
      return IREE_ASYNC_WAITER_CHECK_FAILED;
    }

    // Check epoch.
    int64_t current_epoch =
        iree_atomic_load(&tracker->axis_table.entries[index].current_epoch,
                         iree_memory_order_acquire);
    if ((uint64_t)current_epoch < frontier->entries[i].epoch) {
      return IREE_ASYNC_WAITER_CHECK_PENDING;
    }
  }
  return IREE_ASYNC_WAITER_CHECK_SATISFIED;
}

static void iree_async_frontier_tracker_fail_axis_impl(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status, bool retire);

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_frontier_tracker_create(
    iree_async_frontier_tracker_options_t options, iree_allocator_t allocator,
    iree_async_frontier_tracker_t** out_tracker) {
  IREE_ASSERT_ARGUMENT(out_tracker);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, options.axis_table_capacity);
  *out_tracker = NULL;

  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(sizeof(iree_async_frontier_tracker_t), &total_size,
                         IREE_STRUCT_FIELD_FAM(options.axis_table_capacity,
                                               iree_async_axis_table_entry_t)));

  iree_async_frontier_tracker_t* tracker = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&tracker));
  memset(tracker, 0, total_size);

  iree_atomic_ref_count_init(&tracker->ref_count);
  tracker->allocator = allocator;
  tracker->session_epoch = options.session_epoch;
  tracker->machine_index = options.machine_index;
  iree_async_axis_table_initialize(&tracker->axis_table,
                                   tracker->axis_table_entries,
                                   options.axis_table_capacity);
  iree_slim_mutex_initialize(&tracker->waiters_mutex);
  // Publish an empty head with relaxed ordering; no other thread can observe
  // the tracker until *out_tracker is assigned below.
  iree_async_frontier_tracker_store_waiters_head(tracker, NULL,
                                                 iree_memory_order_relaxed);

  *out_tracker = tracker;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_frontier_tracker_destroy(
    iree_async_frontier_tracker_t* tracker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Steal the full waiter list in a single atomic swap, then dispatch the
  // cancellations outside any lock. No mutex is needed here: by the time
  // destroy runs, the ref-count has hit zero, which means no caller can still
  // be touching the tracker. The acq_rel on the final ref_count decrement
  // already synchronizes with all prior waiter-list updates, so this exchange
  // can be relaxed. The collect-then-dispatch shape matches advance() and
  // fail_axis_impl() for consistency, and keeps us safe if a cancellation
  // callback ever tries to touch another tracker.
  iree_async_frontier_waiter_t* dispatch_head =
      iree_async_frontier_tracker_exchange_waiters_head(
          tracker, NULL, iree_memory_order_relaxed);

  iree_async_frontier_waiter_t* waiter = dispatch_head;
  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;
    waiter->callback(waiter->user_data, iree_make_status(IREE_STATUS_CANCELLED,
                                                         "tracker destroyed"));
    waiter = next;
  }

  // Deinitialize mutex.
  iree_slim_mutex_deinitialize(&tracker->waiters_mutex);

  // Free failure statuses.
  for (uint32_t i = 0; i < tracker->axis_table.capacity; ++i) {
    if (!iree_status_is_ok(tracker->axis_table.entries[i].failure_status)) {
      iree_status_free(tracker->axis_table.entries[i].failure_status);
    }
  }

  iree_allocator_t allocator = tracker->allocator;
  iree_allocator_free(allocator, tracker);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_async_frontier_tracker_retain(
    iree_async_frontier_tracker_t* tracker) {
  if (tracker) {
    iree_atomic_ref_count_inc(&tracker->ref_count);
  }
}

IREE_API_EXPORT void iree_async_frontier_tracker_release(
    iree_async_frontier_tracker_t* tracker) {
  if (tracker && iree_atomic_ref_count_dec(&tracker->ref_count) == 1) {
    iree_async_frontier_tracker_destroy(tracker);
  }
}

IREE_API_EXPORT uint8_t iree_async_frontier_tracker_session_epoch(
    const iree_async_frontier_tracker_t* tracker) {
  IREE_ASSERT_ARGUMENT(tracker);
  return tracker->session_epoch;
}

IREE_API_EXPORT uint8_t iree_async_frontier_tracker_machine_index(
    const iree_async_frontier_tracker_t* tracker) {
  IREE_ASSERT_ARGUMENT(tracker);
  return tracker->machine_index;
}

IREE_API_EXPORT iree_status_t iree_async_frontier_tracker_register_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_async_semaphore_t* semaphore) {
  IREE_ASSERT_ARGUMENT(tracker);
  if (iree_async_axis_table_find(&tracker->axis_table, axis) >= 0) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "frontier tracker axis 0x%016" PRIX64 " already registered", axis);
  }
  int32_t index =
      iree_async_axis_table_add(&tracker->axis_table, axis, semaphore);
  if (index < 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "frontier tracker axis table capacity %u "
                            "exhausted registering axis 0x%016" PRIX64,
                            tracker->axis_table.capacity, axis);
  }
  return iree_ok_status();
}

IREE_API_EXPORT bool iree_async_frontier_tracker_query_epoch(
    const iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    uint64_t epoch) {
  IREE_ASSERT_ARGUMENT(tracker);
  int32_t axis_index = iree_async_axis_table_find(&tracker->axis_table, axis);
  if (axis_index < 0) return false;
  int64_t current_epoch =
      iree_atomic_load(&tracker->axis_table.entries[axis_index].current_epoch,
                       iree_memory_order_acquire);
  return (uint64_t)current_epoch >= epoch;
}

//===----------------------------------------------------------------------===//
// Tracker operations
//===----------------------------------------------------------------------===//

iree_host_size_t iree_async_frontier_tracker_advance(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    uint64_t epoch) {
  // Phase 1: Find axis and update epoch (lock-free).
  int32_t axis_index = iree_async_axis_table_find(&tracker->axis_table, axis);
  if (axis_index < 0) {
    return 0;  // Unknown axis — no-op.
  }

  iree_async_axis_table_entry_t* entry =
      &tracker->axis_table.entries[axis_index];

  // CAS loop to update epoch if advancing. Uses acq_rel ordering so that the
  // subsequent load of waiters_head (lock-free fast path) cannot be reordered
  // before the store. Without this, on weakly-ordered architectures (ARM),
  // advance() could store the epoch and read a stale waiters_head (NULL) while
  // wait() inserts a waiter and reads the old epoch — a classic lost wakeup.
  int64_t current_epoch;
  do {
    current_epoch =
        iree_atomic_load(&entry->current_epoch, iree_memory_order_acquire);
    if (epoch <= (uint64_t)current_epoch) {
      return 0;  // Monotonic — epoch not advancing.
    }
  } while (!iree_atomic_compare_exchange_weak(
      &entry->current_epoch, &current_epoch, (int64_t)epoch,
      iree_memory_order_acq_rel, iree_memory_order_relaxed));

  // Phase 2: Signal semaphore if present.
  if (entry->semaphore != NULL) {
    // Each timeline value must be signaled exactly once. If the signal fails
    // (e.g. semaphore already past this epoch), it indicates a structural
    // error — fail the semaphore so waiters get a proper diagnostic.
    iree_status_t signal_status =
        iree_async_semaphore_signal(entry->semaphore, epoch, NULL);
    if (IREE_UNLIKELY(!iree_status_is_ok(signal_status))) {
      iree_async_semaphore_fail(entry->semaphore, signal_status);
    }
  }

  // Phase 3: Collect satisfied waiters under lock, dispatch after unlock.
  // Callbacks must fire without the mutex held to prevent deadlock if a
  // callback reenters the tracker (e.g., registering a new waiter or
  // failing another axis).

  // Lock-free quick check: if no waiters, skip the mutex entirely.
  // This is safe because wait() re-checks satisfaction after insertion:
  //   - advance() reads waiters_head (sees NULL, stale due to concurrent
  //     insert)
  //   - advance() returns early (never takes lock)
  //   - wait() inserts waiter, re-checks epoch (sees new value from our CAS)
  //   - wait() dispatches immediately
  // The acquire load here pairs with release stores on waiters_head from
  // wait()/cancel_wait()/fail_axis_impl() under the mutex. Combined with the
  // acq_rel CAS on current_epoch above, lost wakeups are prevented on
  // weakly-ordered architectures.
  if (iree_async_frontier_tracker_load_waiters_head(
          tracker, iree_memory_order_acquire) == NULL) {
    return 0;
  }

  iree_slim_mutex_lock(&tracker->waiters_mutex);

  // Re-check under lock: another thread may have emptied the list. Relaxed
  // ordering is sufficient here because the mutex provides the happens-before
  // edge.
  iree_async_frontier_waiter_t* waiter =
      iree_async_frontier_tracker_load_waiters_head(tracker,
                                                    iree_memory_order_relaxed);
  if (waiter == NULL) {
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    return 0;
  }

  // Walk the waiter list, collecting satisfied/failed waiters into a local
  // dispatch list. |prev| tracks the surviving predecessor so unlinks can
  // patch either tracker->waiters_head (prev == NULL) or prev->next.
  iree_async_frontier_waiter_t* dispatch_head = NULL;
  iree_async_frontier_waiter_t** dispatch_tail = &dispatch_head;
  iree_host_size_t dispatched_count = 0;
  iree_async_frontier_waiter_t* prev = NULL;

  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;

    // Quick check: does this waiter care about the axis we just advanced?
    if (iree_async_frontier_find_axis(waiter->frontier, axis) < 0) {
      // This waiter doesn't reference the advanced axis — skip.
      prev = waiter;
      waiter = next;
      continue;
    }

    // Full satisfaction check.
    iree_status_t failure_status = iree_ok_status();
    iree_async_waiter_check_result_t result =
        iree_async_frontier_tracker_check_waiter(tracker, waiter,
                                                 &failure_status);

    if (result == IREE_ASYNC_WAITER_CHECK_SATISFIED ||
        result == IREE_ASYNC_WAITER_CHECK_FAILED) {
      // Unlink from tracker list and append to dispatch list. Release
      // ordering on the head store pairs with the acquire fast-path load.
      if (prev == NULL) {
        iree_async_frontier_tracker_store_waiters_head(
            tracker, next, iree_memory_order_release);
      } else {
        prev->next = next;
      }
      waiter->next = NULL;
      *dispatch_tail = waiter;
      dispatch_tail = &waiter->next;
      waiter->dispatch_status = failure_status;
      ++dispatched_count;
      // |prev| is unchanged — it is still the last surviving predecessor.
    } else {
      // Still pending — keep in list and advance |prev|.
      prev = waiter;
    }

    waiter = next;
  }

  iree_slim_mutex_unlock(&tracker->waiters_mutex);

  // Dispatch all collected waiters outside the lock.
  waiter = dispatch_head;
  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;
    waiter->callback(waiter->user_data, waiter->dispatch_status);
    waiter = next;
  }

  return dispatched_count;
}

iree_status_t iree_async_frontier_tracker_wait(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier,
    iree_async_frontier_waiter_fn_t callback, void* user_data,
    iree_async_frontier_waiter_t* waiter) {
  // Validate: all axes must be in the table.
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    if (iree_async_axis_table_find(&tracker->axis_table,
                                   frontier->entries[i].axis) < 0) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "frontier entry %" PRIu8
                              " references unknown axis 0x%016" PRIX64,
                              i, frontier->entries[i].axis);
    }
  }

  // Populate waiter fields (needed even for immediate dispatch, for
  // consistency).
  waiter->frontier = frontier;
  waiter->callback = callback;
  waiter->user_data = user_data;
  waiter->next = NULL;

  // Take lock and check satisfaction.
  iree_slim_mutex_lock(&tracker->waiters_mutex);

  iree_status_t failure_status = iree_ok_status();
  iree_async_waiter_check_result_t result =
      iree_async_frontier_tracker_check_waiter(tracker, waiter,
                                               &failure_status);

  if (result == IREE_ASYNC_WAITER_CHECK_SATISFIED) {
    // Already satisfied — dispatch immediately under lock.
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    callback(user_data, iree_ok_status());
    return iree_ok_status();
  }

  if (result == IREE_ASYNC_WAITER_CHECK_FAILED) {
    // Axis failed — dispatch immediately with failure.
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    callback(user_data, failure_status);
    return iree_ok_status();
  }

  // Not yet satisfied — insert at the head of the waiter list. Release
  // ordering on the head store pairs with the acquire fast-path load in
  // advance() so the lock-free reader observes the insertion on weakly
  // ordered architectures.
  waiter->next = iree_async_frontier_tracker_load_waiters_head(
      tracker, iree_memory_order_relaxed);
  iree_async_frontier_tracker_store_waiters_head(tracker, waiter,
                                                 iree_memory_order_release);

  // Re-check satisfaction after insertion. This handles the race where:
  //   1. wait() checks epoch (old value), decides to insert
  //   2. advance() updates epoch, does lock-free check (sees NULL), returns
  //   3. wait() inserts waiter (would be stuck without this re-check)
  // The re-check sees the new epoch and dispatches immediately.
  //
  // check_waiter only writes |failure_status| when returning FAILED, so on
  // SATISFIED the value is still the iree_ok_status() sentinel we
  // initialized above, and on FAILED it owns the cloned failure. Either way
  // we can propagate it directly into the callback without a conditional —
  // no status is ever dropped on the floor.
  result = iree_async_frontier_tracker_check_waiter(tracker, waiter,
                                                    &failure_status);
  if (result == IREE_ASYNC_WAITER_CHECK_SATISFIED ||
      result == IREE_ASYNC_WAITER_CHECK_FAILED) {
    // Either epoch advanced or axis failed while we were inserting. The
    // waiter we just pushed is still at the head of the list, so remove it
    // and dispatch inline. Relaxed ordering is fine for this removal — the
    // waiter was only briefly visible and no lock-free reader can hand off
    // to a different code path based on its presence.
    iree_async_frontier_tracker_store_waiters_head(tracker, waiter->next,
                                                   iree_memory_order_relaxed);
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    callback(user_data, failure_status);
    return iree_ok_status();
  }

  iree_slim_mutex_unlock(&tracker->waiters_mutex);
  return iree_ok_status();
}

void iree_async_frontier_tracker_cancel_wait(
    iree_async_frontier_tracker_t* tracker,
    iree_async_frontier_waiter_t* waiter) {
  iree_slim_mutex_lock(&tracker->waiters_mutex);

  // Search for the waiter in the list, tracking the predecessor so we can
  // patch either the head or prev->next on unlink.
  iree_async_frontier_waiter_t* prev = NULL;
  iree_async_frontier_waiter_t* current =
      iree_async_frontier_tracker_load_waiters_head(tracker,
                                                    iree_memory_order_relaxed);
  while (current != NULL) {
    if (current == waiter) {
      // Found — unlink and we're done. Callback will never fire. Release
      // ordering on the head store pairs with advance()'s acquire load.
      if (prev == NULL) {
        iree_async_frontier_tracker_store_waiters_head(
            tracker, current->next, iree_memory_order_release);
      } else {
        prev->next = current->next;
      }
      iree_slim_mutex_unlock(&tracker->waiters_mutex);
      return;
    }
    prev = current;
    current = current->next;
  }

  // Not found — waiter was already dispatched (callback completed or
  // in-flight). This is a no-op.
  iree_slim_mutex_unlock(&tracker->waiters_mutex);
}

void iree_async_frontier_tracker_fail_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status) {
  iree_async_frontier_tracker_fail_axis_impl(tracker, axis, status,
                                             /*retire=*/false);
}

static void iree_async_frontier_tracker_fail_axis_impl(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status, bool retire) {
  // fail_axis/retire_axis are the error-path and clean-teardown entry points
  // respectively — both require a real status from the caller. Silently
  // synthesizing one hides caller bugs and produces misleading diagnostics
  // when an axis turns up "cancelled" without context.
  IREE_ASSERT(!iree_status_is_ok(status),
              "fail/retire require a non-OK status from the caller");

  // Find axis.
  int32_t axis_index = iree_async_axis_table_find(&tracker->axis_table, axis);
  if (axis_index < 0) {
    // Unknown axis — free status and return.
    iree_status_free(status);
    return;
  }

  // Take lock before checking/storing failure status.
  // This protects against:
  //   - Write-write race: two concurrent fail_axis() calls for the same axis
  //   - Read-write race: check_waiter() reads failure_status under lock
  iree_slim_mutex_lock(&tracker->waiters_mutex);

  iree_async_semaphore_t* semaphore =
      tracker->axis_table.entries[axis_index].semaphore;
  if (retire) {
    tracker->axis_table.entries[axis_index].semaphore = NULL;
    semaphore = NULL;
  }

  // Check for existing failure (first-failure-wins).
  if (!iree_status_is_ok(
          tracker->axis_table.entries[axis_index].failure_status)) {
    // Already failed — ignore this failure.
    iree_slim_mutex_unlock(&tracker->waiters_mutex);
    iree_status_free(status);
    return;
  }

  // Store failure status (take ownership).
  tracker->axis_table.entries[axis_index].failure_status = status;

  // Fail the associated semaphore (if any). This dispatches all pending
  // semaphore timepoints with the failure status, bridging axis failure to
  // the proactor's semaphore-based wait infrastructure. Without this, proxy
  // semaphore timepoints would be orphaned when a remote axis fails.
  if (semaphore != NULL) {
    iree_async_semaphore_fail(semaphore, iree_status_clone(status));
  }

  // Collect all waiters that reference this axis into a local dispatch list.
  // Callbacks must fire outside the lock to prevent deadlock if a callback
  // reenters the tracker. |prev| tracks the surviving predecessor so unlinks
  // can patch either the head (via release store to pair with advance()'s
  // acquire fast path) or prev->next (plain store under the mutex).
  iree_async_frontier_waiter_t* dispatch_head = NULL;
  iree_async_frontier_waiter_t** dispatch_tail = &dispatch_head;

  iree_async_frontier_waiter_t* prev = NULL;
  iree_async_frontier_waiter_t* waiter =
      iree_async_frontier_tracker_load_waiters_head(tracker,
                                                    iree_memory_order_relaxed);

  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;

    // Check if this waiter references the failed axis.
    if (iree_async_frontier_find_axis(waiter->frontier, axis) >= 0) {
      // Unlink from tracker list and append to dispatch list.
      if (prev == NULL) {
        iree_async_frontier_tracker_store_waiters_head(
            tracker, next, iree_memory_order_release);
      } else {
        prev->next = next;
      }
      waiter->next = NULL;
      *dispatch_tail = waiter;
      dispatch_tail = &waiter->next;
      // |prev| is unchanged — still the last surviving predecessor.
    } else {
      prev = waiter;
    }

    waiter = next;
  }

  iree_slim_mutex_unlock(&tracker->waiters_mutex);

  // Dispatch all collected waiters outside the lock.
  waiter = dispatch_head;
  while (waiter != NULL) {
    iree_async_frontier_waiter_t* next = waiter->next;
    waiter->callback(waiter->user_data, iree_status_clone(status));
    waiter = next;
  }
}

IREE_API_EXPORT void iree_async_frontier_tracker_retire_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status) {
  iree_async_frontier_tracker_fail_axis_impl(tracker, axis, status,
                                             /*retire=*/true);
}
