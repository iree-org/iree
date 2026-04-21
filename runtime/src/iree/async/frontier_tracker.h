// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_FRONTIER_TRACKER_H_
#define IREE_ASYNC_FRONTIER_TRACKER_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_semaphore_t iree_async_semaphore_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;

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

  // Callback invoked when the frontier becomes ready or failed.
  iree_async_frontier_waiter_fn_t callback;

  // User data passed through to |callback|.
  void* user_data;

  // Scratch field used during dispatch: holds the status to pass to the
  // callback after the waiter has been unlinked from the tracker list.
  // Only valid between unlinking and callback invocation.
  iree_status_t dispatch_status;
} iree_async_frontier_waiter_t;

//===----------------------------------------------------------------------===//
// Frontier tracker
//===----------------------------------------------------------------------===//

// Tracks axis progress and dispatches frontier waiters when satisfied.
//
// The frontier tracker is the runtime component that makes frontiers
// actionable. It answers two questions:
//   1. "Has this frontier been reached yet?" (is_satisfied / wait)
//   2. "An axis just advanced; who should be woken up?" (advance)
//
// ## How it connects to the rest of the system
//
//   Participant completes work
//     → participant timeline signals epoch N
//     → Semaphore timepoint fires
//     → iree_async_frontier_tracker_advance(tracker, axis, N)
//     → Tracker checks all pending waiters
//     → Waiters whose frontiers are now fully satisfied get their callbacks
//     → Callbacks submit dependent operations to their scheduler
//
//   Network receives frontier update from remote machine
//     → Protocol decoder reads {axis, epoch} pairs
//     → iree_async_frontier_tracker_advance(tracker, remote_axis, epoch)
//     → Same flow as above; local waiters on remote axes get woken
//
// ## Thread safety
//
//   advance():     Thread-safe. Multiple threads can advance different axes
//                  concurrently. Same axis from multiple threads is safe
//                  (monotonic: max wins, lower values are no-ops).
//
//   wait():        Thread-safe. Can be called from any thread.
//
//   cancel_wait(): Thread-safe. Reports whether the waiter was unlinked before
//                  dispatch. Callers that need to wait for an
//                  already-dispatched callback must use their own completion
//                  marker.
//
// The internal waiters list is protected by a slim mutex. The advance() hot
// path does an atomic epoch compare first and only takes the lock if waiters
// might be satisfied (avoiding lock contention in the common case where axes
// advance without pending waiters).
//
// ## Lifecycle
//
// Per-causal-domain object. Created when a distributed/local execution context
// establishes its axis namespace and destroyed after all participants using
// that namespace have retired. The axis table is populated during setup and is
// immutable during steady-state operation except for cold-path axis retirement.
//
//   iree_async_frontier_tracker_create(options, allocator, &tracker);
//   // ... populate axis table during session setup ...
//   // ... steady-state: advance() and wait() from multiple threads ...
//   iree_async_frontier_tracker_release(tracker);
//
// ## Axis lookup shape
//
// Axes are registered once during setup, then looked up by advance(), wait(),
// and query_epoch(). The implementation stores them in tracker-owned hash
// slots so steady-state operations do not scan all registered axes:
//
//   ┌───────────────────────────────┐
//   │ iree_async_axis_t             │
//   │ session/machine/domain/ordinal│
//   └──────────────┬────────────────┘
//                  │ hash
//                  ▼
//   ┌──────┬──────┬──────┬──────┬──────┐
//   │ slot │ slot │ slot │ slot │ slot │
//   └──────┴──────┴──────┴──────┴──────┘
//                         │
//                         ▼
//              {epoch, bridge semaphore, failure}
//
// The capacity option below is the maximum number of registered axes. The
// tracker may allocate additional internal slots to keep probes short.
//
// Options for creating a frontier tracker.
typedef struct iree_async_frontier_tracker_options_t {
  // Maximum number of axes that may be registered for the causal domain.
  uint32_t axis_table_capacity;

  // Generation value for the causal domain. This is embedded into axes to
  // prevent stale axis IDs from a previous execution context from comparing
  // equal to axes in a later context that reused the same machine/domain/
  // ordinal values.
  uint8_t session_epoch;

  // Local machine ordinal within the causal domain. This is embedded into axes
  // so frontiers can distinguish participants with the same domain/ordinal
  // values on different machines.
  uint8_t machine_index;
} iree_async_frontier_tracker_options_t;

static inline iree_async_frontier_tracker_options_t
iree_async_frontier_tracker_options_default(void) {
  iree_async_frontier_tracker_options_t options;
  memset(&options, 0, sizeof(options));
  options.axis_table_capacity = 256;
  options.session_epoch = 1;
  options.machine_index = 0;
  return options;
}

// Creates a ref-counted frontier tracker with tracker-owned axis storage.
iree_status_t iree_async_frontier_tracker_create(
    iree_async_frontier_tracker_options_t options, iree_allocator_t allocator,
    iree_async_frontier_tracker_t** out_tracker);

// Retains the given |tracker| for the caller.
void iree_async_frontier_tracker_retain(iree_async_frontier_tracker_t* tracker);

// Releases the given |tracker| from the caller. When the final reference is
// released, all pending waiters are cancelled with IREE_STATUS_CANCELLED.
void iree_async_frontier_tracker_release(
    iree_async_frontier_tracker_t* tracker);

// Returns the local session epoch used for deterministic axis construction.
uint8_t iree_async_frontier_tracker_session_epoch(
    const iree_async_frontier_tracker_t* tracker);

// Returns the local machine index used for deterministic axis construction.
uint8_t iree_async_frontier_tracker_machine_index(
    const iree_async_frontier_tracker_t* tracker);

// Registers an axis for the lifetime of the tracker. Axis IDs are never reused:
// once registered, an axis remains reserved even after retirement.
//
// |semaphore| is a borrowed bridge pointer. The registering participant must
// retire the axis before destroying the borrowed semaphore.
iree_status_t iree_async_frontier_tracker_register_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_async_semaphore_t* semaphore);

// Retires an axis and clears any borrowed bridge semaphore. This marks the
// axis permanently failed/cancelled, dispatches pending waiters that reference
// it, and causes future waits on the axis to fail immediately.
//
// The participant owning the axis must quiesce all concurrent calls to
// iree_async_frontier_tracker_advance() for that axis before retiring it.
// Takes ownership of |status|, which must be a non-OK status describing why
// the axis is being retired (e.g. CANCELLED for clean teardown).
void iree_async_frontier_tracker_retire_axis(
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_status_t status);

// Returns true if |axis| has reached at least |epoch|. Unknown axes return
// false. This is a lock-free progress query for pool/death-frontier dominance;
// it does not treat axis failure as success.
bool iree_async_frontier_tracker_query_epoch(
    const iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    uint64_t epoch);

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
// tracker's axis table (programming error: all axes must be registered during
// session setup).
iree_status_t iree_async_frontier_tracker_wait(
    iree_async_frontier_tracker_t* tracker,
    const iree_async_frontier_t* frontier,
    iree_async_frontier_waiter_fn_t callback, void* user_data,
    iree_async_frontier_waiter_t* waiter);

// Cancels a pending wait.
//
// Returns true if |waiter| was found in the pending list and removed, meaning
// the callback will not fire. Returns false if |waiter| was not found, meaning
// the callback has already been dispatched and may still be executing.
bool iree_async_frontier_tracker_cancel_wait(
    iree_async_frontier_tracker_t* tracker,
    iree_async_frontier_waiter_t* waiter);

// Fails an axis permanently. All pending waiters that reference this axis are
// dispatched with |status|. Future waits on this axis will fail immediately.
// Takes ownership of |status|, which must be a non-OK status describing the
// failure. This API is explicitly for the error path, never for clean
// dispositions.
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
