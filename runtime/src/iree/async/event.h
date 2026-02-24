// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lightweight async signaling primitive.
//
// An iree_async_event_t provides cross-thread signaling that integrates with
// the proactor's event loop. Events are the building block for proactor wake(),
// cross-thread notification, and user-facing synchronization.
//
// Semantics: set() makes the event signaled (thread-safe, may be called from
// any context). The proactor detects the signaled state during poll() and
// delivers the completion. reset() drains the signal after handling.
//
// Platform mapping:
//   Linux:   eventfd (write to signal, read to reset)
//   macOS:   pipe (write end signals, read end monitored by kqueue)
//   Windows: auto-reset Win32 event (SetEvent to signal)

#ifndef IREE_ASYNC_EVENT_H_
#define IREE_ASYNC_EVENT_H_

#include "iree/async/primitive.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_event_pool_t iree_async_event_pool_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;

//===----------------------------------------------------------------------===//
// Event
//===----------------------------------------------------------------------===//

// A proactor-managed signaling primitive. Created via
// iree_async_event_create().
//
// The event uses two primitives: one for monitoring (the proactor polls this)
// and one for signaling (set() writes to this). On some platforms these are
// the same handle; on others they differ:
//
//   Platform   | primitive (poll)   | signal_primitive (set)  | Same?
//   -----------|--------------------|-----------------------------|------
//   Linux      | eventfd            | eventfd (same fd)           | Yes
//   macOS/BSD  | pipe read end      | pipe write end              | No
//   Windows    | Win32 event        | Win32 event (same HANDLE)   | Yes
//
// Implementers: when primitive == signal_primitive, close only once during
// destroy. When they differ (pipe), close both ends independently.
typedef struct iree_async_event_t {
  iree_atomic_ref_count_t ref_count;

  // The proactor this event is bound to. Not retained.
  iree_async_proactor_t* proactor;

  // Primitive monitored by the proactor for readability/signaled state.
  // On Linux: eventfd. On macOS: pipe read end. On Windows: Win32 event.
  iree_async_primitive_t primitive;

  // Primitive written to by set() to trigger the signal.
  // On Linux: same eventfd (write 1). On macOS: pipe write end (write 1 byte).
  // On Windows: same Win32 event (SetEvent).
  iree_async_primitive_t signal_primitive;

  // io_uring fixed file index (-1 if not registered).
  int32_t fixed_file_index;

  // Buffer for linked READ operations that drain the eventfd.
  // Used by io_uring's linked POLL_ADD + READ to auto-reset the event.
  uint64_t drain_buffer;

  // Pool support: home pool for release routing (NULL if not pooled).
  iree_async_event_pool_t* pool;

  // Pool support: intrusive list linkage for acquire_stack and return_stack.
  // This field is used for the LIFO stacks that manage available events.
  struct iree_async_event_t* pool_next;

  // Pool support: intrusive list linkage for all_events cleanup list.
  // This separate field tracks ALL events ever created by the pool,
  // independent of their current stack location, for cleanup during deinit.
  struct iree_async_event_t* pool_all_next;
} iree_async_event_t;

// Creates a new event for cross-thread signaling.
//
// Events are lightweight, waitable objects that can be signaled from any
// thread. Use iree_async_event_wait_operation_t to wait on them asynchronously.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Implementation:
//   io_uring: eventfd (IORING_OP_POLL_ADD for waits)
//   IOCP: Event object (SetEvent/WaitForSingleObject)
//   kqueue: EVFILT_USER
//   generic: eventfd or pipe + poll
//
// Note: For cross-platform notification semantics with richer features,
// consider using iree_async_notification_t (when available) which provides
// a higher-level abstraction over events and futexes.
//
// Returns:
//   IREE_STATUS_OK: Event created successfully.
//   IREE_STATUS_RESOURCE_EXHAUSTED: System resource limit reached.
IREE_API_EXPORT iree_status_t iree_async_event_create(
    iree_async_proactor_t* proactor, iree_async_event_t** out_event);

// Increments the reference count.
IREE_API_EXPORT void iree_async_event_retain(iree_async_event_t* event);

// Decrements the reference count and destroys if it reaches zero.
IREE_API_EXPORT void iree_async_event_release(iree_async_event_t* event);

// Signals the event. Thread-safe, async-signal-safe.
// Wakes the proactor's poll() if it is monitoring this event.
// Idempotent: multiple calls before the wait completes are coalesced.
IREE_API_EXPORT iree_status_t iree_async_event_set(iree_async_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_EVENT_H_
