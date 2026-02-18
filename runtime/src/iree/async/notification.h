// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_NOTIFICATION_H_
#define IREE_ASYNC_NOTIFICATION_H_

#include "iree/async/primitive.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/futex.h"
#include "iree/base/threading/notification.h"

// Compile-time selection for sync notification_wait() implementation.
// When futex is available, sync waiters use futex_wait() on the epoch atomic —
// no eventfd involvement, no drain race with the poll thread. When futex is
// unavailable (macOS), sync waiters use iree_notification_t (condvar-based),
// keeping the eventfd exclusively for the poll thread.
//
// This flag is potentially transient: after measurement across platforms it
// may either track the main IREE_RUNTIME_USE_FUTEX unconditionally or be
// removed entirely.
//
// Set IREE_ASYNC_POSIX_NOTIFICATION_WANT_FUTEX=0 to force the condvar path
// even on platforms that support futex (for benchmarking/testing).
#ifndef IREE_ASYNC_POSIX_NOTIFICATION_WANT_FUTEX
#define IREE_ASYNC_POSIX_NOTIFICATION_WANT_FUTEX 1
#endif  // IREE_ASYNC_POSIX_NOTIFICATION_WANT_FUTEX
#if defined(IREE_RUNTIME_USE_FUTEX) && IREE_ASYNC_POSIX_NOTIFICATION_WANT_FUTEX
#define IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX 1
#endif  // IREE_RUNTIME_USE_FUTEX && IREE_ASYNC_POSIX_NOTIFICATION_WANT_FUTEX

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_notification_wait_operation_t
    iree_async_notification_wait_operation_t;

//===----------------------------------------------------------------------===//
// Notification
//===----------------------------------------------------------------------===//

// Implementation mode for notification wait/wake primitives.
typedef enum iree_async_notification_mode_e {
  // Futex word: wait/wake operate directly on the epoch atomic.
  // Optimal (no fd overhead) but requires kernel futex support.
  // Used by io_uring on Linux 6.7+ with FUTEX_OPERATIONS capability.
  IREE_ASYNC_NOTIFICATION_MODE_FUTEX = 0,

  // Event fd: eventfd (Linux) or pipe (macOS/BSD) with poll-based waits.
  // Used by POSIX backend and io_uring fallback on older kernels.
  IREE_ASYNC_NOTIFICATION_MODE_EVENT = 1,
} iree_async_notification_mode_t;

// Lightweight notification primitive for proactor-integrated thread wakeup.
//
// Provides cross-thread signaling with epoch counting: multiple signals
// coalesce, and waiters observe signals that occurred after their wait was
// submitted. Unlike events (edge-triggered, one signal -> one completion),
// notifications are level-triggered.
//
// Semantics:
//   - signal(): Atomically increments the epoch and wakes waiters.
//     Thread-safe, may be called from any context including callbacks.
//   - wait (async): Completes when the epoch advances past the token captured
//     at submit time. Multiple signals between submit and poll coalesce.
//   - wait (sync): Blocks until the epoch advances or timeout expires.
//     For worker threads outside the proactor.
//
// The epoch counter is the source of truth for signal state. In EVENT mode,
// the eventfd/pipe is purely a wakeup mechanism.
typedef struct iree_async_notification_t {
  iree_atomic_ref_count_t ref_count;

  // The proactor this notification is bound to. Not retained.
  iree_async_proactor_t* proactor;

  // Epoch counter incremented on each signal. Source of truth for signal state.
  // In FUTEX mode, also the address for futex syscalls.
  iree_atomic_int32_t epoch;

  // Implementation mode selected at creation time by the proactor backend.
  iree_async_notification_mode_t mode;

  // Platform-specific resources. Only the creating backend accesses its member.
  union {
    // io_uring backend (Linux).
    // In FUTEX mode: primitive is unused (futex operates on &epoch directly).
    // In EVENT mode: primitive is an eventfd with EFD_SEMAPHORE for linked
    // POLL_ADD + READ SQE patterns.
    struct {
      // Eventfd for poll-based async waits (EVENT mode only).
      iree_async_primitive_t primitive;
      // Buffer target for linked READ SQEs that drain the eventfd.
      uint64_t drain_buffer;
      // Count of relays with in-flight FUTEX_WAIT SQEs on this notification.
      // Incremented when a relay submits a FUTEX_WAIT, decremented when the
      // FUTEX_WAIT CQE is processed. Read from the signal path to compute a
      // precise futex wake count that includes both user waiters and relay
      // waiters. Always zero in EVENT mode (relays use POLL_ADD instead).
      iree_atomic_int32_t futex_relay_count;
    } io_uring;

    // POSIX backend (Linux/macOS/BSD).
    // Uses eventfd (Linux) or pipe (macOS/BSD) as a wakeup mechanism for
    // poll-based synchronous and asynchronous waits.
    struct {
      // Fd monitored for POLLIN by the proactor poll loop and sync waiters.
      // Linux: eventfd. macOS/BSD: pipe read end.
      iree_async_primitive_t primitive;
      // Fd written to by signal() to trigger POLLIN on the monitored end.
      // Linux: same eventfd (bidirectional). macOS/BSD: pipe write end.
      iree_async_primitive_t signal_primitive;
      // Intrusive list of pending async wait operations (poll thread only).
      // Uses iree_async_operation_t::next for linkage.
      iree_async_notification_wait_operation_t* pending_waits;
      // Relays with this notification as their source (poll thread only).
      // Walked on notification fd readiness alongside pending_waits.
      // Uses iree_async_relay_t::platform.posix.notification_relay_next.
      struct iree_async_relay_t* relay_list;
#if !defined(IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX)
      // Condvar-based wakeup for sync waiters. Signal posts here alongside
      // the eventfd write; sync waiters await here instead of poll()+read()
      // on the eventfd. This eliminates the drain race where a sync waiter
      // consumes the eventfd signal before the proactor poll loop sees it.
      iree_notification_t sync_notification;
#endif  // !IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX
    } posix;

    // IOCP backend (Windows).
    // No fd/primitive needed. Sync waiters use WaitOnAddress on &epoch
    // (functionally identical to Linux futex). Async waits are tracked in
    // an intrusive list processed by the poll thread on each iteration.
    struct {
      // Intrusive list of pending async wait operations (poll thread only).
      // Uses iree_async_operation_t::next for linkage.
      iree_async_notification_wait_operation_t* pending_waits;
      // Intrusive linkage for proactor's notifications_with_waits list.
      // Only valid when in_wait_list is true.
      struct iree_async_notification_t* next_with_waits;
      // Whether this notification is currently in the proactor's
      // notifications_with_waits list. Avoids duplicate insertion.
      bool in_wait_list;
      // Intrusive list of relays with this notification as their source
      // (poll thread only). Walked alongside pending_waits during poll
      // to fire relay sinks when the epoch advances.
      // Uses iree_async_relay_t::platform.iocp.notification_relay_next.
      struct iree_async_relay_t* relay_list;
    } iocp;
  } platform;
} iree_async_notification_t;

// Creation flags for notifications. Reserved for future use.
enum iree_async_notification_flag_bits_e {
  IREE_ASYNC_NOTIFICATION_FLAG_NONE = 0u,
};
typedef uint32_t iree_async_notification_flags_t;

// Creates a new notification for cross-thread signaling with epoch semantics.
//
// Notifications are lightweight, waitable objects that provide level-triggered
// signaling: multiple signals coalesce, and waiters observe any signal that
// occurs after their wait was submitted. Use
// iree_async_notification_wait_operation_t for async waits, or
// iree_async_notification_wait() for synchronous blocking waits (worker
// threads).
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Implementation selection (automatic based on capabilities):
//   io_uring 6.7+: Futex word with FUTEX_WAIT/WAKE ops (optimal).
//   io_uring <6.7: Eventfd with linked POLL_ADD + READ.
//   IOCP: WaitOnAddress / SetEvent.
//   kqueue: dispatch_semaphore / kevent.
//   generic: eventfd + poll.
//
// Returns:
//   IREE_STATUS_OK: Notification created successfully.
//   IREE_STATUS_RESOURCE_EXHAUSTED: System resource limit reached.
IREE_API_EXPORT iree_status_t iree_async_notification_create(
    iree_async_proactor_t* proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification);

// Retains a reference to the notification.
IREE_API_EXPORT void iree_async_notification_retain(
    iree_async_notification_t* notification);

// Releases a reference to the notification.
// Destroys when the reference count reaches zero.
IREE_API_EXPORT void iree_async_notification_release(
    iree_async_notification_t* notification);

// Signals the notification, waking up to |wake_count| waiters.
//
// Thread-safe, async-signal-safe. May be called from any context including
// completion callbacks, signal handlers, or other threads.
//
// |wake_count|: Number of waiters to wake.
//   - 1: Wake a single waiter (typical for producer/consumer).
//   - INT32_MAX: Wake all waiters (broadcast).
//
// Implementation:
//   Atomically increments the notification's epoch, then wakes waiters via
//   the platform primitive (futex_wake, eventfd write, etc.).
//
// Idempotent: Multiple signals before any waiter observes them coalesce into
// a single epoch advance. This is intentional for level-triggered semantics.
IREE_API_EXPORT void iree_async_notification_signal(
    iree_async_notification_t* notification, int32_t wake_count);

// Returns the current epoch of the notification.
//
// The epoch is incremented each time the notification is signaled. This can be
// used for polling-style observation of notification state without blocking.
// Typical usage:
//   1. Read epoch before an operation: observed = query_epoch(notification)
//   2. Perform operation that should trigger a signal
//   3. Poll until epoch advances: while (query_epoch(notification) == observed)
//
// Thread safety:
//   May be called from any thread concurrently with signal/wait operations.
//   The returned value represents a point-in-time snapshot.
IREE_API_EXPORT uint32_t
iree_async_notification_query_epoch(iree_async_notification_t* notification);

// Blocks the calling thread until the notification is signaled or timeout
// expires.
//
// This is the synchronous blocking API for worker threads that need to wait
// on a notification outside the proactor's poll loop. The proactor poll thread
// should use NOTIFICATION_WAIT operations instead.
//
// Returns:
//   true: Notification was signaled.
//   false: Timeout expired without signal.
//
// Thread safety:
//   May be called from any thread. Multiple threads may wait concurrently.
//   However, this function blocks the calling thread and should not be called
//   from the proactor's poll thread (it would deadlock).
IREE_API_EXPORT bool iree_async_notification_wait(
    iree_async_notification_t* notification, iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_NOTIFICATION_H_
