// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for POSIX proactor implementation.
//
// This header exposes the proactor struct and internal helpers for use by
// POSIX-specific modules (worker.c, etc.). External users should include
// the public api.h instead.

#ifndef IREE_ASYNC_PLATFORM_POSIX_PROACTOR_H_
#define IREE_ASYNC_PLATFORM_POSIX_PROACTOR_H_

#include "iree/async/notification.h"
#include "iree/async/platform/posix/event_set.h"
#include "iree/async/platform/posix/fd_map.h"
#include "iree/async/platform/posix/timer_list.h"
#include "iree/async/platform/posix/wake.h"
#include "iree/async/platform/posix/worker.h"
#include "iree/async/proactor.h"
#include "iree/async/util/completion_pool.h"
#include "iree/async/util/message_pool.h"
#include "iree/async/util/ready_pool.h"
#include "iree/async/util/sequence_emulation.h"
#include "iree/async/util/signal.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/threading/notification.h"

// Signal backend selection: signalfd on Linux (efficient, kernel-managed),
// self-pipe on all other POSIX platforms (portable fallback).
// Define IREE_ASYNC_SIGNAL_FORCE_SELFPIPE to use self-pipe on Linux for
// testing.
#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_ASYNC_SIGNAL_FORCE_SELFPIPE)
#define IREE_ASYNC_SIGNAL_USE_SIGNALFD 1
#include "iree/async/platform/linux/signal.h"
#else
#define IREE_ASYNC_SIGNAL_USE_SELFPIPE 1
#include "iree/async/platform/posix/signal.h"
#endif  // IREE_ASYNC_SIGNAL_FORCE_SELFPIPE

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default number of worker threads if not specified.
#define IREE_ASYNC_POSIX_DEFAULT_WORKER_COUNT 4

// Default pool capacity for ready and completion entries.
// Sized to accommodate CTS stress tests (200+ concurrent operations).
#define IREE_ASYNC_POSIX_DEFAULT_POOL_CAPACITY 256

//===----------------------------------------------------------------------===//
// Event source (persistent fd monitoring)
//===----------------------------------------------------------------------===//

// Tracks a registered event source for persistent monitoring of an external fd.
// The proactor monitors the fd for readability and invokes the callback on each
// poll cycle where the fd is ready. Doubly-linked list node; proactor owns the
// list.
struct iree_async_event_source_t {
  // Intrusive doubly-linked list for efficient removal.
  struct iree_async_event_source_t* next;
  struct iree_async_event_source_t* prev;

  // Owning proactor (for vtable access in callbacks).
  iree_async_proactor_t* proactor;

  // The monitored fd (not owned by the event source).
  int fd;

  // User callback invoked when the fd is readable.
  iree_async_event_source_callback_t callback;

  // Allocator used to allocate this struct (for deallocation).
  iree_allocator_t allocator;
};

//===----------------------------------------------------------------------===//
// Proactor state
//===----------------------------------------------------------------------===//

// POSIX proactor state using event_set + worker thread pool.
typedef struct iree_async_proactor_posix_t {
  // Must be first for safe casting.
  iree_async_proactor_t base;
  iree_atomic_int32_t shutdown_requested;

  // Poll thread state.
  iree_async_posix_event_set_t* event_set;  // Pluggable: poll/epoll/kqueue.
  iree_async_posix_fd_map_t fd_map;
  iree_async_posix_wake_t wake;

  // MPSC queues.
  iree_atomic_slist_t pending_queue;  // submit() → poll thread.
  iree_atomic_slist_t ready_queue;    // poll thread → workers.
  iree_notification_t ready_notification;
  iree_atomic_slist_t completion_queue;  // workers → poll thread.
  // Timepoint callbacks funneled to poll thread.
  iree_atomic_slist_t pending_semaphore_waits;
  // Fence imports deferred from arbitrary threads to poll thread.
  iree_atomic_slist_t pending_fence_imports;

  // Pre-allocated pools for hot-path queue entries.
  iree_async_posix_ready_pool_t ready_pool;
  iree_async_posix_completion_pool_t completion_pool;

  // Cross-proactor messaging state.
  // The POSIX backend has no native kernel-mediated messaging (unlike io_uring
  // MSG_RING or IOCP PostQueuedCompletionStatus), so messages are delivered via
  // a pre-allocated lock-free pool that senders push to and poll() drains.
  iree_async_message_pool_t message_pool;
  iree_async_proactor_message_callback_t message_callback;

  // Worker pool (points into trailing data of this allocation).
  iree_host_size_t worker_count;         // Configured count.
  iree_host_size_t initialized_workers;  // Actually initialized (for cleanup).
  iree_async_posix_worker_t* workers;

  // Resource lists (poll thread only).
  // Event sources and relays are kept in linked lists for cleanup/destroy.
  // All handler types are ALSO registered in fd_map for O(1) dispatch.
  iree_async_event_source_t* event_sources;
  struct iree_async_relay_t* relays;

  // Signal handling (lazy-initialized on first subscribe).
  // Uses signalfd on Linux (IREE_ASYNC_SIGNAL_USE_SIGNALFD) or self-pipe on
  // all other POSIX platforms (IREE_ASYNC_SIGNAL_USE_SELFPIPE). The signal fd
  // (signalfd or pipe read end) is registered as an event source so the
  // poll/epoll loop wakes when signals arrive.
  struct {
    bool initialized;
#if defined(IREE_ASYNC_SIGNAL_USE_SIGNALFD)
    iree_async_linux_signal_state_t backend_state;
#elif defined(IREE_ASYNC_SIGNAL_USE_SELFPIPE)
    iree_async_selfpipe_signal_state_t backend_state;
#endif  // IREE_ASYNC_SIGNAL_USE_*
    iree_async_signal_dispatch_state_t dispatch_state;
    iree_async_signal_subscription_t* subscriptions[IREE_ASYNC_SIGNAL_COUNT];
    int subscriber_count[IREE_ASYNC_SIGNAL_COUNT];
    // Event source monitoring the signal fd. Owned by the proactor's event
    // source list; stored here for cleanup during destroy.
    iree_async_event_source_t* event_source;
  } signal;

  // Timer state (poll thread only).
  iree_async_posix_timer_list_t timers;

  // Count of fd-registered operations with CANCELLED flag set, waiting for the
  // poll thread to remove them from the fd_map and push CANCELLED completions.
  // Incremented by cancel() from any thread; decremented by the poll thread
  // during drain_pending_fd_cancellations(). Used to avoid scanning the fd_map
  // on every poll iteration when no cancellations are pending.
  iree_atomic_int32_t pending_fd_cancellation_count;

  // Count of timer operations in the timer_list with CANCELLED flag set,
  // waiting for the poll thread to remove them and push CANCELLED completions.
  // Incremented by cancel() from any thread; decremented by the poll thread
  // during drain_pending_timer_cancellations(). Without this, cancelled timers
  // linger in the timer_list until their original deadline expires.
  iree_atomic_int32_t pending_timer_cancellation_count;

  // Singleton constraint: only one READ-access slab may be registered at a
  // time. Mirrors io_uring's fixed buffer table limitation, enforced as a
  // public API contract for portability (see proactor.h:930-934).
  bool has_read_slab_registration;

  // Sequence emulator for IREE_ASYNC_OPERATION_TYPE_SEQUENCE operations.
  // Drives step-by-step execution when step_fn is set. When step_fn is NULL,
  // the LINK path is used instead (no emulator involvement).
  iree_async_sequence_emulator_t sequence_emulator;

  // Configuration.
  iree_async_proactor_capabilities_t capabilities;
  uint32_t numa_node;
} iree_async_proactor_posix_t;

static inline iree_async_proactor_posix_t* iree_async_proactor_posix_cast(
    iree_async_proactor_t* proactor) {
  return (iree_async_proactor_posix_t*)proactor;
}

// Wakes the poll thread (thread-safe, async-signal-safe).
void iree_async_proactor_posix_wake_poll_thread(
    iree_async_proactor_posix_t* proactor);

// Creates a threaded proactor using poll() + worker pool for I/O.
// Uses the platform-default event backend (poll on most systems).
iree_status_t iree_async_proactor_create_posix(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor);

// Creates a threaded proactor with a specific event notification backend.
// Use this for testing different event backends or to force a specific backend
// when the default is not suitable.
iree_status_t iree_async_proactor_create_posix_with_backend(
    iree_async_proactor_options_t options,
    iree_async_posix_event_backend_t event_backend, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_PROACTOR_H_
