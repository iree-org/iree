// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fence import/export operations for POSIX proactor.
//
// Fences bridge external sync primitives (GPU completion events as fds) to
// async semaphores and vice versa, enabling zero-overhead GPU↔network I/O
// coordination in the remoting system.
//
// Import: External fd → semaphore signal (one-shot poll operation)
// Export: Semaphore timepoint → external fd (timepoint callback writes to fd)

#ifndef IREE_ASYNC_PLATFORM_POSIX_FENCE_H_
#define IREE_ASYNC_PLATFORM_POSIX_FENCE_H_

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_posix_t iree_async_proactor_posix_t;

//===----------------------------------------------------------------------===//
// Import fence tracker
//===----------------------------------------------------------------------===//

// Tracker for import fence: bridges external fd → semaphore signal.
// The tracker owns the fence fd and is responsible for closing it on
// completion. The semaphore is retained for the lifetime of the tracker.
//
// Lifecycle: import_fence() allocates and pushes to the pending_fence_imports
// MPSC queue. The poll thread pops, registers in fd_map/event_set, and the
// tracker lives in the fd_map until the fd fires. On completion,
// handle_fence_import() removes from fd_map/event_set and frees.
typedef struct iree_async_posix_fence_import_tracker_t {
  // Intrusive MPSC list link for deferred registration.
  // Used by import_fence() to push to pending_fence_imports and by the poll
  // thread to drain. Not used after registration (fd_map stores the tracker
  // pointer directly).
  iree_atomic_slist_entry_t slist_entry;

  iree_async_semaphore_t* semaphore;  // Retained; released on completion.
  uint64_t signal_value;              // Target value to signal.
  int fence_fd;                       // Owned; closed on completion.
  iree_allocator_t allocator;         // For self-deallocation.
} iree_async_posix_fence_import_tracker_t;

// Handles fence import completion when the fd becomes readable.
// Called from poll dispatch when FENCE_IMPORT fd fires.
// - On POLLIN: signals semaphore
// - On POLLERR/POLLHUP: fails semaphore
// - Always: removes from fd_map/event_set, closes fd, releases semaphore, frees
// tracker
void iree_async_proactor_posix_handle_fence_import(
    iree_async_proactor_posix_t* proactor,
    iree_async_posix_fence_import_tracker_t* tracker, short revents);

//===----------------------------------------------------------------------===//
// Export fence tracker
//===----------------------------------------------------------------------===//

// Tracker for export fence: bridges semaphore timepoint → external fd.
// The tracker does NOT own the eventfd (caller owns it). The semaphore is
// retained for the lifetime of the tracker.
typedef struct iree_async_posix_fence_export_tracker_t {
  iree_async_semaphore_timepoint_t timepoint;  // Embedded; owned by semaphore.
  iree_async_semaphore_t* semaphore;  // Retained; released in callback.
  int eventfd;                        // Caller-owned; never closed.
  iree_allocator_t allocator;         // For self-deallocation.
} iree_async_posix_fence_export_tracker_t;

// Timepoint callback for export fence.
// Fires when the semaphore reaches the target value. On success, writes to the
// eventfd to make it readable. On failure, leaves the fd unreadable.
// Always releases the semaphore and frees the tracker.
// Never closes the eventfd (caller owns it).
void iree_async_posix_fence_export_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status);

//===----------------------------------------------------------------------===//
// Proactor vtable functions
//===----------------------------------------------------------------------===//

// Import fence vtable implementation.
// Thread-safe: defers fd_map/event_set registration to the poll thread via the
// pending_fence_imports MPSC queue. The caller may invoke this from any thread.
iree_status_t iree_async_proactor_posix_import_fence(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value);

// Drains the pending_fence_imports MPSC queue, registering deferred fence
// imports in fd_map and event_set. Must be called from the poll thread.
void iree_async_proactor_posix_drain_pending_fence_imports(
    iree_async_proactor_posix_t* proactor);

// Export fence vtable implementation.
// Bridges semaphore timepoint → external fd. Returns a caller-owned fd that
// becomes readable when the semaphore reaches the target value.
iree_status_t iree_async_proactor_posix_export_fence(
    iree_async_proactor_t* base_proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_FENCE_H_
