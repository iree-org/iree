// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Worker thread for the POSIX proactor.
//
// Workers execute async operations whose fds are ready (non-blocking I/O).
// They wait on the ready_queue, pop operations, execute them, and push
// completions back to the completion_queue for the poll thread to process.
//
// Workers are self-contained structs with an initialize/deinitialize lifecycle.
// The proactor allocates workers as a trailing array and initializes each one.
// Each worker has a back-pointer to the owning proactor for queue access.

#ifndef IREE_ASYNC_PLATFORM_POSIX_WORKER_H_
#define IREE_ASYNC_PLATFORM_POSIX_WORKER_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_async_proactor_posix_t;

// Worker state machine: RUNNING → EXITING → ZOMBIE.
//
// Transition graph:
//   RUNNING → EXITING → ZOMBIE
//
// RUNNING: Worker is idle or actively processing work.
// EXITING: Worker should exit (or is exiting) and will soon enter ZOMBIE.
// ZOMBIE: Worker has exited and is waiting for join/deinitialize.
typedef enum iree_async_posix_worker_state_e {
  IREE_ASYNC_POSIX_WORKER_STATE_RUNNING = 0,
  IREE_ASYNC_POSIX_WORKER_STATE_EXITING = 1,
  IREE_ASYNC_POSIX_WORKER_STATE_ZOMBIE = 2,
} iree_async_posix_worker_state_t;

// A worker within the POSIX proactor's thread pool.
//
// Workers are self-contained and use an initialize/deinitialize lifecycle.
// The proactor allocates workers as a trailing array in its single allocation,
// then initializes each. Each worker has a back-pointer to the proactor.
//
// Workers share proactor->ready_notification for work-stealing: all workers
// wait on the same notification, and enqueue_for_execution posts to it to
// wake exactly one idle worker.
typedef struct iree_async_posix_worker_t {
  // Current state (atomic for cross-thread visibility).
  // Written by request_exit, read by worker loop.
  iree_atomic_int32_t state;

  // Signaled when worker changes state (for await_exit).
  // Worker posts on state transition to ZOMBIE.
  iree_notification_t state_notification;

  // Fields below are thread-local (only touched by this worker's thread).

  // Owning proactor (for queue access, pool access, wake).
  // Set during initialize, never changes.
  struct iree_async_proactor_posix_t* proactor;

  // Worker index within pool (for debugging/tracing).
  iree_host_size_t worker_index;

  // Thread handle (owned, released in deinitialize).
  iree_thread_t* thread;

  // Padding to ensure struct spans at least one cache line (64 bytes) to
  // minimize false sharing when workers are stored in a contiguous array.
  uint8_t _padding[24];
} iree_async_posix_worker_t;

// Verify worker struct spans at least one cache line to minimize false sharing
// when workers are stored in a contiguous array.
static_assert(sizeof(iree_async_posix_worker_t) >=
                  iree_hardware_constructive_interference_size,
              "worker struct should span at least one cache line");

// Initializes a worker and starts its thread.
// |out_worker| must point to caller-allocated storage (typically in the
// proactor's trailing data array).
// On failure, caller should still call deinitialize for cleanup.
iree_status_t iree_async_posix_worker_initialize(
    struct iree_async_proactor_posix_t* proactor, iree_host_size_t worker_index,
    iree_allocator_t allocator, iree_async_posix_worker_t* out_worker);

// Requests that the worker begin exiting.
// Non-blocking; worker will exit after completing current work (if any).
// May be called from any thread.
void iree_async_posix_worker_request_exit(iree_async_posix_worker_t* worker);

// Blocks until the worker has entered ZOMBIE state.
// Must be called before deinitialize.
// May be called from any thread (but not from the worker itself).
void iree_async_posix_worker_await_exit(iree_async_posix_worker_t* worker);

// Deinitializes a worker that has exited (ZOMBIE state).
// Releases thread handle and cleans up notifications.
// The worker must have been requested to exit and awaited first.
void iree_async_posix_worker_deinitialize(iree_async_posix_worker_t* worker);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_WORKER_H_
