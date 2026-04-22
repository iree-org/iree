// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_WORKER_H_
#define IREE_TASK_WORKER_H_

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/prng.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/task/affinity_set.h"
#include "iree/task/executor.h"
#include "iree/task/topology.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Indicates the current state of a worker or, in the case of EXITING, the state
// the worker should transition to.
//
// Transition graph:
//   SUSPENDED -> RUNNING (IDLE<->PROCESSING) -> EXITING -> ZOMBIE
//
// NOTE: state values are ordered such that </> comparisons can be used; ensure
// that for example all states after resuming are > SUSPENDED and all states
// before exiting are < EXITING.
typedef enum iree_task_worker_state_e {
  // Worker is idle or actively processing tasks (either its own or others).
  IREE_TASK_WORKER_STATE_RUNNING = 0,
  // Worker should exit (or is exiting) and will soon enter the zombie state.
  // Coordinators can request workers to exit by setting their state to this and
  // then waking.
  IREE_TASK_WORKER_STATE_EXITING = 1,
  // Worker has exited and entered a zombie state (waiting for join).
  // The thread handle is still valid and must be destroyed.
  IREE_TASK_WORKER_STATE_ZOMBIE = 2,
} iree_task_worker_state_t;

// A worker within the executor pool.
//
// Workers drain processes from two sources:
//   - wake_budget == 1 immediate list: popped exclusively and drained to
//     completion or sleep.
//   - wake_budget > 1 compute slots: scanned round-robin and cooperatively
//     drained.
//
// Cache line layout:
//   Line 0 (cross-thread written): state, wake_notification,
//       state_notification. These fields are written by external threads
//       (wake_workers posts wake_notification, request_exit stores state).
//       Grouped together so cross-thread writes only invalidate this line.
//   Line 1+ (owner-only): executor, worker_index, worker_bit, affinity,
//       compute_slot_scan_start, thread, processor_id, local_memory. These
//       are either read-only after init or written exclusively by the owning
//       worker thread. Isolating them from line 0 means cross-thread wake
//       notifications don't invalidate the worker's hot read state.
//
// The struct is cache-line aligned to prevent false sharing between adjacent
// workers in the executor's contiguous worker array.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_task_worker_t {
  //===--------------------------------------------------------------------===//
  // Cache line 0: cross-thread written
  //===--------------------------------------------------------------------===//

  // Current state of the worker (iree_task_worker_state_t).
  // Written by request_exit (from any thread), read by the worker to check
  // for exit requests.
  iree_atomic_int32_t state;

  // Notification signaled when the worker should wake (if it is idle).
  // Posted by wake_workers (from any thread including other workers),
  // prepare/commit/cancel-waited by the owning worker.
  iree_notification_t wake_notification;

  // Notification signaled when the worker changes any state.
  // Posted by the worker on exit, awaited by request_exit/await_exit callers.
  iree_notification_t state_notification;

  //===--------------------------------------------------------------------===//
  // Cache line 1+: owner-only (read-only after init or worker-exclusive)
  //===--------------------------------------------------------------------===//

  // Parent executor that can be used to access the global work queue or task
  // pool. Executors always outlive the workers they own.
  // Read-only after initialization.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_task_executor_t* executor;

  // Globally unique worker index (worker_base_index + local worker_index).
  // Read-only after initialization.
  iree_host_size_t worker_index;

  // Precomputed bit position the worker represents in executor bitsets.
  // Local to the executor owning the worker.
  // Read-only after initialization.
  iree_task_affinity_bit_t worker_bit;

  // Ideal thread affinity for the worker thread.
  // Read-only after initialization.
  iree_thread_affinity_t ideal_thread_affinity;

  // A bitmask of other group indices that share some level of the cache
  // hierarchy. Workers of this group are more likely to constructively share
  // some cache levels higher up with these other groups. For example, if the
  // workers in a group all share an L2 cache then the groups indicated here may
  // all share the same L3 cache.
  // Read-only after initialization.
  iree_task_affinity_set_t constructive_sharing_mask;

  // Round-robin starting offset for compute slot scanning. Each worker starts
  // from a different slot to distribute scanning pressure evenly. Advanced by
  // one after each scan pass.
  // Written only by the owning worker thread.
  uint32_t compute_slot_scan_start;

  // Thread handle of the worker. If the thread has exited the handle will
  // remain valid so that the executor can query its state.
  // Read-only after initialization (set once, released on deinitialize).
  iree_thread_t* thread;

  // Guess at the current processor ID.
  // This is updated infrequently as it can be semi-expensive to determine
  // (on some platforms at least 1 syscall involved). We always update it upon
  // waking as idle waits are the most likely place the worker will be migrated
  // across processors.
  // Written only by the owning worker thread.
  iree_cpu_processor_id_t processor_id;
  // An opaque tag used to reduce the cost of processor ID queries.
  // Written only by the owning worker thread.
  iree_cpu_processor_tag_t processor_tag;

  // Pointer to local memory available for use exclusively by the worker.
  // The base address should be aligned to avoid false sharing with other
  // workers.
  // Read-only after initialization.
  iree_byte_span_t local_memory;
} iree_task_worker_t;

// Initializes a worker by creating its thread and configuring it for receiving
// tasks. Where supported the worker will be created in a suspended state so
// that we aren't creating a thundering herd on startup:
// https://en.wikipedia.org/wiki/Thundering_herd_problem
iree_status_t iree_task_worker_initialize(
    iree_task_executor_t* executor, iree_host_size_t worker_index,
    const iree_task_topology_group_t* topology_group,
    iree_host_size_t stack_size, iree_byte_span_t local_memory,
    iree_prng_splitmix64_state_t* seed_prng, iree_task_worker_t* out_worker);

// Requests that the worker begin exiting (if it hasn't already).
// If the worker is actively processing tasks it will wait until it has
// completed all it can and is about to go idle prior to exiting.
//
// May be called from any thread (including the worker thread).
void iree_task_worker_request_exit(iree_task_worker_t* worker);

// Blocks the caller until |worker| has exited.
//
// May be called from any thread.
void iree_task_worker_await_exit(iree_task_worker_t* worker);

// Deinitializes a worker that has successfully exited.
// The worker must be in the IREE_TASK_WORKER_STATE_ZOMBIE state.
//
// Expected shutdown sequence:
//  - request_exit on all workers
//  - await_exit on all workers
//  - deinitialize all workers
void iree_task_worker_deinitialize(iree_task_worker_t* worker);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_WORKER_H_
