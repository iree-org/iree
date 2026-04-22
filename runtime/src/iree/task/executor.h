// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_EXECUTOR_H_
#define IREE_TASK_EXECUTOR_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/task/process.h"
#include "iree/task/scope.h"
#include "iree/task/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//==============================================================================
// IREE Task Executor
//==============================================================================
//
// The executor manages a pool of worker threads and provides two scheduling
// mechanisms for processes:
//
// Processes with wake_budget == 1 (immediate list):
//   Sequential processes that need exclusive draining (one worker at a time).
//   Pushed to a lock-free MPSC list; workers pop and drain to completion or
//   sleep. Used for queue management, host callbacks, and retire/signal paths.
//
// Processes with wake_budget > 1 (compute slots):
//   Parallel processes that benefit from multiple workers draining
//   concurrently. Placed into fixed slots (CAS on activation); workers scan
//   round-robin and cooperatively drain bounded work from each occupied slot.
//   The wake budget controls wake fan-out, not admission to drain().
//
// Workers alternate between draining immediate processes and scanning compute
// slots. When no work is available, workers sleep via a notification-based
// Dekker protocol until schedule_process wakes them.
//
// The executor scales from single-worker configurations (useful for
// deterministic testing or embedded systems) to
// IREE_TASK_EXECUTOR_MAX_WORKER_COUNT worker topologies (default 256) with
// NUMA-aware thread pinning. Multiple executors can be composed for systems
// requiring cross-NUMA isolation.

// A bitfield specifying the scheduling mode used for configuring how (or if)
// work is balanced across queues.
enum iree_task_scheduling_mode_bits_t {
  IREE_TASK_SCHEDULING_MODE_RESERVED = 0u,
};
typedef uint32_t iree_task_scheduling_mode_t;

// Options controlling task executor behavior.
typedef struct iree_task_executor_options_t {
  // Specifies the schedule mode used for worker and workload balancing.
  iree_task_scheduling_mode_t scheduling_mode;

  // Base value added to each executor-local worker index.
  // This allows workers to uniquely identify themselves in multi-executor
  // configurations.
  iree_host_size_t worker_base_index;

  // Maximum duration in nanoseconds each worker should spin waiting for
  // additional work. In almost all cases this should be IREE_DURATION_ZERO as
  // spinning is often extremely harmful to system health. Only set to non-zero
  // values when latency is the #1 priority (over thermals, system-wide
  // scheduling, and the environment).
  iree_duration_t worker_spin_ns;

  // Minimum size in bytes of each worker thread stack.
  // The underlying platform may allocate more stack space but _should_
  // guarantee that the available stack space is near this amount. Note that the
  // task system will take some stack space and not all bytes should be assumed
  // usable. Note that as much as possible users should not rely on the stack
  // for storage over ~16-32KB and instead use local workgroup memory.
  iree_host_size_t worker_stack_size;

  // Defines the bytes to be allocated and reserved by each worker to use for
  // local memory operations. Will be rounded up to the next power of two.
  // Dispatches performed will be able to request up to this amount of memory
  // for their invocations and no more. May be 0 if no worker local memory is
  // required.
  // By default the CPU L2 cache size is used if such queries are supported.
  iree_host_size_t worker_local_memory_size;
} iree_task_executor_options_t;

// Initializes |out_options| to default values.
void iree_task_executor_options_initialize(
    iree_task_executor_options_t* out_options);

// Base task system executor interface.
typedef struct iree_task_executor_t iree_task_executor_t;

// Creates a task executor using the specified topology.
// |options| must be initialized with iree_task_executor_options_initialize by
// callers and then overridden as required.
// |topology| is only used during creation and need not live beyond this call.
// |out_executor| must be released by the caller.
iree_status_t iree_task_executor_create(iree_task_executor_options_t options,
                                        const iree_task_topology_t* topology,
                                        iree_allocator_t allocator,
                                        iree_task_executor_t** out_executor);

// Retains the given |executor| for the caller.
void iree_task_executor_retain(iree_task_executor_t* executor);

// Releases the given |executor| from the caller.
void iree_task_executor_release(iree_task_executor_t* executor);

// Returns the NUMA node this executor's workers are pinned to, or
// IREE_TASK_TOPOLOGY_NODE_ID_ANY if the executor was created without a
// specific NUMA node assignment.
iree_task_topology_node_id_t iree_task_executor_node_id(
    iree_task_executor_t* executor);

// Trims pools and caches used by the executor and its workers.
void iree_task_executor_trim(iree_task_executor_t* executor);

// Returns the number of live workers usable by the executor.
// The actual number used for any particular operation is dynamic.
iree_host_size_t iree_task_executor_worker_count(
    iree_task_executor_t* executor);

// Returns a pointer to the executor's desired_wake counter. Processes can
// atomically add wake credits to this counter at region transitions (when
// ramping up worker count). Workers claim credits via relay_wake,
// propagating the wake tree without requiring direct notification posts.
iree_atomic_int32_t* iree_task_executor_desired_wake_ptr(
    iree_task_executor_t* executor);

// Schedules a process for draining by a worker. If the process is idle, it is
// pushed to the appropriate run list and a worker is woken. If the process is
// already queued or being drained, this is a no-op — the draining worker will
// re-check for new work before going idle.
//
// Callers must make work available to the process (e.g., push to its ready
// list) BEFORE calling this. The needs_drain flag is set to ensure the draining
// worker sees the new work even if it's mid-drain.
//
// Thread-safe: may be called from any thread (proactor, semaphore callback,
// completing worker, user thread).
void iree_task_executor_schedule_process(iree_task_executor_t* executor,
                                         iree_task_process_t* process);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_EXECUTOR_H_
