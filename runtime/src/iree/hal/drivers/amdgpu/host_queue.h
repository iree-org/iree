// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_

#include "iree/async/frontier.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/epoch_signal_table.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/notification_ring.h"
#include "iree/hal/drivers/amdgpu/virtual_queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_pending_op_t iree_hal_amdgpu_pending_op_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_queue_t
//===----------------------------------------------------------------------===//

// Maximum number of frontier entries the queue's accumulated frontier can
// track. Each entry is one (axis, epoch) pair representing a causal
// dependency on another queue or device. 64 entries covers rack-scale
// systems (8 machines x 8 GPUs x 4 queues = 256 theoretical axes, but a
// single queue only waits on its collective peers — typically 8-16 axes).
// Overflow is handled gracefully (frontier merge returns false, wait elision
// degrades but correctness is preserved).
//
// Transition snapshots serialize this frontier verbatim, so the queue's
// frontier capacity is tied to the notification ring's snapshot entry limit.
#define IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY \
  IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT

// Host-driven queue with per-queue epoch signal and proactor-drained
// notification ring. Embeds iree_hal_amdgpu_virtual_queue_t at offset 0.
//
// The epoch signal (owned by the notification ring) is a single hsa_signal_t
// set as completion_signal on each submission's last AQL packet. The CP
// decrements it by 1 on completion. The notification ring maps epochs to
// semaphore signals that the proactor drains when the epoch advances.
//
// All queue operations enter through the virtual_queue vtable. There are no
// public methods beyond initialize/deinitialize.
typedef struct iree_hal_amdgpu_host_queue_t {
  // Virtual queue vtable at offset 0.
  iree_hal_amdgpu_virtual_queue_t base;

  // HSA API handle for queue operations. Not retained.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Proactor for async notifications (borrowed from device).
  iree_async_proactor_t* proactor;
  iree_allocator_t host_allocator;

  // Sticky error status from the HSA queue error callback. Non-zero indicates
  // an unrecoverable GPU fault (page fault, invalid packet, ECC error).
  // First-error-wins CAS from the HSA runtime thread; acquire-loaded by the
  // proactor progress callback to fail pending semaphores instead of signaling.
  // Owned by the queue (freed in deinit).
  iree_atomic_intptr_t error_status;

  // Hardware AQL queue created via hsa_queue_create. Owned by this queue.
  hsa_queue_t* hardware_queue;

  // Cached AQL ring state for zero-indirection packet submission.
  // Initialized from hardware_queue at init time.
  iree_hal_amdgpu_aql_ring_t aql_ring;

  // Per-queue kernarg bump allocator backed by HSA coarse-grain memory.
  iree_hal_amdgpu_kernarg_ring_t kernarg_ring;

  // Epoch-driven notification ring mapping submission completions to
  // semaphore signals. The proactor progress callback drains this ring.
  iree_hal_amdgpu_notification_ring_t notification_ring;

  // Proactor bridge. Registers a progress callback that polls the epoch
  // signal via hsa_signal_load each proactor iteration and runs the drain
  // when it advances. An io_uring HSA_SIGNAL_WAIT SQE could replace the
  // polling — the drain callback would be identical, only the wakeup
  // mechanism changes.
  iree_async_progress_entry_t progress_entry;

  //--- Submission pipeline state -------------------------------------------//
  //
  // Threading model: the queue has three execution contexts.
  //
  //   Submission (any thread, serialized by submission_mutex):
  //     AQL slot reservation, packet fill, notification ring push, frontier
  //     snapshot push, queue frontier mutation, last_signal update, kernarg
  //     allocation. Multiple threads may submit to the same queue; the mutex
  //     serializes them. Independent queues do not synchronize.
  //
  //   Proactor (single proactor thread):
  //     Notification ring drain, error_status check. Reads the notification
  //     ring (SPSC consumer) and the atomic error_status. Never writes to
  //     submission-path fields.
  //
  //   HSA error callback (HSA runtime thread):
  //     Writes error_status via atomic CAS. Wakes the proactor.
  //
  // Wait-resolution fast-path contract:
  //   - Same-queue signal-before-wait is elided directly from the semaphore's
  //     last_signal cache when the cached producer axis matches queue->axis.
  //   - Local cross-queue waits use one producer epoch barrier when the
  //     semaphore cache marks that producer frontier as exact and this queue's
  //     frontier does not already dominate that producer axis/epoch.
  //   - The full semaphore-frontier mutex/copy path is reserved for unresolved
  //     waits whose cached producer frontier is not exact (for example, true
  //     multi-producer fan-in) or for conservative fallback after
  //     cache/frontier overflow.
  //   - Wait-before-signal, remote/non-queue-domain axes, and queue teardown
  //   use
  //     software deferral.
  //
  // Signal-commit fast-path contract:
  //   - Each successful AQL submission advances this queue's epoch, merges this
  //     queue axis into queue->frontier, pushes one notification-ring entry per
  //     signal semaphore, and publishes queue->frontier into each signaled
  //     semaphore under that semaphore's mutex.
  //   - AMDGPU semaphores also receive a seqlock-protected last_signal snapshot
  //     plus a PRODUCER_FRONTIER_EXACT flag derived while the semaphore mutex
  //     is held. That keeps consumer-side same-queue and exact-cross-queue
  //     waits off the semaphore-frontier mutex/copy path, but producer-side
  //     semaphore publication is still on the signal hot path today.

  // Serializes the submission path. All queue operations (dispatch, copy,
  // fill, execute, etc.) acquire this before touching submission state and
  // release after signal commit. The proactor thread does not acquire this.
  iree_slim_mutex_t submission_mutex;

  // Set under submission_mutex when queue teardown begins. Deferred ops whose
  // waits race to completion after this point are failed with CANCELLED instead
  // of issuing new AQL packets.
  bool is_shutting_down;

  // False once this queue's accumulated frontier overflows while merging waited
  // axes. After that, the frontier remains a safe lower bound for resolving
  // this queue's own waits, but it is no longer a conservative summary that can
  // be published to signal semaphores. Signal commits therefore clear
  // last_signal and skip semaphore-frontier merges, forcing downstream
  // not-yet-complete waits onto the software path instead of under-barriering.
  bool can_publish_frontier;

  // This queue's axis in the causal graph. Constructed from the system's
  // session epoch + machine index and this queue's device/queue ordinals.
  // Used to identify this queue in frontier entries and epoch signal lookups.
  // Immutable after initialization.
  iree_async_axis_t axis;

  // Shared epoch signal table for cross-queue barrier emission (tier 2 wait
  // resolution). Maps (device_index, queue_index) to hsa_signal_t for each
  // queue's epoch signal. Used to look up peer queues' epoch signals when
  // emitting AQL barrier-value packets for multi-axis dependencies (e.g.,
  // TP collective joins needing barriers on 7 peer queues).
  //
  // Borrowed from the device/system — valid for the lifetime of the queue.
  // This queue's own epoch signal is registered at init and deregistered at
  // deinit. Read-only during normal operation.
  iree_hal_amdgpu_epoch_signal_table_t* epoch_table;

  // Last semaphore pushed to the notification ring and its epoch. Used to
  // detect semaphore transitions for frontier snapshot recording: when a
  // push targets a different semaphore than last_signal.semaphore, the
  // signal commit path writes a frontier snapshot at last_signal.epoch
  // before starting the new span.
  //
  // Protected by submission_mutex (submission-context-only).
  //
  // ABA safety: the semaphore pointer is only compared for identity (not
  // dereferenced) during transition detection. ABA can occur if a semaphore
  // is released and a new one is allocated at the same address between two
  // submissions. This is benign:
  //   - The old semaphore's notification entries must have been drained
  //     before release (notification ring lifetime contract), so no
  //     undrained entries for the old semaphore remain.
  //   - A missed transition causes the new semaphore's entries to be
  //     coalesced with a span that has no pending entries — the drain
  //     produces the correct signal for the new semaphore.
  //   - The frontier snapshot at the end of the coalesced span (when the
  //     next transition occurs, or the fallback frontier at drain end)
  //     captures the queue's accumulated frontier, which is an upper bound
  //     on the actual causal context. Over-attribution (conservative), never
  //     under-attribution (unsafe).
  struct {
    iree_async_semaphore_t* semaphore;
    uint64_t epoch;
  } last_signal;

  // Block pool for arena-allocating deferred operations. NUMA-pinned to the
  // physical device. Borrowed from the physical device; valid for the
  // lifetime of the queue.
  iree_arena_block_pool_t* block_pool;

  // Ordinal of this queue's physical device within the topology. Used to look
  // up device-specific kernel_args from executables via
  // iree_hal_amdgpu_executable_lookup_kernel_args_for_device.
  iree_host_size_t device_ordinal;

  // Intrusive singly-linked list of pending (deferred) operations. Used for
  // cleanup on shutdown and GPU fault propagation. Operations add themselves
  // on deferral and remove themselves on issue/fail/cancel. Protected by
  // submission_mutex.
  iree_hal_amdgpu_pending_op_t* pending_head;

  // Accumulated frontier. Advances on each AQL submission: the queue's own
  // axis entry is set to the current epoch, and cross-queue wait dependencies
  // are merged in. Used for:
  //   - FIFO wait elision (tier 1): queue->frontier dominates the wait
  //     semaphore's frontier → skip the wait entirely.
  //   - Submission-time causal merge: merged into signal semaphores' frontiers
  //     at AQL submission time so that same-queue FIFO elision works before
  //     GPU completion.
  //   - Frontier snapshot recording: snapshotted to the notification ring's
  //     frontier byte ring at semaphore transitions.
  //
  // Inline storage sized for IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY entries.
  // Layout-compatible with iree_async_frontier_t when accessed via
  // iree_hal_amdgpu_host_queue_frontier(). The struct fields duplicate
  // iree_async_frontier_t's layout with a fixed-size entries array in place
  // of the FAM.
  struct {
    uint8_t entry_count;
    uint8_t reserved[7];
    iree_async_frontier_entry_t
        entries[IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY];
  } frontier;
} iree_hal_amdgpu_host_queue_t;

// Returns a pointer to the queue's accumulated frontier. The returned pointer
// is layout-compatible with iree_async_frontier_t and valid for all frontier
// APIs (compare, merge, etc.). Valid for the lifetime of the queue.
static inline iree_async_frontier_t* iree_hal_amdgpu_host_queue_frontier(
    iree_hal_amdgpu_host_queue_t* queue) {
  return (iree_async_frontier_t*)&queue->frontier;
}

// Returns a const pointer to the queue's accumulated frontier.
static inline const iree_async_frontier_t*
iree_hal_amdgpu_host_queue_const_frontier(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return (const iree_async_frontier_t*)&queue->frontier;
}

// Initializes a host queue in caller-provided memory.
// The caller must allocate at least sizeof(iree_hal_amdgpu_host_queue_t).
//
// Creates an HSA hardware queue on |gpu_agent|, initializes the AQL ring from
// it, allocates a kernarg ring from |kernarg_pool|, creates the epoch signal
// and notification ring, and registers the proactor progress callback.
//
// |axis| is this queue's identity in the causal graph, constructed by the
// caller from the system's session/machine identifiers and this queue's
// device/queue ordinals via iree_async_axis_make_queue().
//
// |epoch_table| is the shared epoch signal table for cross-queue barrier
// emission. This queue registers its epoch signal in the table at init and
// deregisters at deinit. The table must outlive the queue.
//
// |aql_queue_capacity| is the power-of-two hardware AQL queue size in packets.
// |notification_capacity| is the power-of-two notification ring size.
// |kernarg_capacity_in_blocks| is the power-of-two kernarg ring size in
// 64-byte blocks. Must satisfy the backpressure invariant:
//   kernarg_capacity >= aql_queue_capacity * max_kernarg_blocks_per_packet.
iree_status_t iree_hal_amdgpu_host_queue_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_async_proactor_t* proactor,
    hsa_agent_t gpu_agent, hsa_amd_memory_pool_t kernarg_pool,
    const iree_hal_amdgpu_topology_t* topology, iree_async_axis_t axis,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_table,
    iree_arena_block_pool_t* block_pool, iree_host_size_t device_ordinal,
    uint32_t aql_queue_capacity, uint32_t notification_capacity,
    uint32_t kernarg_capacity_in_blocks, iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_queue_t* out_queue);

// Deinitializes the queue. Destroys all owned resources and unregisters the
// proactor progress callback.
//
// All in-flight work must have completed and been drained before calling.
// The caller must ensure no concurrent access to the queue during deinit.
void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_
