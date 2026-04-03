// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

// The inline frontier on host_queue_t uses the same {entry_count, reserved[7],
// entries[N]} layout as iree_async_frontier_t + FAM. Verify the entries field
// starts at the FAM offset so iree_hal_amdgpu_host_queue_frontier() is valid.
static_assert(offsetof(iree_hal_amdgpu_host_queue_t, frontier.entries) -
                      offsetof(iree_hal_amdgpu_host_queue_t, frontier) ==
                  sizeof(iree_async_frontier_t),
              "inline frontier entries must align with frontier_t FAM offset");

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable;

//===----------------------------------------------------------------------===//
// Fixed-capacity frontier storage
//===----------------------------------------------------------------------===//

// Stack-allocable frontier with fixed capacity. Layout-compatible with
// iree_async_frontier_t when cast — same {entry_count, reserved, entries[]}
// field layout with a fixed-size array in place of the FAM.
typedef struct iree_hal_amdgpu_fixed_frontier_t {
  uint8_t entry_count;
  uint8_t reserved[7];
  iree_async_frontier_entry_t entries[IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY];
} iree_hal_amdgpu_fixed_frontier_t;

static_assert(offsetof(iree_hal_amdgpu_fixed_frontier_t, entries) ==
                  sizeof(iree_async_frontier_t),
              "fixed frontier entries must align with frontier_t FAM offset");

static inline iree_async_frontier_t* iree_hal_amdgpu_fixed_frontier_as_frontier(
    iree_hal_amdgpu_fixed_frontier_t* storage) {
  return (iree_async_frontier_t*)storage;
}

//===----------------------------------------------------------------------===//
// Wait resolution
//===----------------------------------------------------------------------===//

// Returns true if |frontier| contains |axis| at an epoch >= |target_epoch|.
// Frontier entries are sorted by axis, so the scan can stop as soon as the
// target axis has been passed.
static bool iree_hal_amdgpu_frontier_dominates_axis(
    const iree_async_frontier_t* frontier, iree_async_axis_t axis,
    uint64_t target_epoch) {
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    const iree_async_frontier_entry_t* entry = &frontier->entries[i];
    if (entry->axis < axis) continue;
    return entry->axis == axis && entry->epoch >= target_epoch;
  }
  return false;
}

// A single barrier to emit as an AQL barrier-value packet. Produced by
// resolve_waits for each undominated axis that maps to a local queue.
typedef struct iree_hal_amdgpu_wait_barrier_t {
  iree_async_axis_t axis;
  hsa_signal_t epoch_signal;
  uint64_t target_epoch;
} iree_hal_amdgpu_wait_barrier_t;

// Result of resolving a wait_semaphore_list. Either all waits are resolved
// (barrier_count barriers to emit) or deferral is needed.
//
// barrier_count and needs_deferral are at the top so they share a cache line
// with barriers[0] — the caller checks deferral then iterates barriers[0..N].
typedef struct iree_hal_amdgpu_wait_resolution_t {
  uint8_t barrier_count;
  bool needs_deferral;
  uint8_t reserved[6];
  iree_hal_amdgpu_wait_barrier_t
      barriers[IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY];
} iree_hal_amdgpu_wait_resolution_t;

// Appends a tier-2 barrier to |resolution| while preserving ascending axis
// order and deduplicating repeated axes across multiple waits. If the barrier
// budget is exhausted, returns false so the caller can fall back to software
// deferral instead of relying on a debug-only assert.
//
// Caller must hold submission_mutex.
static bool iree_hal_amdgpu_host_queue_append_wait_barrier(
    iree_hal_amdgpu_wait_resolution_t* resolution, iree_async_axis_t axis,
    hsa_signal_t epoch_signal, uint64_t target_epoch) {
  uint8_t insert_ordinal = 0;
  while (insert_ordinal < resolution->barrier_count &&
         resolution->barriers[insert_ordinal].axis < axis) {
    ++insert_ordinal;
  }

  if (insert_ordinal < resolution->barrier_count &&
      resolution->barriers[insert_ordinal].axis == axis) {
    if (target_epoch > resolution->barriers[insert_ordinal].target_epoch) {
      resolution->barriers[insert_ordinal].target_epoch = target_epoch;
      resolution->barriers[insert_ordinal].epoch_signal = epoch_signal;
    }
    return true;
  }

  if (resolution->barrier_count >= IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY) {
    return false;
  }

  for (uint8_t i = resolution->barrier_count; i > insert_ordinal; --i) {
    resolution->barriers[i] = resolution->barriers[i - 1];
  }

  iree_hal_amdgpu_wait_barrier_t* barrier =
      &resolution->barriers[insert_ordinal];
  barrier->axis = axis;
  barrier->epoch_signal = epoch_signal;
  barrier->target_epoch = target_epoch;
  ++resolution->barrier_count;
  return true;
}

// Resolves a single (semaphore, value) wait. Appends tier 2 barriers to
// |resolution| if needed. Returns true if the wait is resolved (satisfied or
// barriers appended), false if deferral is needed.
//
// Tier 0: timeline_value >= value → already completed.
// Tier 1a: signal submitted by this queue → FIFO-elided directly from
//   last_signal, no semaphore-frontier mutex/copy.
// Tier 1b: signal submitted by a producer epoch that exactly covers the
//   semaphore frontier, and this queue already dominates that producer epoch
//   → FIFO-elided directly from last_signal.
// Tier 1c: signal submitted + queue frontier dominates → FIFO-elided.
// Tier 2a: signal submitted by a local producer epoch that exactly covers the
//   semaphore frontier → append one barrier directly from last_signal.
// Tier 2b: signal submitted + local queue axes from semaphore frontier →
//   barriers appended from the undominated frontier entries.
// Tier 3: anything else → deferral.
//
// Caller must hold submission_mutex.
static bool iree_hal_amdgpu_host_queue_resolve_wait(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_semaphore_t* semaphore,
    uint64_t value, iree_hal_amdgpu_wait_resolution_t* resolution) {
  iree_async_semaphore_t* async_semaphore = (iree_async_semaphore_t*)semaphore;

  // A failed semaphore must take the software deferral path so the timepoint
  // callback propagates the failure to this op's signal semaphores.
  if (iree_async_semaphore_query_status(async_semaphore) != IREE_STATUS_OK) {
    return false;
  }

  // Tier 0: already completed. Cheapest check (one atomic load).
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &async_semaphore->timeline_value, iree_memory_order_acquire);
  if (current_value >= value) return true;

  // Not completed. Must be an AMDGPU semaphore for device-side resolution.
  if (!iree_hal_amdgpu_semaphore_isa(semaphore)) return false;

  // Has the signal for |value| been submitted? The last_signal cache records
  // the most recent signal's value. If it hasn't reached |value|, the signal
  // hasn't been submitted yet (wait-before-signal) and the frontier does not
  // reflect the signal's causal context — frontier dominance would be a
  // false positive.
  iree_hal_amdgpu_last_signal_flags_t signal_flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
  iree_async_axis_t signal_axis = 0;
  uint64_t signal_epoch = 0;
  uint64_t signal_value = 0;
  if (!iree_hal_amdgpu_last_signal_load(
          iree_hal_amdgpu_semaphore_last_signal(semaphore), &signal_flags,
          &signal_axis, &signal_epoch, &signal_value) ||
      signal_value < value) {
    return false;
  }

  // Tier 1a: same-queue FIFO elision from the last_signal cache alone.
  // If the covering signal was submitted by this queue, then submission order
  // under submission_mutex plus AQL FIFO ordering guarantees this operation's
  // packets execute after that signal's completion point. Avoid the semaphore
  // frontier mutex/copy entirely on the dominant HIP-stream-on-one-queue path.
  if (signal_axis == queue->axis) return true;

  // Tier 1b/2a: when the semaphore cache says the producer queue's epoch
  // exactly covers the unresolved semaphore frontier, resolve directly from
  // that producer axis/epoch snapshot. This avoids the semaphore-frontier
  // mutex/copy on common cross-queue handoffs while still refusing to guess
  // on TP fan-in semaphores with independent producers.
  if (signal_flags & IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT) {
    if (iree_hal_amdgpu_frontier_dominates_axis(
            iree_hal_amdgpu_host_queue_const_frontier(queue), signal_axis,
            signal_epoch)) {
      return true;
    }
    hsa_signal_t peer_signal;
    if (!iree_hal_amdgpu_epoch_signal_table_lookup(queue->epoch_table,
                                                   signal_axis, &peer_signal)) {
      return false;
    }
    return iree_hal_amdgpu_host_queue_append_wait_barrier(
        resolution, signal_axis, peer_signal, signal_epoch);
  }

  // Signal submitted, not completed. Copy the semaphore's frontier and find
  // axes that our queue doesn't dominate.
  iree_hal_amdgpu_fixed_frontier_t semaphore_frontier;
  iree_async_semaphore_query_frontier(
      async_semaphore,
      iree_hal_amdgpu_fixed_frontier_as_frontier(&semaphore_frontier),
      IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY);

  iree_async_frontier_entry_t
      undominated[IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY];
  uint8_t undominated_count = iree_async_frontier_find_undominated(
      iree_hal_amdgpu_host_queue_const_frontier(queue),
      iree_hal_amdgpu_fixed_frontier_as_frontier(&semaphore_frontier),
      IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY, undominated);

  // Tier 1c: all axes dominated → FIFO-elided.
  if (undominated_count == 0) return true;

  // Tier 2b: look up each undominated axis in the epoch signal table.
  // If any axis is not a local queue (remote, collective, host), defer.
  for (uint8_t i = 0; i < undominated_count; ++i) {
    hsa_signal_t peer_signal;
    if (!iree_hal_amdgpu_epoch_signal_table_lookup(
            queue->epoch_table, undominated[i].axis, &peer_signal)) {
      return false;
    }
    if (!iree_hal_amdgpu_host_queue_append_wait_barrier(
            resolution, undominated[i].axis, peer_signal,
            undominated[i].epoch)) {
      return false;
    }
  }
  return true;
}

// Resolves a wait_semaphore_list into barriers or deferral.
//
// All-or-nothing: if ANY wait is tier 3, sets needs_deferral and returns
// immediately. No partial barriers are emitted.
//
// Caller must hold submission_mutex.
static void iree_hal_amdgpu_host_queue_resolve_waits(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_wait_resolution_t* out_resolution) {
  out_resolution->barrier_count = 0;
  out_resolution->needs_deferral = false;
  memset(out_resolution->reserved, 0, sizeof(out_resolution->reserved));

  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    if (!iree_hal_amdgpu_host_queue_resolve_wait(
            queue, wait_semaphore_list.semaphores[i],
            wait_semaphore_list.payload_values[i], out_resolution)) {
      out_resolution->needs_deferral = true;
      out_resolution->barrier_count = 0;
      return;
    }
  }
}

// Emits AQL barrier-value packets for resolved waits. Each barrier halts the
// CP until the peer queue's epoch signal reaches the target epoch.
//
// |first_packet_id| is the first AQL slot reserved for barriers. The caller
// must have reserved resolution->barrier_count consecutive slots starting here.
//
// Caller must hold submission_mutex.
static void iree_hal_amdgpu_host_queue_emit_barriers(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    uint64_t first_packet_id) {
  for (uint8_t i = 0; i < resolution->barrier_count; ++i) {
    const iree_hal_amdgpu_wait_barrier_t* barrier = &resolution->barriers[i];
    iree_hal_amdgpu_aql_packet_t* packet =
        iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, first_packet_id + i);

    // The epoch signal starts at INITIAL_VALUE and is decremented by 1 per
    // completion. The barrier fires when:
    //   signal_load(s) < INITIAL_VALUE - target_epoch + 1
    // which is true once current_epoch >= target_epoch.
    iree_hsa_signal_value_t compare_value =
        (iree_hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                                  barrier->target_epoch + 1);

    uint16_t header = iree_hal_amdgpu_aql_emit_barrier_value(
        &packet->barrier_value,
        (iree_hsa_signal_t){.handle = barrier->epoch_signal.handle},
        IREE_HSA_SIGNAL_CONDITION_LT, compare_value,
        ~(iree_hsa_signal_value_t)0, iree_hsa_signal_null());
    iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
  }
}

// Merges the waited axes from resolved barriers into the queue's accumulated
// frontier. This records that the queue has barrier'd on these axes, enabling
// future submissions to FIFO-elide waits on the same axes.
//
// The barrier entries are maintained in ascending axis order by
// append_wait_barrier(), satisfying the frontier's sorted invariant.
//
// Caller must hold submission_mutex.
static void iree_hal_amdgpu_host_queue_merge_barrier_axes(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution) {
  if (resolution->barrier_count == 0 || !queue->can_publish_frontier) return;
  iree_hal_amdgpu_fixed_frontier_t barrier_frontier;
  iree_async_frontier_initialize(
      iree_hal_amdgpu_fixed_frontier_as_frontier(&barrier_frontier),
      resolution->barrier_count);
  for (uint8_t i = 0; i < resolution->barrier_count; ++i) {
    barrier_frontier.entries[i].axis = resolution->barriers[i].axis;
    barrier_frontier.entries[i].epoch = resolution->barriers[i].target_epoch;
  }
  if (!iree_async_frontier_merge(
          iree_hal_amdgpu_host_queue_frontier(queue),
          IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY,
          iree_hal_amdgpu_fixed_frontier_as_frontier(&barrier_frontier))) {
    queue->can_publish_frontier = false;
  }
}

// Returns a conservative upper bound on the number of frontier snapshots that
// commit_signals will push for |signal_semaphore_list|.
//
// Caller must hold submission_mutex.
static iree_host_size_t iree_hal_amdgpu_host_queue_count_frontier_snapshots(
    const iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_host_size_t snapshot_count = 0;
  iree_async_semaphore_t* last_semaphore = queue->last_signal.semaphore;
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_async_semaphore_t* semaphore =
        (iree_async_semaphore_t*)signal_semaphore_list.semaphores[i];
    if (semaphore == last_semaphore) continue;
    if (last_semaphore != NULL) {
      ++snapshot_count;
    }
    last_semaphore = semaphore;
  }
  return snapshot_count;
}

//===----------------------------------------------------------------------===//
// Pending operations (deferred submission)
//===----------------------------------------------------------------------===//
//
// LOCKING PROTOCOL
//
// submission_mutex protects all submission-path state: AQL ring reservation,
// kernarg allocation, packet emission, commit_signals, frontier mutation,
// notification ring push, and the pending list (link/unlink).
//
// The proactor thread (drain, error check) does NOT acquire submission_mutex.
// It reads the notification ring (SPSC consumer) and the atomic error_status.
//
// Deferred operations use a two-phase protocol:
//
//   Phase 1 (under submission_mutex): resolve_waits, allocate pending_op,
//     capture operation parameters, link to pending list.
//
//   Phase 2 (WITHOUT submission_mutex): register timepoints via enqueue_waits.
//     Timepoint callbacks may fire synchronously during acquire_timepoint
//     (when the semaphore value is already reached or the semaphore is already
//     failed). The last callback to fire calls pending_op_issue or
//     pending_op_fail, both of which acquire submission_mutex internally.
//     This is safe because Phase 1 released the mutex before Phase 2 began.
//
// pending_op_issue: acquires submission_mutex to emit AQL packets, transfer
//   retained resources to the reclaim ring, commit signals, and unlink.
//
// pending_op_fail: acquires submission_mutex to unlink. Semaphore failure
//   and resource release happen outside the lock.
//
// pending_op_destroy_under_lock: for capture-time failures (arena allocation
//   errors after pending_op_allocate). Caller already holds submission_mutex.
//   Does NOT re-acquire — unlinks and cleans up directly.

// Operation types corresponding to virtual queue vtable entries. Each type
// has a per-operation capture struct in the pending_op_t union.
typedef enum iree_hal_amdgpu_pending_op_type_e {
  IREE_HAL_AMDGPU_PENDING_OP_FILL,
  IREE_HAL_AMDGPU_PENDING_OP_COPY,
  IREE_HAL_AMDGPU_PENDING_OP_UPDATE,
  IREE_HAL_AMDGPU_PENDING_OP_DISPATCH,
  IREE_HAL_AMDGPU_PENDING_OP_EXECUTE,
  IREE_HAL_AMDGPU_PENDING_OP_ALLOCA,
  IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA,
  IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL,
} iree_hal_amdgpu_pending_op_type_t;

// Completion ownership for a deferred operation.
typedef enum iree_hal_amdgpu_pending_op_lifecycle_e {
  // Waiting callbacks may still resolve the op.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING = 0,
  // Queue shutdown claimed cancellation ownership.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING = 1,
  // The last wait callback claimed completion ownership.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING = 2,
} iree_hal_amdgpu_pending_op_lifecycle_t;

// Per-wait timepoint entry, arena-allocated (one per unsatisfied wait).
// The timepoint callback decrements the operation's atomic wait counter;
// the last callback to fire issues (or fails) the operation.
typedef struct iree_hal_amdgpu_wait_entry_t {
  iree_async_semaphore_timepoint_t timepoint;
  iree_hal_amdgpu_pending_op_t* operation;
  // Set to 1 after the callback's final access to this entry/op completes.
  // Queue shutdown spins on this for callbacks that were already detached from
  // the semaphore before cancel_timepoint() ran.
  iree_atomic_int32_t callback_complete;
} iree_hal_amdgpu_wait_entry_t;

// A deferred queue operation waiting for its waits to become satisfiable.
// Arena-allocated from the queue's block pool. All variable-size captured
// data (semaphore lists, constants, bindings, update source data) lives in
// the arena alongside this struct.
struct iree_hal_amdgpu_pending_op_t {
  // Arena backing this operation and all captured data. Deinitialized on
  // completion, failure, or cancellation — returns blocks to the pool.
  iree_arena_allocator_t arena;

  // Owning queue. Used to acquire submission_mutex and emit AQL packets
  // when all waits are satisfied.
  iree_hal_amdgpu_host_queue_t* queue;

  // Intrusive linked list for the queue's pending list (cleanup/shutdown).
  iree_hal_amdgpu_pending_op_t* next;
  iree_hal_amdgpu_pending_op_t** prev_next;

  // Number of outstanding wait timepoints. Atomically decremented by each
  // wait_entry callback. When this reaches zero, the operation is ready.
  iree_atomic_int32_t wait_count;

  // Completion/cancellation owner. Exactly one path transitions this away from
  // PENDING and then destroys the operation.
  iree_atomic_int32_t lifecycle_state;

  // First error from a failed wait. CAS from 0; the winner owns the status.
  iree_atomic_intptr_t error_status;

  // Wait semaphore list (arena-allocated clone, semaphores retained by the
  // clone). Released when the op is destroyed.
  iree_hal_semaphore_list_t wait_semaphore_list;

  // Signal semaphore list. The semaphores[] pointer points into the first
  // entries of retained_resources (the semaphore pointers are shared, not
  // separately retained). The payload_values[] is a separate arena-allocated
  // array. Used for commit_signals and semaphore_list_fail.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Wait entries registered with the wait semaphores, one per
  // wait_semaphore_list entry. Arena-allocated as one contiguous array so queue
  // shutdown can cancel and await each callback before freeing the arena.
  iree_hal_amdgpu_wait_entry_t* wait_entries;

  // Flat array of all retained HAL resources (arena-allocated). Signal
  // semaphores are stored first (signal_semaphore_list.semaphores aliases
  // this region), followed by operation-specific resources (buffers,
  // executables, command buffers). On successful issue, ownership transfers
  // to the reclaim ring. On failure/cancel, released directly.
  iree_hal_resource_t** retained_resources;
  uint16_t retained_resource_count;

  iree_hal_amdgpu_pending_op_type_t type;
  union {
    struct {
      iree_hal_buffer_t* target_buffer;
      iree_device_size_t target_offset;
      iree_device_size_t length;
      uint64_t pattern_bits;
      iree_host_size_t pattern_length;
      iree_hal_fill_flags_t flags;
    } fill;
    struct {
      iree_hal_buffer_t* source_buffer;
      iree_device_size_t source_offset;
      iree_hal_buffer_t* target_buffer;
      iree_device_size_t target_offset;
      iree_device_size_t length;
      iree_hal_copy_flags_t flags;
    } copy;
    struct {
      // Source data is copied into the arena (not a borrowed pointer).
      const void* source_data;
      iree_host_size_t source_offset;
      iree_hal_buffer_t* target_buffer;
      iree_device_size_t target_offset;
      iree_device_size_t length;
      iree_hal_update_flags_t flags;
    } update;
    struct {
      iree_hal_executable_t* executable;
      iree_hal_executable_export_ordinal_t export_ordinal;
      iree_hal_dispatch_config_t config;
      iree_const_byte_span_t constants;     // Arena-allocated copy.
      iree_hal_buffer_ref_list_t bindings;  // Arena-allocated copy.
      iree_hal_dispatch_flags_t flags;
    } dispatch;
    struct {
      iree_hal_command_buffer_t* command_buffer;
      iree_hal_buffer_binding_table_t binding_table;  // Arena-allocated copy.
      iree_hal_execute_flags_t flags;
    } execute;
    struct {
      iree_hal_allocator_pool_t pool;
      iree_hal_buffer_params_t params;
      iree_device_size_t allocation_size;
      iree_hal_alloca_flags_t flags;
      iree_hal_buffer_t* buffer;
    } alloca_op;
    struct {
      iree_hal_buffer_t* buffer;
      iree_hal_dealloca_flags_t flags;
    } dealloca;
    struct {
      iree_hal_host_call_t call;
      uint64_t args[4];
      iree_hal_host_call_flags_t flags;
    } host_call;
  };
};

static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op);
static iree_status_t iree_hal_amdgpu_pending_op_fail(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status);

// Links a pending op into the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_link(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  op->next = queue->pending_head;
  op->prev_next = &queue->pending_head;
  if (queue->pending_head) {
    queue->pending_head->prev_next = &op->next;
  }
  queue->pending_head = op;
}

// Unlinks a pending op from the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_unlink(
    iree_hal_amdgpu_pending_op_t* op) {
  *op->prev_next = op->next;
  if (op->next) {
    op->next->prev_next = op->prev_next;
  }
  op->next = NULL;
  op->prev_next = NULL;
}

// Retains a resource and appends it to the pending op's retained_resources
// array. The caller must have allocated sufficient capacity in the array
// via the max_resource_count parameter to pending_op_allocate.
static inline void iree_hal_amdgpu_pending_op_retain(
    iree_hal_amdgpu_pending_op_t* op, iree_hal_resource_t* resource) {
  if (IREE_LIKELY(resource)) {
    iree_hal_resource_retain(resource);
    op->retained_resources[op->retained_resource_count++] = resource;
  }
}

// Releases all retained HAL resources in the flat array. Used on failure
// and cancellation paths. On success, retained_resources are transferred
// to the reclaim ring instead (no release here).
static void iree_hal_amdgpu_pending_op_release_retained(
    iree_hal_amdgpu_pending_op_t* op) {
  for (uint16_t i = 0; i < op->retained_resource_count; ++i) {
    iree_hal_resource_release(op->retained_resources[i]);
  }
  op->retained_resource_count = 0;
}

// Destroys a pending operation that failed during capture (arena allocation
// error after pending_op_allocate but before enqueue_waits). Caller MUST hold
// submission_mutex — the op is linked to the pending list by allocate and
// needs the mutex for unlinking.
//
// Unlike pending_op_fail (which acquires submission_mutex internally), this
// function assumes the caller already holds it. This is necessary because the
// capture phase runs under the mutex (Phase 1 of the two-phase protocol).
static void iree_hal_amdgpu_pending_op_destroy_under_lock(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  // Fail signal semaphores so downstream waiters get the error.
  iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  // Unlink from the pending list (caller holds submission_mutex).
  iree_hal_amdgpu_pending_op_unlink(op);
  // Return arena blocks to the pool.
  iree_arena_deinitialize(&op->arena);
}

// Timepoint callback fired when a wait semaphore reaches its target value
// (or fails). Each pending operation has one wait_entry per unsatisfied wait;
// each entry's callback atomically decrements wait_count. The last callback
// to fire (wait_count reaches zero) issues the operation or fails it.
static void iree_hal_amdgpu_wait_entry_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  iree_hal_amdgpu_pending_op_t* op = entry->operation;

  // Record the first failure via CAS.
  if (!iree_status_is_ok(status)) {
    intptr_t expected = 0;
    if (!iree_atomic_compare_exchange_strong(
            &op->error_status, &expected, (intptr_t)status,
            iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
      iree_status_ignore(status);
    }
  }

  // Decrement wait count. The last callback to fire triggers issue or fail.
  int32_t previous_count =
      iree_atomic_fetch_sub(&op->wait_count, 1, iree_memory_order_acq_rel);
  if (previous_count == 1) {
    int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
    if (iree_atomic_compare_exchange_strong(
            &op->lifecycle_state, &expected_state,
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      // Publish callback completion before destroying the op arena.
      iree_atomic_store(&entry->callback_complete, 1,
                        iree_memory_order_release);
      iree_status_t error = (iree_status_t)iree_atomic_exchange(
          &op->error_status, 0, iree_memory_order_acquire);
      if (!iree_status_is_ok(error)) {
        iree_status_ignore(iree_hal_amdgpu_pending_op_fail(op, error));
      } else {
        iree_hal_amdgpu_pending_op_issue(op);
      }
      return;
    }
  }

  iree_atomic_store(&entry->callback_complete, 1, iree_memory_order_release);
}

// Registers timepoints for all waits in the operation's wait semaphore list.
// Sets wait_count and registers one timepoint per wait — callbacks may fire
// synchronously during registration. When all waits are satisfied (the last
// callback fires), the operation is issued or failed.
//
// The wait_semaphore_list on the op (cloned into the arena by allocate)
// retains all semaphores for the lifetime of the op. Wait entries do not
// independently retain semaphores.
static iree_status_t iree_hal_amdgpu_pending_op_enqueue_waits(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_semaphore_list_t wait_semaphores = op->wait_semaphore_list;
  if (wait_semaphores.count == 0) {
    iree_hal_amdgpu_pending_op_issue(op);
    return iree_ok_status();
  }

  iree_host_size_t wait_entry_bytes = 0;
  if (!iree_host_size_checked_mul(wait_semaphores.count,
                                  sizeof(*op->wait_entries),
                                  &wait_entry_bytes)) {
    return iree_hal_amdgpu_pending_op_fail(
        op, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "pending op wait entry allocation overflow"));
  }
  iree_status_t status = iree_arena_allocate(&op->arena, wait_entry_bytes,
                                             (void**)&op->wait_entries);
  if (!iree_status_is_ok(status)) {
    return iree_hal_amdgpu_pending_op_fail(op, status);
  }
  memset(op->wait_entries, 0, wait_entry_bytes);

  // Set wait_count before registering any timepoints. A timepoint callback
  // may fire synchronously during acquire_timepoint.
  iree_atomic_store(&op->wait_count, (int32_t)wait_semaphores.count,
                    iree_memory_order_release);

  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
    entry->operation = op;
    iree_atomic_store(&entry->callback_complete, 0, iree_memory_order_relaxed);
    entry->timepoint.callback = iree_hal_amdgpu_wait_entry_resolved;
    entry->timepoint.user_data = entry;
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)wait_semaphores.semaphores[i],
        wait_semaphores.payload_values[i], &entry->timepoint);

    if (!iree_status_is_ok(status)) {
      // Registration failed at index i. Timepoints 0..i-1 are already
      // registered and their callbacks will fire asynchronously — we cannot
      // destroy the op here. Record the error and subtract the unregistered
      // count so the existing callbacks drain and destroy the op.
      intptr_t expected = 0;
      if (!iree_atomic_compare_exchange_strong(
              &op->error_status, &expected, (intptr_t)status,
              iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
        iree_status_ignore(status);
      }
      int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      int32_t previous_count = iree_atomic_fetch_sub(
          &op->wait_count, unregistered, iree_memory_order_acq_rel);
      if (previous_count == unregistered) {
        iree_status_t error = (iree_status_t)iree_atomic_exchange(
            &op->error_status, 0, iree_memory_order_acquire);
        return iree_hal_amdgpu_pending_op_fail(op, error);
      }
      return iree_ok_status();
    }
  }

  return iree_ok_status();
}

// Allocates and initializes a pending operation from a fresh arena.
// Clones the wait semaphore list. Allocates the retained_resources array
// with |max_resource_count| capacity and populates the first entries with
// signal semaphores (retained). The signal_semaphore_list.semaphores pointer
// aliases into retained_resources so that commit_signals and
// semaphore_list_fail can use it directly.
//
// The caller must push operation-specific resources into retained_resources
// (via the returned op) before calling enqueue_waits.
//
// On failure, the arena is cleaned up and *out_op is set to NULL.
static iree_status_t iree_hal_amdgpu_pending_op_allocate(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_amdgpu_pending_op_type_t type, uint16_t max_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;

  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->block_pool, &arena);

  iree_hal_amdgpu_pending_op_t* op = NULL;
  iree_status_t status = iree_arena_allocate(&arena, sizeof(*op), (void**)&op);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }

  memset(op, 0, sizeof(*op));
  memcpy(&op->arena, &arena, sizeof(arena));
  op->queue = queue;
  op->type = type;
  iree_atomic_store(&op->wait_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
                    iree_memory_order_relaxed);
  iree_atomic_store(&op->error_status, 0, iree_memory_order_relaxed);

  iree_allocator_t arena_allocator = iree_arena_allocator(&op->arena);

  // Clone the wait semaphore list (retains each wait semaphore).
  status = iree_hal_semaphore_list_clone(wait_semaphore_list, arena_allocator,
                                         &op->wait_semaphore_list);

  // Allocate the retained resources array and the signal payload values.
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate(
        &op->arena, max_resource_count * sizeof(iree_hal_resource_t*),
        (void**)&op->retained_resources);
  }
  uint64_t* signal_payload_values = NULL;
  if (iree_status_is_ok(status) && signal_semaphore_list->count > 0) {
    status = iree_arena_allocate(
        &op->arena, signal_semaphore_list->count * sizeof(uint64_t),
        (void**)&signal_payload_values);
  }

  if (iree_status_is_ok(status)) {
    // Signal semaphores occupy the first entries of retained_resources.
    // The signal_semaphore_list.semaphores pointer aliases this region.
    for (iree_host_size_t i = 0; i < signal_semaphore_list->count; ++i) {
      op->retained_resources[i] =
          (iree_hal_resource_t*)signal_semaphore_list->semaphores[i];
      iree_hal_resource_retain(op->retained_resources[i]);
      signal_payload_values[i] = signal_semaphore_list->payload_values[i];
    }
    op->retained_resource_count = (uint16_t)signal_semaphore_list->count;
    op->signal_semaphore_list.count = signal_semaphore_list->count;
    op->signal_semaphore_list.semaphores =
        (iree_hal_semaphore_t**)op->retained_resources;
    op->signal_semaphore_list.payload_values = signal_payload_values;

    iree_hal_amdgpu_pending_op_link(op);
    *out_op = op;
  } else {
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_arena_deinitialize(&op->arena);
  }
  return status;
}

// Issues a deferred operation after all waits are satisfied. All waits are
// tier 0 (timeline_value >= waited_value) — the GPU work producing those
// values has completed. No barriers are needed.
//
// Called from the last wait_entry callback (any thread). Acquires
// submission_mutex to emit AQL packets and commit signals.
static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_slim_mutex_lock(&queue->submission_mutex);

  // All waits are tier 0 — emit operation packets with no barriers.
  // TODO: switch on op->type and emit AQL packets from captured data.
  iree_status_t status =
      queue->is_shutting_down
          ? iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down")
          : iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "pending op issue not yet wired up");

  if (iree_status_is_ok(status)) {
    // Transfer retained resources to the reclaim ring. No new retains —
    // the pending_op's retains become the reclaim ring's retains.
    iree_hal_amdgpu_reclaim_entry_t* reclaim =
        iree_hal_amdgpu_notification_ring_reclaim_entry(
            &queue->notification_ring);
    iree_hal_resource_t** resources = NULL;
    status = iree_hal_amdgpu_reclaim_entry_prepare(
        reclaim, queue->block_pool, op->retained_resource_count, &resources);
    if (iree_status_is_ok(status)) {
      memcpy(resources, op->retained_resources,
             op->retained_resource_count * sizeof(iree_hal_resource_t*));
      reclaim->count = op->retained_resource_count;
      op->retained_resource_count = 0;

      // TODO: commit_signals + doorbell here.
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
    iree_hal_amdgpu_pending_op_release_retained(op);
  }

  // Clean up the pending op. Wait semaphore list is released (the clone
  // holds separate retains). Signal semaphore list is NOT released — the
  // semaphore pointers are in retained_resources (either transferred to
  // reclaim or released above).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_arena_deinitialize(&op->arena);

  iree_slim_mutex_unlock(&queue->submission_mutex);
}

// Fails a deferred operation. Propagates the error to all signal semaphores
// so downstream waiters receive the failure instead of hanging. Takes
// ownership of |status|. Returns iree_ok_status() so callers can tail-call:
//   return iree_hal_amdgpu_pending_op_fail(op, status);
static iree_status_t iree_hal_amdgpu_pending_op_fail(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  // Fail signal semaphores (records error, does not release our retains).
  iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  iree_arena_deinitialize(&op->arena);
  return iree_ok_status();
}

// Cancels all pending operations on a queue with the given status.
// Takes ownership of |status| (clones for each op, frees original).
// Called during deinitialize or on unrecoverable GPU fault.
// Caller must ensure no concurrent submissions (shutdown path).
static void iree_hal_amdgpu_host_queue_cancel_pending(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_t status) {
  iree_slim_mutex_lock(&queue->submission_mutex);
  queue->is_shutting_down = true;
  iree_slim_mutex_unlock(&queue->submission_mutex);

  for (;;) {
    iree_hal_amdgpu_pending_op_t* op = NULL;
    iree_slim_mutex_lock(&queue->submission_mutex);
    for (iree_hal_amdgpu_pending_op_t* candidate = queue->pending_head;
         candidate != NULL; candidate = candidate->next) {
      int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
      if (iree_atomic_compare_exchange_strong(
              &candidate->lifecycle_state, &expected_state,
              IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING,
              iree_memory_order_acq_rel, iree_memory_order_acquire)) {
        iree_hal_amdgpu_pending_op_unlink(candidate);
        op = candidate;
        break;
      }
    }
    bool has_pending_ops = queue->pending_head != NULL;
    iree_slim_mutex_unlock(&queue->submission_mutex);

    if (op == NULL) {
      if (!has_pending_ops) break;
      continue;
    }

    for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
      iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
      if (iree_async_semaphore_cancel_timepoint(entry->timepoint.semaphore,
                                                &entry->timepoint)) {
        continue;
      }
      while (iree_atomic_load(&entry->callback_complete,
                              iree_memory_order_acquire) == 0) {
      }
    }

    iree_status_t op_status = (iree_status_t)iree_atomic_exchange(
        &op->error_status, 0, iree_memory_order_acquire);
    if (iree_status_is_ok(op_status)) {
      op_status = iree_status_clone(status);
    }
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, op_status);
    iree_hal_amdgpu_pending_op_release_retained(op);
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_arena_deinitialize(&op->arena);
  }
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Initialization / deinitialization
//===----------------------------------------------------------------------===//

// Proactor progress callback. Drains completed notification entries and
// reclaims kernarg space on each proactor iteration. If the GPU queue has
// faulted (error_status is set), fails all pending entries instead of
// draining normally.
static iree_host_size_t iree_hal_amdgpu_host_queue_progress_fn(
    void* user_data) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;

  // Check for GPU queue error (set by the HSA error callback on another
  // thread). If the queue has faulted, no further epochs will advance —
  // fail all pending entries so waiters get the actual GPU error instead
  // of hanging or timing out.
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &queue->error_status, iree_memory_order_acquire);
  uint64_t kernarg_reclaim_position = 0;
  iree_host_size_t count = 0;
  if (IREE_UNLIKELY(error)) {
    count = iree_hal_amdgpu_notification_ring_fail_all(
        &queue->notification_ring, error, &kernarg_reclaim_position);
  } else {
    count = iree_hal_amdgpu_notification_ring_drain(&queue->notification_ring,
                                                    /*fallback_frontier=*/NULL,
                                                    &kernarg_reclaim_position);
  }
  if (kernarg_reclaim_position > 0) {
    iree_hal_amdgpu_kernarg_ring_reclaim(&queue->kernarg_ring,
                                         kernarg_reclaim_position);
  }
  return count;
}

// HSA queue error callback. Called by the HSA runtime (on an internal thread)
// when the queue encounters an unrecoverable error (page fault, invalid AQL
// packet, ECC error). Stores the error atomically on the queue so the proactor
// progress callback can fail pending semaphores with the actual GPU error.
static void iree_hal_amdgpu_host_queue_error_callback(hsa_status_t status,
                                                      hsa_queue_t* source,
                                                      void* data) {
  iree_hal_amdgpu_host_queue_t* queue = (iree_hal_amdgpu_host_queue_t*)data;

  // Convert the HSA error to an IREE status with diagnostic information.
  iree_status_t error = iree_status_from_hsa_status(
      __FILE__, __LINE__, status, "hsa_queue_error_callback",
      "GPU queue encountered an unrecoverable error");

  // First-error-wins: store the error with release semantics so the status
  // payload (heap-allocated string, backtrace) is visible to any thread that
  // loads with acquire. If another error already won the race, free ours.
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &queue->error_status, &expected, (intptr_t)error,
          iree_memory_order_release, iree_memory_order_acquire)) {
    iree_status_free(error);
    return;
  }

  // Wake the proactor so it picks up the error promptly. The proactor already
  // busy-polls when progress callbacks are registered, but it may be inside a
  // blocking wait at the exact moment of the fault.
  iree_async_proactor_wake(queue->proactor);
}

iree_status_t iree_hal_amdgpu_host_queue_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_async_proactor_t* proactor,
    hsa_agent_t gpu_agent, hsa_amd_memory_pool_t kernarg_pool,
    const iree_hal_amdgpu_topology_t* topology, iree_async_axis_t axis,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_table,
    iree_arena_block_pool_t* block_pool, iree_host_size_t device_ordinal,
    uint32_t aql_queue_capacity, uint32_t notification_capacity,
    uint32_t kernarg_capacity_in_blocks, iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(epoch_table);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_queue);

  if (!iree_host_size_is_power_of_two(aql_queue_capacity) ||
      !iree_host_size_is_power_of_two(notification_capacity) ||
      !iree_host_size_is_power_of_two(kernarg_capacity_in_blocks)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "all capacities must be powers of two");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_queue, 0, sizeof(*out_queue));
  out_queue->base.vtable = &iree_hal_amdgpu_host_queue_vtable;
  out_queue->libhsa = libhsa;
  out_queue->proactor = proactor;
  out_queue->host_allocator = host_allocator;

  // Submission pipeline state.
  iree_slim_mutex_initialize(&out_queue->submission_mutex);
  out_queue->axis = axis;
  out_queue->last_signal.semaphore = NULL;
  out_queue->last_signal.epoch = 0;
  out_queue->block_pool = block_pool;
  out_queue->can_publish_frontier = true;
  out_queue->device_ordinal = device_ordinal;
  out_queue->pending_head = NULL;
  iree_async_frontier_initialize(iree_hal_amdgpu_host_queue_frontier(out_queue),
                                 /*entry_count=*/0);

  // Create the HSA hardware AQL queue.
  hsa_queue_t* hardware_queue = NULL;
  iree_status_t status = iree_hsa_queue_create(
      IREE_LIBHSA(libhsa), gpu_agent, aql_queue_capacity, HSA_QUEUE_TYPE_SINGLE,
      iree_hal_amdgpu_host_queue_error_callback,
      /*data=*/out_queue,
      /*private_segment_size=*/UINT32_MAX,
      /*group_segment_size=*/UINT32_MAX, &hardware_queue);

  // Initialize the AQL ring from the hardware queue.
  if (iree_status_is_ok(status)) {
    out_queue->hardware_queue = hardware_queue;
    iree_hal_amdgpu_aql_ring_initialize((iree_amd_queue_t*)hardware_queue,
                                        &out_queue->aql_ring);
  }

  // Initialize the kernarg ring from the HSA kernarg memory pool.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_kernarg_ring_initialize(
        libhsa, gpu_agent, kernarg_pool, kernarg_capacity_in_blocks, topology,
        &out_queue->kernarg_ring);
  }

  // Initialize the notification ring (creates epoch signal + entry buffer).
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_notification_ring_initialize(
        libhsa, block_pool, notification_capacity, host_allocator,
        &out_queue->notification_ring);
  }

  // Register this queue's epoch signal in the shared table for cross-queue
  // barrier emission lookups. Must happen after notification ring init (which
  // creates the epoch signal) and before any submissions.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_epoch_signal_table_register(
        epoch_table, iree_async_axis_device_index(axis),
        iree_async_axis_queue_index(axis),
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &out_queue->notification_ring));
    out_queue->epoch_table = epoch_table;
  }

  if (iree_status_is_ok(status)) {
    memset(&out_queue->progress_entry, 0, sizeof(out_queue->progress_entry));
    out_queue->progress_entry.fn = iree_hal_amdgpu_host_queue_progress_fn;
    out_queue->progress_entry.user_data = out_queue;
    iree_async_proactor_register_progress(proactor, &out_queue->progress_entry);
  } else {
    iree_hal_amdgpu_host_queue_deinitialize(out_queue);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_unregister_progress(queue->proactor,
                                          &queue->progress_entry);

  // Cancel all pending (deferred) operations. Their signal semaphores are
  // failed with CANCELLED so downstream waiters don't hang.
  if (queue->pending_head) {
    iree_hal_amdgpu_host_queue_cancel_pending(
        queue, iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down"));
  }

  // Process any remaining notification entries before destroying resources.
  // If the GPU faulted, fail all pending entries so waiters get the actual
  // error. Otherwise drain normally (entries completed but not yet processed).
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &queue->error_status, iree_memory_order_acquire);
  uint64_t kernarg_reclaim_position = 0;
  if (!iree_status_is_ok(error)) {
    iree_hal_amdgpu_notification_ring_fail_all(&queue->notification_ring, error,
                                               &kernarg_reclaim_position);
    iree_status_free(error);
  } else {
    iree_hal_amdgpu_notification_ring_drain(&queue->notification_ring,
                                            /*fallback_frontier=*/NULL,
                                            &kernarg_reclaim_position);
  }
  if (kernarg_reclaim_position > 0) {
    iree_hal_amdgpu_kernarg_ring_reclaim(&queue->kernarg_ring,
                                         kernarg_reclaim_position);
  }

  // Deregister from the epoch signal table before destroying the notification
  // ring (which owns the epoch signal). Guarded by epoch_table != NULL to
  // handle partial initialization (init failed before registration).
  if (queue->epoch_table) {
    iree_hal_amdgpu_epoch_signal_table_deregister(
        queue->epoch_table, iree_async_axis_device_index(queue->axis),
        iree_async_axis_queue_index(queue->axis));
    queue->epoch_table = NULL;
  }

  iree_hal_amdgpu_notification_ring_deinitialize(&queue->notification_ring);

  iree_hal_amdgpu_kernarg_ring_deinitialize(queue->libhsa,
                                            &queue->kernarg_ring);

  if (queue->hardware_queue) {
    IREE_IGNORE_ERROR(iree_hsa_queue_destroy(IREE_LIBHSA(queue->libhsa),
                                             queue->hardware_queue));
    queue->hardware_queue = NULL;
  }

  iree_slim_mutex_deinitialize(&queue->submission_mutex);

  IREE_TRACE_ZONE_END(z0);
}

// Commits the signal side of an AQL submission. Called after AQL packets are
// written but before the doorbell is rung. Caller must hold submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_commit_signals(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    uint64_t kernarg_write_position) {
  IREE_ASSERT(signal_semaphore_list.count > 0,
              "empty signal list — operation should have been elided");

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_notification_ring_reserve(
      &queue->notification_ring, signal_semaphore_list.count,
      iree_hal_amdgpu_host_queue_count_frontier_snapshots(
          queue, signal_semaphore_list)));

  // Advance epoch and merge this queue's axis into the accumulated frontier.
  uint64_t epoch = iree_hal_amdgpu_notification_ring_advance_epoch(
      &queue->notification_ring);
  iree_async_single_frontier_t self_frontier;
  iree_async_single_frontier_initialize(&self_frontier, queue->axis, epoch);
  iree_async_frontier_merge(
      iree_hal_amdgpu_host_queue_frontier(queue),
      IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY,
      iree_async_single_frontier_as_const_frontier(&self_frontier));

  const iree_async_frontier_t* queue_frontier =
      iree_hal_amdgpu_host_queue_const_frontier(queue);

  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_hal_semaphore_t* hal_semaphore = signal_semaphore_list.semaphores[i];
    uint64_t value = signal_semaphore_list.payload_values[i];
    iree_async_semaphore_t* async_semaphore =
        (iree_async_semaphore_t*)hal_semaphore;
    bool is_amdgpu_semaphore = iree_hal_amdgpu_semaphore_isa(hal_semaphore);

    // Detect semaphore transition for frontier snapshot recording.
    if (async_semaphore != queue->last_signal.semaphore) {
      if (queue->last_signal.semaphore != NULL) {
        iree_hal_amdgpu_notification_ring_push_frontier_snapshot(
            &queue->notification_ring, queue->last_signal.epoch,
            queue_frontier);
      }
      queue->last_signal.semaphore = async_semaphore;
    }

    // Push notification entry for drain → signal_untainted on completion.
    iree_hal_amdgpu_notification_ring_push(&queue->notification_ring, epoch,
                                           async_semaphore, value,
                                           kernarg_write_position);

    // Submission-time causal marker: merge queue's frontier into the
    // semaphore's frontier so same-queue FIFO wait elision works before
    // GPU completion.
    bool did_publish_frontier = queue->can_publish_frontier;
    if (did_publish_frontier) {
      if (is_amdgpu_semaphore) {
        did_publish_frontier = iree_hal_amdgpu_semaphore_publish_signal(
            hal_semaphore, queue->axis, queue_frontier, epoch, value);
      } else {
        did_publish_frontier = iree_async_semaphore_merge_frontier(
            async_semaphore, queue_frontier);
      }
    }
    if (!did_publish_frontier) {
      // The semaphore's frontier storage overflowed, so its frontier is no
      // longer a conservative summary of this signal's causal dependencies.
      // Clear the last-signal cache to force future waits down the software
      // deferral path instead of unsafely eliding or under-barriering them.
      if (is_amdgpu_semaphore) {
        iree_hal_amdgpu_semaphore_clear_last_signal(hal_semaphore);
      }
      continue;
    }
  }

  queue->last_signal.epoch = epoch;
  return iree_ok_status();
}

static void iree_hal_amdgpu_host_queue_trim(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {}

//===----------------------------------------------------------------------===//
// Queue operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_host_queue_alloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue alloca not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_dealloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue dealloca not yet implemented");
}

// Captures a fill operation into a pending op. Does NOT call enqueue_waits —
// the caller must release submission_mutex before calling enqueue_waits on the
// returned op (Phase 2 of the two-phase protocol).
// Caller must hold submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = (uint16_t)signal_semaphore_list->count + 1;
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_FILL, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);
  op->fill.target_buffer = target_buffer;
  op->fill.target_offset = target_offset;
  op->fill.length = length;
  op->fill.pattern_bits = pattern_bits;
  op->fill.pattern_length = pattern_length;
  op->fill.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_fill(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  // Phase 1 (under mutex): resolve waits and capture if deferred.
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_fill(
        queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
        target_offset, length, pattern_bits, pattern_length, flags,
        &deferred_op);
  } else {
    status =
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "inline fill AQL emission");
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  // Phase 2 (without mutex): register timepoints. Callbacks may fire
  // synchronously and re-acquire submission_mutex via pending_op_issue/fail.
  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

// Captures a copy operation into a pending op.
// Caller must hold submission_mutex. Caller must call enqueue_waits after
// releasing submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = (uint16_t)signal_semaphore_list->count + 2;
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_COPY, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)source_buffer);
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);
  op->copy.source_buffer = source_buffer;
  op->copy.source_offset = source_offset;
  op->copy.target_buffer = target_buffer;
  op->copy.target_offset = target_offset;
  op->copy.length = length;
  op->copy.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_copy(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_copy(
        queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        &deferred_op);
  } else {
    status =
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "inline copy AQL emission");
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

// Captures an update operation into a pending op. Copies the source host data
// into the arena (the caller's buffer may be freed after this returns).
// Caller must hold submission_mutex. Caller must call enqueue_waits after
// releasing submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = (uint16_t)signal_semaphore_list->count + 1;
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_UPDATE, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);

  // Copy the source host data into the arena. The caller's buffer may be
  // freed after this call returns.
  void* source_copy = NULL;
  iree_status_t status =
      iree_arena_allocate(&op->arena, (iree_host_size_t)length, &source_copy);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op, status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "arena allocation failed during defer_update");
  }
  memcpy(source_copy, (const uint8_t*)source_buffer + source_offset, length);
  op->update.source_data = source_copy;
  op->update.source_offset = 0;
  op->update.target_buffer = target_buffer;
  op->update.target_offset = target_offset;
  op->update.length = length;
  op->update.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_update(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_update(
        queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        &deferred_op);
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "inline update AQL emission");
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

// Captures a dispatch operation into a pending op. Copies constants and
// bindings into the arena.
// Caller must hold submission_mutex. Caller must call enqueue_waits after
// releasing submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  // 1 executable + up to bindings.count buffers.
  uint16_t max_resources =
      (uint16_t)signal_semaphore_list->count + 1 + (uint16_t)bindings.count;
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_DISPATCH, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)executable);
  op->dispatch.executable = executable;
  op->dispatch.export_ordinal = export_ordinal;
  op->dispatch.config = config;
  op->dispatch.flags = flags;

  // Copy constants into the arena.
  iree_status_t status = iree_ok_status();
  if (constants.data_length > 0) {
    void* constants_copy = NULL;
    status =
        iree_arena_allocate(&op->arena, constants.data_length, &constants_copy);
    if (iree_status_is_ok(status)) {
      memcpy(constants_copy, constants.data, constants.data_length);
      op->dispatch.constants.data = (const uint8_t*)constants_copy;
      op->dispatch.constants.data_length = constants.data_length;
    }
  }

  // Copy bindings array and retain all bound buffers.
  if (iree_status_is_ok(status) && bindings.count > 0) {
    iree_hal_buffer_ref_t* bindings_copy = NULL;
    status = iree_arena_allocate(&op->arena,
                                 bindings.count * sizeof(iree_hal_buffer_ref_t),
                                 (void**)&bindings_copy);
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, bindings.values,
             bindings.count * sizeof(iree_hal_buffer_ref_t));
      for (iree_host_size_t i = 0; i < bindings.count; ++i) {
        iree_hal_amdgpu_pending_op_retain(
            op, (iree_hal_resource_t*)bindings_copy[i].buffer);
      }
      op->dispatch.bindings.count = bindings.count;
      op->dispatch.bindings.values = bindings_copy;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op, status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "arena allocation failed during defer_dispatch");
  }
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_dispatch(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_dispatch(
        queue, &wait_semaphore_list, &signal_semaphore_list, executable,
        export_ordinal, config, constants, bindings, flags, &deferred_op);
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "inline dispatch AQL emission");
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

// Captures an execute operation into a pending op. Copies binding table into
// the arena.
// Caller must hold submission_mutex. Caller must call enqueue_waits after
// releasing submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_execute(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags, iree_hal_amdgpu_pending_op_t** out_op) {
  // 1 command buffer + up to binding_table.count buffers.
  uint16_t max_resources = (uint16_t)signal_semaphore_list->count + 1 +
                           (uint16_t)binding_table.count;
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_EXECUTE, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)command_buffer);
  op->execute.command_buffer = command_buffer;
  op->execute.flags = flags;

  // Copy binding table and retain all bound buffers.
  iree_status_t status = iree_ok_status();
  if (binding_table.count > 0) {
    iree_hal_buffer_binding_t* bindings_copy = NULL;
    status = iree_arena_allocate(
        &op->arena, binding_table.count * sizeof(iree_hal_buffer_binding_t),
        (void**)&bindings_copy);
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, binding_table.bindings,
             binding_table.count * sizeof(iree_hal_buffer_binding_t));
      for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
        iree_hal_amdgpu_pending_op_retain(
            op, (iree_hal_resource_t*)bindings_copy[i].buffer);
      }
      op->execute.binding_table.count = binding_table.count;
      op->execute.binding_table.bindings = bindings_copy;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op, status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "arena allocation failed during defer_execute");
  }
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_execute(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_execute(
        queue, &wait_semaphore_list, &signal_semaphore_list, command_buffer,
        binding_table, flags, &deferred_op);
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "inline execute AQL emission");
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_read(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue read not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_write(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue write not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_host_call(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue host_call not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_flush(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Virtual queue vtable
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_host_queue_deinitialize_vtable(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {
  iree_hal_amdgpu_host_queue_deinitialize(
      (iree_hal_amdgpu_host_queue_t*)base_queue);
}

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable = {
        .deinitialize = iree_hal_amdgpu_host_queue_deinitialize_vtable,
        .trim = iree_hal_amdgpu_host_queue_trim,
        .alloca = iree_hal_amdgpu_host_queue_alloca,
        .dealloca = iree_hal_amdgpu_host_queue_dealloca,
        .fill = iree_hal_amdgpu_host_queue_fill,
        .update = iree_hal_amdgpu_host_queue_update,
        .copy = iree_hal_amdgpu_host_queue_copy,
        .read = iree_hal_amdgpu_host_queue_read,
        .write = iree_hal_amdgpu_host_queue_write,
        .host_call = iree_hal_amdgpu_host_queue_host_call,
        .dispatch = iree_hal_amdgpu_host_queue_dispatch,
        .execute = iree_hal_amdgpu_host_queue_execute,
        .flush = iree_hal_amdgpu_host_queue_flush,
};
