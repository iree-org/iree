// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue.h"

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amdgpu/host_queue_blit.h"
#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"
#include "iree/hal/drivers/amdgpu/host_queue_file.h"
#include "iree/hal/drivers/amdgpu/host_queue_host_call.h"
#include "iree/hal/drivers/amdgpu/host_queue_memory.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"
#include "iree/hal/drivers/amdgpu/host_queue_waits.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

// The inline frontier on host_queue_t uses the same {entry_count, reserved[7],
// entries[N]} layout as iree_async_frontier_t + FAM. Verify the entries field
// starts at the FAM offset so iree_hal_amdgpu_host_queue_frontier() is valid.
static_assert(offsetof(iree_hal_amdgpu_host_queue_t, frontier.entries) -
                      offsetof(iree_hal_amdgpu_host_queue_t, frontier) ==
                  sizeof(iree_async_frontier_t),
              "inline frontier entries must align with frontier_t FAM offset");

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable;

static iree_status_t iree_hal_amdgpu_host_queue_allocate_pm4_ib_slots(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t gpu_agent,
    hsa_amd_memory_pool_t pm4_ib_pool, uint32_t aql_queue_capacity,
    iree_hal_amdgpu_host_queue_t* out_queue) {
  iree_host_size_t pm4_ib_size = 0;
  if (!iree_host_size_checked_mul(aql_queue_capacity,
                                  sizeof(iree_hal_amdgpu_pm4_ib_slot_t),
                                  &pm4_ib_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 IB slot buffer size overflow");
  }
  if (IREE_UNLIKELY(!pm4_ib_pool.handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 IB memory pool is required");
  }
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(libhsa), pm4_ib_pool, pm4_ib_size,
      HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, (void**)&pm4_ib_slots));
  iree_status_t status = iree_hsa_amd_agents_allow_access(
      IREE_LIBHSA(libhsa), /*num_agents=*/1, &gpu_agent, /*flags=*/NULL,
      pm4_ib_slots);
  if (iree_status_is_ok(status)) {
    memset(pm4_ib_slots, 0, pm4_ib_size);
    out_queue->pm4_ib_slots = pm4_ib_slots;
  } else {
    IREE_IGNORE_ERROR(
        iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), pm4_ib_slots));
  }
  return status;
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
// The completion thread (drain, error check) does NOT acquire
// submission_mutex. It reads the notification ring (SPSC consumer) and the
// atomic error_status.
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
  IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION,
} iree_hal_amdgpu_pending_op_type_t;

// Completion ownership for a deferred operation.
typedef enum iree_hal_amdgpu_pending_op_lifecycle_e {
  // Waiting callbacks may still resolve the op.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING = 0,
  // Queue shutdown claimed cancellation ownership.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING = 1,
  // The last wait callback claimed completion ownership.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING = 2,
  // The issuing thread is registering a cold alloca memory-readiness wait.
  // Cancellation only claims PENDING ops; the arming thread publishes PENDING
  // after registration or observes a synchronous callback as COMPLETING.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT = 3,
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

typedef enum iree_hal_amdgpu_alloca_memory_wait_kind_e {
  // No active memory wait or held reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE = 0,
  // Waiting for a copied pool death frontier while holding a reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER = 1,
  // Waiting for a pool release notification before retrying reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION = 2,
} iree_hal_amdgpu_alloca_memory_wait_kind_t;

// Cold-path alloca memory-readiness wait. Allocated inside a pending op's
// arena only after user semaphore waits have resolved and the pool cannot
// produce immediately-usable bytes.
typedef struct iree_hal_amdgpu_alloca_memory_wait_t {
  // Active wait source.
  iree_hal_amdgpu_alloca_memory_wait_kind_t kind;

  // Set to 1 after the callback's final access to this wait/op completes.
  iree_atomic_int32_t callback_complete;

  // State for a held reservation blocked on a pool death frontier.
  struct {
    // Queue-owned reservation held while waiting for its death frontier.
    iree_hal_pool_reservation_t reservation;

    // Arena-owned copy of the pool-owned death frontier.
    iree_async_frontier_t* wait_frontier;

    // Tracker waiter storage for |wait_frontier|.
    iree_async_frontier_waiter_t waiter;
  } frontier;

  // State for reservation retry after pool release notifications.
  struct {
    // Borrowed notification returned by the pool.
    iree_async_notification_t* notification;

    // Notification epoch observed before the reservation retry.
    uint32_t wait_token;

    // Wait operations rotated so a callback can arm a retry before returning.
    iree_async_notification_wait_operation_t wait_ops[2];

    // Index of the active wait operation in |wait_ops|.
    uint8_t wait_slot;
  } pool_notification;
} iree_hal_amdgpu_alloca_memory_wait_t;

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

  // Wakes cancellation when a detached wait callback finishes touching the op.
  iree_notification_t callback_notification;

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

  // Number of entries currently owned in |retained_resources|.
  uint16_t retained_resource_count;

  // Operation payload selector.
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
      // Borrowed pool resolved during queue_alloca capture. The pool owner
      // must outlive all queued transient allocations.
      iree_hal_pool_t* pool;

      // Buffer parameters captured from queue_alloca.
      iree_hal_buffer_params_t params;

      // Requested allocation size in bytes.
      iree_device_size_t allocation_size;

      // HAL allocation flags captured from queue_alloca.
      iree_hal_alloca_flags_t flags;

      // Pool reservation flags used when probing the selected pool.
      iree_hal_pool_reserve_flags_t reserve_flags;

      // Transient buffer returned to the caller and committed on success.
      iree_hal_buffer_t* buffer;

      // Cold memory-readiness sidecar allocated only after user waits resolve.
      iree_hal_amdgpu_alloca_memory_wait_t* memory_wait;
    } alloca_op;
    struct {
      iree_hal_buffer_t* buffer;
    } dealloca;
    struct {
      iree_hal_host_call_t call;
      uint64_t args[4];
      iree_hal_host_call_flags_t flags;
    } host_call;
    struct {
      iree_hal_amdgpu_reclaim_action_t action;
    } host_action;
  };
};

static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op);
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status);
static iree_status_t iree_hal_amdgpu_host_queue_submit_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op);

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

static void iree_hal_amdgpu_pending_op_fail_host_action(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION ||
      !op->host_action.action.fn) {
    return;
  }
  op->host_action.action.fn(/*entry=*/NULL, op->host_action.action.user_data,
                            status);
  op->host_action.action.fn = NULL;
  op->host_action.action.user_data = NULL;
}

// Releases any queue-owned alloca memory-readiness reservation. This runs only
// on failure/cancellation paths or after ownership has not transferred into the
// transient buffer.
static void iree_hal_amdgpu_pending_op_release_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) return;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) return;

  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_pool_release_reservation(op->alloca_op.pool,
                                        &wait->frontier.reservation,
                                        wait->frontier.wait_frontier);
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
}

// Clears the queued marker for a deferred dealloca that never published a
// completion epoch. Successful deallocas transfer ownership to the reclaim ring
// and must not call this.
static void iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA) return;
  iree_hal_amdgpu_transient_buffer_abort_dealloca(op->dealloca.buffer);
}

static bool iree_hal_amdgpu_alloca_memory_wait_callback_is_complete(
    void* user_data) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait =
      (iree_hal_amdgpu_alloca_memory_wait_t*)user_data;
  return iree_atomic_load(&wait->callback_complete,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_release);
  iree_notification_post(&op->callback_notification, IREE_ALL_WAITERS);
}

// Publishes a prepared memory-readiness wait as ARMING. The release store on
// lifecycle_state makes the initialized sidecar fields visible to the callback
// or cancellation path that observes the state transition.
static void iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
    iree_hal_amdgpu_pending_op_t* op,
    iree_hal_amdgpu_alloca_memory_wait_t* wait,
    iree_hal_amdgpu_alloca_memory_wait_kind_t kind) {
  wait->kind = kind;
  iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_relaxed);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT,
                    iree_memory_order_release);
}

// Cancels any active alloca memory-readiness wait before destroying the op.
static void iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) return;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) return;

  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER: {
      const bool cancelled = iree_async_frontier_tracker_cancel_wait(
          op->queue->frontier_tracker, &wait->frontier.waiter);
      if (!cancelled) {
        iree_notification_await(
            &op->callback_notification,
            iree_hal_amdgpu_alloca_memory_wait_callback_is_complete, wait,
            iree_infinite_timeout());
      }
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION: {
      // Shutdown is allowed to prod the pool notification: it is a broad wake,
      // but prevents teardown from depending on a future dealloca. The callback
      // observes the CANCELLING lifecycle state and only publishes completion.
      iree_async_notification_signal(wait->pool_notification.notification,
                                     INT32_MAX);
      iree_notification_await(
          &op->callback_notification,
          iree_hal_amdgpu_alloca_memory_wait_callback_is_complete, wait,
          iree_infinite_timeout());
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
}

static bool iree_hal_amdgpu_wait_entry_callback_is_complete(void* user_data) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  return iree_atomic_load(&entry->callback_complete,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_amdgpu_wait_entry_publish_callback_complete(
    iree_hal_amdgpu_wait_entry_t* entry) {
  iree_atomic_store(&entry->callback_complete, 1, iree_memory_order_release);
  iree_notification_post(&entry->operation->callback_notification,
                         IREE_ALL_WAITERS);
}

static bool iree_hal_amdgpu_pending_op_wait_callbacks_are_complete(
    void* user_data) {
  iree_hal_amdgpu_pending_op_t* op = (iree_hal_amdgpu_pending_op_t*)user_data;
  for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
    iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
    if (!iree_hal_amdgpu_wait_entry_callback_is_complete(entry)) {
      return false;
    }
  }
  return true;
}

// Records the first asynchronous wait failure. Takes ownership of |status|,
// storing it for the completion owner or dropping it if another failure won.
static void iree_hal_amdgpu_pending_op_record_error_status(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &op->error_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

static bool iree_hal_amdgpu_pending_op_mark_waits_resolved(
    iree_hal_amdgpu_pending_op_t* op) {
  int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
  return iree_atomic_compare_exchange_strong(
      &op->lifecycle_state, &expected_state,
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

static void iree_hal_amdgpu_pending_op_complete_resolved_waits(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_notification_await(
      &op->callback_notification,
      iree_hal_amdgpu_pending_op_wait_callbacks_are_complete, op,
      iree_infinite_timeout());
  iree_status_t error = (iree_status_t)iree_atomic_exchange(
      &op->error_status, 0, iree_memory_order_acquire);
  if (!iree_status_is_ok(error)) {
    iree_hal_amdgpu_pending_op_fail(op, error);
  } else {
    iree_hal_amdgpu_pending_op_issue(op);
  }
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
  iree_hal_amdgpu_pending_op_fail_host_action(op, status);
  // Fail signal semaphores so downstream waiters get the error.
  iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
  iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
  // Release any queue-owned memory reservation before releasing op resources.
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  // Unlink from the pending list (caller holds submission_mutex).
  iree_hal_amdgpu_pending_op_unlink(op);
  // Tear down callback wake state before returning arena blocks to the pool.
  iree_notification_deinitialize(&op->callback_notification);
  // Return arena blocks to the pool.
  iree_arena_deinitialize(&op->arena);
}

// Timepoint callback fired when a wait semaphore reaches its target value or
// fails. The last resolved wait claims completion, waits for all callbacks to
// finish touching arena-owned entries, and then issues or fails the operation.
static void iree_hal_amdgpu_wait_entry_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  iree_hal_amdgpu_pending_op_t* op = entry->operation;

  iree_hal_amdgpu_pending_op_record_error_status(op, status);

  int32_t previous_count =
      iree_atomic_fetch_sub(&op->wait_count, 1, iree_memory_order_acq_rel);
  bool owns_completion = false;
  if (previous_count == 1) {
    owns_completion = iree_hal_amdgpu_pending_op_mark_waits_resolved(op);
  }

  iree_hal_amdgpu_wait_entry_publish_callback_complete(entry);
  if (owns_completion) {
    iree_hal_amdgpu_pending_op_complete_resolved_waits(op);
  }
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
    iree_hal_amdgpu_pending_op_fail(
        op, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "pending op wait entry allocation overflow"));
    return iree_ok_status();
  }
  iree_status_t status = iree_arena_allocate(&op->arena, wait_entry_bytes,
                                             (void**)&op->wait_entries);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    return iree_ok_status();
  }
  memset(op->wait_entries, 0, wait_entry_bytes);
  // Unregistered entries never receive callbacks, so they start complete.
  // Active registrations flip their entry incomplete until the callback exits.
  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_atomic_store(&op->wait_entries[i].callback_complete, 1,
                      iree_memory_order_relaxed);
  }

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
      iree_hal_amdgpu_pending_op_record_error_status(op, status);
      int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      iree_atomic_store(&entry->callback_complete, 1,
                        iree_memory_order_release);
      int32_t previous_count = iree_atomic_fetch_sub(
          &op->wait_count, unregistered, iree_memory_order_acq_rel);
      if (previous_count == unregistered) {
        if (iree_hal_amdgpu_pending_op_mark_waits_resolved(op)) {
          iree_hal_amdgpu_pending_op_complete_resolved_waits(op);
        }
      }
      return iree_ok_status();
    }
  }

  return iree_ok_status();
}

static void iree_hal_amdgpu_alloca_memory_wait_resolved(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (iree_status_is_ok(status) &&
      wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION) {
    wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  }

  int32_t expected_state =
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_record_error_status(op, status);
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
    return;
  }

  expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdgpu_pending_op_fail(op, status);
    } else {
      iree_hal_amdgpu_pending_op_issue(op);
    }
    return;
  }

  iree_hal_amdgpu_pending_op_record_error_status(op, status);
  iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
}

static void iree_hal_amdgpu_alloca_frontier_wait_resolved(
    void* user_data, iree_status_t status) {
  iree_hal_amdgpu_alloca_memory_wait_resolved(
      (iree_hal_amdgpu_pending_op_t*)user_data, status);
}

static void iree_hal_amdgpu_alloca_pool_notification_wait_resolved(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_amdgpu_alloca_memory_wait_resolved(
      (iree_hal_amdgpu_pending_op_t*)user_data, status);
}

static void iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  int32_t expected_state =
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT;
  if (iree_status_is_ok(status)) {
    if (iree_atomic_compare_exchange_strong(
            &op->lifecycle_state, &expected_state,
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      return;
    }
    if (expected_state == IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING) {
      iree_status_t error = (iree_status_t)iree_atomic_exchange(
          &op->error_status, 0, iree_memory_order_acquire);
      if (!iree_status_is_ok(error)) {
        iree_hal_amdgpu_pending_op_fail(op, error);
      } else {
        iree_hal_amdgpu_pending_op_issue(op);
      }
    }
    return;
  }

  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    return;
  }
  iree_hal_amdgpu_pending_op_record_error_status(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_complete, 0, iree_memory_order_relaxed);
  iree_status_t status = iree_async_frontier_tracker_wait(
      queue->frontier_tracker, wait->frontier.wait_frontier,
      iree_hal_amdgpu_alloca_frontier_wait_resolved, op,
      &wait->frontier.waiter);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_pool_notification_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_async_notification_wait_operation_t* wait_op =
      &wait->pool_notification.wait_ops[wait->pool_notification.wait_slot];
  iree_async_operation_zero(&wait_op->base, sizeof(*wait_op));
  iree_async_operation_initialize(
      &wait_op->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_amdgpu_alloca_pool_notification_wait_resolved, op);
  wait_op->notification = wait->pool_notification.notification;
  wait_op->wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op->wait_token = wait->pool_notification.wait_token;

  iree_atomic_store(&wait->callback_complete, 0, iree_memory_order_relaxed);
  iree_status_t status =
      iree_async_proactor_submit_one(op->queue->proactor, &wait_op->base);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      iree_hal_amdgpu_pending_op_enqueue_alloca_pool_notification_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      iree_hal_amdgpu_pending_op_fail(
          op, iree_make_status(IREE_STATUS_INTERNAL,
                               "pending alloca has no memory wait to enqueue"));
      break;
  }
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
  iree_notification_initialize(&op->callback_notification);

  iree_allocator_t arena_allocator = iree_arena_allocator(&op->arena);

  // Clone the wait semaphore list (retains each wait semaphore).
  status = iree_hal_semaphore_list_clone(wait_semaphore_list, arena_allocator,
                                         &op->wait_semaphore_list);

  // Allocate the retained resources array and the signal payload values.
  if (iree_status_is_ok(status) && max_resource_count > 0) {
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
    iree_notification_deinitialize(&op->callback_notification);
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

  iree_status_t status = iree_ok_status();
  if (queue->is_shutting_down) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  } else {
    // All waits are tier 0 — emit operation packets with no dependency
    // barriers.
    iree_hal_amdgpu_wait_resolution_t resolution;
    resolution.barrier_count = 0;
    resolution.needs_deferral = false;
    memset(resolution.reserved, 0, sizeof(resolution.reserved));
    resolution.inline_acquire_scope = op->wait_semaphore_list.count > 0
                                          ? IREE_HSA_FENCE_SCOPE_SYSTEM
                                          : IREE_HSA_FENCE_SCOPE_NONE;
    resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
    switch (op->type) {
      case IREE_HAL_AMDGPU_PENDING_OP_FILL:
        status = iree_hal_amdgpu_host_queue_submit_fill(
            queue, &resolution, op->signal_semaphore_list,
            op->fill.target_buffer, op->fill.target_offset, op->fill.length,
            op->fill.pattern_bits, op->fill.pattern_length, op->fill.flags,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_COPY:
        status = iree_hal_amdgpu_host_queue_submit_copy(
            queue, &resolution, op->signal_semaphore_list,
            op->copy.source_buffer, op->copy.source_offset,
            op->copy.target_buffer, op->copy.target_offset, op->copy.length,
            op->copy.flags, IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_UPDATE:
        status = iree_hal_amdgpu_host_queue_submit_update(
            queue, &resolution, op->signal_semaphore_list,
            op->update.source_data, /*source_offset=*/0,
            op->update.target_buffer, op->update.target_offset,
            op->update.length, op->update.flags,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_DISPATCH:
        status = iree_hal_amdgpu_host_queue_submit_dispatch(
            queue, &resolution, op->signal_semaphore_list,
            op->dispatch.executable, op->dispatch.export_ordinal,
            op->dispatch.config, op->dispatch.constants, op->dispatch.bindings,
            op->dispatch.flags,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_EXECUTE:
        if (op->execute.command_buffer) {
          status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                    "pending command buffer execute not yet "
                                    "wired up");
          break;
        }
        status = iree_hal_amdgpu_host_queue_submit_barrier(
            queue, &resolution, op->signal_semaphore_list,
            (iree_hal_amdgpu_reclaim_action_t){0},
            /*operation_resources=*/NULL,
            /*operation_resource_count=*/0,
            /*post_commit_fn=*/NULL, /*post_commit_user_data=*/NULL,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_ALLOCA: {
        iree_hal_amdgpu_pending_op_t* memory_wait_op = NULL;
        status = iree_hal_amdgpu_host_queue_submit_alloca(
            queue, &resolution, op->signal_semaphore_list, op->alloca_op.pool,
            op->alloca_op.params, op->alloca_op.allocation_size,
            op->alloca_op.flags, op->alloca_op.reserve_flags,
            op->alloca_op.buffer,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, op,
            &memory_wait_op);
        if (iree_status_is_ok(status) && memory_wait_op) {
          iree_slim_mutex_unlock(&queue->submission_mutex);
          iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(memory_wait_op);
          return;
        }
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      }
      case IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA:
        status = iree_hal_amdgpu_host_queue_submit_dealloca(
            queue, &resolution, op->signal_semaphore_list, op->dealloca.buffer,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL:
        status = iree_hal_amdgpu_host_queue_submit_host_call(
            queue, &resolution, op->signal_semaphore_list, op->host_call.call,
            op->host_call.args, op->host_call.flags);
        if (iree_status_is_ok(status)) {
          iree_hal_amdgpu_pending_op_release_retained(op);
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION:
        status = iree_hal_amdgpu_host_queue_submit_barrier(
            queue, &resolution, iree_hal_semaphore_list_empty(),
            op->host_action.action, op->retained_resources,
            op->retained_resource_count,
            /*post_commit_fn=*/NULL, /*post_commit_user_data=*/NULL,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "pending op issue not yet wired up");
        break;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail_host_action(op, status);
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
    iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
    iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
    iree_hal_amdgpu_pending_op_release_retained(op);
  }

  // Clean up the pending op. Wait semaphore list is released (the clone
  // holds separate retains). Signal semaphore list is NOT released — the
  // semaphore pointers are in retained_resources (either transferred to
  // reclaim or released above).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_notification_deinitialize(&op->callback_notification);
  iree_arena_deinitialize(&op->arena);

  iree_slim_mutex_unlock(&queue->submission_mutex);
}

// Fails a deferred operation. Propagates the error to all signal semaphores
// so downstream waiters receive the failure instead of hanging. Takes
// ownership of |status|.
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_pending_op_fail_host_action(op, status);
  // Fail signal semaphores (records error, does not release our retains).
  iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
  iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
  // Release any queue-owned memory reservation before releasing op resources.
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  iree_notification_deinitialize(&op->callback_notification);
  iree_arena_deinitialize(&op->arena);
}

// Cancels all pending operations on a queue with the given failure details.
// Creates a status only for operations that do not already carry a wait error.
// Called during deinitialize or on unrecoverable GPU fault.
// Caller must ensure no concurrent submissions (shutdown path).
static void iree_hal_amdgpu_host_queue_cancel_pending(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_code_t status_code,
    const char* status_message) {
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
      iree_thread_yield();
      continue;
    }

    for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
      iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
      if (iree_hal_amdgpu_wait_entry_callback_is_complete(entry)) continue;
      if (iree_async_semaphore_cancel_timepoint(entry->timepoint.semaphore,
                                                &entry->timepoint)) {
        continue;
      }
      iree_notification_await(&op->callback_notification,
                              iree_hal_amdgpu_wait_entry_callback_is_complete,
                              entry, iree_infinite_timeout());
    }
    iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(op);

    iree_status_t op_status = (iree_status_t)iree_atomic_exchange(
        &op->error_status, 0, iree_memory_order_acquire);
    if (iree_status_is_ok(op_status)) {
      op_status = iree_make_status(status_code, "%s", status_message);
    }
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, op_status);
    iree_hal_amdgpu_pending_op_abort_unsubmitted_dealloca(op);
    iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
    iree_hal_amdgpu_pending_op_release_retained(op);
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_notification_deinitialize(&op->callback_notification);
    iree_arena_deinitialize(&op->arena);
  }
}

//===----------------------------------------------------------------------===//
// Initialization / deinitialization
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_post_drain_action_t* action,
    iree_hal_amdgpu_host_queue_post_drain_fn_t fn, void* user_data) {
  action->next = NULL;
  action->fn = fn;
  action->user_data = user_data;

  iree_slim_mutex_lock(&queue->post_drain_mutex);
  if (queue->post_drain_tail) {
    queue->post_drain_tail->next = action;
  } else {
    queue->post_drain_head = action;
  }
  queue->post_drain_tail = action;
  iree_slim_mutex_unlock(&queue->post_drain_mutex);
}

static void iree_hal_amdgpu_host_queue_run_post_drain_actions(
    iree_hal_amdgpu_host_queue_t* queue) {
  for (;;) {
    iree_slim_mutex_lock(&queue->post_drain_mutex);
    iree_hal_amdgpu_host_queue_post_drain_action_t* action =
        queue->post_drain_head;
    if (action) {
      queue->post_drain_head = action->next;
      if (!queue->post_drain_head) {
        queue->post_drain_tail = NULL;
      }
      action->next = NULL;
    }
    iree_slim_mutex_unlock(&queue->post_drain_mutex);

    if (!action) break;
    action->fn(action->user_data);
  }
}

// Drains completed notification entries and reclaims kernarg space. If the GPU
// queue has faulted (error_status is set), fails all pending entries instead of
// draining normally.
static iree_host_size_t iree_hal_amdgpu_host_queue_drain_completions(
    iree_hal_amdgpu_host_queue_t* queue) {
  // Check for GPU queue error (set by the HSA error callback on another
  // thread). If the queue has faulted, no further epochs will advance —
  // fail all pending entries so waiters get the actual GPU error instead
  // of hanging or timing out.
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &queue->error_status, iree_memory_order_acquire);
  const uint64_t previous_epoch = (uint64_t)iree_atomic_load(
      &queue->notification_ring.epoch.last_drained, iree_memory_order_relaxed);
  uint64_t kernarg_reclaim_position = 0;
  iree_host_size_t count = 0;
  if (IREE_UNLIKELY(error)) {
    count = iree_hal_amdgpu_notification_ring_fail_all(
        &queue->notification_ring, error, &kernarg_reclaim_position);
    iree_async_frontier_tracker_fail_axis(
        queue->frontier_tracker, queue->axis,
        iree_status_from_code(iree_status_code(error)));
  } else {
    count = iree_hal_amdgpu_notification_ring_drain(&queue->notification_ring,
                                                    /*fallback_frontier=*/NULL,
                                                    &kernarg_reclaim_position);
    const uint64_t current_epoch =
        (uint64_t)iree_atomic_load(&queue->notification_ring.epoch.last_drained,
                                   iree_memory_order_acquire);
    if (current_epoch > previous_epoch) {
      iree_async_frontier_tracker_advance(queue->frontier_tracker, queue->axis,
                                          current_epoch);
    }
  }
  if (kernarg_reclaim_position > 0) {
    iree_hal_amdgpu_kernarg_ring_reclaim(&queue->kernarg_ring,
                                         kernarg_reclaim_position);
  }
  iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);
  return count;
}

static bool iree_hal_amdgpu_host_queue_has_error(
    iree_hal_amdgpu_host_queue_t* queue) {
  return iree_atomic_load(&queue->error_status, iree_memory_order_acquire) != 0;
}

static bool iree_hal_amdgpu_host_queue_store_error(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_t error) {
  intptr_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &queue->error_status, &expected, (intptr_t)error,
          iree_memory_order_release, iree_memory_order_acquire)) {
    return true;
  }
  iree_status_free(error);
  return false;
}

static void iree_hal_amdgpu_host_queue_request_completion_thread_stop(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (queue->completion_thread_stop_signal.handle) {
    iree_hsa_signal_store_screlease(IREE_LIBHSA(queue->libhsa),
                                    queue->completion_thread_stop_signal, 1);
  }
}

static hsa_signal_value_t iree_hal_amdgpu_host_queue_last_drained_signal_value(
    iree_hal_amdgpu_host_queue_t* queue) {
  const uint64_t last_drained_epoch = (uint64_t)iree_atomic_load(
      &queue->notification_ring.epoch.last_drained, iree_memory_order_acquire);
  return (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                              last_drained_epoch);
}

// Completion thread entry point. Blocks in HSA until either the queue epoch
// signal changes or teardown/error signals the stop signal. Completion wakeups
// drain normally; stop/error wakeups perform one final drain/fail before exit.
static int iree_hal_amdgpu_host_queue_completion_thread_main(void* entry_arg) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)entry_arg;

  enum {
    IREE_HAL_AMDGPU_COMPLETION_WAIT_EPOCH_SIGNAL = 0,
    IREE_HAL_AMDGPU_COMPLETION_WAIT_STOP_SIGNAL = 1,
    IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT = 2,
  };

  hsa_signal_t epoch_signal =
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring);
  hsa_signal_t stop_signal = queue->completion_thread_stop_signal;
  hsa_signal_value_t last_epoch_value =
      iree_hal_amdgpu_host_queue_last_drained_signal_value(queue);

  bool keep_running = true;
  while (keep_running) {
    hsa_signal_t signals[IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT] = {
        epoch_signal,
        stop_signal,
    };
    hsa_signal_condition_t
        conditions[IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT] = {
            HSA_SIGNAL_CONDITION_NE,
            HSA_SIGNAL_CONDITION_NE,
        };
    hsa_signal_value_t values[IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT] = {
        last_epoch_value,
        0,
    };
    const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
        IREE_LIBHSA(queue->libhsa),
        IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT, signals, conditions,
        values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED,
        /*satisfying_value=*/NULL);

    if (signal_index == IREE_HAL_AMDGPU_COMPLETION_WAIT_EPOCH_SIGNAL) {
      iree_hal_amdgpu_host_queue_drain_completions(queue);
      // Arm the next wait from the epoch we actually drained, not from a raw
      // HSA signal load. A GPU completion can race with the drain and update
      // the signal after drain() sampled it; observing that newer value here
      // would mark an undrained epoch as already seen and could sleep forever
      // with a user semaphore still pending.
      last_epoch_value =
          iree_hal_amdgpu_host_queue_last_drained_signal_value(queue);
    }

    if (signal_index == IREE_HAL_AMDGPU_COMPLETION_WAIT_STOP_SIGNAL ||
        iree_hal_amdgpu_host_queue_has_error(queue)) {
      iree_hal_amdgpu_host_queue_drain_completions(queue);
      keep_running = false;
    } else if (IREE_UNLIKELY(signal_index >=
                             IREE_HAL_AMDGPU_COMPLETION_WAIT_SIGNAL_COUNT)) {
      iree_status_t error = iree_make_status(
          IREE_STATUS_INTERNAL,
          "hsa_amd_signal_wait_any returned invalid signal index %u",
          signal_index);
      iree_hal_amdgpu_host_queue_store_error(queue, error);
      iree_hal_amdgpu_host_queue_drain_completions(queue);
      keep_running = false;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return 0;
}

// HSA queue error callback. Called by the HSA runtime (on an internal thread)
// when the queue encounters an unrecoverable error (page fault, invalid AQL
// packet, ECC error). Stores the error atomically on the queue so the
// completion thread can fail pending semaphores with the actual GPU error.
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
  if (iree_hal_amdgpu_host_queue_store_error(queue, error)) {
    iree_hal_amdgpu_host_queue_request_completion_thread_stop(queue);
  }
}

iree_status_t iree_hal_amdgpu_host_queue_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_hal_device_t* logical_device,
    iree_async_proactor_t* proactor, hsa_agent_t gpu_agent,
    hsa_amd_memory_pool_t kernarg_pool, hsa_amd_memory_pool_t pm4_ib_pool,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
    iree_hal_queue_affinity_t queue_affinity,
    iree_thread_affinity_t completion_thread_affinity,
    iree_hal_amdgpu_wait_barrier_strategy_t wait_barrier_strategy,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_table,
    iree_arena_block_pool_t* block_pool,
    const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context,
    iree_hal_pool_t* default_pool, iree_hal_amdgpu_staging_pool_t* staging_pool,
    iree_host_size_t device_ordinal, uint32_t aql_queue_capacity,
    uint32_t notification_capacity, uint32_t kernarg_capacity_in_blocks,
    iree_allocator_t host_allocator, iree_hal_amdgpu_host_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  IREE_ASSERT_ARGUMENT(epoch_table);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(transfer_context);
  IREE_ASSERT_ARGUMENT(default_pool);
  IREE_ASSERT_ARGUMENT(out_queue);

  if (!iree_host_size_is_power_of_two(aql_queue_capacity) ||
      !iree_host_size_is_power_of_two(notification_capacity) ||
      !iree_host_size_is_power_of_two(kernarg_capacity_in_blocks)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "all capacities must be powers of two");
  }
  if (kernarg_capacity_in_blocks / 2u < aql_queue_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernarg ring capacity must be at least 2x the AQL ring capacity "
        "to cover one tail-padding gap at wrap (got kernarg_blocks=%u, "
        "aql_packets=%u)",
        kernarg_capacity_in_blocks, aql_queue_capacity);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_queue, 0, sizeof(*out_queue));
  out_queue->base.vtable = &iree_hal_amdgpu_host_queue_vtable;
  out_queue->libhsa = libhsa;
  out_queue->logical_device = logical_device;
  out_queue->proactor = proactor;
  out_queue->frontier_tracker = frontier_tracker;
  out_queue->host_allocator = host_allocator;

  // Submission pipeline state.
  iree_slim_mutex_initialize(&out_queue->submission_mutex);
  iree_slim_mutex_initialize(&out_queue->post_drain_mutex);
  out_queue->axis = axis;
  out_queue->wait_barrier_strategy = wait_barrier_strategy;
  out_queue->queue_affinity = queue_affinity;
  out_queue->last_signal.semaphore = NULL;
  out_queue->last_signal.epoch = 0;
  out_queue->block_pool = block_pool;
  out_queue->can_publish_frontier = true;
  out_queue->transfer_context = transfer_context;
  out_queue->default_pool = default_pool;
  out_queue->staging_pool = staging_pool;
  out_queue->device_ordinal = device_ordinal;
  out_queue->pending_head = NULL;
  iree_async_frontier_initialize(iree_hal_amdgpu_host_queue_frontier(out_queue),
                                 /*entry_count=*/0);

  // The optional tracker semaphore is an iree_async_semaphore_t bridge for
  // CPU-side wait integration. The queue's GPU-visible HSA epoch signal is
  // created by the notification ring below and registered in the epoch table.
  iree_status_t status = iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL);

  // Create the host-only stop signal before the hardware queue so the HSA error
  // callback always has a valid signal to wake if queue creation races with an
  // asynchronous fault.
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(libhsa), /*initial_value=*/0,
        /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0,
        &out_queue->completion_thread_stop_signal);
  }

  // Create the HSA hardware AQL queue.
  //
  // HSA_QUEUE_TYPE_MULTI is required (not just an optimization). Once command
  // buffers start performing device-side enqueue, the CP itself becomes a
  // concurrent producer alongside the host submission path, so the queue must
  // permit multiple concurrent producers. The host-side reserve already uses
  // an atomic fetch_add on the write index, which is well-defined only on
  // MULTI queues.
  hsa_queue_t* hardware_queue = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hsa_queue_create(
        IREE_LIBHSA(libhsa), gpu_agent, aql_queue_capacity,
        HSA_QUEUE_TYPE_MULTI, iree_hal_amdgpu_host_queue_error_callback,
        /*data=*/out_queue,
        /*private_segment_size=*/UINT32_MAX,
        /*group_segment_size=*/UINT32_MAX, &hardware_queue);
  }

  // Initialize the AQL ring from the hardware queue.
  if (iree_status_is_ok(status)) {
    out_queue->hardware_queue = hardware_queue;
    iree_hal_amdgpu_aql_ring_initialize((iree_amd_queue_t*)hardware_queue,
                                        &out_queue->aql_ring);
  }

  // Initialize the kernarg ring from the HSA kernarg memory pool.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_kernarg_ring_initialize(
        libhsa, gpu_agent, kernarg_pool, kernarg_capacity_in_blocks,
        &out_queue->kernarg_ring);
  }

  // Initialize the optional PM4 IB slot buffer. The buffer is indexed by AQL
  // packet id and inherits AQL ring backpressure/reuse; there is no separate
  // PM4 producer or reclaim position.
  if (iree_status_is_ok(status) &&
      wait_barrier_strategy ==
          IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64) {
    status = iree_hal_amdgpu_host_queue_allocate_pm4_ib_slots(
        libhsa, gpu_agent, pm4_ib_pool, aql_queue_capacity, out_queue);
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
    iree_thread_create_params_t thread_params;
    memset(&thread_params, 0, sizeof(thread_params));
    thread_params.name = iree_make_cstring_view("iree-hal-amdgpu-complete");
    thread_params.initial_affinity = completion_thread_affinity;
    status = iree_thread_create(
        iree_hal_amdgpu_host_queue_completion_thread_main, out_queue,
        thread_params, host_allocator, &out_queue->completion_thread);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_deinitialize(out_queue);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&queue->submission_mutex);
  queue->is_shutting_down = true;
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (queue->completion_thread) {
    iree_hal_amdgpu_host_queue_request_completion_thread_stop(queue);
    // There is only one owner for the thread, so this also joins the thread.
    iree_thread_release(queue->completion_thread);
    queue->completion_thread = NULL;
  }

  // Destroy the hardware queue before the remaining host-side resources so the
  // HSA runtime cannot race a late error callback against signal teardown.
  if (queue->hardware_queue) {
    IREE_IGNORE_ERROR(iree_hsa_queue_destroy(IREE_LIBHSA(queue->libhsa),
                                             queue->hardware_queue));
    queue->hardware_queue = NULL;
  }

  // Cancel all pending (deferred) operations. Their signal semaphores are
  // failed with CANCELLED so downstream waiters don't hang.
  if (queue->pending_head) {
    iree_hal_amdgpu_host_queue_cancel_pending(queue, IREE_STATUS_CANCELLED,
                                              "queue shutting down");
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
  iree_hal_amdgpu_host_queue_run_post_drain_actions(queue);

  // Deregister from the epoch signal table before destroying the notification
  // ring (which owns the epoch signal). Guarded by epoch_table != NULL to
  // handle partial initialization (init failed before registration).
  if (queue->epoch_table) {
    iree_hal_amdgpu_epoch_signal_table_deregister(
        queue->epoch_table, iree_async_axis_device_index(queue->axis),
        iree_async_axis_queue_index(queue->axis));
    queue->epoch_table = NULL;
  }

  if (queue->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->frontier_tracker, queue->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    queue->frontier_tracker = NULL;
    queue->axis = 0;
  }

  iree_hal_amdgpu_notification_ring_deinitialize(&queue->notification_ring);

  iree_hal_amdgpu_kernarg_ring_deinitialize(queue->libhsa,
                                            &queue->kernarg_ring);

  if (queue->pm4_ib_slots) {
    IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(queue->libhsa),
                                                    queue->pm4_ib_slots));
    queue->pm4_ib_slots = NULL;
  }

  if (queue->completion_thread_stop_signal.handle) {
    IREE_IGNORE_ERROR(iree_hsa_signal_destroy(
        IREE_LIBHSA(queue->libhsa), queue->completion_thread_stop_signal));
    queue->completion_thread_stop_signal.handle = 0;
  }

  iree_slim_mutex_deinitialize(&queue->post_drain_mutex);
  iree_slim_mutex_deinitialize(&queue->submission_mutex);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_host_queue_trim(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {}

//===----------------------------------------------------------------------===//
// Queue operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_host_queue_execute(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags);

static iree_status_t iree_hal_amdgpu_host_queue_signal_empty_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status)) {
    // Signal outside submission_mutex: semaphore signaling dispatches satisfied
    // timepoints, and those callbacks may submit additional queue work.
    status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                            /*frontier=*/NULL);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_ALLOCA, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)buffer);
  op->alloca_op.pool = pool;
  op->alloca_op.params = params;
  op->alloca_op.allocation_size = allocation_size;
  op->alloca_op.flags = flags;
  op->alloca_op.reserve_flags = reserve_flags;
  op->alloca_op.buffer = buffer;
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op,
    iree_hal_amdgpu_alloca_memory_wait_t** out_wait) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&op->arena, sizeof(*wait), (void**)&wait));
    memset(wait, 0, sizeof(*wait));
    iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_relaxed);
    op->alloca_op.memory_wait = wait;
  }
  *out_wait = wait;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_alloca_reservation_t* alloca_reservation) {
  const iree_async_frontier_t* wait_frontier =
      alloca_reservation->acquire_info.wait_frontier;
  if (IREE_UNLIKELY(!wait_frontier)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca waitable pool reservation did not provide a frontier");
  }

  iree_host_size_t wait_frontier_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_frontier_size(wait_frontier->entry_count,
                                                &wait_frontier_size));
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  iree_async_frontier_t* wait_frontier_copy = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&op->arena, wait_frontier_size,
                                           (void**)&wait_frontier_copy));

  memcpy(wait_frontier_copy, wait_frontier, wait_frontier_size);
  wait->frontier.reservation = alloca_reservation->reservation;
  wait->frontier.wait_frontier = wait_frontier_copy;
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pending_op_prepare_alloca_pool_notification_wait(
    iree_hal_amdgpu_pending_op_t* op, iree_async_notification_t* notification,
    uint32_t wait_token) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  wait->pool_notification.notification = notification;
  wait->pool_notification.wait_token = wait_token;
  wait->pool_notification.wait_slot =
      (uint8_t)((wait->pool_notification.wait_slot + 1u) & 1u);
  iree_hal_amdgpu_pending_op_begin_alloca_memory_wait_arming(
      op, wait, IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_alloca_memory_wait_t* memory_wait) {
  iree_hal_amdgpu_alloca_reservation_t alloca_reservation = {
      .readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY,
      .acquire_result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT,
      .reservation = memory_wait->frontier.reservation,
      .acquire_info =
          {
              .wait_frontier = memory_wait->frontier.wait_frontier,
          },
      .wait_resolution = *resolution,
  };
  memory_wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  return iree_hal_amdgpu_host_queue_submit_alloca_reservation(
      queue, &alloca_reservation, signal_semaphore_list, allocation_pool,
      params, buffer, submission_flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  *out_memory_wait_op = NULL;
  if (pending_op) {
    *out_memory_wait_op = pending_op;
    return iree_ok_status();
  }

  iree_hal_semaphore_list_t empty_wait_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_defer_alloca(
      queue, &empty_wait_list, &signal_semaphore_list, allocation_pool, params,
      allocation_size, flags, reserve_flags, buffer, out_memory_wait_op));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    const iree_hal_amdgpu_alloca_reservation_t* alloca_reservation,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, params, allocation_size,
      flags, reserve_flags, buffer, pending_op, &memory_wait_op);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(
        memory_wait_op, alloca_reservation);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else {
    iree_hal_pool_release_reservation(
        allocation_pool, &alloca_reservation->reservation,
        alloca_reservation->acquire_info.wait_frontier);
    if (!pending_op && memory_wait_op) {
      iree_hal_amdgpu_pending_op_destroy_under_lock(memory_wait_op,
                                                    iree_status_clone(status));
    }
  }

  return status;
}

static iree_status_t
iree_hal_amdgpu_host_queue_defer_alloca_pool_notification_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  iree_async_notification_t* notification =
      iree_hal_pool_notification(allocation_pool);
  if (IREE_UNLIKELY(!notification)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca exhausted pool did not provide a notification");
  }

  const uint32_t wait_token = iree_async_notification_query_epoch(notification);
  iree_hal_amdgpu_alloca_reservation_t alloca_reservation;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_acquire_alloca_reservation(
      queue, resolution, allocation_pool, params, allocation_size, flags,
      reserve_flags, &alloca_reservation));
  switch (alloca_reservation.readiness) {
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY:
      return iree_hal_amdgpu_host_queue_submit_alloca_reservation(
          queue, &alloca_reservation, signal_semaphore_list, allocation_pool,
          params, buffer, submission_flags);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT:
      return iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
          queue, signal_semaphore_list, allocation_pool, params,
          allocation_size, flags, reserve_flags, buffer, &alloca_reservation,
          pending_op, out_memory_wait_op);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION:
      break;
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized alloca reservation readiness %u",
                              alloca_reservation.readiness);
  }

  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, params, allocation_size,
      flags, reserve_flags, buffer, pending_op, &memory_wait_op);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pending_op_prepare_alloca_pool_notification_wait(
        memory_wait_op, notification, wait_token);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else if (!pending_op && memory_wait_op) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(memory_wait_op,
                                                  iree_status_clone(status));
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  *out_memory_wait_op = NULL;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  iree_hal_amdgpu_alloca_memory_wait_t* memory_wait =
      pending_op ? pending_op->alloca_op.memory_wait : NULL;
  if (memory_wait &&
      memory_wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER) {
    return iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
        queue, resolution, signal_semaphore_list, allocation_pool, params,
        buffer, submission_flags, memory_wait);
  }

  iree_hal_amdgpu_alloca_reservation_t alloca_reservation;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_acquire_alloca_reservation(
      queue, resolution, allocation_pool, params, allocation_size, flags,
      reserve_flags, &alloca_reservation));
  switch (alloca_reservation.readiness) {
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY:
      return iree_hal_amdgpu_host_queue_submit_alloca_reservation(
          queue, &alloca_reservation, signal_semaphore_list, allocation_pool,
          params, buffer, submission_flags);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT:
      return iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
          queue, signal_semaphore_list, allocation_pool, params,
          allocation_size, flags, reserve_flags, buffer, &alloca_reservation,
          pending_op, out_memory_wait_op);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION:
      return iree_hal_amdgpu_host_queue_defer_alloca_pool_notification_wait(
          queue, resolution, signal_semaphore_list, allocation_pool, params,
          allocation_size, flags, reserve_flags, buffer, submission_flags,
          pending_op, out_memory_wait_op);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized alloca reservation readiness %u",
                              alloca_reservation.readiness);
  }
}

static iree_status_t iree_hal_amdgpu_host_queue_alloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_hal_pool_t* allocation_pool = NULL;
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_alloca_wrapper(
      queue, pool, &params, allocation_size, flags, &allocation_pool, &buffer));
  // Always ask the pool to surface waitable death-frontier candidates so the
  // queue can distinguish true pool pressure from a dependency the caller did
  // not authorize. The HAL alloca flag is checked before consuming any
  // OK_NEEDS_WAIT reservation.
  const iree_hal_pool_reserve_flags_t reserve_flags =
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  iree_hal_amdgpu_pending_op_t* memory_wait_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_alloca(
        queue, &wait_semaphore_list, &signal_semaphore_list, allocation_pool,
        params, allocation_size, flags, reserve_flags, buffer, &deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_alloca(
        queue, &resolution, signal_semaphore_list, allocation_pool, params,
        allocation_size, flags, reserve_flags, buffer,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        /*pending_op=*/NULL, &memory_wait_op);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  if (iree_status_is_ok(status) && memory_wait_op) {
    iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(memory_wait_op);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)buffer);
  op->dealloca.buffer = buffer;
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_dealloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  if (IREE_UNLIKELY(
          iree_any_bit_set(flags, ~(IREE_HAL_DEALLOCA_FLAG_NONE |
                                    IREE_HAL_DEALLOCA_FLAG_PREFER_ORIGIN)))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported dealloca flags: 0x%" PRIx64, flags);
  }

  // iree_hal_device_queue_dealloca() applies PREFER_ORIGIN before vtable
  // dispatch by rewriting the device and queue affinity from the buffer's
  // allocation placement. Transient wrappers created by queue_alloca carry this
  // queue's one-bit affinity in that placement, so this host-queue path can use
  // |base_queue| directly.
  if (!iree_hal_amdgpu_transient_buffer_isa(buffer)) {
    return iree_hal_amdgpu_host_queue_execute(
        base_queue, wait_semaphore_list, signal_semaphore_list,
        /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  if (IREE_UNLIKELY(!iree_hal_amdgpu_transient_buffer_begin_dealloca(buffer))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has already been queued for deallocation");
  }

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_dealloca(
        queue, &wait_semaphore_list, &signal_semaphore_list, buffer,
        &deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_dealloca(
        queue, &resolution, signal_semaphore_list, buffer,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_transient_buffer_abort_dealloca(buffer);
  }
  return status;
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
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
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
    status = iree_hal_amdgpu_host_queue_submit_fill(
        queue, &resolution, signal_semaphore_list, target_buffer, target_offset,
        length, pattern_bits, pattern_length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
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
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/2, &max_resources));
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
    status = iree_hal_amdgpu_host_queue_submit_copy(
        queue, &resolution, signal_semaphore_list, source_buffer, source_offset,
        target_buffer, target_offset, length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
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
  const uint8_t* source_bytes = NULL;
  iree_host_size_t source_length = 0;
  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_update_copy(
      target_buffer, target_offset, source_buffer, source_offset, length, flags,
      &source_bytes, &source_length, &target_device_ptr));
  (void)target_device_ptr;

  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_UPDATE, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);

  // Copy the source host data into the arena. The caller's buffer may be
  // freed after this call returns.
  void* source_copy = NULL;
  iree_status_t status =
      iree_arena_allocate(&op->arena, source_length, &source_copy);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op, status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "arena allocation failed during defer_update");
  }
  memcpy(source_copy, source_bytes, source_length);
  op->update.source_data = source_copy;
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
    status = iree_hal_amdgpu_host_queue_submit_update(
        queue, &resolution, signal_semaphore_list, source_buffer, source_offset,
        target_buffer, target_offset, length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

static bool iree_hal_amdgpu_host_queue_is_noop_dispatch(
    const iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags) {
  return !iree_hal_dispatch_uses_indirect_parameters(flags) &&
         (config.workgroup_count[0] | config.workgroup_count[1] |
          config.workgroup_count[2]) == 0;
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
  if (IREE_UNLIKELY(!command_buffer && binding_table.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "barrier-only queue_execute must not provide a binding table "
        "(count=%" PRIhsz ")",
        binding_table.count);
  }

  // Optional command buffer + up to binding_table.count buffers.
  iree_host_size_t operation_resource_count = command_buffer ? 1 : 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(operation_resource_count,
                                                binding_table.count,
                                                &operation_resource_count))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "execute retains too many resources (bindings=%" PRIhsz ")",
        binding_table.count);
  }
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count, operation_resource_count, &max_resources));
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

  if (iree_status_is_ok(status)) {
    *out_op = op;
  } else {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op,
                                                  iree_status_clone(status));
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
  iree_host_size_t operation_resource_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_dispatch(
      queue, executable, export_ordinal, config, constants, bindings, flags,
      &operation_resource_count));
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count, operation_resource_count, &max_resources));
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
  if (iree_status_is_ok(status) && bindings.count > 0 &&
      !iree_any_bit_set(flags,
                        IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
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

  if (iree_status_is_ok(status)) {
    *out_op = op;
  } else {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op,
                                                  iree_status_clone(status));
  }
  return status;
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
  const bool is_noop_dispatch =
      iree_hal_amdgpu_host_queue_is_noop_dispatch(config, flags);

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    if (is_noop_dispatch) {
      status = iree_hal_amdgpu_host_queue_defer_execute(
          queue, &wait_semaphore_list, &signal_semaphore_list,
          /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
          IREE_HAL_EXECUTE_FLAG_NONE, &deferred_op);
    } else {
      status = iree_hal_amdgpu_host_queue_defer_dispatch(
          queue, &wait_semaphore_list, &signal_semaphore_list, executable,
          export_ordinal, config, constants, bindings, flags, &deferred_op);
    }
  } else if (is_noop_dispatch) {
    status = iree_hal_amdgpu_host_queue_submit_barrier(
        queue, &resolution, signal_semaphore_list,
        (iree_hal_amdgpu_reclaim_action_t){0},
        /*operation_resources=*/NULL,
        /*operation_resource_count=*/0,
        /*post_commit_fn=*/NULL, /*post_commit_user_data=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_dispatch(
        queue, &resolution, signal_semaphore_list, executable, export_ordinal,
        config, constants, bindings, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
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

  if (!command_buffer && wait_semaphore_list.count == 0) {
    if (IREE_UNLIKELY(binding_table.count != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "barrier-only queue_execute must not provide a binding table "
          "(count=%" PRIhsz ")",
          binding_table.count);
    }
    return iree_hal_amdgpu_host_queue_signal_empty_barrier(
        queue, signal_semaphore_list);
  }

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
  } else if (!command_buffer) {
    if (IREE_UNLIKELY(binding_table.count != 0)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "barrier-only queue_execute must not provide a binding table "
          "(count=%" PRIhsz ")",
          binding_table.count);
    } else {
      status = iree_hal_amdgpu_host_queue_submit_barrier(
          queue, &resolution, signal_semaphore_list,
          (iree_hal_amdgpu_reclaim_action_t){0},
          /*operation_resources=*/NULL,
          /*operation_resource_count=*/0,
          /*post_commit_fn=*/NULL, /*post_commit_user_data=*/NULL,
          IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
    }
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
  return iree_hal_amdgpu_host_queue_read_file(
      base_queue, wait_semaphore_list, signal_semaphore_list, source_file,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_write(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_hal_amdgpu_host_queue_write_file(
      base_queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_file, target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_host_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      /*signal_semaphore_count=*/0,
      /*operation_resource_count=*/operation_resource_count, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  const iree_hal_semaphore_list_t empty_signal_list =
      iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, &empty_signal_list,
      IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION, max_resources, &op));
  for (iree_host_size_t i = 0; i < operation_resource_count; ++i) {
    iree_hal_amdgpu_pending_op_retain(op, operation_resources[i]);
  }
  op->host_action.action = action;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_enqueue_host_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count) {
  if (IREE_UNLIKELY(!action.fn)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host action callback must be non-null");
  }
  if (IREE_UNLIKELY(operation_resource_count > 0 && !operation_resources)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host action resources must be non-null");
  }

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  // Host actions execute on CPU threads and must observe device-produced
  // host-visible memory even when a semaphore edge itself is device-local.
  resolution.inline_acquire_scope = iree_hal_amdgpu_host_queue_max_fence_scope(
      resolution.inline_acquire_scope, IREE_HSA_FENCE_SCOPE_SYSTEM);
  resolution.barrier_acquire_scope = iree_hal_amdgpu_host_queue_max_fence_scope(
      resolution.barrier_acquire_scope, IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_host_action(
        queue, &wait_semaphore_list, action, operation_resources,
        operation_resource_count, &deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_barrier(
        queue, &resolution, iree_hal_semaphore_list_empty(), action,
        operation_resources, operation_resource_count,
        /*post_commit_fn=*/NULL, /*post_commit_user_data=*/NULL,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_host_call(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags, iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/0, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL, max_resources, &op));
  op->host_call.call = call;
  memcpy(op->host_call.args, args, sizeof(op->host_call.args));
  op->host_call.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_host_call(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_host_call(call, args, flags));

  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_host_call(
        queue, &wait_semaphore_list, &signal_semaphore_list, call, args, flags,
        &deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_host_call(
        queue, &resolution, signal_semaphore_list, call, args, flags);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
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
