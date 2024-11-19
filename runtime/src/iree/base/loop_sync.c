// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/loop_sync.h"

#include "iree/base/internal/math.h"
#include "iree/base/internal/wait_handle.h"

//===----------------------------------------------------------------------===//
// iree_loop_sync_t utilities
//===----------------------------------------------------------------------===//

// Amount of time that can remain in a wait-until while still retiring.
// This prevents additional system sleeps when the remaining time before the
// deadline is less than the granularity the system is likely able to sleep for.
// Some platforms may have as much as 10-15ms of potential slop and sleeping for
// 1ms may result in 10-15ms.
#define IREE_LOOP_SYNC_DELAY_SLOP_NS (2 /*ms*/ * 1000000)

// NOTE: all callbacks should be at offset 0. This allows for easily zipping
// through the params lists and issuing callbacks.
static_assert(offsetof(iree_loop_call_params_t, callback) == 0,
              "callback must be at offset 0");
static_assert(offsetof(iree_loop_dispatch_params_t, callback) == 0,
              "callback must be at offset 0");
static_assert(offsetof(iree_loop_wait_until_params_t, callback) == 0,
              "callback must be at offset 0");
static_assert(offsetof(iree_loop_wait_one_params_t, callback) == 0,
              "callback must be at offset 0");
static_assert(offsetof(iree_loop_wait_multi_params_t, callback) == 0,
              "callback must be at offset 0");

static void iree_loop_sync_abort_scope(iree_loop_sync_t* loop_sync,
                                       iree_loop_sync_scope_t* scope);

//===----------------------------------------------------------------------===//
// iree_loop_run_ring_t
//===----------------------------------------------------------------------===//

// Represents an operation in the loop run ringbuffer.
// Note that the storage may be reallocated at any time and all pointers must be
// external to the storage in order to remain valid.
typedef struct iree_loop_run_op_t {
  union {
    iree_loop_callback_t callback;  // asserted at offset 0 above
    union {
      iree_loop_call_params_t call;
      iree_loop_dispatch_params_t dispatch;
    } params;
  };
  iree_loop_command_t command;
  iree_loop_sync_scope_t* scope;

  // Set on calls when we are issuing a callback for an operation.
  // Unlike other pointers in the params this is owned by the ring.
  iree_status_t status;
} iree_loop_run_op_t;

// Ringbuffer containing pending ready to run callback operations.
//
// Generally this works as a FIFO but we allow for head-of-ring replacement
// for high priority tail calls. New operations are appended to the ring and
// removed as drained; if the ringbuffer capacity is exceeded then the storage
// will be reallocated up to the maximum capacity specified at creation time.
typedef iree_alignas(iree_max_align_t) struct iree_loop_run_ring_t {
  // Current storage capacity of |ops|.
  uint32_t capacity;
  // Index into |ops| where the next operation to be dequeued is located.
  uint32_t read_head;
  // Index into |ops| where the last operation to be enqueued is located.
  uint32_t write_head;
  // Ringbuffer storage.
  iree_loop_run_op_t ops[0];
} iree_loop_run_ring_t;

static iree_host_size_t iree_loop_run_ring_storage_size(
    iree_loop_sync_options_t options) {
  return sizeof(iree_loop_run_ring_t) +
         options.max_queue_depth * sizeof(iree_loop_run_op_t);
}

static inline uint32_t iree_loop_run_ring_mask(
    const iree_loop_run_ring_t* run_ring) {
  return run_ring->capacity - 1;
}

static iree_host_size_t iree_loop_run_ring_size(
    const iree_loop_run_ring_t* run_ring) {
  return run_ring->write_head >= run_ring->read_head
             ? (run_ring->write_head - run_ring->read_head)
             : (run_ring->write_head + run_ring->capacity -
                run_ring->read_head);
}

static bool iree_loop_run_ring_is_empty(const iree_loop_run_ring_t* run_ring) {
  return run_ring->read_head == run_ring->write_head;
}

static bool iree_loop_run_ring_is_full(const iree_loop_run_ring_t* run_ring) {
  const uint32_t mask = iree_loop_run_ring_mask(run_ring);
  return ((run_ring->write_head - run_ring->read_head) & mask) == mask;
}

static void iree_loop_run_ring_initialize(iree_loop_sync_options_t options,
                                          iree_loop_run_ring_t* out_run_ring) {
  IREE_TRACE_ZONE_BEGIN(z0);

  out_run_ring->capacity = (uint32_t)options.max_queue_depth;
  out_run_ring->read_head = 0;
  out_run_ring->write_head = 0;

  IREE_TRACE_ZONE_END(z0);
}

static void iree_loop_run_ring_deinitialize(iree_loop_run_ring_t* run_ring) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Expected abort to be called.
  IREE_ASSERT(iree_loop_run_ring_is_empty(run_ring));

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_loop_run_ring_enqueue(iree_loop_run_ring_t* run_ring,
                                                iree_loop_run_op_t op) {
  if (iree_loop_run_ring_is_full(run_ring)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "run ringbuffer capacity %u exceeded; reduce the amount of concurrent "
        "work or use a full loop implementation",
        run_ring->capacity);
  }

  IREE_TRACE_PLOT_VALUE_I64("iree_loop_queue_depth",
                            iree_loop_run_ring_size(run_ring));

  // Reserve a slot for the new operation.
  uint32_t slot = run_ring->write_head;
  run_ring->write_head =
      (run_ring->write_head + 1) & iree_loop_run_ring_mask(run_ring);

  // Copy the operation in; the params are on the stack and won't be valid after
  // the caller returns.
  run_ring->ops[slot] = op;

  ++op.scope->pending_count;

  IREE_TRACE_PLOT_VALUE_I64("iree_loop_queue_depth",
                            iree_loop_run_ring_size(run_ring));
  return iree_ok_status();
}

static bool iree_loop_run_ring_dequeue(iree_loop_run_ring_t* run_ring,
                                       iree_loop_run_op_t* out_op) {
  if (iree_loop_run_ring_is_empty(run_ring)) return false;

  IREE_TRACE_PLOT_VALUE_I64("iree_loop_queue_depth",
                            iree_loop_run_ring_size(run_ring));

  // Acquire the next operation.
  uint32_t slot = run_ring->read_head;
  run_ring->read_head =
      (run_ring->read_head + 1) & iree_loop_run_ring_mask(run_ring);

  // Copy out the parameters; the operation we execute may overwrite them by
  // enqueuing more work.
  *out_op = run_ring->ops[slot];

  --out_op->scope->pending_count;

  IREE_TRACE_PLOT_VALUE_I64("iree_loop_queue_depth",
                            iree_loop_run_ring_size(run_ring));
  return true;
}

// Aborts all ops that are part of |scope|.
// A NULL |scope| indicates all work from all scopes should be aborted.
static void iree_loop_run_ring_abort_scope(iree_loop_run_ring_t* run_ring,
                                           iree_loop_sync_scope_t* scope) {
  if (iree_loop_run_ring_is_empty(run_ring)) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Do a single pass over the ring and abort all ops matching the scope.
  // To keep things simple and preserve dense ordered ops in the ringbuffer we
  // dequeue all ops and re-enqueue any that don't match. When complete the ring
  // may be at a different offset but will contain only those ops we didn't
  // abort in their original order.
  iree_host_size_t count = iree_loop_run_ring_size(run_ring);
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_loop_run_op_t op;
    if (!iree_loop_run_ring_dequeue(run_ring, &op)) break;
    if (scope && op.scope != scope) {
      // Not part of the scope we are aborting; re-enqueue to the ring.
      iree_status_ignore(iree_loop_run_ring_enqueue(run_ring, op));
    } else {
      // Part of the scope to abort.
      --op.scope->pending_count;
      iree_status_ignore(op.status);
      iree_status_ignore(op.callback.fn(op.callback.user_data, iree_loop_null(),
                                        iree_make_status(IREE_STATUS_ABORTED)));
    }
  }

  IREE_TRACE_PLOT_VALUE_I64("iree_loop_queue_depth",
                            iree_loop_run_ring_size(run_ring));
  IREE_TRACE_ZONE_END(z0);
}

// Aborts all ops from all scopes.
static void iree_loop_run_ring_abort_all(iree_loop_run_ring_t* run_ring) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_loop_run_ring_abort_scope(run_ring, /*scope=*/NULL);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_loop_wait_list_t
//===----------------------------------------------------------------------===//

// Represents an operation in the loop wait list.
// Note that the storage may be reallocated at any time and all pointers must be
// external to the storage in order to remain valid.
typedef struct iree_loop_wait_op_t {
  union {
    iree_loop_callback_t callback;  // asserted at offset 0 above
    union {
      iree_loop_wait_until_params_t wait_until;
      iree_loop_wait_one_params_t wait_one;
      iree_loop_wait_multi_params_t wait_multi;
    } params;
  };
  iree_loop_command_t command;
  iree_loop_sync_scope_t* scope;
} iree_loop_wait_op_t;

// Dense list of pending wait operations.
// We don't care about the order here as we put them all into a wait set for
// multi-wait anyway. iree_wait_set_t should really be rewritten such that this
// is not required (custom data on registered handles, etc).
typedef iree_alignas(iree_max_align_t) struct iree_loop_wait_list_t {
  // System wait set used to perform multi-waits.
  iree_wait_set_t* wait_set;
  // Current storage capacity of |ops|.
  uint32_t capacity;
  // Current count of valid |ops|.
  uint32_t count;
  // Pending wait operations.
  iree_loop_wait_op_t ops[0];
} iree_loop_wait_list_t;

static iree_host_size_t iree_loop_wait_list_storage_size(
    iree_loop_sync_options_t options) {
  return sizeof(iree_loop_wait_list_t) +
         options.max_wait_count * sizeof(iree_loop_wait_op_t);
}

static bool iree_loop_wait_list_is_empty(iree_loop_wait_list_t* wait_list) {
  return wait_list->count == 0;
}

static iree_status_t iree_loop_wait_list_initialize(
    iree_loop_sync_options_t options, iree_allocator_t allocator,
    iree_loop_wait_list_t* out_wait_list) {
  IREE_TRACE_ZONE_BEGIN(z0);

  out_wait_list->capacity = (uint32_t)options.max_wait_count;
  out_wait_list->count = 0;

  iree_status_t status = iree_wait_set_allocate(
      options.max_wait_count, allocator, &out_wait_list->wait_set);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_loop_wait_list_deinitialize(iree_loop_wait_list_t* wait_list) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Expected abort to be called.
  IREE_ASSERT(iree_loop_wait_list_is_empty(wait_list));

  iree_wait_set_free(wait_list->wait_set);
  wait_list->wait_set = NULL;

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_loop_wait_list_register_wait_source(
    iree_loop_wait_list_t* wait_list, iree_wait_source_t* wait_source) {
  if (iree_wait_source_is_immediate(*wait_source)) {
    // Task has been neutered and is treated as an immediately resolved wait.
    return iree_ok_status();
  } else if (iree_wait_source_is_delay(*wait_source)) {
    // We can't easily support delays as registered wait sources; we need to be
    // able to snoop the tasks to find the earliest sleep time and can't easily
    // do that if we tried to put them in the wait set.
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "delays must come from wait-until ops");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();

  // Acquire a wait handle and insert it into the wait set.
  // We swap out the wait source with the handle so that we don't export it
  // again and can find it on wake.
  iree_wait_handle_t wait_handle = iree_wait_handle_immediate();
  iree_wait_handle_t* wait_handle_ptr =
      iree_wait_handle_from_source(wait_source);
  if (wait_handle_ptr) {
    // Already a wait handle - can directly insert it.
    wait_handle = *wait_handle_ptr;
  } else {
    iree_wait_primitive_t wait_primitive = iree_wait_primitive_immediate();
    status = iree_wait_source_export(*wait_source, IREE_WAIT_PRIMITIVE_TYPE_ANY,
                                     iree_immediate_timeout(), &wait_primitive);
    if (iree_status_is_ok(status)) {
      // Swap the wait handle with the exported handle so we can wake it later.
      // It'd be ideal if we retained the wait handle separate so that we could
      // still do fast queries for local wait sources.
      iree_wait_handle_wrap_primitive(wait_primitive.type, wait_primitive.value,
                                      &wait_handle);
      status = iree_wait_source_import(wait_primitive, wait_source);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_wait_set_insert(wait_list->wait_set, wait_handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_loop_wait_list_unregister_wait_source(
    iree_loop_wait_list_t* wait_list, iree_wait_source_t* wait_source) {
  if (iree_wait_source_is_immediate(*wait_source) ||
      iree_wait_source_is_delay(*wait_source)) {
    // Not registered or it's already been unregistered.
    return;
  }
  iree_wait_handle_t* wait_handle = iree_wait_handle_from_source(wait_source);
  if (wait_handle) {
    iree_wait_set_erase(wait_list->wait_set, *wait_handle);
  }
  *wait_source = iree_wait_source_immediate();
}

static void iree_loop_wait_list_unregister_wait_sources(
    iree_loop_wait_list_t* wait_list, iree_loop_wait_op_t* op) {
  switch (op->command) {
    case IREE_LOOP_COMMAND_WAIT_ONE:
      iree_loop_wait_list_unregister_wait_source(
          wait_list, &op->params.wait_one.wait_source);
      break;
    case IREE_LOOP_COMMAND_WAIT_ANY:
    case IREE_LOOP_COMMAND_WAIT_ALL:
      for (iree_host_size_t i = 0; i < op->params.wait_multi.count; ++i) {
        iree_loop_wait_list_unregister_wait_source(
            wait_list, &op->params.wait_multi.wait_sources[i]);
      }
      break;
    default:
    case IREE_LOOP_COMMAND_WAIT_UNTIL:
      break;
  }
}

static iree_status_t iree_loop_wait_list_insert(
    iree_loop_wait_list_t* wait_list, iree_loop_wait_op_t op) {
  if (wait_list->count + 1 >= wait_list->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "wait list capacity %u reached",
                            wait_list->capacity);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_PLOT_VALUE_I64("iree_loop_wait_depth", wait_list->count);

  uint32_t slot = wait_list->count++;
  wait_list->ops[slot] = op;

  iree_status_t status = iree_ok_status();
  switch (op.command) {
    case IREE_LOOP_COMMAND_WAIT_UNTIL:
      // No entry in the wait set; we just need it in the list in order to scan.
      break;
    case IREE_LOOP_COMMAND_WAIT_ONE: {
      status = iree_loop_wait_list_register_wait_source(
          wait_list, &op.params.wait_one.wait_source);
      break;
    }
    case IREE_LOOP_COMMAND_WAIT_ALL:
    case IREE_LOOP_COMMAND_WAIT_ANY: {
      for (iree_host_size_t i = 0;
           i < op.params.wait_multi.count && iree_status_is_ok(status); ++i) {
        status = iree_loop_wait_list_register_wait_source(
            wait_list, &op.params.wait_multi.wait_sources[i]);
      }
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unhandled wait list command");
      break;
  }

  if (iree_status_is_ok(status)) {
    ++op.scope->pending_count;
  }

  IREE_TRACE_PLOT_VALUE_I64("iree_loop_wait_depth", wait_list->count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_loop_wait_list_notify_wake(
    iree_loop_wait_list_t* wait_list, iree_loop_run_ring_t* run_ring,
    iree_host_size_t i, iree_status_t status) {
  IREE_TRACE_PLOT_VALUE_I64("iree_loop_wait_depth", wait_list->count);

  // Unregister all wait handles from the wait set.
  iree_loop_wait_list_unregister_wait_sources(wait_list, &wait_list->ops[i]);

  // Since we make no guarantees about the order of the lists we can just swap
  // with the last value. Note that we need to preserve the callback.
  iree_loop_sync_scope_t* scope = wait_list->ops[i].scope;
  --scope->pending_count;
  iree_loop_callback_t callback = wait_list->ops[i].callback;
  int tail_index = (int)wait_list->count - 1;
  if (tail_index > i) {
    memcpy(&wait_list->ops[i], &wait_list->ops[tail_index],
           sizeof(*wait_list->ops));
  }
  --wait_list->count;

  IREE_TRACE_PLOT_VALUE_I64("iree_loop_wait_depth", wait_list->count);

  // Enqueue the callback on the run ring - this ensures it gets sequenced with
  // other runnable work and keeps ordering easier to reason about.
  return iree_loop_run_ring_enqueue(
      run_ring, (iree_loop_run_op_t){
                    .command = IREE_LOOP_COMMAND_CALL,
                    .scope = scope,
                    .params =
                        {
                            .call =
                                {
                                    .callback = callback,
                                    // TODO(benvanik): elevate callback priority
                                    // to reduce latency?
                                    .priority = IREE_LOOP_PRIORITY_DEFAULT,
                                },
                        },
                    .status = status,
                });
}

// Returns DEFERRED if unresolved, OK if resolved, and an error otherwise.
// If resolved (successful or not) the caller must erase the wait.
static iree_status_t iree_loop_wait_list_scan_wait_until(
    iree_loop_wait_list_t* wait_list, iree_loop_wait_until_params_t* params,
    iree_time_t now_ns, iree_time_t* earliest_deadline_ns) {
  // Task is a delay until some future time; factor that in to our earliest
  // deadline so that we'll wait in the system until that time. If we wake
  // earlier because another wait resolved it's still possible for the delay
  // to have been reached before we get back to this check.
  if (params->deadline_ns <= now_ns + IREE_LOOP_SYNC_DELAY_SLOP_NS) {
    // Wait deadline reached.
    return iree_ok_status();
  } else {
    // Still waiting.
    *earliest_deadline_ns =
        iree_min(*earliest_deadline_ns, params->deadline_ns);
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }
}

// Returns DEFERRED if unresolved, OK if resolved, and an error otherwise.
// If resolved (successful or not) the caller must erase the wait.
static iree_status_t iree_loop_wait_list_scan_wait_one(
    iree_loop_wait_list_t* wait_list, iree_loop_wait_one_params_t* params,
    iree_time_t now_ns, iree_time_t* earliest_deadline_ns) {
  // Query the status.
  iree_status_code_t wait_status_code = IREE_STATUS_OK;
  IREE_RETURN_IF_ERROR(
      iree_wait_source_query(params->wait_source, &wait_status_code));

  if (wait_status_code != IREE_STATUS_OK) {
    if (params->deadline_ns <= now_ns) {
      // Deadline reached without having resolved.
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    } else {
      // Still waiting.
      *earliest_deadline_ns =
          iree_min(*earliest_deadline_ns, params->deadline_ns);
    }
  }

  return iree_status_from_code(wait_status_code);
}

// Returns DEFERRED if unresolved, OK if resolved, and an error otherwise.
// If resolved (successful or not) the caller must erase the wait.
static iree_status_t iree_loop_wait_list_scan_wait_any(
    iree_loop_wait_list_t* wait_list, iree_loop_wait_multi_params_t* params,
    iree_time_t now_ns, iree_time_t* earliest_deadline_ns) {
  for (iree_host_size_t i = 0; i < params->count; ++i) {
    iree_status_code_t wait_status_code = IREE_STATUS_OK;
    IREE_RETURN_IF_ERROR(
        iree_wait_source_query(params->wait_sources[i], &wait_status_code));
    if (wait_status_code == IREE_STATUS_OK) {
      return iree_ok_status();  // one resolved, wait-any satisfied
    }
  }
  if (params->deadline_ns <= now_ns) {
    // Deadline reached without having resolved any.
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  } else {
    // Still waiting.
    *earliest_deadline_ns =
        iree_min(*earliest_deadline_ns, params->deadline_ns);
  }
  return iree_status_from_code(IREE_STATUS_DEFERRED);  // none resolved
}

// Returns DEFERRED if unresolved, OK if resolved, and an error otherwise.
// If resolved (successful or not) the caller must erase the wait.
static iree_status_t iree_loop_wait_list_scan_wait_all(
    iree_loop_wait_list_t* wait_list, iree_loop_wait_multi_params_t* params,
    iree_time_t now_ns, iree_time_t* earliest_deadline_ns) {
  bool any_unresolved = false;
  for (iree_host_size_t i = 0; i < params->count; ++i) {
    if (iree_wait_source_is_immediate(params->wait_sources[i])) continue;
    iree_status_code_t wait_status_code = IREE_STATUS_OK;
    IREE_RETURN_IF_ERROR(
        iree_wait_source_query(params->wait_sources[i], &wait_status_code));
    if (wait_status_code == IREE_STATUS_OK) {
      // Wait resolved; remove it from the wait set so that we don't wait on it
      // again. We do this by neutering the handle.
      iree_wait_handle_t* wait_handle =
          iree_wait_handle_from_source(&params->wait_sources[i]);
      if (wait_handle) {
        iree_wait_set_erase(wait_list->wait_set, *wait_handle);
      }
      params->wait_sources[i] = iree_wait_source_immediate();
    } else {
      // Wait not yet resolved.
      if (params->deadline_ns <= now_ns) {
        // Deadline reached without having resolved all.
        return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      } else {
        // Still waiting.
        *earliest_deadline_ns =
            iree_min(*earliest_deadline_ns, params->deadline_ns);
        any_unresolved = true;
      }
    }
  }
  return any_unresolved ? iree_status_from_code(IREE_STATUS_DEFERRED)
                        : iree_ok_status();
}

static void iree_loop_wait_list_handle_wake(iree_loop_wait_list_t* wait_list,
                                            iree_loop_run_ring_t* run_ring,
                                            iree_wait_handle_t wake_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): scan the list. We need a way to map wake_handle back to
  // the zero or more tasks that match it but don't currently store the
  // handle. Ideally we'd have the wait set tell us precisely which things
  // woke - possibly by having a bitmap of original insertions that match the
  // handle - but for now we just eat the extra query syscall.
  int woken_tasks = 0;

  (void)woken_tasks;
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, woken_tasks);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_loop_wait_list_scan(
    iree_loop_wait_list_t* wait_list, iree_loop_run_ring_t* run_ring,
    iree_time_t* out_earliest_deadline_ns) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_earliest_deadline_ns = IREE_TIME_INFINITE_FUTURE;

  iree_time_t now_ns = iree_time_now();
  iree_status_t scan_status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < wait_list->count && iree_status_is_ok(scan_status); ++i) {
    iree_status_t wait_status = iree_ok_status();
    switch (wait_list->ops[i].command) {
      case IREE_LOOP_COMMAND_WAIT_UNTIL:
        wait_status = iree_loop_wait_list_scan_wait_until(
            wait_list, &wait_list->ops[i].params.wait_until, now_ns,
            out_earliest_deadline_ns);
        break;
      case IREE_LOOP_COMMAND_WAIT_ONE:
        wait_status = iree_loop_wait_list_scan_wait_one(
            wait_list, &wait_list->ops[i].params.wait_one, now_ns,
            out_earliest_deadline_ns);
        break;
      case IREE_LOOP_COMMAND_WAIT_ANY:
        wait_status = iree_loop_wait_list_scan_wait_any(
            wait_list, &wait_list->ops[i].params.wait_multi, now_ns,
            out_earliest_deadline_ns);
        break;
      case IREE_LOOP_COMMAND_WAIT_ALL:
        wait_status = iree_loop_wait_list_scan_wait_all(
            wait_list, &wait_list->ops[i].params.wait_multi, now_ns,
            out_earliest_deadline_ns);
        break;
    }
    if (!iree_status_is_deferred(wait_status)) {
      // Wait completed/failed - erase from the wait set and op list.
      scan_status =
          iree_loop_wait_list_notify_wake(wait_list, run_ring, i, wait_status);
      --i;  // item i removed

      // Don't commit the wait if we woke something; we want the callback to be
      // issued ASAP and will let the main loop pump again to actually wait if
      // needed.
      *out_earliest_deadline_ns = IREE_TIME_INFINITE_PAST;
    }
  }

  IREE_TRACE_PLOT_VALUE_I64("iree_loop_wait_depth", wait_list->count);
  IREE_TRACE_ZONE_END(z0);
  return scan_status;
}

static iree_status_t iree_loop_wait_list_commit(
    iree_loop_wait_list_t* wait_list, iree_loop_run_ring_t* run_ring,
    iree_time_t deadline_ns) {
  if (iree_wait_set_is_empty(wait_list->wait_set) == 0) {
    // No wait handles; this is a sleep.
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_loop_wait_list_commit_sleep");
    iree_status_t status =
        iree_wait_until(deadline_ns)
            ? iree_ok_status()
            : iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Real system wait.
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)wait_list->count);

  // Enter the system wait API.
  iree_wait_handle_t wake_handle = iree_wait_handle_immediate();
  iree_status_t status =
      iree_wait_any(wait_list->wait_set, deadline_ns, &wake_handle);
  if (iree_status_is_ok(status)) {
    // One or more waiters is ready. We don't support multi-wake right now so
    // we'll just take the one we got back and try again.
    //
    // To avoid extra syscalls we scan the list and mark whatever tasks were
    // using the handle the wait set reported waking as completed. On the next
    // scan they'll be retired immediately. Ideally we'd have the wait set be
    // able to tell us this precise list.
    if (iree_wait_handle_is_immediate(wake_handle)) {
      // No-op wait - ignore.
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "nop");
    } else {
      // Route to zero or more tasks using this handle.
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "task(s)");
      iree_loop_wait_list_handle_wake(wait_list, run_ring, wake_handle);
    }
  } else if (iree_status_is_deadline_exceeded(status)) {
    // Indicates nothing was woken within the deadline. We gracefully bail here
    // and let the scan check for per-op deadline exceeded events or delay
    // completion.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "deadline exceeded");
  } else {
    // (Spurious?) error during wait.
    // TODO(#4026): propagate failure to all scopes involved.
    // Failures during waits are serious: ignoring them could lead to live-lock
    // as tasks further in the pipeline expect them to have completed or - even
    // worse - user code/other processes/drivers/etc may expect them to
    // complete.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "failure");
    IREE_ASSERT_TRUE(iree_status_is_ok(status));
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Aborts all waits that are part of |scope|.
// A NULL |scope| indicates all work from all scopes should be aborted.
static void iree_loop_wait_list_abort_scope(iree_loop_wait_list_t* wait_list,
                                            iree_loop_sync_scope_t* scope) {
  if (!wait_list->count) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_PLOT_VALUE_I64("iree_loop_wait_depth", wait_list->count);

  // Issue the completion callback of each op to notify it of the abort.
  // To prevent enqueuing more work while aborting we pass in a NULL loop.
  // We can't do anything with the errors so we ignore them.
  for (iree_host_size_t i = 0; i < wait_list->count; ++i) {
    if (scope && wait_list->ops[i].scope != scope) continue;

    --wait_list->ops[i].scope->pending_count;
    iree_loop_callback_t callback = wait_list->ops[i].callback;
    iree_status_t status = callback.fn(callback.user_data, iree_loop_null(),
                                       iree_make_status(IREE_STATUS_ABORTED));
    iree_status_ignore(status);

    // Since we make no guarantees about the order of the lists we can just swap
    // with the last value.
    int tail_index = (int)wait_list->count - 1;
    if (tail_index > i) {
      memcpy(&wait_list->ops[i], &wait_list->ops[tail_index],
             sizeof(*wait_list->ops));
    }
    --wait_list->count;
    --i;
  }

  IREE_TRACE_PLOT_VALUE_I64("iree_loop_wait_depth", wait_list->count);
  IREE_TRACE_ZONE_END(z0);
}

// Aborts all waits from all scopes.
static void iree_loop_wait_list_abort_all(iree_loop_wait_list_t* wait_list) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_loop_wait_list_abort_scope(wait_list, /*scope=*/NULL);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_loop_sync_scope_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_loop_sync_scope_initialize(
    iree_loop_sync_t* loop_sync, iree_loop_sync_error_fn_t error_fn,
    void* error_user_data, iree_loop_sync_scope_t* out_scope) {
  memset(out_scope, 0, sizeof(*out_scope));
  out_scope->loop_sync = loop_sync;
  out_scope->pending_count = 0;
  out_scope->error_fn = error_fn;
  out_scope->error_user_data = error_user_data;
}

IREE_API_EXPORT void iree_loop_sync_scope_deinitialize(
    iree_loop_sync_scope_t* scope) {
  IREE_ASSERT_ARGUMENT(scope);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (scope->loop_sync) {
    iree_loop_sync_abort_scope(scope->loop_sync, scope);
  }

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_loop_sync_t
//===----------------------------------------------------------------------===//

typedef struct iree_loop_sync_t {
  iree_allocator_t allocator;

  iree_loop_run_ring_t* run_ring;
  iree_loop_wait_list_t* wait_list;

  // Trailing data:
  // + iree_loop_run_ring_storage_size
  // + iree_loop_wait_list_storage_size
} iree_loop_sync_t;

IREE_API_EXPORT iree_status_t iree_loop_sync_allocate(
    iree_loop_sync_options_t options, iree_allocator_t allocator,
    iree_loop_sync_t** out_loop_sync) {
  IREE_ASSERT_ARGUMENT(out_loop_sync);

  // The run queue must be a power of two due to the ringbuffer masking
  // technique we use.
  options.max_queue_depth =
      iree_math_round_up_to_pow2_u32((uint32_t)options.max_queue_depth);
  if (options.max_queue_depth > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue depth exceeds maximum");
  }

  // Wait sets also have a handle limit but we may want to allow more
  // outstanding wait operations even if we can't wait on them all
  // simultaneously.
  if (IREE_UNLIKELY(options.max_wait_count > UINT16_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wait list depth exceeds maximum");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t loop_sync_size =
      iree_host_align(sizeof(iree_loop_sync_t), iree_max_align_t);
  const iree_host_size_t run_ring_size = iree_host_align(
      iree_loop_run_ring_storage_size(options), iree_max_align_t);
  const iree_host_size_t wait_list_size = iree_host_align(
      iree_loop_wait_list_storage_size(options), iree_max_align_t);
  const iree_host_size_t total_storage_size =
      loop_sync_size + run_ring_size + wait_list_size;

  uint8_t* storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, total_storage_size, (void**)&storage));
  iree_loop_sync_t* loop_sync = (iree_loop_sync_t*)storage;
  loop_sync->allocator = allocator;
  loop_sync->run_ring = (iree_loop_run_ring_t*)(storage + loop_sync_size);
  loop_sync->wait_list =
      (iree_loop_wait_list_t*)(storage + loop_sync_size + run_ring_size);

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    iree_loop_run_ring_initialize(options, loop_sync->run_ring);
  }
  if (iree_status_is_ok(status)) {
    status = iree_loop_wait_list_initialize(options, allocator,
                                            loop_sync->wait_list);
  }

  if (iree_status_is_ok(status)) {
    *out_loop_sync = loop_sync;
  } else {
    iree_loop_sync_free(loop_sync);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_loop_sync_free(iree_loop_sync_t* loop_sync) {
  IREE_ASSERT_ARGUMENT(loop_sync);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = loop_sync->allocator;

  // Abort all pending operations.
  // This will issue callbacks for each operation that was aborted directly
  // with IREE_STATUS_ABORTED.
  // To ensure we don't enqueue more work while aborting we NULL out the lists.
  iree_loop_run_ring_t* run_ring = loop_sync->run_ring;
  iree_loop_wait_list_t* wait_list = loop_sync->wait_list;
  loop_sync->run_ring = NULL;
  loop_sync->wait_list = NULL;
  iree_loop_wait_list_abort_all(wait_list);
  iree_loop_run_ring_abort_all(run_ring);

  // After all operations are cleared we can release the data structures.
  iree_loop_run_ring_deinitialize(run_ring);
  iree_loop_wait_list_deinitialize(wait_list);
  iree_allocator_free(allocator, loop_sync);

  IREE_TRACE_ZONE_END(z0);
}

// Aborts all operations in the loop attributed to |scope|.
static void iree_loop_sync_abort_scope(iree_loop_sync_t* loop_sync,
                                       iree_loop_sync_scope_t* scope) {
  iree_loop_wait_list_abort_scope(loop_sync->wait_list, scope);
  iree_loop_run_ring_abort_scope(loop_sync->run_ring, scope);
}

// Emits |status| to the given |loop| scope and aborts associated operations.
static void iree_loop_sync_emit_error(iree_loop_t loop, iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(status)));

  iree_loop_sync_scope_t* scope = (iree_loop_sync_scope_t*)loop.self;
  iree_loop_sync_t* loop_sync = scope->loop_sync;

  if (scope->error_fn) {
    scope->error_fn(scope->error_user_data, status);
  } else {
    iree_status_ignore(status);
  }

  iree_loop_sync_abort_scope(loop_sync, scope);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_loop_sync_run_call(iree_loop_sync_t* loop_sync,
                                    iree_loop_t loop,
                                    const iree_loop_call_params_t params,
                                    iree_status_t op_status) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      params.callback.fn(params.callback.user_data, loop, op_status);
  if (!iree_status_is_ok(status)) {
    iree_loop_sync_emit_error(loop, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_loop_sync_run_dispatch(
    iree_loop_sync_t* loop_sync, iree_loop_t loop,
    const iree_loop_dispatch_params_t params) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // We run all workgroups before issuing the completion callback.
  // If any workgroup fails we exit early and pass the failing status back to
  // the completion handler exactly once.
  uint32_t workgroup_count_x = params.workgroup_count_xyz[0];
  uint32_t workgroup_count_y = params.workgroup_count_xyz[1];
  uint32_t workgroup_count_z = params.workgroup_count_xyz[2];
  iree_status_t workgroup_status = iree_ok_status();
  for (uint32_t z = 0; z < workgroup_count_z; ++z) {
    for (uint32_t y = 0; y < workgroup_count_y; ++y) {
      for (uint32_t x = 0; x < workgroup_count_x; ++x) {
        workgroup_status =
            params.workgroup_fn(params.callback.user_data, loop, x, y, z);
        if (!iree_status_is_ok(workgroup_status)) goto workgroup_failed;
      }
    }
  }
workgroup_failed:

  // Fire the completion callback with either success or the first error hit by
  // a workgroup.
  status =
      params.callback.fn(params.callback.user_data, loop, workgroup_status);
  if (!iree_status_is_ok(status)) {
    iree_loop_sync_emit_error(loop, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

// Drains work from the loop until all work in |scope| has completed.
// A NULL |scope| indicates all work from all scopes should be drained.
static iree_status_t iree_loop_sync_drain_scope(iree_loop_sync_t* loop_sync,
                                                iree_loop_sync_scope_t* scope,
                                                iree_time_t deadline_ns) {
  IREE_TRACE_ZONE_BEGIN(z0);

  do {
    // If we are draining a particular scope we can bail whenever there's no
    // more work remaining.
    if (scope && !scope->pending_count) break;

    // Run an op from the runnable queue.
    // We dequeue operations here so that re-entrant enqueuing works.
    // We only want to run one op at a time before checking our deadline so that
    // we don't get into infinite loops or exceed the deadline (too much).
    iree_loop_run_op_t run_op;
    if (iree_loop_run_ring_dequeue(loop_sync->run_ring, &run_op)) {
      iree_loop_t loop = {
          .self = run_op.scope,
          .ctl = iree_loop_sync_ctl,
      };
      switch (run_op.command) {
        case IREE_LOOP_COMMAND_CALL:
          iree_loop_sync_run_call(loop_sync, loop, run_op.params.call,
                                  run_op.status);
          break;
        case IREE_LOOP_COMMAND_DISPATCH:
          iree_loop_sync_run_dispatch(loop_sync, loop, run_op.params.dispatch);
          break;
      }
      continue;  // loop back around only if under the deadline
    }

    // -- if here then the run ring is currently empty --

    // If there are no pending waits then the drain has completed.
    if (iree_loop_wait_list_is_empty(loop_sync->wait_list)) {
      break;
    }

    // Scan the wait list and check for resolved ops.
    // If there are any waiting ops the next earliest timeout is returned. An
    // immediate timeout indicates that there's work in the run ring and we
    // shouldn't perform a wait operation this go around the loop.
    iree_time_t earliest_deadline_ns = IREE_TIME_INFINITE_FUTURE;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_loop_wait_list_scan(loop_sync->wait_list, loop_sync->run_ring,
                                     &earliest_deadline_ns));
    if (earliest_deadline_ns != IREE_TIME_INFINITE_PAST) {
      // Commit the wait operation, waiting up until the minimum of the user
      // specified and wait list derived values.
      iree_time_t wait_deadline_ns = earliest_deadline_ns < deadline_ns
                                         ? earliest_deadline_ns
                                         : deadline_ns;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_loop_wait_list_commit(
                  loop_sync->wait_list, loop_sync->run_ring, wait_deadline_ns));
    }
  } while (iree_time_now() < deadline_ns);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_loop_sync_wait_idle(iree_loop_sync_t* loop_sync, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(loop_sync);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_status_t status =
      iree_loop_sync_drain_scope(loop_sync, /*scope=*/NULL, deadline_ns);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Control function for the synchronous loop.
// |self| must be an iree_loop_sync_scope_t.
IREE_API_EXPORT iree_status_t iree_loop_sync_ctl(void* self,
                                                 iree_loop_command_t command,
                                                 const void* params,
                                                 void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(self);
  iree_loop_sync_scope_t* scope = (iree_loop_sync_scope_t*)self;
  iree_loop_sync_t* loop_sync = scope->loop_sync;

  if (IREE_UNLIKELY(!loop_sync->run_ring)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "new work cannot be enqueued while the loop is shutting down");
  }

  // NOTE: we return immediately to make this all (hopefully) tail calls.
  switch (command) {
    case IREE_LOOP_COMMAND_CALL:
      return iree_loop_run_ring_enqueue(
          loop_sync->run_ring,
          (iree_loop_run_op_t){
              .command = command,
              .scope = scope,
              .params =
                  {
                      .call = *(const iree_loop_call_params_t*)params,
                  },
          });
    case IREE_LOOP_COMMAND_DISPATCH:
      return iree_loop_run_ring_enqueue(
          loop_sync->run_ring,
          (iree_loop_run_op_t){
              .command = command,
              .scope = scope,
              .params =
                  {
                      .dispatch = *(const iree_loop_dispatch_params_t*)params,
                  },
          });
    case IREE_LOOP_COMMAND_WAIT_UNTIL:
      return iree_loop_wait_list_insert(
          loop_sync->wait_list,
          (iree_loop_wait_op_t){
              .command = command,
              .scope = scope,
              .params =
                  {
                      .wait_until =
                          *(const iree_loop_wait_until_params_t*)params,
                  },
          });
    case IREE_LOOP_COMMAND_WAIT_ONE:
      return iree_loop_wait_list_insert(
          loop_sync->wait_list,
          (iree_loop_wait_op_t){
              .command = command,
              .scope = scope,
              .params =
                  {
                      .wait_one = *(const iree_loop_wait_one_params_t*)params,
                  },
          });
    case IREE_LOOP_COMMAND_WAIT_ALL:
    case IREE_LOOP_COMMAND_WAIT_ANY:
      return iree_loop_wait_list_insert(
          loop_sync->wait_list,
          (iree_loop_wait_op_t){
              .command = command,
              .scope = scope,
              .params =
                  {
                      .wait_multi =
                          *(const iree_loop_wait_multi_params_t*)params,
                  },
          });
    case IREE_LOOP_COMMAND_DRAIN:
      return iree_loop_sync_drain_scope(
          loop_sync, scope,
          ((const iree_loop_drain_params_t*)params)->deadline_ns);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented loop command");
  }
}
