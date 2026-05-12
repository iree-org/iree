// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_queue.h"

#include <stddef.h>
#include <string.h>

#include "iree/async/file.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/async/span.h"
#include "iree/base/threading/processor.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/local_task/block_builder.h"
#include "iree/hal/drivers/local_task/block_command_buffer.h"
#include "iree/hal/drivers/local_task/block_command_ops.h"
#include "iree/hal/drivers/local_task/block_processor.h"
#include "iree/hal/local/local_executable.h"
#include "iree/hal/local/transient_buffer.h"
#include "iree/hal/utils/resource_set.h"

#if !defined(NDEBUG)
static void iree_hal_task_queue_debug_record_submit(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  operation->debug.ordinal =
      (uint64_t)iree_atomic_fetch_add(&queue->debug.next_operation_ordinal, 1,
                                      iree_memory_order_relaxed) +
      1;
  iree_atomic_fetch_add(&queue->debug.submitted_operation_count, 1,
                        iree_memory_order_relaxed);
  iree_atomic_store(&queue->debug.last_submitted_operation_ordinal,
                    (int64_t)operation->debug.ordinal,
                    iree_memory_order_relaxed);
}

static void iree_hal_task_queue_debug_record_complete(
    iree_hal_task_queue_t* queue, const iree_hal_task_queue_op_t* operation) {
  iree_atomic_fetch_add(&queue->debug.completed_operation_count, 1,
                        iree_memory_order_relaxed);
  iree_atomic_store(&queue->debug.last_completed_operation_ordinal,
                    (int64_t)operation->debug.ordinal,
                    iree_memory_order_relaxed);
}

static void iree_hal_task_queue_debug_record_fail(
    iree_hal_task_queue_t* queue, const iree_hal_task_queue_op_t* operation) {
  iree_atomic_fetch_add(&queue->debug.failed_operation_count, 1,
                        iree_memory_order_relaxed);
  iree_atomic_store(&queue->debug.last_failed_operation_ordinal,
                    (int64_t)operation->debug.ordinal,
                    iree_memory_order_relaxed);
}

static void iree_hal_task_queue_debug_record_destroy(
    iree_hal_task_queue_t* queue, const iree_hal_task_queue_op_t* operation,
    iree_status_t failure_status) {
  if (iree_status_is_ok(failure_status)) {
    iree_atomic_fetch_add(&queue->debug.destroyed_ok_operation_count, 1,
                          iree_memory_order_relaxed);
  } else {
    iree_atomic_fetch_add(&queue->debug.destroyed_failed_operation_count, 1,
                          iree_memory_order_relaxed);
  }
  iree_atomic_store(&queue->debug.last_destroyed_operation_ordinal,
                    (int64_t)operation->debug.ordinal,
                    iree_memory_order_relaxed);
}
#else
static void iree_hal_task_queue_debug_record_submit(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  (void)queue;
  (void)operation;
}

static void iree_hal_task_queue_debug_record_complete(
    iree_hal_task_queue_t* queue, const iree_hal_task_queue_op_t* operation) {
  (void)queue;
  (void)operation;
}

static void iree_hal_task_queue_debug_record_fail(
    iree_hal_task_queue_t* queue, const iree_hal_task_queue_op_t* operation) {
  (void)queue;
  (void)operation;
}

static void iree_hal_task_queue_debug_record_destroy(
    iree_hal_task_queue_t* queue, const iree_hal_task_queue_op_t* operation,
    iree_status_t failure_status) {
  (void)queue;
  (void)operation;
  (void)failure_status;
}
#endif  // !defined(NDEBUG)

// Each submission creates an arena-allocated operation that flows through:
//
//  submit()
//    |
//    |   +--------------+
//    +-> | +--------------+  Unsatisfied waits register async semaphore
//    .   +-|  sema waits  |  timepoints. Same-queue semaphores are typically
//    .     +--------------+  already satisfied (fast path skips entirely).
//    .        | | | | |
//    +--------+-+-+-+-+
//    |
//    v
//  +--------------------+
//  |   ready list       |  Operations with all waits satisfied. MPSC slist.
//  +--------------------+
//    |
//    v
//  +--------------------+    Queue process (wake_budget == 1) pops operations
//  and |   queue drain()    |    handles them by type: command buffers are
//  issued
//  +--------------------+    as separate compute processes; barriers and host
//    |                       calls are handled inline.
//    v
//  +--------------------+    CB process completion (or inline for barriers/
//  |   completion       |    host calls): signal semaphores, advance frontier,
//  +--------------------+    end scope, free arena.

//===----------------------------------------------------------------------===//
// Profiling helpers
//===----------------------------------------------------------------------===//

// Optional trailing profiling state for one queue operation. This is allocated
// from the operation arena only while a local CPU profile session is active so
// the disabled path does not grow task_queue_op_t or its memset footprint.
typedef struct iree_hal_task_queue_profile_operation_t {
  // Recorder captured at operation allocation time.
  iree_hal_local_profile_recorder_t* recorder;

  // Queue identity captured at operation allocation time.
  iree_hal_local_profile_queue_scope_t scope;

  // Queue operation type shared by emitted records.
  iree_hal_profile_queue_event_type_t type;

  // Queue event flags captured for emitted queue records.
  iree_hal_profile_queue_event_flags_t queue_flags;

  // Host execution flags captured for emitted execution records.
  iree_hal_profile_host_execution_event_flags_t host_flags;

  // Strategy used for wait dependencies on this operation.
  iree_hal_profile_queue_dependency_strategy_t dependency_strategy;

  // Submission id assigned to this queue operation, or 0 when absent.
  uint64_t submission_id;

  // Command-buffer id assigned to this queue operation, or 0 when absent.
  uint64_t command_buffer_id;

  // Executable id assigned to this operation, or 0 when absent.
  uint64_t executable_id;

  // Allocation id assigned to this operation, or 0 when absent.
  uint64_t allocation_id;

  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;

  // Number of encoded payload operations represented by this queue operation.
  uint32_t operation_count;

  // Number of wait semaphores supplied to the queue operation.
  uint32_t wait_count;

  // Number of signal semaphores supplied to the queue operation.
  uint32_t signal_count;

  // Executable export ordinal for dispatch-like spans, or UINT32_MAX.
  uint32_t export_ordinal;

  // Workgroup counts submitted for dispatch-like spans.
  uint32_t workgroup_count[3];

  // Workgroup sizes submitted for dispatch-like spans.
  uint32_t workgroup_size[3];

  // Host timestamp captured at queue submission.
  iree_time_t submit_host_time_ns;

  // Host timestamp captured when the operation became ready to execute.
  iree_time_t ready_host_time_ns;

  // First host execution start timestamp, or 0 when no span began.
  iree_atomic_int64_t start_host_time_ns;

  // True after the queue event append has been attempted.
  bool queue_event_recorded;
} iree_hal_task_queue_profile_operation_t;

static iree_host_size_t iree_hal_task_queue_profile_operation_offset(void) {
  return iree_host_align(sizeof(iree_hal_task_queue_op_t),
                         iree_alignof(iree_hal_task_queue_profile_operation_t));
}

static uint32_t iree_hal_task_queue_profile_count(iree_host_size_t count) {
  return count > UINT32_MAX ? UINT32_MAX : (uint32_t)count;
}

static iree_hal_profile_queue_event_type_t iree_hal_task_queue_profile_type(
    iree_hal_task_queue_op_type_t type) {
  switch (type) {
    case IREE_HAL_TASK_QUEUE_OP_COMMANDS:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
    case IREE_HAL_TASK_QUEUE_OP_BARRIER:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER;
    case IREE_HAL_TASK_QUEUE_OP_HOST_CALL:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL;
    case IREE_HAL_TASK_QUEUE_OP_ALLOCA:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA;
    case IREE_HAL_TASK_QUEUE_OP_DEALLOCA:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA;
    case IREE_HAL_TASK_QUEUE_OP_READ:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ;
    case IREE_HAL_TASK_QUEUE_OP_WRITE:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE;
    case IREE_HAL_TASK_QUEUE_OP_FILL:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL;
    case IREE_HAL_TASK_QUEUE_OP_COPY:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY;
    case IREE_HAL_TASK_QUEUE_OP_UPDATE:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE;
    case IREE_HAL_TASK_QUEUE_OP_DISPATCH:
      return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  }
  return IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
}

static uint32_t iree_hal_task_queue_profile_operation_count(
    iree_hal_task_queue_op_type_t type) {
  return type == IREE_HAL_TASK_QUEUE_OP_BARRIER ? 0u : 1u;
}

static iree_hal_task_queue_profile_operation_t*
iree_hal_task_queue_profile_operation(iree_hal_task_queue_op_t* operation) {
  if (IREE_LIKELY(!operation->queue->profile_recorder)) return NULL;
  const iree_host_size_t profile_operation_offset =
      iree_hal_task_queue_profile_operation_offset();
  return (iree_hal_task_queue_profile_operation_t*)((uint8_t*)operation +
                                                    profile_operation_offset);
}

static void iree_hal_task_queue_profile_operation_initialize(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_type_t type,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_hal_task_queue_profile_operation_t* profile_operation) {
  memset(profile_operation, 0, sizeof(*profile_operation));
  profile_operation->recorder = queue->profile_recorder;
  profile_operation->scope = queue->profile_scope;
  profile_operation->type = iree_hal_task_queue_profile_type(type);
  profile_operation->operation_count =
      iree_hal_task_queue_profile_operation_count(type);
  profile_operation->signal_count =
      iree_hal_task_queue_profile_count(signal_semaphores->count);
  profile_operation->export_ordinal = UINT32_MAX;
  if (iree_hal_local_profile_recorder_is_enabled(
          profile_operation->recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    profile_operation->submit_host_time_ns = iree_time_now();
  }
  iree_atomic_store(&profile_operation->start_host_time_ns, 0,
                    iree_memory_order_relaxed);
  if (queue->profile_submission_counter) {
    profile_operation->submission_id = (uint64_t)iree_atomic_fetch_add(
        queue->profile_submission_counter, 1, iree_memory_order_relaxed);
  }
}

static void iree_hal_task_queue_profile_set_waits(
    iree_hal_task_queue_op_t* operation, iree_host_size_t original_wait_count,
    iree_host_size_t unsatisfied_wait_count) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return;
  profile_operation->wait_count =
      iree_hal_task_queue_profile_count(original_wait_count);
  if (original_wait_count == 0) {
    profile_operation->dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE;
  } else if (unsatisfied_wait_count == 0) {
    profile_operation->dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE;
  } else {
    profile_operation->dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
    profile_operation->queue_flags |=
        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
  }
}

static void iree_hal_task_queue_profile_force_software_defer(
    iree_hal_task_queue_op_t* operation) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return;
  profile_operation->dependency_strategy =
      IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER;
  profile_operation->queue_flags |=
      IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED;
}

static void iree_hal_task_queue_profile_set_payload(
    iree_hal_task_queue_op_t* operation, uint64_t payload_length) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return;
  profile_operation->payload_length = payload_length;
}

static void iree_hal_task_queue_profile_set_transient_buffer(
    iree_hal_task_queue_op_t* operation, iree_hal_buffer_t* transient_buffer) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return;
  profile_operation->allocation_id =
      iree_hal_local_transient_buffer_profile_id(transient_buffer);
  profile_operation->payload_length =
      iree_hal_buffer_byte_length(transient_buffer);
}

static void iree_hal_task_queue_profile_populate_memory_event_pool_stats(
    iree_hal_pool_t* pool, iree_hal_profile_memory_event_t* event) {
  if (!pool) return;
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  event->flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_STATS;
  event->pool_bytes_reserved = stats.bytes_reserved;
  event->pool_bytes_free = stats.bytes_free;
  event->pool_bytes_committed = stats.bytes_committed;
  event->pool_budget_limit = stats.budget_limit;
  event->pool_reservation_count = stats.reservation_count;
  event->pool_slab_count = stats.slab_count;
}

static void iree_hal_task_queue_profile_record_memory_event(
    iree_hal_task_queue_op_t* operation,
    iree_hal_profile_memory_event_type_t type,
    iree_hal_profile_memory_event_flags_t flags, uint32_t result,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    uint64_t frontier_entry_count) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation || !iree_hal_local_profile_recorder_is_enabled(
                                profile_operation->recorder,
                                IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS)) {
    return;
  }

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = type;
  event.flags = flags;
  event.result = result;
  event.allocation_id = profile_operation->allocation_id;
  event.pool_id = (uint64_t)(uintptr_t)pool;
  event.submission_id = profile_operation->submission_id;
  event.physical_device_ordinal =
      profile_operation->scope.physical_device_ordinal;
  event.queue_ordinal = profile_operation->scope.queue_ordinal;
  event.frontier_entry_count =
      (uint32_t)iree_min(frontier_entry_count, (uint64_t)UINT32_MAX);
  event.memory_type = params.type;
  event.buffer_usage = params.usage;
  event.length = profile_operation->payload_length;
  event.alignment = params.min_alignment ? params.min_alignment : 1;
  if (reservation) {
    event.flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION;
    event.backing_id = reservation->block_handle;
    event.offset = reservation->offset;
    if (type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA &&
        type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA) {
      event.length = reservation->length;
    }
  }
  iree_hal_task_queue_profile_populate_memory_event_pool_stats(pool, &event);
  iree_hal_local_profile_recorder_append_memory_event(
      profile_operation->recorder, &event, /*out_event_id=*/NULL);
}

static void iree_hal_task_queue_profile_add_host_flags(
    iree_hal_task_queue_op_t* operation,
    iree_hal_profile_host_execution_event_flags_t host_flags) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return;
  profile_operation->host_flags |= host_flags;
}

static iree_status_t iree_hal_task_queue_profile_set_dispatch(
    iree_hal_task_queue_op_t* operation, iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_record_executable(
      profile_operation->recorder, executable));
  iree_hal_local_executable_t* local_executable =
      iree_hal_local_executable_cast(executable);
  profile_operation->executable_id =
      iree_hal_local_executable_profile_id(local_executable);
  profile_operation->export_ordinal = export_ordinal;
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    profile_operation->host_flags |=
        IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_INDIRECT_PARAMETERS;
  } else {
    memcpy(profile_operation->workgroup_count, config.workgroup_count,
           sizeof(profile_operation->workgroup_count));
  }
  profile_operation->workgroup_size[0] =
      config.workgroup_size[0] ? config.workgroup_size[0] : 1;
  profile_operation->workgroup_size[1] =
      config.workgroup_size[1] ? config.workgroup_size[1] : 1;
  profile_operation->workgroup_size[2] =
      config.workgroup_size[2] ? config.workgroup_size[2] : 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_task_queue_profile_record_command_buffer(
    iree_hal_task_queue_op_t* operation,
    iree_hal_command_buffer_t* command_buffer) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return iree_ok_status();
  if (!iree_hal_local_profile_recorder_is_enabled(
          profile_operation->recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA)) {
    return iree_ok_status();
  }

  iree_hal_profile_command_buffer_record_t command_buffer_record;
  const iree_hal_profile_command_operation_record_t* operations = NULL;
  iree_host_size_t operation_count = 0;
  iree_hal_block_command_buffer_profile_metadata(
      command_buffer, profile_operation->scope.physical_device_ordinal,
      &command_buffer_record, &operations, &operation_count);
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_record_command_buffer(
      profile_operation->recorder, &command_buffer_record, operation_count,
      operations));

  iree_host_size_t executable_count = 0;
  iree_hal_executable_t* const* executables =
      iree_hal_block_command_buffer_profile_executables(command_buffer,
                                                        &executable_count);
  for (iree_host_size_t i = 0; i < executable_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_record_executable(
        profile_operation->recorder, executables[i]));
  }
  return iree_ok_status();
}

static void iree_hal_task_queue_profile_set_command_buffer(
    iree_hal_task_queue_op_t* operation,
    iree_hal_command_buffer_t* command_buffer) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return;
  profile_operation->command_buffer_id =
      command_buffer ? iree_hal_command_buffer_profile_id(command_buffer) : 0;
  if (command_buffer && iree_hal_block_command_buffer_isa(command_buffer)) {
    profile_operation->operation_count = iree_hal_task_queue_profile_count(
        iree_hal_block_command_buffer_profile_operation_count(command_buffer));
  }
}

static void iree_hal_task_queue_profile_record_queue_event(
    iree_hal_task_queue_op_t* operation, iree_time_t ready_host_time_ns) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation || profile_operation->queue_event_recorded) return;
  if (!iree_hal_local_profile_recorder_is_enabled(
          profile_operation->recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    return;
  }
  profile_operation->queue_event_recorded = true;
  profile_operation->ready_host_time_ns = ready_host_time_ns;

  iree_hal_local_profile_queue_event_info_t event_info =
      iree_hal_local_profile_queue_event_info_default();
  event_info.type = profile_operation->type;
  event_info.flags = profile_operation->queue_flags;
  event_info.dependency_strategy = profile_operation->dependency_strategy;
  event_info.scope = profile_operation->scope;
  event_info.host_time_ns = profile_operation->submit_host_time_ns;
  event_info.ready_host_time_ns = profile_operation->ready_host_time_ns;
  event_info.submission_id = profile_operation->submission_id;
  event_info.command_buffer_id = profile_operation->command_buffer_id;
  event_info.allocation_id = profile_operation->allocation_id;
  event_info.wait_count = profile_operation->wait_count;
  event_info.signal_count = profile_operation->signal_count;
  event_info.operation_count = profile_operation->operation_count;
  event_info.payload_length = profile_operation->payload_length;
  iree_hal_local_profile_recorder_append_queue_event(
      profile_operation->recorder, &event_info, /*out_event_id=*/NULL);
}

static void iree_hal_task_queue_profile_record_ready(
    iree_hal_task_queue_op_t* operation) {
  if (operation->type == IREE_HAL_TASK_QUEUE_OP_ALLOCA) return;
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation || !iree_hal_local_profile_recorder_is_enabled(
                                profile_operation->recorder,
                                IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    return;
  }
  iree_hal_task_queue_profile_record_queue_event(operation, iree_time_now());
}

static void iree_hal_task_queue_profile_record_failed_before_ready(
    iree_hal_task_queue_op_t* operation) {
  iree_hal_task_queue_profile_record_queue_event(operation,
                                                 /*ready_host_time_ns=*/0);
}

static void iree_hal_task_queue_profile_start_host_execution(
    iree_hal_task_queue_op_t* operation) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return;
  if (!iree_hal_local_profile_recorder_is_enabled(
          profile_operation->recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    return;
  }
  int64_t expected = 0;
  const iree_time_t start_host_time_ns = iree_time_now();
  iree_atomic_compare_exchange_strong(
      &profile_operation->start_host_time_ns, &expected, start_host_time_ns,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

static void iree_hal_task_queue_profile_finish_host_execution(
    iree_hal_task_queue_op_t* operation, iree_status_t operation_status) {
  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  if (!profile_operation) return;
  if (!iree_hal_local_profile_recorder_is_enabled(
          profile_operation->recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    return;
  }
  const iree_time_t start_host_time_ns = iree_atomic_load(
      &profile_operation->start_host_time_ns, iree_memory_order_acquire);
  if (start_host_time_ns == 0) return;

  iree_hal_local_profile_host_execution_event_info_t event_info =
      iree_hal_local_profile_host_execution_event_info_default();
  event_info.type = profile_operation->type;
  event_info.flags = profile_operation->host_flags;
  event_info.status_code = iree_status_code(operation_status);
  event_info.scope = profile_operation->scope;
  event_info.submission_id = profile_operation->submission_id;
  event_info.command_buffer_id = profile_operation->command_buffer_id;
  event_info.executable_id = profile_operation->executable_id;
  event_info.allocation_id = profile_operation->allocation_id;
  event_info.export_ordinal = profile_operation->export_ordinal;
  memcpy(event_info.workgroup_count, profile_operation->workgroup_count,
         sizeof(event_info.workgroup_count));
  memcpy(event_info.workgroup_size, profile_operation->workgroup_size,
         sizeof(event_info.workgroup_size));
  event_info.start_host_time_ns = start_host_time_ns;
  event_info.end_host_time_ns = iree_time_now();
  event_info.payload_length = profile_operation->payload_length;
  event_info.operation_count = profile_operation->operation_count;
  iree_hal_local_profile_recorder_append_host_execution_event(
      profile_operation->recorder, &event_info, /*out_event_id=*/NULL);
}

//===----------------------------------------------------------------------===//
// Operation lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_task_queue_op_release_alloca_memory_wait(
    iree_hal_task_queue_op_t* operation);

// Unmaps any SCOPED binding table mappings before user-visible completion is
// published. Dealloca can legally wait on command completion and immediately
// decommit transient backing memory, so command mappings must be finalized
// before signal semaphores/frontiers become observable.
static iree_status_t iree_hal_task_queue_op_unmap_binding_mappings(
    iree_hal_task_queue_op_t* operation, iree_status_t status) {
  if (operation->type != IREE_HAL_TASK_QUEUE_OP_COMMANDS ||
      !operation->commands.binding_mappings) {
    return status;
  }

  iree_hal_buffer_mapping_t* binding_mappings =
      operation->commands.binding_mappings;
  const iree_host_size_t binding_mapping_count =
      operation->commands.binding_mapping_count;
  operation->commands.binding_mappings = NULL;
  operation->commands.binding_mapping_count = 0;

  for (iree_host_size_t i = 0; i < binding_mapping_count; ++i) {
    iree_status_t unmap_status =
        iree_hal_buffer_unmap_range(&binding_mappings[i]);
    if (!iree_status_is_ok(unmap_status)) {
      if (iree_status_is_ok(status)) {
        status = unmap_status;
      } else {
        status = iree_status_join(status, unmap_status);
      }
    }
  }
  return status;
}

// Destroys an operation, failing signal semaphores if |failure_status| is
// non-OK. Releases all retained resources, ends the scope, and deinitializes
// the arena (which frees the operation itself and all transient allocations).
static void iree_hal_task_queue_op_destroy(iree_hal_task_queue_op_t* operation,
                                           iree_status_t failure_status) {
  iree_hal_task_queue_debug_record_destroy(operation->queue, operation,
                                           failure_status);
  iree_hal_task_queue_profile_finish_host_execution(operation, failure_status);

  // Failure/early-destroy backstop: success completion finalizes mappings
  // before signaling, but issue failures can destroy the operation directly.
  //
  // The mappings array is 1:1 with resolved block binding entries; NULL-buffer
  // slots have zeroed mappings that unmap_range handles as no-ops.
  failure_status =
      iree_hal_task_queue_op_unmap_binding_mappings(operation, failure_status);

  iree_hal_task_queue_op_release_alloca_memory_wait(operation);

  // Fail signal semaphores on error (stores the error status in each), then
  // always release the semaphore references regardless of success/failure.
  if (!iree_status_is_ok(failure_status)) {
    iree_hal_semaphore_list_fail(operation->signal_semaphores, failure_status);
  }
  iree_hal_semaphore_list_release(operation->signal_semaphores);

  // Release retained resources (command buffers, binding table buffers).
  if (operation->resource_set) {
    iree_hal_resource_set_free(operation->resource_set);
    operation->resource_set = NULL;
  }

  // Capture scope before arena deinitialization frees the operation.
  iree_task_scope_t* scope = operation->scope;

  // Free all transient memory (including this operation).
  iree_arena_allocator_t arena = operation->arena;
  operation = NULL;
  iree_arena_deinitialize(&arena);

  // Notify the scope that this operation has completed. Must happen after
  // all operation memory is freed so that idle waiters can safely deallocate.
  if (scope) {
    iree_task_scope_end(scope);
  }
}

// Fails an operation: propagates the error to the frontier tracker (if present)
// and destroys the operation with the given failure status.
static void iree_hal_task_queue_op_fail(iree_hal_task_queue_op_t* operation,
                                        iree_status_t status) {
  iree_hal_task_queue_debug_record_fail(operation->queue, operation);

  iree_async_frontier_tracker_fail_axis(
      operation->frontier_tracker, operation->axis,
      iree_status_from_code(iree_status_code(status)));
  iree_hal_task_queue_op_destroy(operation, status);
}

static uint64_t iree_hal_task_queue_op_reserve_completion_epoch(
    iree_hal_task_queue_op_t* operation) {
  return (uint64_t)iree_atomic_fetch_add(operation->epoch_counter, 1,
                                         iree_memory_order_acq_rel) +
         1;
}

static void iree_hal_task_queue_op_advance_frontier(
    iree_hal_task_queue_op_t* operation, uint64_t epoch) {
  iree_async_frontier_tracker_advance(operation->frontier_tracker,
                                      operation->axis, epoch);
}

static void iree_hal_task_queue_op_complete_with_epoch(
    iree_hal_task_queue_op_t* operation, uint64_t epoch) {
  iree_status_t status = iree_hal_task_queue_op_unmap_binding_mappings(
      operation, iree_ok_status());
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(operation->signal_semaphores,
                                            /*frontier=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_advance_frontier(operation, epoch);
    iree_hal_task_queue_debug_record_complete(operation->queue, operation);
    iree_hal_task_queue_op_destroy(operation, status);
  } else {
    iree_hal_task_queue_op_fail(operation, status);
  }
}

// Completes an operation successfully: signals semaphores, advances the
// frontier, then destroys the operation (freeing the arena).
static void iree_hal_task_queue_op_complete(
    iree_hal_task_queue_op_t* operation) {
  iree_status_t status = iree_hal_task_queue_op_unmap_binding_mappings(
      operation, iree_ok_status());
  if (iree_status_is_ok(status)) {
    // Signal all semaphores to their new values.
    status = iree_hal_semaphore_list_signal(operation->signal_semaphores,
                                            /*frontier=*/NULL);
  }

  // Advance the frontier tracker after signaling.
  if (iree_status_is_ok(status)) {
    uint64_t epoch = iree_hal_task_queue_op_reserve_completion_epoch(operation);
    iree_hal_task_queue_op_advance_frontier(operation, epoch);
    iree_hal_task_queue_debug_record_complete(operation->queue, operation);
    iree_hal_task_queue_op_destroy(operation, status);
  } else {
    // If signaling failed, fail the frontier and propagate the error.
    iree_hal_task_queue_op_fail(operation, status);
  }
}

// Allocates and initializes a queue operation from a fresh arena.
// The operation is allocated from the arena (so the arena owns the memory).
// Signal semaphores are cloned into the arena, and a resource set is created.
static iree_status_t iree_hal_task_queue_op_allocate(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_type_t type,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_hal_task_queue_op_t** out_operation) {
  *out_operation = NULL;

  // Create the arena that owns all transient memory for this operation.
  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->small_block_pool, &arena);

  // Allocate the operation from the arena.
  iree_hal_task_queue_op_t* operation = NULL;
  const bool has_profile_operation =
      IREE_UNLIKELY(queue->profile_recorder != NULL);
  const iree_host_size_t operation_size =
      has_profile_operation
          ? iree_hal_task_queue_profile_operation_offset() +
                sizeof(iree_hal_task_queue_profile_operation_t)
          : sizeof(*operation);
  iree_status_t status =
      iree_arena_allocate(&arena, operation_size, (void**)&operation);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }

  memset(operation, 0, sizeof(*operation));
  operation->type = type;
  memcpy(&operation->arena, &arena, sizeof(arena));
  operation->scope = &queue->scope;
  operation->frontier_tracker = queue->frontier_tracker;
  operation->axis = queue->axis;
  operation->epoch_counter = &queue->epoch;
  operation->queue = queue;
  iree_hal_task_queue_debug_record_submit(queue, operation);
  if (has_profile_operation) {
    iree_hal_task_queue_profile_operation_t* profile_operation =
        iree_hal_task_queue_profile_operation(operation);
    iree_hal_task_queue_profile_operation_initialize(
        queue, type, signal_semaphores, profile_operation);
  }

  // Clone signal semaphores into the arena.
  status = iree_hal_semaphore_list_clone(
      signal_semaphores, iree_arena_allocator(&operation->arena),
      &operation->signal_semaphores);

  // Create a resource set for retaining command buffers and binding table
  // buffers through completion.
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_allocate(queue->small_block_pool,
                                            &operation->resource_set);
  }

  if (iree_status_is_ok(status)) {
    *out_operation = operation;
  } else {
    iree_hal_semaphore_list_release(operation->signal_semaphores);
    if (operation->resource_set) {
      iree_hal_resource_set_free(operation->resource_set);
    }
    iree_arena_deinitialize(&operation->arena);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Semaphore wait integration
//===----------------------------------------------------------------------===//

// Arena-allocated wrapper for a semaphore timepoint that feeds completed waits
// into the queue's ready list. When all waits on an operation are satisfied,
// the operation is pushed to the ready list and the queue process is woken.
typedef struct iree_hal_task_queue_wait_entry_t {
  // Timepoint registered with the async semaphore.
  iree_async_semaphore_timepoint_t timepoint;
  // The operation waiting on this semaphore.
  iree_hal_task_queue_op_t* operation;
  // Retained reference to the HAL semaphore.
  iree_hal_semaphore_t* semaphore;
} iree_hal_task_queue_wait_entry_t;

static bool iree_hal_task_queue_is_shutting_down(iree_hal_task_queue_t* queue) {
  return iree_atomic_load(&queue->shutting_down, iree_memory_order_acquire) !=
         0;
}

static iree_status_t iree_hal_task_queue_make_shutdown_status(void) {
  return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
}

// Callback fired when a semaphore timepoint is resolved (value reached or
// semaphore failed). Decrements the operation's wait_count and, if this was
// the last outstanding wait, either pushes the operation to the ready list
// (success) or destroys it (failure).
static void iree_hal_task_queue_wait_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_task_queue_wait_entry_t* entry =
      (iree_hal_task_queue_wait_entry_t*)user_data;
  iree_hal_task_queue_op_t* operation = entry->operation;
  iree_hal_task_queue_t* queue = operation->queue;
  iree_hal_semaphore_t* semaphore = entry->semaphore;

  // Record the first failure via CAS.
  if (!iree_status_is_ok(status)) {
    intptr_t expected = 0;
    if (!iree_atomic_compare_exchange_strong(
            &operation->error_status, &expected, (intptr_t)status,
            iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
      // Another wait already recorded an error; free ours.
      iree_status_free(status);
    }
  }

  // Decrement wait count regardless of success/failure.
  int32_t previous_count = iree_atomic_fetch_sub(&operation->wait_count, 1,
                                                 iree_memory_order_acq_rel);
  if (previous_count == 1) {
    // Last wait resolved. Check for errors.
    iree_status_t error = (iree_status_t)iree_atomic_exchange(
        &operation->error_status, 0, iree_memory_order_acquire);
    if (!iree_status_is_ok(error)) {
      // At least one wait failed. Fail signals and destroy.
      iree_hal_task_queue_profile_record_failed_before_ready(operation);
      iree_hal_task_queue_op_destroy(operation, error);
    } else if (iree_hal_task_queue_is_shutting_down(operation->queue)) {
      // Shutdown owns all remaining unresolved work. Late wait callbacks must
      // resolve the operation inline instead of scheduling a terminal process.
      iree_hal_task_queue_profile_record_failed_before_ready(operation);
      iree_hal_task_queue_op_destroy(
          operation, iree_hal_task_queue_make_shutdown_status());
    } else {
      // All waits satisfied. Push to ready list and wake queue.
      iree_hal_task_queue_profile_record_ready(operation);
      iree_hal_task_queue_op_slist_push(&queue->ready_list, operation);
      iree_task_executor_schedule_process(queue->executor, &queue->process);
    }
  }

  // Release retained semaphore reference. Must happen after all operation
  // access — the semaphore does not depend on the operation, but we should
  // not access the entry (arena memory) after the operation might be freed.
  iree_hal_semaphore_release(semaphore);
}

// Queries wait semaphores front-to-back, removing already-satisfied entries.
// If all are satisfied, sets count to 0 so the caller skips wait registration.
// If entry i is the first unsatisfied, slices the list to [i, count) so only
// the unsatisfied remainder gets timepoint registration.
//
// Returns a failure status if any queried semaphore has been failed.
static iree_status_t iree_hal_task_queue_try_satisfy_waits(
    iree_hal_semaphore_list_t* wait_semaphores) {
  for (iree_host_size_t i = 0; i < wait_semaphores->count; ++i) {
    uint64_t current_value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(
        wait_semaphores->semaphores[i], &current_value));
    if (current_value < wait_semaphores->payload_values[i]) {
      // Slice past the satisfied entries we already verified.
      wait_semaphores->semaphores += i;
      wait_semaphores->payload_values += i;
      wait_semaphores->count -= i;
      return iree_ok_status();
    }
  }
  wait_semaphores->count = 0;
  return iree_ok_status();
}

// Registers semaphore timepoints for each unsatisfied wait. The operation's
// wait_count is set to the number of unsatisfied waits. Each timepoint callback
// decrements wait_count; the last one to fire pushes the operation to the
// ready list.
//
// If all waits are already satisfied (count == 0), the operation is pushed
// directly to the ready list.
static iree_status_t iree_hal_task_queue_enqueue_waits(
    iree_hal_task_queue_op_t* operation,
    iree_hal_semaphore_list_t wait_semaphores) {
  if (wait_semaphores.count == 0) {
    if (iree_hal_task_queue_is_shutting_down(operation->queue)) {
      return iree_hal_task_queue_make_shutdown_status();
    }

    // All waits already satisfied — push directly.
    iree_hal_task_queue_profile_record_ready(operation);
    iree_hal_task_queue_op_slist_push(&operation->queue->ready_list, operation);
    iree_task_executor_schedule_process(operation->queue->executor,
                                        &operation->queue->process);
    return iree_ok_status();
  }

  // Set the wait count before registering any timepoints. A timepoint callback
  // may fire synchronously during acquire_timepoint if the semaphore value
  // was reached between try_satisfy_waits and here.
  iree_atomic_store(&operation->wait_count, (int32_t)wait_semaphores.count,
                    iree_memory_order_release);

  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_hal_semaphore_t* semaphore = wait_semaphores.semaphores[i];
    uint64_t minimum_value = wait_semaphores.payload_values[i];

    // Allocate the wait entry from the operation's arena.
    iree_hal_task_queue_wait_entry_t* entry = NULL;
    iree_status_t status =
        iree_arena_allocate(&operation->arena, sizeof(*entry), (void**)&entry);

    // Register the timepoint. The callback may fire synchronously.
    if (iree_status_is_ok(status)) {
      entry->operation = operation;
      entry->semaphore = semaphore;
      iree_hal_semaphore_retain(semaphore);
      entry->timepoint.callback = iree_hal_task_queue_wait_resolved;
      entry->timepoint.user_data = entry;
      status = iree_async_semaphore_acquire_timepoint(
          (iree_async_semaphore_t*)semaphore, minimum_value, &entry->timepoint);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_release(semaphore);
      }
    }

    if (!iree_status_is_ok(status)) {
      // Registration failed at index i. Timepoints 0..i-1 are already
      // registered and their callbacks will fire asynchronously. We cannot
      // destroy the operation here — the callbacks would access freed arena
      // memory. Instead, record the error and subtract the unregistered
      // count from wait_count so the existing callbacks can drain and
      // destroy the operation when the last one completes.
      intptr_t expected = 0;
      if (!iree_atomic_compare_exchange_strong(
              &operation->error_status, &expected, (intptr_t)status,
              iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
        iree_status_free(status);
      }
      // Subtract count for this entry and all remaining unregistered ones.
      int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      int32_t previous_count = iree_atomic_fetch_sub(
          &operation->wait_count, unregistered, iree_memory_order_acq_rel);
      if (previous_count == unregistered) {
        // All registered timepoints already fired. We are the last
        // decrement — destroy the operation with the recorded error.
        iree_status_t error = (iree_status_t)iree_atomic_exchange(
            &operation->error_status, 0, iree_memory_order_acquire);
        iree_hal_task_queue_profile_record_failed_before_ready(operation);
        iree_hal_task_queue_op_destroy(operation, error);
      }
      // Return OK: the operation is owned by the timepoint callbacks (or
      // already destroyed above). The caller must not touch it.
      return iree_ok_status();
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Alloca memory readiness
//===----------------------------------------------------------------------===//

typedef enum iree_hal_task_queue_alloca_memory_wait_kind_e {
  IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE = 0,
  IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER = 1,
  IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION = 2,
} iree_hal_task_queue_alloca_memory_wait_kind_t;

// Cold-path alloca memory-readiness wait. Allocated inside the queue
// operation's arena only after user semaphore waits have resolved and the pool
// cannot produce immediately-usable bytes.
struct iree_hal_task_queue_alloca_memory_wait_t {
  // Active wait source.
  iree_hal_task_queue_alloca_memory_wait_kind_t kind;

  // State for a held reservation blocked on a pool death frontier.
  struct {
    // Queue-owned reservation held while waiting for its death frontier.
    iree_hal_pool_reservation_t reservation;

    // Pool-owned death frontier borrowed while the reservation is held.
    const iree_async_frontier_t* wait_frontier;

    // Tracker waiter storage for |wait_frontier|.
    iree_async_frontier_waiter_t waiter;
  } frontier;

  // State for reservation retry after pool release notifications.
  struct {
    // Wait operations rotated so a callback can arm a retry before returning.
    iree_async_notification_wait_operation_t wait_ops[2];

    // Index of the active wait operation in |wait_ops|.
    uint8_t wait_slot;
  } pool_notification;
};

static iree_status_t iree_hal_task_queue_alloca_memory_wait_ensure(
    iree_hal_task_queue_op_t* operation,
    iree_hal_task_queue_alloca_memory_wait_t** out_wait) {
  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  if (!wait) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&operation->arena, sizeof(*wait), (void**)&wait));
    memset(wait, 0, sizeof(*wait));
    operation->alloca.memory_wait = wait;
  }
  *out_wait = wait;
  return iree_ok_status();
}

static void iree_hal_task_queue_op_release_alloca_memory_wait(
    iree_hal_task_queue_op_t* operation) {
  if (operation->type != IREE_HAL_TASK_QUEUE_OP_ALLOCA) {
    return;
  }
  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  if (!wait) {
    return;
  }

  switch (wait->kind) {
    case IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_pool_release_reservation(operation->alloca.pool,
                                        &wait->frontier.reservation,
                                        wait->frontier.wait_frontier);
      wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
}

static void iree_hal_task_queue_alloca_memory_wait_resolved(
    iree_hal_task_queue_op_t* operation, iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_fail(operation, status);
    return;
  }

  if (iree_hal_task_queue_is_shutting_down(operation->queue)) {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_destroy(operation,
                                   iree_hal_task_queue_make_shutdown_status());
    return;
  }

  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  if (wait &&
      wait->kind == IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION) {
    wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
  }
  iree_hal_task_queue_op_slist_push(&operation->queue->ready_list, operation);
  iree_task_executor_schedule_process(operation->queue->executor,
                                      &operation->queue->process);
}

static void iree_hal_task_queue_alloca_frontier_wait_resolved(
    void* user_data, iree_status_t status) {
  iree_hal_task_queue_alloca_memory_wait_resolved(
      (iree_hal_task_queue_op_t*)user_data, status);
}

static void iree_hal_task_queue_alloca_pool_notification_wait_resolved(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_task_queue_alloca_memory_wait_resolved(
      (iree_hal_task_queue_op_t*)user_data, status);
}

static iree_status_t iree_hal_task_queue_alloca_wait_for_frontier(
    iree_hal_task_queue_op_t* operation,
    const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* wait_frontier) {
  if (IREE_UNLIKELY(!wait_frontier)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca waitable pool reservation did not provide a frontier");
  }

  iree_hal_task_queue_alloca_memory_wait_t* wait = NULL;
  iree_status_t status =
      iree_hal_task_queue_alloca_memory_wait_ensure(operation, &wait);
  if (iree_status_is_ok(status)) {
    wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER;
    wait->frontier.reservation = *reservation;
    wait->frontier.wait_frontier = wait_frontier;
    status = iree_async_frontier_tracker_wait(
        operation->frontier_tracker, wait_frontier,
        iree_hal_task_queue_alloca_frontier_wait_resolved, operation,
        &wait->frontier.waiter);
    if (iree_status_is_ok(status)) {
      iree_hal_task_queue_profile_force_software_defer(operation);
      iree_hal_task_queue_profile_record_memory_event(
          operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER,
          IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT, operation->alloca.pool,
          operation->alloca.params, reservation, wait_frontier->entry_count);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_pool_release_reservation(operation->alloca.pool, reservation,
                                      wait_frontier);
    if (wait) {
      wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
    }
  }
  return status;
}

static iree_status_t iree_hal_task_queue_alloca_wait_for_pool_notification(
    iree_hal_task_queue_op_t* operation,
    iree_hal_pool_acquire_result_t acquire_result,
    iree_async_notification_t* notification, uint32_t wait_token) {
  iree_hal_task_queue_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_task_queue_alloca_memory_wait_ensure(operation, &wait));
  wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION;
  wait->pool_notification.wait_slot =
      (uint8_t)((wait->pool_notification.wait_slot + 1u) & 1u);

  iree_async_notification_wait_operation_t* wait_op =
      &wait->pool_notification.wait_ops[wait->pool_notification.wait_slot];
  iree_async_operation_zero(&wait_op->base, sizeof(*wait_op));
  iree_async_operation_initialize(
      &wait_op->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_task_queue_alloca_pool_notification_wait_resolved, operation);
  wait_op->notification = notification;
  wait_op->wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op->wait_token = wait_token;

  iree_status_t status = iree_async_proactor_submit_one(
      operation->queue->proactor, &wait_op->base);
  if (!iree_status_is_ok(status)) {
    wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
  } else {
    iree_hal_task_queue_profile_force_software_defer(operation);
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
            IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION,
        acquire_result, operation->alloca.pool, operation->alloca.params,
        /*reservation=*/NULL, /*frontier_entry_count=*/0);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Per-operation drain functions
//===----------------------------------------------------------------------===//

// Sentinel value for compute_current when no recording is active (NULL).
#define IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_NULL ((intptr_t)0)

// Low bit of queue->compute_current_revision while a writer is publishing a
// new queue->compute_current pointer.
#define IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_REVISION_UPDATING ((int64_t)1)

// Increment applied when one compute_current publication completes.
#define IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_REVISION_INCREMENT ((int64_t)2)

// Returns the lifecycle generation stored in the high 32 bits of
// item->drainers.
static int64_t iree_hal_task_queue_compute_item_generation(
    iree_hal_task_queue_compute_item_t* item) {
  return iree_atomic_load(&item->drainers, iree_memory_order_acquire) &
         ~(int64_t)UINT32_MAX;
}

// Attempts to begin an exclusive compute_current publication.
//
// Returns true if the publication lock was acquired and stores the previous
// even revision in |out_prior_revision|. The caller must complete publication
// by calling iree_hal_task_queue_end_compute_current_update with that value.
//
// Returns false immediately if another writer is already publishing or if this
// worker lost the CAS race. Hot compute drainers use this nonblocking path so a
// delayed publisher does not turn an empty-queue check into full-pool spinning.
static bool iree_hal_task_queue_try_begin_compute_current_update(
    iree_hal_task_queue_t* queue, int64_t* out_prior_revision) {
  int64_t current_revision = iree_atomic_load(&queue->compute_current_revision,
                                              iree_memory_order_acquire);
  if (IREE_UNLIKELY(current_revision &
                    IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_REVISION_UPDATING)) {
    return false;
  }

  int64_t expected_revision = current_revision;
  if (!iree_atomic_compare_exchange_strong(
          &queue->compute_current_revision, &expected_revision,
          current_revision |
              IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_REVISION_UPDATING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    return false;
  }

  *out_prior_revision = current_revision;
  return true;
}

// Begins an exclusive compute_current publication.
//
// The returned revision is always even. The caller must complete publication
// by calling iree_hal_task_queue_end_compute_current_update with this value.
static int64_t iree_hal_task_queue_begin_compute_current_update(
    iree_hal_task_queue_t* queue) {
  int64_t prior_revision = 0;
  while (!iree_hal_task_queue_try_begin_compute_current_update(
      queue, &prior_revision)) {
    iree_processor_yield();
  }
  return prior_revision;
}

// Finishes a compute_current publication started by
// iree_hal_task_queue_begin_compute_current_update.
static void iree_hal_task_queue_end_compute_current_update(
    iree_hal_task_queue_t* queue, int64_t prior_revision) {
  iree_atomic_store(
      &queue->compute_current_revision,
      prior_revision + IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_REVISION_INCREMENT,
      iree_memory_order_release);
}

// Publishes |item| as queue->compute_current or NULL if no recording is active.
static void iree_hal_task_queue_store_compute_current(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_compute_item_t* item) {
  int64_t prior_revision =
      iree_hal_task_queue_begin_compute_current_update(queue);
  iree_atomic_store(
      &queue->compute_current,
      item ? (intptr_t)item : IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_NULL,
      iree_memory_order_release);
  iree_hal_task_queue_end_compute_current_update(queue, prior_revision);
}

// Installs |item| as queue->compute_current only if no recording is currently
// active. Returns false if another worker already published a current item.
static bool iree_hal_task_queue_try_store_compute_current_if_empty(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_compute_item_t* item) {
  if (iree_atomic_load(&queue->compute_current, iree_memory_order_acquire) !=
      IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_NULL) {
    return false;
  }

  int64_t prior_revision = 0;
  if (!iree_hal_task_queue_try_begin_compute_current_update(queue,
                                                            &prior_revision)) {
    return false;
  }

  const bool did_store =
      iree_atomic_load(&queue->compute_current, iree_memory_order_relaxed) ==
      IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_NULL;
  if (did_store) {
    iree_atomic_store(&queue->compute_current, (intptr_t)item,
                      iree_memory_order_release);
    // A stale compute-process drainer can race this direct publication and
    // consume an empty snapshot before schedule_process runs. Leave a wake
    // obligation behind before dropping the publication lock so the process
    // cannot transition to IDLE with compute_current already non-NULL.
    iree_atomic_store(&queue->compute_process.needs_drain, 1,
                      iree_memory_order_release);
  }
  iree_hal_task_queue_end_compute_current_update(queue, prior_revision);
  return did_store;
}

// Promotes one pending recording into compute_current if no recording is
// currently active. Returns true if compute_current is non-NULL after the
// locked check/promotion and false only if both compute_current and
// compute_pending were empty while the publication lock was held, or if another
// writer already owns publication and will either schedule the compute process
// with new work or leave it empty.
//
// This helper must be used by the compute process's no-current path instead of
// a lock-free "load current, then pop pending" sequence. Otherwise one drainer
// can observe a stale empty snapshot and return did_work=false while another
// thread publishes a new current item under the revision lock, allowing the
// last process drainer to clear needs_drain and put the process to sleep with
// live compute work still published.
static bool iree_hal_task_queue_promote_pending_to_compute_current(
    iree_hal_task_queue_t* queue) {
  int64_t prior_revision = 0;
  if (!iree_hal_task_queue_try_begin_compute_current_update(queue,
                                                            &prior_revision)) {
    return false;
  }

  bool has_current =
      iree_atomic_load(&queue->compute_current, iree_memory_order_relaxed) !=
      IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_NULL;
  if (!has_current) {
    iree_hal_task_queue_compute_item_t* next =
        iree_hal_task_queue_compute_item_slist_pop(&queue->compute_pending);
    if (next) {
      iree_atomic_store(&queue->compute_current, (intptr_t)next,
                        iree_memory_order_release);
      has_current = true;
    }
  }

  iree_hal_task_queue_end_compute_current_update(queue, prior_revision);
  if (has_current) {
    iree_task_executor_schedule_process(queue->executor,
                                        &queue->compute_process);
  }
  return has_current;
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
static void iree_hal_task_queue_trace_compute_drain_event(void) {
  IREE_TRACE_ZONE_BEGIN_NAMED(z_drain, "iree_hal_local_task_compute_drain");
  IREE_TRACE_ZONE_END(z_drain);
}
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

// Allocates a new compute recording item from the queue's arena with a
// trailing worker_states FAM sized to compute_worker_count. The item is NOT
// pushed to the free pool — the caller uses it directly.
static iree_status_t iree_hal_task_queue_compute_item_allocate(
    iree_hal_task_queue_t* queue,
    iree_hal_task_queue_compute_item_t** out_item) {
  *out_item = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t item_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(iree_sizeof_struct(*(*out_item)), &item_size,
                             IREE_STRUCT_FIELD_FAM(
                                 queue->compute_worker_count,
                                 iree_hal_cmd_block_processor_worker_state_t)));

  iree_hal_task_queue_compute_item_t* item = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate_aligned(
              &queue->compute_item_arena, item_size,
              iree_hardware_destructive_interference_size, (void**)&item));
  memset(item, 0, item_size);

  // Link into the all-items list for shutdown cleanup enumeration.
  item->next_allocated = queue->compute_item_head;
  queue->compute_item_head = item;

  *out_item = item;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Routes a recording through the compute process for deferred execution.
// Acquires a compute item, allocates the processor context, and schedules the
// compute process. The processor context always references a recording that
// remains live until the compute item's final release.
//
// If |owned_recording| is non-NULL, the compute item takes ownership and
// releases the blocks in its final release path. If NULL, the caller
// retains ownership (e.g., the recording lives inside a command buffer that
// the resource_set keeps alive).
static iree_status_t iree_hal_task_queue_drain_recording(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation,
    const iree_hal_cmd_block_recording_t* recording,
    iree_hal_cmd_block_recording_t* owned_recording,
    const iree_hal_cmd_binding_entry_t* binding_table,
    iree_host_size_t binding_table_length) {
  iree_hal_task_queue_profile_add_host_flags(
      operation, IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_DEFERRED);
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(queue->device_allocator);

  uint32_t worker_count =
      (uint32_t)iree_task_executor_worker_count(queue->executor);
  if (worker_count == 0) worker_count = 1;

  iree_hal_task_queue_profile_operation_t* profile_operation =
      iree_hal_task_queue_profile_operation(operation);
  iree_hal_cmd_block_processor_profile_dispatch_t* profile_dispatches = NULL;
  iree_host_size_t profile_dispatch_capacity = 0;
  iree_status_t status = iree_ok_status();
  if (worker_count > 1 && profile_operation &&
      recording->max_region_dispatch_count > 0 &&
      iree_hal_local_profile_recorder_is_enabled(
          profile_operation->recorder,
          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    profile_dispatch_capacity = recording->max_region_dispatch_count;
    iree_host_size_t dispatch_profile_bytes = 0;
    if (IREE_LIKELY(iree_host_size_checked_mul(profile_dispatch_capacity,
                                               sizeof(*profile_dispatches),
                                               &dispatch_profile_bytes))) {
      status = iree_arena_allocate(&operation->arena, dispatch_profile_bytes,
                                   (void**)&profile_dispatches);
    } else {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "local-task profile dispatch aggregation storage is too large");
    }
    if (iree_status_is_ok(status)) {
      memset(profile_dispatches, 0, dispatch_profile_bytes);
    }
  }
  IREE_RETURN_IF_ERROR(status);

  // Acquire a recording item from the free pool, growing the pool if empty.
  iree_hal_task_queue_compute_item_t* item =
      iree_hal_task_queue_compute_item_slist_pop(&queue->compute_free_pool);
  if (!item) {
    IREE_RETURN_IF_ERROR(
        iree_hal_task_queue_compute_item_allocate(queue, &item));
  }

  // Transfer ownership before allocating the processor context so the context
  // never captures a pointer to a caller stack recording.
  const iree_hal_cmd_block_recording_t* processor_recording = recording;
  if (owned_recording) {
    item->recording = *owned_recording;
    memset(owned_recording, 0, sizeof(*owned_recording));
    processor_recording = &item->recording;
  } else {
    memset(&item->recording, 0, sizeof(item->recording));
  }

  // Allocate the block processor execution context.
  iree_hal_cmd_block_processor_context_t* processor_context = NULL;
  status = iree_hal_cmd_block_processor_context_allocate(
      processor_recording, binding_table, binding_table_length, worker_count,
      host_allocator, &processor_context);
  if (!iree_status_is_ok(status)) {
    if (owned_recording) {
      iree_hal_cmd_block_recording_release(&item->recording);
      memset(&item->recording, 0, sizeof(item->recording));
    }
    iree_hal_task_queue_compute_item_slist_push(&queue->compute_free_pool,
                                                item);
    return status;
  }

  // Set budget tracking pointers so the block processor can update
  // wake_budget at region transitions and wake additional workers. A
  // no-block recording has no processor context and completes immediately via
  // the null-safe block processor drain path.
  if (processor_context) {
    processor_context->wake_budget_ptr = &queue->compute_process.wake_budget;
    processor_context->wake_executor = queue->executor;
    processor_context->retention_epoch_ptr =
        &queue->compute_process.retention_epoch;
    processor_context->retention_sleepers_ptr =
        &queue->compute_process.retention_sleepers;
    processor_context->retention_spin_deadline_ns_ptr =
        &queue->compute_process.retention_spin_deadline_ns;
    processor_context->retention_transition_spin_ns =
        IREE_TASK_WARM_WAIT_TRANSITION_SPIN_NS;
    if (profile_operation) {
      iree_hal_cmd_block_processor_context_set_profile_recorder(
          processor_context, profile_operation->recorder,
          profile_operation->scope, profile_operation->submission_id,
          profile_operation->command_buffer_id, profile_dispatches,
          profile_dispatch_capacity);
    }
  }

  // Seed scheduling with the first active region before publishing the item so
  // the schedule_process call below wakes the right number of workers.
  iree_atomic_store(
      &queue->compute_process.wake_budget,
      iree_hal_cmd_block_processor_context_wake_budget(processor_context),
      iree_memory_order_relaxed);

  // Fill the recording item.
  item->processor_context = processor_context;
  item->worker_count = worker_count;
  item->operation = operation;
  item->host_allocator = host_allocator;
  memset(item->worker_states, 0, sizeof(item->worker_states[0]) * worker_count);
  // drainers retains generation from previous lifecycle. Count and CLOSED
  // were cleared during the previous cleanup. First use: memset zeroed it.

  // Install the item directly into compute_current if the slot is empty.
  // This eliminates the pending→current hop that would otherwise cost a full
  // worker pump cycle: workers scanning the compute slot see the item
  // immediately rather than having to pop it from the pending list first.
  if (!iree_hal_task_queue_try_store_compute_current_if_empty(queue, item)) {
    // Slot busy (a recording is still being drained). Buffer in pending —
    // the completer will promote it when the current recording finishes.
    iree_hal_task_queue_compute_item_slist_push(&queue->compute_pending, item);
  }

  // Schedule the compute process (places in executor slot on first call,
  // subsequent calls just wake workers via the wake tree).
  iree_task_executor_schedule_process(queue->executor, &queue->compute_process);

  return iree_ok_status();
}

static iree_status_t iree_hal_task_queue_resolve_binding_entry(
    const iree_hal_buffer_binding_t* binding,
    iree_hal_buffer_mapping_t* mapping,
    iree_hal_cmd_binding_entry_t* out_entry) {
  if (binding->buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        binding->buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_ANY, binding->offset, binding->length, mapping));
    out_entry->base = mapping->contents.data;
    out_entry->length = mapping->contents.data_length;
  } else {
    out_entry->base = NULL;
    out_entry->length = 0;
  }
  return iree_ok_status();
}

// Handles a COMMANDS operation: extracts the recording from the command buffer
// and routes it through the compute process.
static iree_status_t iree_hal_task_queue_drain_commands(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  // Verify the command buffer is a block command buffer.
  if (!iree_hal_block_command_buffer_isa(operation->commands.command_buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue only accepts block command buffers; got unrecognized type");
  }

  // Get the recording from the command buffer. The CB is retained in the
  // operation's resource_set, and the operation stays alive until the compute
  // item's final release completes after the last drainer exits.
  const iree_hal_cmd_block_recording_t* recording =
      iree_hal_block_command_buffer_recording(
          operation->commands.command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_profile_record_command_buffer(
      operation, operation->commands.command_buffer));

  // Resolve the HAL binding table to block ISA binding entries. External
  // entries come first at their declared binding table slots, followed by
  // late direct transient entries recorded by the command buffer.
  iree_host_size_t late_binding_count = 0;
  const iree_hal_buffer_binding_t* late_bindings =
      iree_hal_block_command_buffer_late_bindings(
          operation->commands.command_buffer, &late_binding_count);

  // The entries and SCOPED mappings are allocated from the operation's arena.
  // Mappings are tracked on the operation for unmap in op_destroy.
  const iree_hal_cmd_binding_entry_t* binding_table = NULL;
  iree_host_size_t binding_table_length = 0;
  const iree_hal_buffer_binding_table_t hal_table =
      operation->commands.binding_table;
  const iree_host_size_t late_binding_base =
      operation->commands.command_buffer->binding_capacity;
  const iree_host_size_t resolved_binding_count =
      iree_max(hal_table.count, late_binding_base + late_binding_count);
  if (resolved_binding_count > 0) {
    iree_hal_cmd_binding_entry_t* entries = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &operation->arena, resolved_binding_count * sizeof(*entries),
        (void**)&entries));
    memset(entries, 0, resolved_binding_count * sizeof(*entries));
    iree_hal_buffer_mapping_t* mappings = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &operation->arena, resolved_binding_count * sizeof(*mappings),
        (void**)&mappings));
    memset(mappings, 0, resolved_binding_count * sizeof(*mappings));
    operation->commands.binding_mappings = mappings;
    operation->commands.binding_mapping_count = resolved_binding_count;
    for (iree_host_size_t i = 0; i < hal_table.count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_task_queue_resolve_binding_entry(
          &hal_table.bindings[i], &mappings[i], &entries[i]));
    }
    for (iree_host_size_t i = 0; i < late_binding_count; ++i) {
      const iree_host_size_t binding_index = late_binding_base + i;
      IREE_RETURN_IF_ERROR(iree_hal_task_queue_resolve_binding_entry(
          &late_bindings[i], &mappings[binding_index],
          &entries[binding_index]));
    }
    binding_table = entries;
    binding_table_length = resolved_binding_count;
  }

  return iree_hal_task_queue_drain_recording(
      queue, operation, recording,
      /*owned_recording=*/NULL, binding_table, binding_table_length);
}

// Handles a HOST_CALL operation: executes the user function inline and
// completes the operation.
static iree_status_t iree_hal_task_queue_drain_host_call(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  const bool is_nonblocking = iree_any_bit_set(
      operation->host_call.flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);

  // Non-blocking calls signal before executing.
  iree_status_t status = iree_ok_status();
  if (is_nonblocking) {
    status = iree_hal_semaphore_list_signal(operation->signal_semaphores,
                                            /*frontier=*/NULL);
  }

  // Execute the user function.
  if (iree_status_is_ok(status)) {
    iree_hal_host_call_context_t context = {
        .device = operation->host_call.device,
        .queue_affinity = operation->host_call.queue_affinity,
        .signal_semaphore_list = is_nonblocking
                                     ? iree_hal_semaphore_list_empty()
                                     : operation->signal_semaphores,
    };
    iree_status_t call_status =
        operation->host_call.call.fn(operation->host_call.call.user_data,
                                     operation->host_call.args, &context);
    if (is_nonblocking || iree_status_is_deferred(call_status)) {
      // User callback will signal in the future (or fire-and-forget).
    } else if (iree_status_is_ok(call_status)) {
      // Signal callback completed synchronously.
      if (!is_nonblocking) {
        status = iree_hal_semaphore_list_signal(operation->signal_semaphores,
                                                /*frontier=*/NULL);
      }
    } else {
      // Callback failed; propagate failure to signal semaphores, release the
      // references, and clear the list to prevent op_destroy from
      // double-signaling or double-releasing. Clone the status before
      // passing to semaphore_list_fail (which consumes it for the last
      // semaphore) so we retain a copy for frontier failure below.
      status = iree_status_clone(call_status);
      iree_hal_semaphore_list_fail(operation->signal_semaphores, call_status);
      iree_hal_semaphore_list_release(operation->signal_semaphores);
      operation->signal_semaphores = iree_hal_semaphore_list_empty();
    }
  }

  // Advance frontier and clean up.
  if (iree_status_is_ok(status)) {
    uint64_t epoch =
        (uint64_t)iree_atomic_fetch_add(operation->epoch_counter, 1,
                                        iree_memory_order_acq_rel) +
        1;
    iree_async_frontier_tracker_advance(operation->frontier_tracker,
                                        operation->axis, epoch);
  } else {
    iree_async_frontier_tracker_fail_axis(
        operation->frontier_tracker, operation->axis,
        iree_status_from_code(iree_status_code(status)));
  }

  iree_hal_task_queue_op_destroy(operation, status);
  return iree_ok_status();
}

static iree_status_t iree_hal_task_queue_drain_alloca_submit_reservation(
    iree_hal_task_queue_op_t* operation,
    iree_hal_pool_acquire_result_t acquire_result,
    const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* reservation_failure_frontier) {
  iree_hal_task_queue_profile_record_queue_event(operation, iree_time_now());
  iree_hal_task_queue_profile_start_host_execution(operation);
  iree_hal_local_transient_buffer_attach_reservation(
      operation->alloca.transient_buffer, operation->alloca.pool, reservation);

  iree_hal_buffer_t* backing_buffer = NULL;
  iree_status_t status = iree_hal_pool_materialize_reservation(
      operation->alloca.pool, operation->alloca.params, reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &backing_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_local_transient_buffer_stage_backing(
        operation->alloca.transient_buffer, backing_buffer);
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, acquire_result,
        operation->alloca.pool, operation->alloca.params, reservation,
        reservation_failure_frontier ? reservation_failure_frontier->entry_count
                                     : 0);
  }
  iree_hal_buffer_release(backing_buffer);

  if (iree_status_is_ok(status)) {
    iree_hal_local_transient_buffer_commit(operation->alloca.transient_buffer);
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, acquire_result,
        operation->alloca.pool, operation->alloca.params, reservation,
        reservation_failure_frontier ? reservation_failure_frontier->entry_count
                                     : 0);
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_local_transient_buffer_release_reservation(
        operation->alloca.transient_buffer, reservation_failure_frontier);
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
        iree_status_code(status), operation->alloca.pool,
        operation->alloca.params, reservation,
        reservation_failure_frontier ? reservation_failure_frontier->entry_count
                                     : 0);
  }
  return status;
}

static iree_status_t iree_hal_task_queue_drain_alloca_acquire(
    iree_hal_task_queue_op_t* operation,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_acquire_info,
    iree_hal_pool_acquire_result_t* out_acquire_result) {
  const iree_device_size_t min_alignment =
      operation->alloca.params.min_alignment
          ? operation->alloca.params.min_alignment
          : 1;
  iree_status_t status = iree_hal_pool_acquire_reservation(
      operation->alloca.pool, operation->alloca.allocation_size, min_alignment,
      /*requester_frontier=*/NULL, operation->alloca.reserve_flags,
      out_reservation, out_acquire_info, out_acquire_result);
  if (iree_status_is_ok(status)) {
    iree_hal_profile_memory_event_flags_t flags =
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
    if (*out_acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
      flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER;
    } else if (*out_acquire_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
               *out_acquire_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
      flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION;
    }
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE, flags,
        *out_acquire_result, operation->alloca.pool, operation->alloca.params,
        *out_acquire_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
                *out_acquire_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET
            ? NULL
            : out_reservation,
        out_acquire_info->wait_frontier
            ? out_acquire_info->wait_frontier->entry_count
            : 0);
  }
  return status;
}

static iree_status_t iree_hal_task_queue_drain_alloca_on_acquire_result(
    iree_hal_task_queue_op_t* operation,
    const iree_hal_pool_reservation_t* reservation,
    const iree_hal_pool_acquire_info_t* acquire_info,
    iree_hal_pool_acquire_result_t acquire_result);

static iree_status_t
iree_hal_task_queue_drain_alloca_wait_for_pool_notification(
    iree_hal_task_queue_op_t* operation) {
  iree_async_notification_t* notification =
      iree_hal_pool_notification(operation->alloca.pool);
  if (IREE_UNLIKELY(!notification)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "queue_alloca exhausted pool did not provide a "
                            "notification");
  }

  const uint32_t wait_token =
      iree_async_notification_begin_observe(notification);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;

  iree_status_t status = iree_hal_task_queue_drain_alloca_acquire(
      operation, &reservation, &acquire_info, &acquire_result);
  if (iree_status_is_ok(status)) {
    switch (acquire_result) {
      case IREE_HAL_POOL_ACQUIRE_OK:
      case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
        status = iree_hal_task_queue_drain_alloca_on_acquire_result(
            operation, &reservation, &acquire_info, acquire_result);
        break;
      case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
      case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
        status = iree_hal_task_queue_alloca_wait_for_pool_notification(
            operation, acquire_result, notification, wait_token);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "unrecognized pool acquire result %u",
                                  acquire_result);
        break;
    }
  }
  iree_async_notification_end_observe(notification);
  return status;
}

static iree_status_t iree_hal_task_queue_drain_alloca_on_acquire_result(
    iree_hal_task_queue_op_t* operation,
    const iree_hal_pool_reservation_t* reservation,
    const iree_hal_pool_acquire_info_t* acquire_info,
    iree_hal_pool_acquire_result_t acquire_result) {
  switch (acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      return iree_hal_task_queue_drain_alloca_submit_reservation(
          operation, acquire_result, reservation,
          /*reservation_failure_frontier=*/NULL);
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      if (!iree_all_bits_set(operation->alloca.flags,
                             IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)) {
        iree_hal_pool_release_reservation(operation->alloca.pool, reservation,
                                          acquire_info->wait_frontier);
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "queue_alloca recycled pool memory requires "
            "IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER");
      }
      return iree_hal_task_queue_alloca_wait_for_frontier(
          operation, reservation, acquire_info->wait_frontier);
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      return iree_hal_task_queue_drain_alloca_wait_for_pool_notification(
          operation);
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unrecognized pool acquire result %u",
                          acquire_result);
}

// Handles an ALLOCA operation: acquires pool memory after user waits have
// resolved, parks on hidden memory-readiness waits when needed, and publishes
// the transient buffer backing only once the bytes are safe to use.
static iree_status_t iree_hal_task_queue_drain_alloca(
    iree_hal_task_queue_op_t* operation) {
  iree_hal_task_queue_alloca_memory_wait_t* wait =
      operation->alloca.memory_wait;
  if (wait && wait->kind == IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER) {
    const iree_hal_pool_reservation_t reservation = wait->frontier.reservation;
    const iree_async_frontier_t* wait_frontier = wait->frontier.wait_frontier;
    wait->kind = IREE_HAL_TASK_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
    return iree_hal_task_queue_drain_alloca_submit_reservation(
        operation, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT, &reservation,
        /*reservation_failure_frontier=*/wait_frontier);
  }

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_drain_alloca_acquire(
      operation, &reservation, &acquire_info, &acquire_result));
  return iree_hal_task_queue_drain_alloca_on_acquire_result(
      operation, &reservation, &acquire_info, acquire_result);
}

// Handles a DEALLOCA operation: decommits the transient buffer and releases the
// reservation with a queue frontier that gates future pool reuse.
static void iree_hal_task_queue_drain_dealloca(
    iree_hal_task_queue_op_t* operation) {
  iree_hal_pool_t* pool = NULL;
  iree_hal_pool_reservation_t reservation;
  const bool has_reservation =
      iree_hal_local_transient_buffer_query_reservation(
          operation->dealloca.transient_buffer, &pool, &reservation);
  const iree_hal_buffer_params_t params = {
      .type = iree_hal_buffer_memory_type(operation->dealloca.transient_buffer),
      .access =
          iree_hal_buffer_allowed_access(operation->dealloca.transient_buffer),
      .usage =
          iree_hal_buffer_allowed_usage(operation->dealloca.transient_buffer),
  };

  iree_hal_local_transient_buffer_decommit(
      operation->dealloca.transient_buffer);

  const uint64_t epoch =
      iree_hal_task_queue_op_reserve_completion_epoch(operation);
  iree_async_single_frontier_t death_frontier;
  iree_async_single_frontier_initialize(&death_frontier, operation->axis,
                                        epoch);
  iree_hal_local_transient_buffer_release_reservation(
      operation->dealloca.transient_buffer,
      iree_async_single_frontier_as_const_frontier(&death_frontier));
  iree_hal_task_queue_profile_record_memory_event(
      operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX, pool,
      params, has_reservation ? &reservation : NULL,
      death_frontier.entry_count);
  if (has_reservation) {
    iree_hal_task_queue_profile_record_memory_event(
        operation, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX, pool,
        params, &reservation, death_frontier.entry_count);
  }
  iree_hal_task_queue_op_complete_with_epoch(operation, epoch);
}

//===----------------------------------------------------------------------===//
// Inline recording execution (fill, copy, update, inline dispatch)
//===----------------------------------------------------------------------===//

// Executes a block recording synchronously with a single worker.
// Used by the drain handlers for fill/copy/update and inline dispatch.
// On success, completes the operation. On failure, destroys it with the error.
//
// The processor context is stack-allocated. The .data state is allocated from
// the operation's arena (which is backed by the small block pool — typically
// fits in the same 4KB block that already holds the operation).
static void iree_hal_task_queue_execute_recording_inline(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation,
    iree_hal_cmd_block_recording_t* recording,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context) {
  iree_status_t status = iree_ok_status();

  // Allocate .data from the operation's arena.
  const iree_host_size_t state_size = iree_hal_cmd_block_state_size(
      recording->max_region_dispatch_count, recording->max_total_binding_count);
  void* state_storage = NULL;
  if (recording->first_block && state_size > 0) {
    status = iree_arena_allocate_aligned(
        &operation->arena, state_size, iree_alignof(iree_hal_cmd_block_state_t),
        &state_storage);
  }

  if (iree_status_is_ok(status) && recording->first_block) {
    // Context on the stack — no heap allocation.
    iree_hal_cmd_block_processor_context_t processor_context;
    iree_hal_cmd_block_processor_context_initialize(
        &processor_context, recording, /*binding_table=*/NULL,
        /*binding_table_length=*/0, (iree_hal_cmd_block_state_t*)state_storage,
        state_size);

    iree_hal_cmd_block_processor_worker_state_t worker_state = {0};
    iree_hal_cmd_block_processor_drain_result_t result;
    iree_hal_cmd_block_processor_drain(&processor_context, worker_context,
                                       &worker_state, &result);
    status =
        iree_hal_cmd_block_processor_context_consume_result(&processor_context);
  }

  iree_hal_cmd_block_recording_release(recording);

  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_destroy(operation, status);
  }
}

// Handles a FILL operation. Small fills (below the queue's
// inline_transfer_threshold) delegate to iree_hal_buffer_map_fill, which
// handles mapping, pattern expansion, cache flush for non-coherent memory,
// and unmap in one call — the block processor setup cost dominates the fill
// itself at those sizes. Larger fills build a single-command recording and
// run it through the inline block processor.
static iree_status_t iree_hal_task_queue_drain_fill(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context) {
  // Fast path: below the threshold, bypass the cmd builder entirely.
  if (operation->fill.length < queue->inline_transfer_threshold) {
    iree_status_t status = iree_hal_buffer_map_fill(
        operation->fill.target_buffer, operation->fill.target_offset,
        operation->fill.length, operation->fill.pattern,
        operation->fill.pattern_length);
    if (iree_status_is_ok(status)) {
      iree_hal_task_queue_op_complete(operation);
    } else {
      iree_hal_task_queue_op_destroy(operation, status);
    }
    return iree_ok_status();
  }

  // Framework path: map the target, build a single-command recording, and
  // execute it inline via the block processor.
  iree_hal_buffer_mapping_t mapping = {{0}};
  iree_status_t status = iree_hal_buffer_map_range(
      operation->fill.target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, operation->fill.target_offset,
      operation->fill.length, &mapping);

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(queue->large_block_pool, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_begin(&builder);
  }

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_build_fill(
        &builder, operation->fill.length, operation->fill.pattern,
        operation->fill.pattern_length, &fixups, &token);
  }
  if (iree_status_is_ok(status)) {
    fixups[0].host_ptr = mapping.contents.data;
    fixups[0].offset = 0;
    fixups[0].length = mapping.contents.data_length;
    fixups[0].slot = 0;
    fixups[0].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
  }

  iree_hal_cmd_block_recording_t recording;
  memset(&recording, 0, sizeof(recording));
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_end(&builder, &recording);
  }

  iree_hal_cmd_block_builder_deinitialize(&builder);
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_execute_recording_inline(queue, operation, &recording,
                                                 worker_context);
  }

  // Unmap the buffer after inline execution completes. Safe on zero-initialized
  // mappings (no-op if map_range was never called or failed).
  iree_hal_buffer_unmap_range(&mapping);
  return status;
}

// Handles a COPY operation. Small copies (below the queue's
// inline_transfer_threshold) delegate to iree_hal_buffer_map_copy, which
// maps both buffers, performs the memcpy, flushes non-coherent memory, and
// unmaps in one call — the block processor setup cost dominates the copy
// itself at those sizes. Larger copies build a single-command recording and
// run it through the inline block processor.
static iree_status_t iree_hal_task_queue_drain_copy(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context) {
  // Fast path: below the threshold, bypass the cmd builder entirely.
  if (operation->copy.length < queue->inline_transfer_threshold) {
    iree_status_t status = iree_hal_buffer_map_copy(
        operation->copy.source_buffer, operation->copy.source_offset,
        operation->copy.target_buffer, operation->copy.target_offset,
        operation->copy.length);
    if (iree_status_is_ok(status)) {
      iree_hal_task_queue_op_complete(operation);
    } else {
      iree_hal_task_queue_op_destroy(operation, status);
    }
    return iree_ok_status();
  }

  // Framework path: map both buffers, build a single-command recording,
  // and execute it inline via the block processor.
  iree_hal_buffer_mapping_t source_mapping = {{0}};
  iree_hal_buffer_mapping_t target_mapping = {{0}};
  iree_status_t status = iree_hal_buffer_map_range(
      operation->copy.source_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, operation->copy.source_offset,
      operation->copy.length, &source_mapping);
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        operation->copy.target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, operation->copy.target_offset,
        operation->copy.length, &target_mapping);
  }

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(queue->large_block_pool, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_begin(&builder);
  }

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_build_copy(&builder, operation->copy.length, &fixups,
                                     &token);
  }
  if (iree_status_is_ok(status)) {
    fixups[0].host_ptr = source_mapping.contents.data;
    fixups[0].offset = 0;
    fixups[0].length = source_mapping.contents.data_length;
    fixups[0].slot = 0;
    fixups[0].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
    fixups[1].host_ptr = target_mapping.contents.data;
    fixups[1].offset = 0;
    fixups[1].length = target_mapping.contents.data_length;
    fixups[1].slot = 0;
    fixups[1].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
  }

  iree_hal_cmd_block_recording_t recording;
  memset(&recording, 0, sizeof(recording));
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_end(&builder, &recording);
  }

  iree_hal_cmd_block_builder_deinitialize(&builder);
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_execute_recording_inline(queue, operation, &recording,
                                                 worker_context);
  }

  iree_hal_buffer_unmap_range(&source_mapping);
  iree_hal_buffer_unmap_range(&target_mapping);
  return status;
}

// Handles an UPDATE operation: maps the target buffer and copies the inline
// source data directly. The queue op already captured the caller's source data,
// so a single memcpy avoids extra command builder + processor dispatch
// overhead.
static iree_status_t iree_hal_task_queue_drain_update(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  iree_hal_buffer_mapping_t mapping = {{0}};
  iree_status_t status = iree_hal_buffer_map_range(
      operation->update.target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, operation->update.target_offset,
      operation->update.length, &mapping);
  if (iree_status_is_ok(status)) {
    memcpy(mapping.contents.data, (const uint8_t*)operation->update.source_data,
           (size_t)operation->update.length);
  }
  iree_hal_buffer_unmap_range(&mapping);

  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_destroy(operation, status);
  }
  return iree_ok_status();
}

// Handles a DISPATCH operation: builds a single-dispatch recording and either
// executes it inline (ALLOW_INLINE_EXECUTION) or routes it through the compute
// process for multi-worker tile distribution.
static iree_status_t iree_hal_task_queue_drain_dispatch(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context);

//===----------------------------------------------------------------------===//
// Compute process (data plane)
//===----------------------------------------------------------------------===//

// Publishes the next recording item after this item is closed. Called by the
// first worker to set CLOSED in item->drainers. This helper intentionally does
// not touch item->operation or item->processor_context: the last drainer owns
// result consumption, operation completion/failure, and teardown.
static void iree_hal_task_queue_compute_item_publish_next(
    iree_hal_task_queue_t* queue) {
  iree_hal_task_queue_compute_item_t* next =
      iree_hal_task_queue_compute_item_slist_pop(&queue->compute_pending);
  iree_hal_task_queue_store_compute_current(queue, next);
  if (next) {
    // Publishing a new current item from inside the already-running compute
    // process must reschedule that same process through the executor's normal
    // wake/sleep protocol. A raw needs_drain store is not enough here: the last
    // drainer of the previous item may be concurrently deciding whether to
    // release the process slot and transition to IDLE, and schedule_process is
    // the one API that closes that race and wakes idle workers.
    iree_task_executor_schedule_process(queue->executor,
                                        &queue->compute_process);
  }
}

// Fires final release for a recording item. Called by the one worker that
// successfully claims RELEASE_CLAIMED after CLOSED is set and the drainer
// count reaches zero. Consumes the processor result exactly once, completes or
// fails the owning operation, tears down item-owned execution state, bumps the
// generation, and returns the item to the free pool.
//
// Exclusive access invariant: winning the gen|CLOSED →
// gen|CLOSED|RELEASE_CLAIMED CAS in compute_item_leave guarantees no other
// worker is inside processor_drain() for this item, and any late CLOSED bailer
// that arrives is bounded to a fetch_add/fetch_sub pair on the drainers counter
// and never touches operation/processor_context/recording. That lets this
// function read and clear those fields without additional synchronization.
static void iree_hal_task_queue_compute_item_release(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_compute_item_t* item) {
  // Capture and clear item-owned state atomically with respect to readers of
  // the item: from here on, the item looks "free" to shutdown scans and to
  // anyone else inspecting the fields, even though the drainers tag still
  // reports CLOSED|RELEASE_CLAIMED until the generation CAS below.
  iree_hal_task_queue_op_t* operation = item->operation;
  iree_hal_cmd_block_processor_context_t* processor_context =
      item->processor_context;
  iree_hal_cmd_block_recording_t recording = item->recording;
  item->operation = NULL;
  item->processor_context = NULL;
  memset(&item->recording, 0, sizeof(item->recording));

  // Consume the processor's error before freeing the context. The write to
  // context->error_status happened on whichever worker observed
  // processor_result.completed; its fetch_or(CLOSED) and fetch_sub
  // transitively synchronize the write to this worker via the drainers chain.
  // Both the consume and the free tolerate a null processor_context
  // (no-block recordings skip context allocation entirely).
  iree_status_t processor_status =
      iree_hal_cmd_block_processor_context_consume_result(processor_context);
  iree_hal_cmd_block_processor_context_free(processor_context,
                                            item->host_allocator);

  // Release item-owned recording blocks. Command buffer recordings have
  // first_block == NULL because the CB retains its own recording through
  // operation->resource_set; recording_release() is a no-op in that case.
  iree_hal_cmd_block_recording_release(&recording);

  // Reset drainers: advance the generation and clear the count + all flags.
  //
  // Late CLOSED bailers may still have passed the compute_current quick check
  // before this item was unpublished. They register with fetch_add(1), observe
  // a negative low32 count from CLOSED|RELEASE_CLAIMED, then call
  // compute_item_leave() to undo the increment. A plain store here could race
  // with such a fetch_add and lose the bailer's claim, letting the stale
  // worker decrement the next lifecycle's count from zero to -1 (a CLOSED
  // bit spill into the generation bits). Spin on a CAS targeting exactly
  // gen|CLOSED|RELEASE_CLAIMED so we only reset once every outstanding
  // bailer has paired its fetch_add with a fetch_sub.
  //
  // compare_exchange_weak overwrites |expected_release_claimed| with the
  // current value on failure (including on spurious failures), so we
  // unconditionally rebuild it at the top of the loop body instead of
  // reusing whatever a bailer left behind.
  int64_t old_drainers =
      iree_atomic_load(&item->drainers, iree_memory_order_relaxed);
  const int64_t generation_bits = old_drainers & ~(int64_t)UINT32_MAX;
  const int64_t next_gen =
      generation_bits + IREE_HAL_TASK_QUEUE_ITEM_GEN_INCREMENT;
  const int64_t release_claimed_tag =
      generation_bits | IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT |
      IREE_HAL_TASK_QUEUE_ITEM_RELEASE_CLAIMED_BIT;
  int64_t expected_release_claimed = release_claimed_tag;
  while (!iree_atomic_compare_exchange_weak(
      &item->drainers, &expected_release_claimed, next_gen,
      iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_processor_yield();
    expected_release_claimed = release_claimed_tag;
  }
  iree_hal_task_queue_compute_item_slist_push(&queue->compute_free_pool, item);

  // Complete or fail the operation AFTER returning the item to the free pool.
  // op_complete/op_fail signal semaphores and run scope_end on the shared
  // queue->scope, which can wake a thread blocked on scope_wait_idle in
  // iree_hal_task_queue_deinitialize. That thread is still gated on the
  // queue's other scope references — in particular, the compute process
  // scope_begin at queue initialization is only released by the compute
  // process release callback after the last slot drainer exits — so the
  // queue will not actually be freed until this worker unwinds back out of
  // the compute slot. We still want the item back in the free pool first so
  // that if the waiter does wake and immediately resubmit on the same queue
  // (rather than tearing it down), the freshly-recycled item is available.
  if (iree_status_is_ok(processor_status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_fail(operation, processor_status);
  }
}

// Releases one drainer claim and runs final item cleanup if this was the
// last worker to leave after the item was closed.
//
// This helper must be used on every path that drops an item drainer claim,
// including early-bail races after CLOSED or compute_current identity changes.
// Otherwise a worker can decrement CLOSED_BIT|1 to CLOSED_BIT and strand the
// item forever with processor_context/operation still attached.
static void iree_hal_task_queue_compute_item_leave(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_compute_item_t* item,
    int64_t registered_generation) {
  int64_t exit_prev =
      iree_atomic_fetch_sub(&item->drainers, 1, iree_memory_order_acq_rel);
  IREE_ASSERT((exit_prev & ~(int64_t)UINT32_MAX) == registered_generation,
              "item generation changed while a drainer claim was still live");
  if ((int32_t)exit_prev !=
      ((int32_t)IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT | 1)) {
    return;
  }

  // A late CLOSED bailer can observe drainers=CLOSED_BIT between the real last
  // drainer's fetch_sub and its cleanup call. Claiming release via CAS ensures
  // exactly one worker runs final cleanup.
  int64_t generation = exit_prev & ~(int64_t)UINT32_MAX;
  int64_t expected_closed = generation | IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT;
  int64_t release_claimed =
      expected_closed | IREE_HAL_TASK_QUEUE_ITEM_RELEASE_CLAIMED_BIT;
  if (iree_atomic_compare_exchange_strong(
          &item->drainers, &expected_closed, release_claimed,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_task_queue_compute_item_release(queue, item);
  }
}

// Compute process drain function. Called by workers from the compute slot.
// Loads the current recording item, drains it cooperatively via the block
// processor, and handles per-recording two-phase completion.
//
// Memory ordering protocol for compute_current:
//   queue->compute_current_revision is a seqlock-style publication sequence for
//   queue->compute_current. Writers set the low bit while publishing and then
//   advance to the next even revision. Workers snapshot the revision, load
//   compute_current, register a drainer, and re-check both revision and pointer
//   before entering the block processor. This rejects same-address recycled
//   items without relying on copying item generations into a fragile sideband.
//
// Per-recording drainer protocol:
//   Workers increment the drainer count in item->drainers before entering
//   processor_drain and decrement it after. The increment uses acq_rel; the
//   item generation snapshot and the post-increment compute_current revision
//   re-check close the TOCTOU window where the item could be recycled between
//   the initial load and the increment.
static iree_status_t iree_hal_task_queue_compute_process_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  iree_hal_task_queue_t* queue = (iree_hal_task_queue_t*)process->user_data;

  // Check for shutdown.
  if (iree_hal_task_queue_is_shutting_down(queue)) {
    IREE_TRACE(iree_hal_task_queue_trace_compute_drain_event());
    out_result->did_work = false;
    out_result->completed = true;
    return iree_ok_status();
  }

  int64_t current_revision = iree_atomic_load(&queue->compute_current_revision,
                                              iree_memory_order_acquire);
  if (IREE_UNLIKELY(current_revision &
                    IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_REVISION_UPDATING)) {
    // A publication owner is in the middle of swapping compute_current.
    // Non-owner drainers have no useful work to do until that writer finishes,
    // and reporting did_work=true here turns one delayed publisher into a
    // full-pool livelock where every worker spins on the odd revision bit.
    // Returning did_work=false lets non-owner workers quiesce; if the writer
    // publishes a new item it will schedule the compute process through the
    // normal wake path, and if it publishes NULL there is no work to preserve.
    IREE_TRACE(iree_hal_task_queue_trace_compute_drain_event());
    out_result->did_work = false;
    out_result->completed = false;
    return iree_ok_status();
  }

  // Load the current recording item.
  iree_hal_task_queue_compute_item_t* item =
      (iree_hal_task_queue_compute_item_t*)iree_atomic_load(
          &queue->compute_current, iree_memory_order_acquire);

  if (item) {
    int64_t item_generation = iree_hal_task_queue_compute_item_generation(item);

    // Register as a drainer via fetch_add on the 64-bit drainers field.
    // Items have stable addresses (arena-allocated, never freed during
    // operation), so the pointer is always valid.
    // If the CLOSED flag is set (bit 31 of the low 32 bits), the recording
    // is completing. Bail, but ask the worker to stay active and retry instead
    // of sleeping. The closer has already installed the next pending recording
    // into compute_current (or null if none). On the next pump iteration the
    // worker will find the new item or properly sleep from the null path.
    int64_t prev_drainers =
        iree_atomic_fetch_add(&item->drainers, 1, iree_memory_order_acq_rel);
    int64_t registered_generation = prev_drainers & ~(int64_t)UINT32_MAX;
    if (IREE_UNLIKELY((prev_drainers & ~(int64_t)UINT32_MAX) !=
                          item_generation ||
                      (int32_t)prev_drainers < 0)) {
      IREE_TRACE(iree_hal_task_queue_trace_compute_drain_event());
      iree_hal_task_queue_compute_item_leave(queue, item,
                                             registered_generation);
      out_result->did_work = false;
      out_result->keep_active = true;
      out_result->completed = false;
      return iree_ok_status();
    }

    // Re-check compute_current and its publication revision for identity
    // safety. If this item was recycled to the same address, compute_current
    // may still compare equal but compute_current_revision will have changed.
    if (iree_atomic_load(&queue->compute_current_revision,
                         iree_memory_order_acquire) != current_revision ||
        (iree_hal_task_queue_compute_item_t*)iree_atomic_load(
            &queue->compute_current, iree_memory_order_acquire) != item) {
      IREE_TRACE(iree_hal_task_queue_trace_compute_drain_event());
      iree_hal_task_queue_compute_item_leave(queue, item,
                                             registered_generation);
      out_result->did_work = false;
      out_result->keep_active = true;
      out_result->completed = false;
      return iree_ok_status();
    }

    IREE_TRACE_ZONE_BEGIN_NAMED(z_drain, "iree_hal_local_task_compute_drain");
    if (IREE_UNLIKELY(queue->profile_recorder)) {
      iree_hal_task_queue_profile_start_host_execution(item->operation);
    }
    const uint32_t local_worker_index =
        worker_context->worker_index % item->worker_count;
    iree_hal_cmd_block_processor_worker_state_t* worker_state =
        &item->worker_states[local_worker_index];
    const iree_hal_cmd_block_processor_worker_context_t
        processor_worker_context = {
            .worker_index = local_worker_index,
            .processor_id = worker_context->processor_id,
            .local_memory = worker_context->local_memory,
        };
    iree_hal_cmd_block_processor_drain_result_t processor_result;
    memset(&processor_result, 0, sizeof(processor_result));
    iree_hal_cmd_block_processor_drain(item->processor_context,
                                       &processor_worker_context, worker_state,
                                       &processor_result);

    // Handle per-recording completion via CLOSED flag. The first worker to set
    // CLOSED only publishes the next recording; the last drainer consumes the
    // processor result and finalizes this item after all drainers have exited.
    if (processor_result.completed) {
      int64_t close_prev = iree_atomic_fetch_or(
          &item->drainers, IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT,
          iree_memory_order_acq_rel);
      if (!((int32_t)close_prev &
            (int32_t)IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT)) {
        iree_hal_task_queue_compute_item_publish_next(queue);
      }
    }

    bool processor_did_work = true;
    bool processor_keep_active = false;
    bool processor_publish_keep_active = false;
    bool processor_keep_warm = false;
    int32_t processor_keep_warm_retainer_limit = 0;
    bool processor_keep_warm_spin = false;
    int32_t processor_keep_warm_epoch = 0;
    if (!processor_result.completed && processor_result.tiles_executed == 0) {
      // Sample the process epoch before validating the processor state. If a
      // transition races after this sample, the warm waiter will observe the
      // epoch change immediately; if the transition already happened,
      // context_did_advance catches the stale no-work observation.
      processor_keep_warm_epoch = iree_task_process_retention_epoch(process);
      if (iree_hal_cmd_block_processor_context_did_advance(
              item->processor_context, &processor_result)) {
        processor_keep_active = true;
        processor_publish_keep_active = true;
      } else {
        int32_t expected_retain_drain = 0;
        if (iree_atomic_compare_exchange_strong(
                &process->retain_drain, &expected_retain_drain, 1,
                iree_memory_order_acq_rel, iree_memory_order_acquire)) {
          // A warm wait needs some drainer to advance the command-buffer
          // retention epoch. Elect one no-work/no-advance drainer to stay
          // active so peer drainers can park warm without stranding progress.
          processor_keep_active = true;
        } else {
          processor_keep_warm = true;
          processor_keep_warm_retainer_limit =
              processor_result.warm_retainer_limit;
          processor_keep_warm_spin = processor_result.prefer_warm_spin;
        }
      }
      processor_did_work = false;
      if (IREE_UNLIKELY(
              item->processor_context->profile.command_region.events_enabled)) {
        iree_hal_cmd_block_processor_context_profile_record_retention(
            item->processor_context, processor_keep_active,
            processor_publish_keep_active, processor_keep_warm);
      }
    }

    // Release our drainer claim. If we were the last drainer after close,
    // fire final cleanup.
    iree_hal_task_queue_compute_item_leave(queue, item, registered_generation);
    IREE_TRACE_ZONE_END(z_drain);

    out_result->did_work = processor_did_work;
    out_result->keep_active = processor_keep_active;
    out_result->publish_keep_active = processor_publish_keep_active;
    out_result->keep_warm = processor_keep_warm;
    out_result->keep_warm_retainer_limit = processor_keep_warm_retainer_limit;
    out_result->keep_warm_spin = processor_keep_warm_spin;
    out_result->keep_warm_epoch = processor_keep_warm_epoch;
    out_result->completed = false;
    return iree_ok_status();
  }

  // No current item. Try to promote one pending recording under the
  // compute_current publication lock.
  if (iree_hal_task_queue_promote_pending_to_compute_current(queue)) {
    IREE_TRACE(iree_hal_task_queue_trace_compute_drain_event());
    out_result->did_work = true;
    out_result->completed = false;
    return iree_ok_status();
  }

  // Nothing available. Workers will park via the sleeping protocol.
  IREE_TRACE(iree_hal_task_queue_trace_compute_drain_event());
  out_result->did_work = false;
  out_result->completed = false;
  return iree_ok_status();
}

// Cleans up a compute item during shutdown. Called from the compute process
// release callback, which only runs after all slot drainers have exited, so
// no worker can be inside processor_drain() or compute_item_leave() for this
// queue. Unlike the normal release path, the item is not returned to the
// free pool — the pool itself is about to be torn down.
static void iree_hal_task_queue_compute_item_cleanup(
    iree_hal_task_queue_compute_item_t* item) {
  // Capture and clear owned state in the same order as compute_item_release
  // so readers see a consistent "cleared" item after this returns.
  iree_hal_task_queue_op_t* operation = item->operation;
  iree_hal_cmd_block_processor_context_t* processor_context =
      item->processor_context;
  iree_hal_cmd_block_recording_t recording = item->recording;
  item->operation = NULL;
  item->processor_context = NULL;
  memset(&item->recording, 0, sizeof(item->recording));

  // Consume the processor's error before freeing the context. Both calls
  // tolerate a null processor_context; no-block recordings never allocated
  // one, and shutdown may catch an item whose context was already freed by
  // a concurrent normal release path that lost the slot release race.
  iree_status_t processor_status =
      iree_hal_cmd_block_processor_context_consume_result(processor_context);
  iree_hal_cmd_block_processor_context_free(processor_context,
                                            item->host_allocator);

  // Release item-owned recording blocks. No-op if first_block is NULL
  // (command buffer recordings, or items that already cleared it).
  iree_hal_cmd_block_recording_release(&recording);

  // If an operation is still attached, its semaphores were never signaled;
  // destroy it with the processor's error if there was one, otherwise a
  // CANCELLED status so waiters see a concrete shutdown reason.
  if (operation) {
    iree_status_t operation_status = processor_status;
    if (iree_status_is_ok(operation_status)) {
      operation_status =
          iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
    }
    iree_hal_task_queue_op_destroy(operation, operation_status);
  } else {
    iree_status_free(processor_status);
  }
}

// Compute process release callback. Fires when the last worker exits the
// compute slot (slot->active_drainers sentinel CAS succeeds in worker.c).
// At that point no worker can be inside processor_drain() or
// compute_item_leave() for this queue, so this is the one safe place to
// finalize any in-flight items and to release the compute process's
// scope_begin claim on queue->scope.
//
// scope_end is deferred to here (rather than to a completion_fn) because
// completion_fn can fire on the first worker to observe termination while
// other workers are still inside drain. Releasing the scope there would let
// a thread blocked in iree_hal_task_queue_deinitialize / scope_wait_idle
// wake up and free the queue out from under those still-draining workers.
// The scope_end call at the bottom of this function is therefore required
// to be the last access to |queue|.
static void iree_hal_task_queue_compute_process_release(
    iree_task_process_t* process) {
  iree_hal_task_queue_t* queue = (iree_hal_task_queue_t*)process->user_data;

  // Scan all allocated pool items for any that still have unfinalized state.
  // An item can be in one of three states when shutdown reaches this point:
  //
  //   (a) Free pool: operation == NULL, processor_context == NULL,
  //       recording.first_block == NULL. Nothing to do.
  //   (b) Mid-drain: currently installed as compute_current or queued in
  //       compute_pending, with its operation still attached. Shutdown
  //       interrupted processing before the item closed, so we must destroy
  //       the operation (signaling its semaphores with CANCELLED) and
  //       release item-owned execution state.
  //   (c) Closed-but-unreleased: the first drainer set CLOSED and
  //       publish_next rotated compute_current away, but the final release
  //       lost the slot release race with shutdown. These items are not
  //       reachable from compute_current or compute_pending anymore, so
  //       only a full-pool scan can find them.
  //
  // All three classes are handled uniformly by compute_item_cleanup.
  //
  // The pool scan is race-free because the slot release only fires after
  // slot->active_drainers reaches the sentinel (no workers in drain), and
  // the compute process's schedule_state transitions to IDLE exclusively in
  // release_compute_process (never from drain), so no other slot lifetime
  // can overlap this callback.
  for (iree_hal_task_queue_compute_item_t* item = queue->compute_item_head;
       item != NULL; item = item->next_allocated) {
    if (item->operation || item->processor_context ||
        item->recording.first_block) {
      iree_hal_task_queue_compute_item_cleanup(item);
    }
  }

  // Reset compute_current (may have been pointing to a cleaned-up item).
  iree_atomic_store(&queue->compute_current,
                    IREE_HAL_TASK_QUEUE_COMPUTE_CURRENT_NULL,
                    iree_memory_order_relaxed);
  iree_atomic_store(&queue->compute_current_revision, 0,
                    iree_memory_order_relaxed);

  // Discard the pending list. Items here were filled by the wake_budget == 1
  // process but never installed as compute_current. They were already cleaned
  // up in the pool scan above; just clear the slist head.
  iree_hal_task_queue_compute_item_slist_discard(&queue->compute_pending);

  // End the scope for the compute process (paired with scope_begin at init).
  // After this call, scope_wait_idle may unblock; the process-release count
  // below is the final queue lifetime barrier for the callback itself.
  iree_task_scope_end(&queue->scope);
  iree_atomic_fetch_sub(&queue->pending_process_release_count, 1,
                        iree_memory_order_release);
}

//===----------------------------------------------------------------------===//
// File I/O (proactor-based async read/write)
//===----------------------------------------------------------------------===//

// Context for an in-flight file read or write operation. Arena-allocated from
// the queue operation's arena and valid until the proactor completion callback
// fires. The callback unmaps the buffer and completes/destroys the operation,
// which frees the arena (and this context with it).
typedef struct iree_hal_task_queue_io_context_t {
  iree_hal_task_queue_op_t* operation;
  iree_async_proactor_t* proactor;
  iree_hal_buffer_mapping_t mapping;
  // Total bytes transferred across all resubmissions (the operation's
  // bytes_read/bytes_written field is per-submission).
  iree_host_size_t total_bytes_transferred;
  // Original requested length (for short read detection at completion).
  iree_host_size_t requested_length;
  union {
    iree_async_file_read_operation_t read_op;
    iree_async_file_write_operation_t write_op;
  };
} iree_hal_task_queue_io_context_t;

// Proactor completion callback for file read operations. Handles partial
// reads by advancing the buffer/offset and resubmitting until the full
// requested length is transferred (or EOF/error).
static void iree_hal_task_queue_io_read_completion(
    void* user_data, iree_async_operation_t* base_op, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_task_queue_io_context_t* io_context =
      (iree_hal_task_queue_io_context_t*)user_data;
  iree_hal_task_queue_op_t* operation = io_context->operation;

  // Accumulate bytes from this submission and check for partial reads.
  if (iree_status_is_ok(status) && io_context->read_op.bytes_read > 0) {
    iree_host_size_t bytes_read = io_context->read_op.bytes_read;
    io_context->total_bytes_transferred += bytes_read;

    // If we haven't read the full amount, advance and resubmit.
    if (io_context->total_bytes_transferred < io_context->requested_length) {
      iree_async_file_t* file = io_context->read_op.file;
      uint64_t next_offset = io_context->read_op.offset + bytes_read;
      iree_async_span_t next_buffer = iree_async_span_from_ptr(
          (uint8_t*)iree_async_span_ptr(io_context->read_op.buffer) +
              bytes_read,
          io_context->read_op.buffer.length - bytes_read);
      iree_async_operation_zero(&io_context->read_op.base,
                                sizeof(io_context->read_op));
      iree_async_operation_initialize(
          &io_context->read_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
          IREE_ASYNC_OPERATION_FLAG_NONE,
          iree_hal_task_queue_io_read_completion, io_context);
      io_context->read_op.file = file;
      io_context->read_op.offset = next_offset;
      io_context->read_op.buffer = next_buffer;
      iree_status_t resubmit_status = iree_async_proactor_submit_one(
          io_context->proactor, &io_context->read_op.base);
      if (iree_status_is_ok(resubmit_status)) return;
      status = resubmit_status;
    }
  } else if (iree_status_is_ok(status) && io_context->total_bytes_transferred <
                                              io_context->requested_length) {
    // bytes_read == 0 but we haven't reached the requested length: EOF.
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "short read: requested %" PRIhsz " bytes, got %" PRIhsz,
        io_context->requested_length, io_context->total_bytes_transferred);
  }

  // Flush non-coherent memory: proactor wrote data into the mapped buffer
  // and the device needs to see it.
  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(
          iree_hal_buffer_memory_type(io_context->mapping.buffer),
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_status_join(
        status,
        iree_hal_buffer_mapping_flush_range(
            &io_context->mapping, 0, io_context->mapping.contents.data_length));
  }

  // Unmap the buffer.
  status = iree_status_join(status,
                            iree_hal_buffer_unmap_range(&io_context->mapping));

  // Complete or fail the operation. This frees the arena (and io_context).
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_fail(operation, status);
  }
}

// Proactor completion callback for file write operations.
static void iree_hal_task_queue_io_write_completion(
    void* user_data, iree_async_operation_t* base_op, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_task_queue_io_context_t* io_context =
      (iree_hal_task_queue_io_context_t*)user_data;
  iree_hal_task_queue_op_t* operation = io_context->operation;

  // Accumulate bytes from this submission and check for partial writes.
  if (iree_status_is_ok(status) && io_context->write_op.bytes_written > 0) {
    iree_host_size_t bytes_written = io_context->write_op.bytes_written;
    io_context->total_bytes_transferred += bytes_written;

    // If we haven't written the full amount, advance and resubmit.
    if (io_context->total_bytes_transferred < io_context->requested_length) {
      iree_async_file_t* file = io_context->write_op.file;
      uint64_t next_offset = io_context->write_op.offset + bytes_written;
      iree_async_span_t next_buffer = iree_async_span_from_ptr(
          (uint8_t*)iree_async_span_ptr(io_context->write_op.buffer) +
              bytes_written,
          io_context->write_op.buffer.length - bytes_written);
      iree_async_operation_zero(&io_context->write_op.base,
                                sizeof(io_context->write_op));
      iree_async_operation_initialize(
          &io_context->write_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
          IREE_ASYNC_OPERATION_FLAG_NONE,
          iree_hal_task_queue_io_write_completion, io_context);
      io_context->write_op.file = file;
      io_context->write_op.offset = next_offset;
      io_context->write_op.buffer = next_buffer;
      iree_status_t resubmit_status = iree_async_proactor_submit_one(
          io_context->proactor, &io_context->write_op.base);
      if (iree_status_is_ok(resubmit_status)) return;
      status = resubmit_status;
    }
  } else if (iree_status_is_ok(status) && io_context->total_bytes_transferred <
                                              io_context->requested_length) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "short write: requested %" PRIhsz " bytes, wrote %" PRIhsz,
        io_context->requested_length, io_context->total_bytes_transferred);
  }

  // Unmap the buffer.
  status = iree_status_join(status,
                            iree_hal_buffer_unmap_range(&io_context->mapping));

  // Complete or fail the operation. This frees the arena (and io_context).
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_fail(operation, status);
  }
}

// Synchronous fallback for READ when no async file handle is available.
// Calls iree_hal_file_read() inline on the worker thread. Blocks the worker
// for the duration of the read but is always correct.
static iree_status_t iree_hal_task_queue_drain_read_sync(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  iree_status_t status =
      iree_hal_file_read(operation->read.hal_file, operation->read.file_offset,
                         operation->read.buffer, operation->read.buffer_offset,
                         operation->read.length);
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_fail(operation, status);
  }
  return iree_ok_status();
}

// Handles a READ operation: maps the target buffer, submits an async proactor
// read, and returns immediately. The proactor callback handles unmapping and
// operation completion.
static iree_status_t iree_hal_task_queue_drain_read(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  // Synchronous fallback when async import failed or is unavailable.
  if (!operation->read.async_file) {
    return iree_hal_task_queue_drain_read_sync(queue, operation);
  }
  iree_hal_task_queue_profile_add_host_flags(
      operation, IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_DEFERRED);

  // Map the target buffer for writing.
  iree_hal_task_queue_io_context_t* io_context = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &operation->arena, sizeof(*io_context), (void**)&io_context));
  memset(io_context, 0, sizeof(*io_context));
  io_context->operation = operation;
  io_context->proactor = queue->proactor;
  io_context->requested_length = (iree_host_size_t)operation->read.length;

  iree_status_t status = iree_hal_buffer_map_range(
      operation->read.buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, operation->read.buffer_offset,
      operation->read.length, &io_context->mapping);
  if (!iree_status_is_ok(status)) return status;

  // Initialize the proactor read operation.
  iree_async_operation_zero(&io_context->read_op.base,
                            sizeof(io_context->read_op));
  iree_async_operation_initialize(
      &io_context->read_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_task_queue_io_read_completion,
      io_context);
  io_context->read_op.file = operation->read.async_file;
  io_context->read_op.offset = operation->read.file_offset;
  io_context->read_op.buffer =
      iree_async_span_from_ptr(io_context->mapping.contents.data,
                               (iree_host_size_t)operation->read.length);

  // Submit to the proactor and return immediately.
  iree_status_t submit_status = iree_async_proactor_submit_one(
      queue->proactor, &io_context->read_op.base);
  if (!iree_status_is_ok(submit_status)) {
    iree_hal_buffer_unmap_range(&io_context->mapping);
  }
  return submit_status;
}

// Synchronous fallback for WRITE when no async file handle is available.
static iree_status_t iree_hal_task_queue_drain_write_sync(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  iree_status_t status = iree_hal_file_write(
      operation->write.hal_file, operation->write.file_offset,
      operation->write.buffer, operation->write.buffer_offset,
      operation->write.length);
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_op_complete(operation);
  } else {
    iree_hal_task_queue_op_fail(operation, status);
  }
  return iree_ok_status();
}

// Handles a WRITE operation: maps the source buffer, submits an async proactor
// write, and returns immediately. The proactor callback handles unmapping and
// operation completion.
static iree_status_t iree_hal_task_queue_drain_write(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation) {
  if (!operation->write.async_file) {
    return iree_hal_task_queue_drain_write_sync(queue, operation);
  }
  iree_hal_task_queue_profile_add_host_flags(
      operation, IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_DEFERRED);

  // Map the source buffer for reading.
  iree_hal_task_queue_io_context_t* io_context = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &operation->arena, sizeof(*io_context), (void**)&io_context));
  memset(io_context, 0, sizeof(*io_context));
  io_context->operation = operation;
  io_context->proactor = queue->proactor;
  io_context->requested_length = (iree_host_size_t)operation->write.length;

  iree_status_t status = iree_hal_buffer_map_range(
      operation->write.buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, operation->write.buffer_offset,
      operation->write.length, &io_context->mapping);
  if (!iree_status_is_ok(status)) return status;

  // Invalidate non-coherent memory: the device may have written data into
  // the buffer and we need to see it before reading for the file write.
  if (!iree_all_bits_set(
          iree_hal_buffer_memory_type(io_context->mapping.buffer),
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_invalidate_range(
        &io_context->mapping, 0, io_context->mapping.contents.data_length);
    if (!iree_status_is_ok(status)) {
      iree_hal_buffer_unmap_range(&io_context->mapping);
      return status;
    }
  }

  // Initialize the proactor write operation.
  iree_async_operation_zero(&io_context->write_op.base,
                            sizeof(io_context->write_op));
  iree_async_operation_initialize(
      &io_context->write_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_task_queue_io_write_completion,
      io_context);
  io_context->write_op.file = operation->write.async_file;
  io_context->write_op.offset = operation->write.file_offset;
  io_context->write_op.buffer =
      iree_async_span_from_ptr(io_context->mapping.contents.data,
                               (iree_host_size_t)operation->write.length);

  // Submit to the proactor and return immediately.
  iree_status_t submit_status = iree_async_proactor_submit_one(
      queue->proactor, &io_context->write_op.base);
  if (!iree_status_is_ok(submit_status)) {
    iree_hal_buffer_unmap_range(&io_context->mapping);
  }
  return submit_status;
}

//===----------------------------------------------------------------------===//
// Control process (wake_budget == 1 queue drain)
//===----------------------------------------------------------------------===//

// Queue process release callback. The queue process is wake_budget == 1, so
// exactly one worker drains it; release fires on that worker immediately after
// iree_task_process_complete returns, and worker.c makes no further access to
// |process| after this callback. Ending the scope reference here (rather than
// in a completion_fn) keeps scope_wait_idle a true lifetime barrier for queue
// and device memory: ending the scope from completion_fn would unblock waiters
// while the owning worker still has a live stack frame for the embedded
// process.
//
// completion_fn is intentionally unset — the base process code consumes the
// terminal status for us when no callback is installed.
static void iree_hal_task_queue_process_release(iree_task_process_t* process) {
  iree_hal_task_queue_t* queue = (iree_hal_task_queue_t*)process->user_data;
  iree_task_scope_end(&queue->scope);
  iree_atomic_fetch_sub(&queue->pending_process_release_count, 1,
                        iree_memory_order_release);
}

// Queue process drain function. Pops one operation from the ready list and
// handles it by type. Returns did_work=false when the ready list is empty
// (the executor's sleeping protocol will park the process). During
// deinitialize, cancels remaining ready work before returning completed=true.
static iree_status_t iree_hal_task_queue_process_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  iree_hal_task_queue_t* queue = (iree_hal_task_queue_t*)process->user_data;
  const iree_hal_cmd_block_processor_worker_context_t processor_worker_context =
      {
          .worker_index = worker_context->worker_index,
          .processor_id = worker_context->processor_id,
          .local_memory = worker_context->local_memory,
      };

  iree_hal_task_queue_op_t* operation =
      iree_hal_task_queue_op_slist_pop(&queue->ready_list);
  if (!operation) {
    out_result->did_work = false;
    out_result->completed = iree_hal_task_queue_is_shutting_down(queue);
    return iree_ok_status();
  }

  if (iree_hal_task_queue_is_shutting_down(queue)) {
    if (operation->type == IREE_HAL_TASK_QUEUE_OP_DEALLOCA) {
      iree_hal_task_queue_profile_start_host_execution(operation);
      iree_hal_task_queue_drain_dealloca(operation);
    } else {
      iree_hal_task_queue_op_destroy(
          operation, iree_hal_task_queue_make_shutdown_status());
    }
    out_result->did_work = true;
    out_result->completed = false;
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  switch (operation->type) {
    case IREE_HAL_TASK_QUEUE_OP_COMMANDS:
      status = iree_hal_task_queue_drain_commands(queue, operation);
      if (!iree_status_is_ok(status)) {
        // Issue failed — destroy the operation with the error.
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();  // Don't fail the queue process itself.
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_BARRIER:
      // Barriers just signal and clean up — no CB to execute.
      iree_hal_task_queue_profile_start_host_execution(operation);
      iree_hal_task_queue_op_complete(operation);
      break;
    case IREE_HAL_TASK_QUEUE_OP_HOST_CALL:
      iree_hal_task_queue_profile_start_host_execution(operation);
      status = iree_hal_task_queue_drain_host_call(queue, operation);
      break;
    case IREE_HAL_TASK_QUEUE_OP_ALLOCA:
      status = iree_hal_task_queue_drain_alloca(operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_profile_record_failed_before_ready(operation);
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_DEALLOCA:
      iree_hal_task_queue_profile_start_host_execution(operation);
      iree_hal_task_queue_drain_dealloca(operation);
      break;
    case IREE_HAL_TASK_QUEUE_OP_READ:
      iree_hal_task_queue_profile_start_host_execution(operation);
      status = iree_hal_task_queue_drain_read(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_WRITE:
      iree_hal_task_queue_profile_start_host_execution(operation);
      status = iree_hal_task_queue_drain_write(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_FILL:
      iree_hal_task_queue_profile_start_host_execution(operation);
      status = iree_hal_task_queue_drain_fill(queue, operation,
                                              &processor_worker_context);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_COPY:
      iree_hal_task_queue_profile_start_host_execution(operation);
      status = iree_hal_task_queue_drain_copy(queue, operation,
                                              &processor_worker_context);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_UPDATE:
      iree_hal_task_queue_profile_start_host_execution(operation);
      status = iree_hal_task_queue_drain_update(queue, operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_DISPATCH:
      status = iree_hal_task_queue_drain_dispatch(queue, operation,
                                                  &processor_worker_context);
      if (!iree_status_is_ok(status)) {
        iree_hal_task_queue_op_destroy(operation, status);
        status = iree_ok_status();
      }
      break;
  }

  out_result->did_work = true;
  out_result->completed = false;
  return status;
}

//===----------------------------------------------------------------------===//
// Submit paths
//===----------------------------------------------------------------------===//

// Phase 1 of submit: allocates the operation and begins scope tracking.
// The caller fills type-specific union fields and retains resources on the
// returned operation, then calls submit_op_finish to register waits and
// enqueue. On failure between begin and finish, the caller must call
// op_destroy on the operation.
static iree_status_t iree_hal_task_queue_submit_op_begin(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_type_t type,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_hal_task_queue_op_t** out_operation) {
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_op_allocate(
      queue, type, signal_semaphores, out_operation));
  iree_task_scope_begin(&queue->scope);
  return iree_ok_status();
}

// Phase 2 of submit: retains resources, registers semaphore waits, and
// enqueues the operation. On failure, destroys the operation.
static iree_status_t iree_hal_task_queue_submit_op_finish(
    iree_hal_task_queue_op_t* operation,
    iree_hal_semaphore_list_t wait_semaphores, iree_host_size_t resource_count,
    iree_hal_resource_t* const* resources) {
  iree_status_t status = iree_ok_status();
  const iree_host_size_t original_wait_count = wait_semaphores.count;
  if (resource_count > 0) {
    status = iree_hal_resource_set_insert(operation->resource_set,
                                          resource_count, resources);
  }
  if (iree_status_is_ok(status) && wait_semaphores.count > 0) {
    status = iree_hal_task_queue_try_satisfy_waits(&wait_semaphores);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_task_queue_profile_set_waits(operation, original_wait_count,
                                          wait_semaphores.count);
    status = iree_hal_task_queue_enqueue_waits(operation, wait_semaphores);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
  }
  return status;
}

// Convenience wrapper: allocates, retains resources, and enqueues in one call.
// For operations with no type-specific union fields (barriers, host calls
// with simple parameters).
static iree_status_t iree_hal_task_queue_submit_op(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_type_t type,
    iree_hal_semaphore_list_t wait_semaphores,
    const iree_hal_semaphore_list_t* signal_semaphores,
    iree_host_size_t resource_count, iree_hal_resource_t* const* resources) {
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, type, signal_semaphores, &operation));
  return iree_hal_task_queue_submit_op_finish(operation, wait_semaphores,
                                              resource_count, resources);
}

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_task_queue_initialize(
    iree_string_view_t identifier, iree_hal_queue_affinity_t affinity,
    iree_task_scope_flags_t scope_flags, iree_task_executor_t* executor,
    iree_async_proactor_t* proactor,
    iree_device_size_t inline_transfer_threshold,
    iree_arena_block_pool_t* small_block_pool,
    iree_arena_block_pool_t* large_block_pool,
    iree_hal_allocator_t* device_allocator, iree_hal_task_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, identifier.data, identifier.size);

  memset(out_queue, 0, sizeof(*out_queue));

  out_queue->affinity = affinity;
  out_queue->executor = executor;
  iree_task_executor_retain(out_queue->executor);
  out_queue->proactor = proactor;
  iree_atomic_store(&out_queue->epoch, 0, iree_memory_order_relaxed);
  out_queue->inline_transfer_threshold = inline_transfer_threshold;
  out_queue->small_block_pool = small_block_pool;
  out_queue->large_block_pool = large_block_pool;
  out_queue->device_allocator = device_allocator;
  iree_hal_allocator_retain(out_queue->device_allocator);

  iree_task_scope_initialize(identifier, scope_flags, &out_queue->scope);
  iree_atomic_store(&out_queue->pending_process_release_count, 2,
                    iree_memory_order_relaxed);

  // Initialize the ready list.
  iree_hal_task_queue_op_slist_initialize(&out_queue->ready_list);

  // Initialize the queue process. It uses wake_budget == 1 and starts
  // suspended with suspend_count=0 (immediately runnable but not scheduled
  // until the first submission arrives).
  iree_task_process_initialize(iree_hal_task_queue_process_drain,
                               /*suspend_count=*/0, /*wake_budget=*/1,
                               &out_queue->process);
  out_queue->process.user_data = out_queue;
  // completion_fn is intentionally NULL — process.c consumes the terminal
  // status when no completion callback is installed. scope_end is deferred to
  // release_fn so it fires after the owning worker has exited the drain stack.
  out_queue->process.release_fn = iree_hal_task_queue_process_release;

  // The queue process participates in the scope so that scope_wait_idle
  // blocks until the process has fully completed (no worker touching
  // queue/device memory). The matching scope_end fires in the release callback
  // after the owning worker has exited the drain stack.
  iree_task_scope_begin(&out_queue->scope);

  // Initialize the compute process. It uses wake_budget > 1 with N equal to
  // the worker count. Not scheduled until the first recording is pushed to
  // compute_pending by drain_commands. The first schedule_process call places
  // it in a compute slot; subsequent calls just wake workers.
  //
  // The matching scope_end fires in the RELEASE callback (not the completion
  // callback). For wake_budget > 1 processes, completion fires eagerly while
  // other workers may still be draining; scope_end there would allow the main
  // thread to free the queue prematurely. The release callback fires only after
  // the last drainer exits, making it safe to unblock scope_wait_idle.
  iree_task_process_initialize(
      iree_hal_task_queue_compute_process_drain,
      /*suspend_count=*/0,
      /*wake_budget=*/(int32_t)iree_task_executor_worker_count(executor),
      &out_queue->compute_process);
  iree_task_process_set_flags(&out_queue->compute_process,
                              IREE_TASK_PROCESS_FLAG_COMPUTE_SLOT);
  out_queue->compute_process.user_data = out_queue;
  // completion_fn is intentionally NULL. For wake_budget > 1 processes,
  // completion fires eagerly on the first worker that observes termination
  // while other workers may still be inside drain; doing any queue teardown
  // there would race them. All teardown (item cleanup + scope_end) lives in
  // release_fn, which fires only after the last slot drainer exits.
  out_queue->compute_process.release_fn =
      iree_hal_task_queue_compute_process_release;
  iree_task_scope_begin(&out_queue->scope);

  // Initialize the compute pending and free pool lists.
  iree_hal_task_queue_compute_item_slist_initialize(
      &out_queue->compute_pending);
  iree_hal_task_queue_compute_item_slist_initialize(
      &out_queue->compute_free_pool);

  // compute_current is NULL (no active recording) — memset handles this.

  // Cache worker count for item FAM sizing.
  uint32_t worker_count = (uint32_t)iree_task_executor_worker_count(executor);
  if (worker_count == 0) worker_count = 1;
  out_queue->compute_worker_count = worker_count;

  // Initialize the arena for item allocation from the large block pool.
  iree_arena_initialize(large_block_pool, &out_queue->compute_item_arena);

  // Pre-allocate initial pool items.
  for (uint32_t i = 0; i < IREE_HAL_TASK_QUEUE_COMPUTE_INITIAL_POOL_SIZE; ++i) {
    iree_hal_task_queue_compute_item_t* item = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_task_queue_compute_item_allocate(out_queue, &item));
    iree_hal_task_queue_compute_item_slist_push(&out_queue->compute_free_pool,
                                                item);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_task_queue_assign_frontier(
    iree_hal_task_queue_t* queue,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis) {
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL));
  queue->frontier_tracker = frontier_tracker;
  queue->axis = axis;
  return iree_ok_status();
}

void iree_hal_task_queue_retire_frontier(iree_hal_task_queue_t* queue) {
  if (queue->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->frontier_tracker, queue->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    queue->frontier_tracker = NULL;
    queue->axis = 0;
  }
}

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Signal both processes to complete. The next drain call on each sees this
  // flag and returns completed=true, triggering normal process completion.
  iree_atomic_store(&queue->shutting_down, 1, iree_memory_order_release);

  // Schedule both processes so workers pick them up and see the shutdown flag.
  // If a process is already being drained, schedule_process sets needs_drain
  // and the worker will see shutting_down on the next iteration. If IDLE,
  // schedule_process pushes it to the appropriate run list.
  iree_task_executor_schedule_process(queue->executor, &queue->process);
  iree_task_executor_schedule_process(queue->executor, &queue->compute_process);

  // Wait for all outstanding operations and both processes to complete.
  // The wake_budget == 1 process's scope_end fires in its release callback.
  // The compute process's scope_end fires in its release callback (after
  // the last drainer exits), which also cleans up any in-flight items.
  // Each submitted operation has its own scope_begin/end pair. scope_wait_idle
  // returns only when all of these have resolved — meaning every worker has
  // finished touching queue/device resources.
  iree_status_t idle_status =
      iree_task_scope_wait_idle(&queue->scope, IREE_TIME_INFINITE_FUTURE);
  IREE_ASSERT(iree_status_is_ok(idle_status),
              "scope idle wait must not fail with an infinite deadline");
  iree_status_free(idle_status);

  while (iree_atomic_load(&queue->pending_process_release_count,
                          iree_memory_order_acquire) != 0) {
    iree_thread_yield();
  }

  // Drain and destroy any remaining operations in the ready list.
  // These are operations that were queued but never drained (e.g.,
  // submitted after the queue process went idle but before shutdown was
  // signaled).
  iree_hal_task_queue_op_t* remaining = NULL;
  while ((remaining = iree_hal_task_queue_op_slist_pop(&queue->ready_list)) !=
         NULL) {
    iree_hal_task_queue_op_destroy(
        remaining,
        iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down"));
  }
  iree_hal_task_queue_op_slist_deinitialize(&queue->ready_list);

  // Deinitialize the compute lists and free all items via the arena.
  iree_hal_task_queue_compute_item_slist_deinitialize(&queue->compute_pending);
  iree_hal_task_queue_compute_item_slist_deinitialize(
      &queue->compute_free_pool);
  iree_arena_deinitialize(&queue->compute_item_arena);

  iree_task_scope_deinitialize(&queue->scope);
  iree_hal_task_queue_retire_frontier(queue);
  iree_hal_allocator_release(queue->device_allocator);
  iree_task_executor_release(queue->executor);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_task_queue_trim(iree_hal_task_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  iree_task_executor_trim(queue->executor);
}

void iree_hal_task_queue_set_profile_recorder(
    iree_hal_task_queue_t* queue,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t profile_scope,
    iree_atomic_int64_t* submission_counter) {
  if (profile_recorder) {
    queue->profile_scope = profile_scope;
    queue->profile_submission_counter = submission_counter;
    queue->profile_recorder = profile_recorder;
  } else {
    queue->profile_recorder = NULL;
    queue->profile_scope = profile_scope;
    queue->profile_submission_counter = NULL;
  }
}

iree_status_t iree_hal_task_queue_submit_barrier(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_task_queue_submit_op(
      queue, IREE_HAL_TASK_QUEUE_OP_BARRIER, wait_semaphores,
      &signal_semaphores, 0, NULL);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_task_queue_submit_commands(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_task_submission_batch_t* batches) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    const iree_hal_task_submission_batch_t* batch = &batches[i];

    // Handle NULL command buffer as a barrier.
    if (batch->command_buffer == NULL) {
      status = iree_hal_task_queue_submit_op(
          queue, IREE_HAL_TASK_QUEUE_OP_BARRIER, batch->wait_semaphores,
          &batch->signal_semaphores, 0, NULL);
      if (!iree_status_is_ok(status)) break;
      continue;
    }

    // Allocate the operation.
    iree_hal_task_queue_op_t* operation = NULL;
    status =
        iree_hal_task_queue_op_allocate(queue, IREE_HAL_TASK_QUEUE_OP_COMMANDS,
                                        &batch->signal_semaphores, &operation);
    if (!iree_status_is_ok(status)) break;

    iree_task_scope_begin(&queue->scope);

    // Retain the command buffer.
    status = iree_hal_resource_set_insert(
        operation->resource_set, 1,
        (iree_hal_resource_t* const*)&batch->command_buffer);

    // Store the command buffer and binding table in the operation.
    operation->commands.command_buffer = batch->command_buffer;
    operation->commands.binding_table = iree_hal_buffer_binding_table_empty();
    operation->commands.binding_mappings = NULL;
    operation->commands.binding_mapping_count = 0;
    iree_hal_task_queue_profile_add_host_flags(
        operation, IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_COMMAND_BUFFER);
    iree_hal_task_queue_profile_set_command_buffer(operation,
                                                   batch->command_buffer);

    // Copy binding table entries into the arena if present.
    if (iree_status_is_ok(status) && batch->binding_table.count > 0) {
      iree_hal_buffer_binding_t* bindings = NULL;
      status = iree_arena_allocate(
          &operation->arena, batch->binding_table.count * sizeof(*bindings),
          (void**)&bindings);
      if (iree_status_is_ok(status)) {
        memcpy(bindings, batch->binding_table.bindings,
               batch->binding_table.count * sizeof(*bindings));
        operation->commands.binding_table.count = batch->binding_table.count;
        operation->commands.binding_table.bindings = bindings;

        // Retain all binding table buffer references.
        status = iree_hal_resource_set_insert_strided(
            operation->resource_set, batch->binding_table.count, bindings,
            offsetof(iree_hal_buffer_binding_t, buffer),
            sizeof(iree_hal_buffer_binding_t));
      }
    }

    if (iree_status_is_ok(status)) {
      status = iree_hal_task_queue_submit_op_finish(
          operation, batch->wait_semaphores, /*resource_count=*/0,
          /*resources=*/NULL);
    } else {
      iree_hal_task_queue_profile_record_failed_before_ready(operation);
      iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_task_queue_submit_host_call(
    iree_hal_task_queue_t* queue, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the operation.
  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_HOST_CALL, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  // Store host call parameters.
  operation->host_call.device = device;
  operation->host_call.queue_affinity = queue_affinity;
  operation->host_call.call = call;
  memcpy(operation->host_call.args, args, sizeof(operation->host_call.args));
  operation->host_call.flags = flags;

  status = iree_hal_task_queue_submit_op_finish(
      operation, wait_semaphores, /*resource_count=*/0, /*resources=*/NULL);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_task_queue_submit_alloca(
    iree_hal_task_queue_t* queue, iree_hal_pool_t* pool,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_alloca_flags_t flags, iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_buffer_t* transient_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_ALLOCA, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  operation->alloca.pool = pool;
  operation->alloca.params = params;
  operation->alloca.allocation_size = allocation_size;
  operation->alloca.flags = flags;
  operation->alloca.reserve_flags = reserve_flags;
  operation->alloca.transient_buffer = transient_buffer;
  operation->alloca.memory_wait = NULL;
  iree_hal_task_queue_profile_set_transient_buffer(operation, transient_buffer);

  // Retain the transient buffer until the operation completes. The selected
  // pool is borrowed: pool lifetimes must outlive all allocations sourced from
  // them, which avoids per-allocation pool refcount traffic in hot paths.
  status = iree_hal_resource_set_insert(
      operation->resource_set, 1,
      (iree_hal_resource_t* const*)&transient_buffer);

  const iree_host_size_t original_wait_count = wait_semaphores.count;

  // Fast path: check if all wait semaphores are already satisfied.
  if (iree_status_is_ok(status) && wait_semaphores.count > 0) {
    status = iree_hal_task_queue_try_satisfy_waits(&wait_semaphores);
  }

  // If user-visible waits are already satisfied, acquire or park the memory
  // reservation before returning so pool state changes are causally part of the
  // queue_alloca submission.
  if (iree_status_is_ok(status) && wait_semaphores.count == 0) {
    iree_hal_task_queue_profile_set_waits(operation, original_wait_count,
                                          wait_semaphores.count);
    status = iree_hal_task_queue_drain_alloca(operation);
  } else if (iree_status_is_ok(status)) {
    iree_hal_task_queue_profile_set_waits(operation, original_wait_count,
                                          wait_semaphores.count);
    status = iree_hal_task_queue_enqueue_waits(operation, wait_semaphores);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_task_queue_submit_dealloca(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* transient_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_DEALLOCA, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  // Store the transient buffer for decommit in the drain handler.
  operation->dealloca.transient_buffer = transient_buffer;
  iree_hal_task_queue_profile_set_transient_buffer(operation, transient_buffer);

  // Retain the transient buffer so it survives until the operation completes.
  status = iree_hal_resource_set_insert(
      operation->resource_set, 1,
      (iree_hal_resource_t* const*)&transient_buffer);

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_queue_submit_op_finish(
        operation, wait_semaphores, /*resource_count=*/0, /*resources=*/NULL);
  } else {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_task_queue_submit_read(
    iree_hal_task_queue_t* queue, iree_hal_file_t* source_file,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_READ, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  operation->read.hal_file = source_file;
  operation->read.async_file = iree_hal_file_async_handle(source_file);
  operation->read.file_offset = source_offset;
  operation->read.buffer = target_buffer;
  operation->read.buffer_offset = target_offset;
  operation->read.length = length;
  iree_hal_task_queue_profile_set_payload(operation, length);

  // Retain the HAL file and buffer through the operation lifetime.
  iree_hal_resource_t* resources[2] = {
      (iree_hal_resource_t*)source_file,
      (iree_hal_resource_t*)target_buffer,
  };
  status = iree_hal_resource_set_insert(operation->resource_set, 2, resources);

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_queue_submit_op_finish(
        operation, wait_semaphores, /*resource_count=*/0, /*resources=*/NULL);
  } else {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_task_queue_submit_write(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_file_t* target_file,
    uint64_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_queue_op_t* operation = NULL;
  iree_status_t status = iree_hal_task_queue_op_allocate(
      queue, IREE_HAL_TASK_QUEUE_OP_WRITE, &signal_semaphores, &operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_task_scope_begin(&queue->scope);

  operation->write.hal_file = target_file;
  operation->write.async_file = iree_hal_file_async_handle(target_file);
  operation->write.file_offset = target_offset;
  operation->write.buffer = source_buffer;
  operation->write.buffer_offset = source_offset;
  operation->write.length = length;
  iree_hal_task_queue_profile_set_payload(operation, length);

  // Retain the HAL file and buffer through the operation lifetime.
  iree_hal_resource_t* resources[2] = {
      (iree_hal_resource_t*)target_file,
      (iree_hal_resource_t*)source_buffer,
  };
  status = iree_hal_resource_set_insert(operation->resource_set, 2, resources);

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_queue_submit_op_finish(
        operation, wait_semaphores, /*resource_count=*/0, /*resources=*/NULL);
  } else {
    iree_hal_task_queue_profile_record_failed_before_ready(operation);
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Native queue fill/copy/update/dispatch submit
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_task_queue_submit_fill(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, IREE_HAL_TASK_QUEUE_OP_FILL, &signal_semaphores, &operation));
  operation->fill.target_buffer = target_buffer;
  operation->fill.target_offset = target_offset;
  operation->fill.length = length;
  operation->fill.pattern_length = (uint8_t)pattern_length;
  memcpy(operation->fill.pattern, pattern, pattern_length);
  iree_hal_task_queue_profile_set_payload(operation, length);
  return iree_hal_task_queue_submit_op_finish(
      operation, wait_semaphores, 1,
      (iree_hal_resource_t* const*)&target_buffer);
}

iree_status_t iree_hal_task_queue_submit_copy(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, IREE_HAL_TASK_QUEUE_OP_COPY, &signal_semaphores, &operation));
  operation->copy.source_buffer = source_buffer;
  operation->copy.source_offset = source_offset;
  operation->copy.target_buffer = target_buffer;
  operation->copy.target_offset = target_offset;
  operation->copy.length = length;
  iree_hal_task_queue_profile_set_payload(operation, length);
  iree_hal_resource_t* copy_resources[2] = {
      (iree_hal_resource_t*)source_buffer,
      (iree_hal_resource_t*)target_buffer,
  };
  return iree_hal_task_queue_submit_op_finish(operation, wait_semaphores, 2,
                                              copy_resources);
}

iree_status_t iree_hal_task_queue_submit_update(
    iree_hal_task_queue_t* queue, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, IREE_HAL_TASK_QUEUE_OP_UPDATE, &signal_semaphores, &operation));

  // Arena-allocate source data copy (caller's buffer may not outlive submit).
  void* source_data_copy = NULL;
  iree_status_t status = iree_arena_allocate(
      &operation->arena, (iree_host_size_t)length, &source_data_copy);
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
    return status;
  }
  memcpy(source_data_copy, (const uint8_t*)source_buffer + source_offset,
         (size_t)length);
  operation->update.target_buffer = target_buffer;
  operation->update.target_offset = target_offset;
  operation->update.length = length;
  operation->update.source_data = source_data_copy;
  iree_hal_task_queue_profile_set_payload(operation, length);

  return iree_hal_task_queue_submit_op_finish(
      operation, wait_semaphores, 1,
      (iree_hal_resource_t* const*)&target_buffer);
}

iree_status_t iree_hal_task_queue_submit_dispatch(
    iree_hal_task_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_t* bindings, iree_host_size_t binding_count,
    iree_hal_dispatch_flags_t flags, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  const bool uses_indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  if (uses_indirect_parameters && !config.workgroup_count_ref.buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue_dispatch requires a direct workgroup count buffer when using "
        "indirect parameters");
  }
  if (uses_indirect_parameters &&
      (config.workgroup_count_ref.offset % sizeof(uint32_t)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "workgroup count offset does not match the required natural alignment "
        "of uint32_t");
  }
  if (uses_indirect_parameters &&
      config.workgroup_count_ref.length != IREE_HAL_WHOLE_BUFFER &&
      config.workgroup_count_ref.length < sizeof(iree_hal_dispatch_params_t)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "workgroup count buffer does not have the capacity to store the "
        "required 3 uint32_t values");
  }

  iree_hal_task_queue_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_op_begin(
      queue, IREE_HAL_TASK_QUEUE_OP_DISPATCH, &signal_semaphores, &operation));

  operation->dispatch.executable = executable;
  operation->dispatch.export_ordinal = export_ordinal;
  operation->dispatch.config = config;
  operation->dispatch.binding_count = binding_count;
  operation->dispatch.flags = flags;

  // Arena-allocate copies of constants and bindings.
  iree_status_t status = iree_hal_task_queue_profile_set_dispatch(
      operation, executable, export_ordinal, config, flags);
  if (iree_status_is_ok(status) && constants.data_length > 0) {
    uint32_t* constants_copy = NULL;
    status = iree_arena_allocate(&operation->arena, constants.data_length,
                                 (void**)&constants_copy);
    if (iree_status_is_ok(status)) {
      memcpy(constants_copy, constants.data, constants.data_length);
      operation->dispatch.constants = constants_copy;
      operation->dispatch.constant_count =
          (uint16_t)(constants.data_length / sizeof(uint32_t));
    }
  }
  if (iree_status_is_ok(status) && binding_count > 0) {
    iree_hal_buffer_ref_t* bindings_copy = NULL;
    status = iree_arena_allocate(&operation->arena,
                                 binding_count * sizeof(iree_hal_buffer_ref_t),
                                 (void**)&bindings_copy);
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, bindings,
             binding_count * sizeof(iree_hal_buffer_ref_t));
      operation->dispatch.bindings = bindings_copy;
    }
  }

  // Retain executable and all bound buffers unless the caller guarantees that
  // resource lifetimes extend until the dispatch signals completion.
  const bool borrow_resource_lifetimes =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES);
  if (iree_status_is_ok(status) && !borrow_resource_lifetimes) {
    status = iree_hal_resource_set_insert(
        operation->resource_set, 1, (iree_hal_resource_t* const*)&executable);
  }
  if (iree_status_is_ok(status) && !borrow_resource_lifetimes &&
      binding_count > 0) {
    status = iree_hal_resource_set_insert_strided(
        operation->resource_set, binding_count, bindings,
        offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t));
  }
  if (iree_status_is_ok(status) && !borrow_resource_lifetimes &&
      iree_hal_dispatch_uses_indirect_parameters(flags)) {
    status = iree_hal_resource_set_insert(operation->resource_set, 1,
                                          &config.workgroup_count_ref.buffer);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_op_destroy(operation, iree_status_clone(status));
    return status;
  }

  return iree_hal_task_queue_submit_op_finish(operation, wait_semaphores, 0,
                                              NULL);
}

//===----------------------------------------------------------------------===//
// Dispatch drain handler
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_task_queue_drain_dispatch(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_t* operation,
    const iree_hal_cmd_block_processor_worker_context_t* worker_context) {
  const bool allow_inline = iree_any_bit_set(
      operation->dispatch.flags, IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION);
  if (allow_inline) {
    iree_hal_task_queue_profile_start_host_execution(operation);
  }
  const bool uses_indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(operation->dispatch.flags);
  const iree_host_size_t binding_count = operation->dispatch.binding_count;

  // For inline execution, track mappings on the stack so we can unmap after.
  // For non-inline, persistent mappings are used (pointer must survive across
  // threads until the compute process finishes).
  iree_hal_buffer_mapping_t* mappings = NULL;
  iree_status_t status = iree_ok_status();
  if (allow_inline && binding_count > 0) {
    mappings = (iree_hal_buffer_mapping_t*)iree_alloca(binding_count *
                                                       sizeof(*mappings));
    memset(mappings, 0, binding_count * sizeof(*mappings));
  }

  // Map all binding buffers. Host pointers and lengths are stored directly
  // in the fixups below (no span indirection needed).
  void** host_ptrs = NULL;
  size_t* host_lengths = NULL;
  if (binding_count > 0) {
    if (allow_inline) {
      host_ptrs = (void**)iree_alloca(binding_count * sizeof(*host_ptrs));
      host_lengths =
          (size_t*)iree_alloca(binding_count * sizeof(*host_lengths));
    } else {
      status = iree_arena_allocate(&operation->arena,
                                   binding_count * sizeof(*host_ptrs),
                                   (void**)&host_ptrs);
      if (iree_status_is_ok(status)) {
        status = iree_arena_allocate(&operation->arena,
                                     binding_count * sizeof(*host_lengths),
                                     (void**)&host_lengths);
      }
    }
  }
  iree_hal_mapping_mode_t mapping_mode = allow_inline
                                             ? IREE_HAL_MAPPING_MODE_SCOPED
                                             : IREE_HAL_MAPPING_MODE_PERSISTENT;
  for (iree_host_size_t i = 0; i < binding_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_buffer_ref_t* binding = &operation->dispatch.bindings[i];
    iree_hal_buffer_mapping_t mapping = {{0}};
    status = iree_hal_buffer_map_range(
        binding->buffer, mapping_mode, IREE_HAL_MEMORY_ACCESS_ANY,
        binding->offset, binding->length, &mapping);
    if (iree_status_is_ok(status)) {
      host_ptrs[i] = mapping.contents.data;
      host_lengths[i] = mapping.contents.data_length;
      if (mappings) mappings[i] = mapping;
    }
  }
  iree_hal_buffer_mapping_t parameter_mapping = {{0}};
  bool has_parameter_mapping = false;
  void* parameter_host_ptr = NULL;
  size_t parameter_host_length = 0;
  if (iree_status_is_ok(status) && uses_indirect_parameters) {
    const iree_hal_buffer_ref_t* parameter_ref =
        &operation->dispatch.config.workgroup_count_ref;
    status = iree_hal_buffer_map_range(
        parameter_ref->buffer, mapping_mode, IREE_HAL_MEMORY_ACCESS_READ,
        parameter_ref->offset, sizeof(iree_hal_dispatch_params_t),
        &parameter_mapping);
    if (iree_status_is_ok(status)) {
      parameter_host_ptr = parameter_mapping.contents.data;
      parameter_host_length = parameter_mapping.contents.data_length;
      has_parameter_mapping = allow_inline;
    }
  }

  // Build a single-dispatch recording.
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(queue->large_block_pool, &builder);
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_begin(&builder);
  }

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  if (iree_status_is_ok(status)) {
    iree_const_byte_span_t dispatch_constants = {
        .data = (const uint8_t*)operation->dispatch.constants,
        .data_length = operation->dispatch.constant_count * sizeof(uint32_t),
    };
    status = iree_hal_cmd_build_dispatch(
        &builder, operation->dispatch.executable,
        operation->dispatch.export_ordinal, operation->dispatch.config,
        dispatch_constants, binding_count, operation->dispatch.flags, &fixups,
        &token);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      fixups[i].host_ptr = host_ptrs[i];
      fixups[i].offset = 0;
      fixups[i].length = host_lengths[i];
      fixups[i].slot = 0;
      fixups[i].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
    }
    if (uses_indirect_parameters) {
      fixups[binding_count].host_ptr = parameter_host_ptr;
      fixups[binding_count].offset = 0;
      fixups[binding_count].length = parameter_host_length;
      fixups[binding_count].slot = 0;
      fixups[binding_count].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
    }
  }

  iree_hal_cmd_block_recording_t recording;
  memset(&recording, 0, sizeof(recording));
  if (iree_status_is_ok(status)) {
    status = iree_hal_cmd_block_builder_end(&builder, &recording);
  }
  iree_hal_cmd_block_builder_deinitialize(&builder);

  if (iree_status_is_ok(status)) {
    if (allow_inline) {
      // execute_recording_inline executes the processor then signals
      // semaphores via op_complete. We unmap AFTER because the processor
      // reads through the host pointers during execution. For local_task
      // (cache-coherent CPU), unmap is a no-op — no flush needed.
      iree_hal_task_queue_execute_recording_inline(queue, operation, &recording,
                                                   worker_context);
      for (iree_host_size_t i = 0; i < binding_count; ++i) {
        status =
            iree_status_join(status, iree_hal_buffer_unmap_range(&mappings[i]));
      }
      if (has_parameter_mapping) {
        status = iree_status_join(
            status, iree_hal_buffer_unmap_range(&parameter_mapping));
      }
      return status;
    }
    // drain_recording takes ownership on success (via owned_recording).
    // Non-inline uses persistent mappings — no unmap needed (the buffer
    // retain in the resource_set keeps the pointer valid).
    status = iree_hal_task_queue_drain_recording(
        queue, operation, &recording, &recording,
        /*binding_table=*/NULL, /*binding_table_length=*/0);
  }

  // On any failure path, release the recording to prevent leaking block pool
  // blocks. Safe on zero-initialized recordings (first_block == NULL → no-op).
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_block_recording_release(&recording);
    // Unmap any scoped bindings that were mapped before the error.
    if (mappings) {
      for (iree_host_size_t i = 0; i < binding_count; ++i) {
        status =
            iree_status_join(status, iree_hal_buffer_unmap_range(&mappings[i]));
      }
    }
    if (has_parameter_mapping) {
      status = iree_status_join(
          status, iree_hal_buffer_unmap_range(&parameter_mapping));
    }
  }

  return status;
}
