// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/queue.h"

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/command_buffer.h"

static iree_status_t iree_hal_amdgpu_queue_initialize_scheduler(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    uint32_t signal_count, iree_hsa_signal_t* signals);
static iree_status_t iree_hal_amdgpu_queue_deinitialize_scheduler(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list);

//===----------------------------------------------------------------------===//
// HAL API Utilities
//===----------------------------------------------------------------------===//

// Resolves a HAL buffer binding to a device-side buffer reference.
// Verifies (roughly) that it's usable but not that it's accessible to any
// particular agent.
static iree_status_t iree_hal_amdgpu_resolve_binding(
    iree_hal_buffer_binding_t binding,
    iree_hal_amdgpu_device_buffer_ref_t* out_device_ref) {
  iree_hal_amdgpu_device_buffer_type_t type = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer(
      binding.buffer, &type, &out_device_ref->value.bits));
  out_device_ref->type = type;
  out_device_ref->offset = binding.offset;
  out_device_ref->length = binding.length;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_queue_initialize(
    iree_hal_amdgpu_host_worker_t* host_worker, hsa_agent_t device_agent,
    iree_host_size_t device_ordinal, iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    iree_allocator_t host_allocator, iree_hal_amdgpu_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(host_worker);
  IREE_ASSERT_ARGUMENT(buffer_pool);
  IREE_ASSERT_ARGUMENT(out_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_queue, 0, sizeof(*out_queue));

  out_queue->host_worker = host_worker;

  out_queue->device_agent = device_agent;
  out_queue->device_ordinal = device_ordinal;

  out_queue->buffer_pool = buffer_pool;

  // out_queue->control_queue;
  // out_queue->execution_queue;

  // out_queue->trace_buffer;
  // out_queue->scheduler;

  // DO NOT SUBMIT
  iree_status_t status = iree_ok_status();

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_queue_deinitialize(iree_hal_amdgpu_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT

  // queue->scheduler;
  // queue->trace_buffer;

  // queue->execution_queue;
  // queue->control_queue;

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_queue_reserve_entry(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_device_queue_entry_type_t type, iree_host_size_t total_size,
    iree_host_size_t max_kernarg_capacity, void** out_entry) {
  // DO NOT SUBMIT

  iree_hal_amdgpu_device_queue_entry_header_t* entry = NULL;

  entry->type = type;
  entry->flags = IREE_HAL_AMDGPU_DEVICE_DEVICE_QUEUE_ENTRY_FLAG_NONE;

  // DO NOT SUBMIT
  // increasing value used to age entries
  entry->epoch = 0;

  // Used to reserve kernarg space on device. Not allocated until issued.
  entry->max_kernarg_capacity = max_kernarg_capacity;

  entry->active_bit_index = 0;  // managed by device scheduler
  entry->kernarg_offset = 0;    // managed by device scheduler
  entry->list_next = NULL;      // managed by device scheduler

  // DO NOT SUBMIT
  // suballocate lists
  // wait_list
  // signal_list

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static void iree_hal_amdgpu_queue_abort_entry(
    iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* entry) {
  // DO NOT SUBMIT
}

static iree_status_t iree_hal_amdgpu_queue_commit_entry(
    iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* entry) {
  // DO NOT SUBMIT
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_queue_initialize_scheduler(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    uint32_t signal_count, iree_hsa_signal_t* signals) {
  IREE_ASSERT_ARGUMENT(queue);

  const iree_host_size_t max_kernarg_capacity = 0;  // unused
  iree_hal_amdgpu_device_queue_initialize_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_INITIALIZE, sizeof(*entry),
      max_kernarg_capacity, (void**)&entry));

  entry->signal_count = signal_count;
  entry->signals = signals;

  return iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
}

static iree_status_t iree_hal_amdgpu_queue_deinitialize_scheduler(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  IREE_ASSERT_ARGUMENT(queue);

  const iree_host_size_t max_kernarg_capacity = 0;  // unused
  iree_hal_amdgpu_device_queue_deinitialize_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_DEINITIALIZE, sizeof(*entry),
      max_kernarg_capacity, (void**)&entry));

  // DO NOT SUBMIT

  return iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
}

iree_status_t iree_hal_amdgpu_queue_alloca(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(queue);

  // TODO(benvanik): use params.queue_affinity to restrict access? By default
  // the device-side allocation pool is accessible to all devices in the system
  // but this can be inefficient.

  // Allocate placeholder HAL buffer handle. This has no backing storage beyond
  // the allocation handle in the device memory pool.
  iree_hal_buffer_t* buffer = NULL;
  iree_hal_amdgpu_device_allocation_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_buffer_pool_acquire(
      queue->buffer_pool, params, allocation_size, &buffer, &handle));

  // NOTE: if entry reserve/commit fails we need to clean up the buffer.
  const iree_host_size_t max_kernarg_capacity =
      IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE;
  iree_hal_amdgpu_device_queue_alloca_entry_t* entry = NULL;
  iree_status_t status = iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_ALLOCA, sizeof(*entry),
      max_kernarg_capacity, (void**)&entry);
  if (iree_status_is_ok(status)) {
    entry->pool = pool;
    entry->min_alignment = params.min_alignment;
    entry->allocation_size = allocation_size;
    entry->handle = handle;
    status = iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_queue_dealloca(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(queue);

  // Must be a transient buffer. This will fail for other buffer types.
  uint32_t pool = 0;
  iree_hal_amdgpu_device_allocation_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_resolve_transient_buffer(buffer, &pool, &handle));

  const iree_host_size_t max_kernarg_capacity =
      IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE;
  iree_hal_amdgpu_device_queue_dealloca_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_DEALLOCA, sizeof(*entry),
      max_kernarg_capacity, (void**)&entry));

  entry->pool = pool;
  entry->handle = handle;

  return iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
}

iree_status_t iree_hal_amdgpu_queue_fill(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_binding_t target_ref, uint64_t pattern,
    uint8_t pattern_length) {
  IREE_ASSERT_ARGUMENT(queue);

  iree_hal_amdgpu_device_buffer_ref_t device_target_ref;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_resolve_binding(target_ref, &device_target_ref),
      "resolving `target_ref`");

  const iree_host_size_t max_kernarg_capacity =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE +
      IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE;
  iree_hal_amdgpu_device_queue_fill_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_FILL, sizeof(*entry),
      max_kernarg_capacity, (void**)&entry));

  entry->target_ref = device_target_ref;
  entry->pattern = pattern;
  entry->pattern_length = pattern_length;

  return iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
}

iree_status_t iree_hal_amdgpu_queue_copy(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_binding_t source_ref,
    iree_hal_buffer_binding_t target_ref) {
  IREE_ASSERT_ARGUMENT(queue);

  iree_hal_amdgpu_device_buffer_ref_t device_source_ref;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_resolve_binding(source_ref, &device_source_ref),
      "resolving `source_ref`");
  iree_hal_amdgpu_device_buffer_ref_t device_target_ref;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_resolve_binding(target_ref, &device_target_ref),
      "resolving `target_ref`");

  const iree_host_size_t max_kernarg_capacity =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE +
      IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE;
  iree_hal_amdgpu_device_queue_copy_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_COPY, sizeof(*entry),
      max_kernarg_capacity, (void**)&entry));

  entry->source_ref = device_source_ref;
  entry->target_ref = device_target_ref;

  return iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
}

iree_status_t iree_hal_amdgpu_queue_read(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  // TODO(benvanik): support device-side reads by relaying through the host
  // worker. We should be able to enqueue the operation and have the device ask
  // the host to read buffer ranges at the appropriate time. Today the
  // emulation is blocking.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "device-side reads not yet implemented");
}

iree_status_t iree_hal_amdgpu_queue_write(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  // TODO(benvanik): support device-side writes by relaying through the host
  // worker. We should be able to enqueue the operation and have the device ask
  // the host to write buffer ranges at the appropriate time. Today the
  // emulation is blocking.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "device-side writes not yet implemented");
}

static iree_status_t iree_hal_amdgpu_queue_barrier(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  IREE_ASSERT_ARGUMENT(queue);

  const iree_host_size_t max_kernarg_capacity = 0;  // unused
  iree_hal_amdgpu_device_queue_barrier_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_BARRIER, sizeof(*entry),
      max_kernarg_capacity, (void**)&entry));

  // No-op.

  return iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
}

iree_status_t iree_hal_amdgpu_queue_execute(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables) {
  IREE_ASSERT_ARGUMENT(queue);

  // Fast-path barriers.
  if (command_buffer_count == 0) {
    return iree_hal_amdgpu_queue_barrier(queue, wait_semaphore_list,
                                         signal_semaphore_list);
  }

  // TODO(benvanik): either support multiple command buffers or remove them from
  // the API. The compiler can't use multiple command buffers today and there's
  // no real need in the C side for them (we don't care about multi-threaded
  // recording and such).
  if (command_buffer_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "multiple command buffers in a single submission are not supported");
  }
  iree_hal_command_buffer_t* command_buffer = command_buffers[0];
  const iree_hal_buffer_binding_table_t binding_table =
      binding_tables ? binding_tables[0] : (iree_hal_buffer_binding_table_t){0};

  // Query the device-side resource requirements and per-device copy of the
  // command buffer information. All other information is handled device-side
  // during the issue of the execute operation.
  iree_hal_amdgpu_device_command_buffer_t* device_command_buffer = NULL;
  iree_host_size_t command_buffer_max_kernarg_capacity = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_command_buffer_query_execution_state(
      command_buffer, queue->device_ordinal, &device_command_buffer,
      &command_buffer_max_kernarg_capacity));

  // Kernarg requirements are for the device-side execution control dispatches
  // as well as execution of any block in the command buffer.
  const iree_host_size_t max_kernarg_capacity =
      IREE_HAL_AMDGPU_DEVICE_EXECUTION_CONTROL_KERNARG_SIZE +
      command_buffer_max_kernarg_capacity;

  iree_hal_amdgpu_device_queue_execute_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_reserve_entry(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_EXECUTE,
      sizeof(*entry) +
          binding_table.count * sizeof(iree_hal_amdgpu_device_buffer_ref_t),
      max_kernarg_capacity, (void**)&entry));

  // NOTE: we only need to populate the flags and command buffer/binding table.
  // Other fields are setup when the operation is issued on device.

  // DO NOT SUBMIT

  // TODO(benvanik): allow
  iree_hal_amdgpu_device_execution_flags_t flags =
      IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_NONE;
  // IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_SERIALIZE
  // IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_UNCACHED
  // IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL
  // IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_DISPATCH
  entry->state.flags = flags;

  // Execution will begin at the entry block (block[0]).
  entry->state.command_buffer = device_command_buffer;

  // Resolve all provided binding table entries to their device handles or
  // pointers. Note that this may fail if any binding is invalid and we need to
  // clean up the allocated queue entry (we do this so that we can resolve
  // in-place and not need an extra allocation).
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    status = iree_hal_amdgpu_resolve_binding(binding_table.bindings[i],
                                             &entry->state.bindings[i]);
    if (!iree_status_is_ok(status)) break;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_queue_abort_entry(queue, &entry->header);
    return status;
  }

  return iree_hal_amdgpu_queue_commit_entry(queue, &entry->header);
}

iree_status_t iree_hal_amdgpu_queue_flush(iree_hal_amdgpu_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  // No-op as we don't do any host-side buffering (today).
  return iree_ok_status();
}
