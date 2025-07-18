// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_queue_t
//===----------------------------------------------------------------------===//

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable;

typedef struct iree_hal_amdgpu_host_queue_t {
  iree_hal_amdgpu_virtual_queue_t base;

  // Optional callback issued when an asynchronous queue error occurs.
  iree_hal_amdgpu_error_callback_t error_callback;
} iree_hal_amdgpu_host_queue_t;

static iree_hal_amdgpu_host_queue_t* iree_hal_amdgpu_host_queue_cast(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue) {
  IREE_ASSERT_ARGUMENT(virtual_queue);
  IREE_ASSERT_EQ(virtual_queue->vtable, &iree_hal_amdgpu_host_queue_vtable);
  return (iree_hal_amdgpu_host_queue_t*)virtual_queue;
}

iree_host_size_t iree_hal_amdgpu_host_queue_calculate_size(
    const iree_hal_amdgpu_queue_options_t* options) {
  IREE_ASSERT_EQ(options->placement, IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST);
  // TODO(benvanik): factor in dynamic sizes (execution queue count, etc).
  return sizeof(iree_hal_amdgpu_host_queue_t);
}

iree_status_t iree_hal_amdgpu_host_queue_initialize(
    iree_hal_amdgpu_system_t* system, iree_hal_amdgpu_queue_options_t options,
    hsa_agent_t device_agent, iree_host_size_t device_ordinal,
    iree_hal_amdgpu_host_service_t* host_service,
    iree_arena_block_pool_t* host_block_pool,
    iree_hal_amdgpu_block_allocators_t* block_allocators,
    iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    iree_hal_amdgpu_error_callback_t error_callback,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_EQ(options.placement, IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST);
  IREE_ASSERT_ARGUMENT(host_service);
  IREE_ASSERT_ARGUMENT(host_block_pool);
  IREE_ASSERT_ARGUMENT(block_allocators);
  IREE_ASSERT_ARGUMENT(buffer_pool);
  IREE_ASSERT_ARGUMENT(out_queue);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, device_ordinal);

  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)out_queue;
  queue->base.vtable = &iree_hal_amdgpu_host_queue_vtable;
  queue->error_callback = error_callback;

  // TODO(benvanik): implement the host queue.
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "host-side queuing not yet implemented");

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)queue;

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_host_queue_trim(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)queue;

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_host_queue_alloca(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue_alloca");
  (void)queue;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_dealloca(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue_dealloca");
  (void)queue;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_fill(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue_fill");
  (void)queue;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_update(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue_update");
  (void)queue;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_copy(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue_copy");
  (void)queue;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_read(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue_read");
  (void)queue;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_write(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue_write");
  (void)queue;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_execute(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue_execute");
  (void)queue;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_flush(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "queue_flush");
  (void)queue;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable = {
        .deinitialize = iree_hal_amdgpu_host_queue_deinitialize,
        .trim = iree_hal_amdgpu_host_queue_trim,
        .alloca = iree_hal_amdgpu_host_queue_alloca,
        .dealloca = iree_hal_amdgpu_host_queue_dealloca,
        .fill = iree_hal_amdgpu_host_queue_fill,
        .update = iree_hal_amdgpu_host_queue_update,
        .copy = iree_hal_amdgpu_host_queue_copy,
        .read = iree_hal_amdgpu_host_queue_read,
        .write = iree_hal_amdgpu_host_queue_write,
        .execute = iree_hal_amdgpu_host_queue_execute,
        .flush = iree_hal_amdgpu_host_queue_flush,
};
