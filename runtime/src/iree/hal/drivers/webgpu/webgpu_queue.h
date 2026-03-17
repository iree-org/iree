// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebGPU queue abstraction.
//
// Encapsulates the per-queue state and all queue operation logic. Each queue
// owns its block pool, scratch builder, frontier tracking state, and the
// bridge handles needed to submit work. The device is a thin vtable layer
// that selects a queue (today: always queue[0]) and delegates.
//
// Designed for queue[N] even though WebGPU currently exposes a single queue
// per device. The clean separation ensures the device file stays small and
// queue-level concerns (async state, instruction building, epoch tracking)
// are self-contained.

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_QUEUE_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_QUEUE_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/handle_table.h"
#include "iree/hal/drivers/webgpu/webgpu_builder.h"
#include "iree/hal/drivers/webgpu/webgpu_builtins.h"

typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_queue_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_queue_t {
  // Bridge handles for the GPUDevice and its GPUQueue.
  iree_hal_webgpu_handle_t device_handle;
  iree_hal_webgpu_handle_t queue_handle;

  // Borrowed from the device. Must outlive the queue.
  const iree_hal_webgpu_builtins_t* builtins;

  // Proactor for async I/O (semaphore waits, onSubmittedWorkDone). Borrowed
  // from the proactor pool via the device.
  iree_async_proactor_t* proactor;

  // Shared frontier tracker for cross-device causal ordering. Borrowed from
  // the session. NULL if frontier-based fast paths are not enabled.
  iree_async_frontier_tracker_t* frontier_tracker;

  // This queue's axis and monotonic epoch counter for frontier tracking.
  iree_async_axis_t axis;
  iree_atomic_int64_t epoch;

  // Block pool shared by the scratch builder and command buffers created on
  // this queue. 64KB blocks.
  iree_arena_block_pool_t block_pool;

  // Scratch builder for single-command queue operations (fill, copy, update,
  // dispatch). All slots are static (dynamic_count = 0). Reset and reused for
  // each queue operation.
  iree_hal_webgpu_builder_t scratch_builder;

  iree_allocator_t host_allocator;
} iree_hal_webgpu_queue_t;

// Initializes a queue. All borrowed pointers (builtins, proactor,
// frontier_tracker) must outlive the queue.
iree_status_t iree_hal_webgpu_queue_initialize(
    iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_handle_t queue_handle,
    const iree_hal_webgpu_builtins_t* builtins, iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
    iree_allocator_t host_allocator, iree_hal_webgpu_queue_t* out_queue);

// Deinitializes the queue, releasing the scratch builder and block pool.
void iree_hal_webgpu_queue_deinitialize(iree_hal_webgpu_queue_t* queue);

//===----------------------------------------------------------------------===//
// Queue operations
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_webgpu_queue_alloca(
    iree_hal_webgpu_queue_t* queue, iree_hal_allocator_t* device_allocator,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer);

iree_status_t iree_hal_webgpu_queue_dealloca(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags);

iree_status_t iree_hal_webgpu_queue_fill(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

iree_status_t iree_hal_webgpu_queue_update(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags);

iree_status_t iree_hal_webgpu_queue_copy(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags);

iree_status_t iree_hal_webgpu_queue_read(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags);

iree_status_t iree_hal_webgpu_queue_write(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags);

// |device| is passed through to the host call context — the queue does not
// retain it. The caller (device vtable) guarantees it remains valid.
iree_status_t iree_hal_webgpu_queue_host_call(
    iree_hal_webgpu_queue_t* queue, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags);

iree_status_t iree_hal_webgpu_queue_dispatch(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags);

iree_status_t iree_hal_webgpu_queue_execute(
    iree_hal_webgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags);

iree_status_t iree_hal_webgpu_queue_flush(iree_hal_webgpu_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_QUEUE_H_
