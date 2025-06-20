// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_VIRTUAL_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_VIRTUAL_QUEUE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_options_t
//===----------------------------------------------------------------------===//

// Power-of-two number of hardware execution queues per HAL queue.
// Each execution queue maps to an independent HSA queue and allows operations
// to execute concurrently.
#define IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_COUNT (2)

// Power-of-two size of the scheduler control queue in entries.
// This is used exclusively for scheduling operations (queue operations, command
// buffer management, etc). It only needs to be as large as the number of
// outstanding control kernel launches - usually 1 to 100.
#define IREE_HAL_AMDGPU_DEFAULT_CONTROL_QUEUE_CAPACITY (512)

// Power-of-two size of the scheduler/command buffer execution queue in entries.
// All DMA operations from the queue as well as all command buffer commands
// execute from this queue. This should be as large as is reasonable so that we
// don't have to subdivide command buffers too much. Note that command buffers
// must be recorded with a block size at or under this value.
#define IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_CAPACITY (64 * 1024)

// Power-of-two size of the kernarg ringbuffer used by each queue.
// This limits the maximum kernarg use of command buffer blocks.
#define IREE_HAL_AMDGPU_DEFAULT_KERNARG_RINGBUFFER_CAPACITY (16 * 1024 * 1024)

// Power-of-two size for the per-queue trace ringbuffer in bytes.
// This limits the amount of tracing data that can be captured before the need
// for a device->host flush.
#define IREE_HAL_AMDGPU_DEFAULT_TRACE_BUFFER_CAPACITY (16 * 1024 * 1024)

// Returns in |out_placement| the optimal location for queue execution based on
// the agent capabilities.
iree_status_t iree_hal_amdgpu_queue_infer_placement(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    hsa_agent_t gpu_agent, iree_hal_amdgpu_queue_placement_t* out_placement);

// Controls queue behavior.
typedef uint32_t iree_hal_amdgpu_queue_flags_t;
enum iree_hal_amdgpu_queue_flag_bits_e {
  IREE_HAL_AMDGPU_QUEUE_FLAG_NONE = 0u,
  // Enable tracing of dispatches (when device tracing is enabled).
  IREE_HAL_AMDGPU_QUEUE_FLAG_TRACE_EXECUTION = 1u << 0,
};

// Controls scheduler behavior.
typedef uint64_t iree_hal_amdgpu_queue_scheduling_mode_t;
enum iree_hal_amdgpu_queue_scheduling_mode_bits_e {
  IREE_HAL_AMDGPU_QUEUE_SCHEDULING_MODE_DEFAULT = 0u,

  // Only one queue entry is allowed to be active at a time. Others will wait in
  // the ready list and execute in epoch order.
  IREE_HAL_AMDGPU_QUEUE_SCHEDULING_MODE_EXCLUSIVE = 1ull << 0,

  // Attempt to schedule entries out-of-order to fill available resources.
  // This may reduce overall latency if small entries come in while large ones
  // are outstanding - or may make things worse as large entries may come in
  // and acquire resources just before a prior entry completes.
  // https://en.wikipedia.org/wiki/Work-conserving_scheduler
  IREE_HAL_AMDGPU_QUEUE_SCHEDULING_MODE_WORK_CONSERVING = 1ull << 1,
};

// Options used to construct a queue.
// These may vary per queue within a physical device or across physical devices.
// Flags and modes apply only to work scheduled on the queue.
typedef struct iree_hal_amdgpu_queue_options_t {
  // Specifies where the queue executes.
  iree_hal_amdgpu_queue_placement_t placement;
  // Flags controlling queue behavior.
  iree_hal_amdgpu_queue_flags_t flags;
  // Scheduling behavior.
  iree_hal_amdgpu_queue_scheduling_mode_t mode;
  // Power-of-two total size of the control queue in entries.
  // IREE_HAL_AMDGPU_DEFAULT_CONTROL_QUEUE_CAPACITY by default.
  iree_host_size_t control_queue_capacity;
  // Power-of-two number of hardware execution queues per HAL queue.
  // IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_COUNT by default.
  iree_host_size_t execution_queue_count;
  // Power-of-two total size of the execution queue in entries.
  // IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_CAPACITY by default.
  iree_host_size_t execution_queue_capacity;
  // Power-of-two total size of the kernarg ringbuffer in bytes.
  iree_device_size_t kernarg_ringbuffer_capacity;
  // Power-of-two total size of the trace buffer, in bytes, if tracing is
  // enabled. IREE_HAL_AMDGPU_DEFAULT_TRACE_BUFFER_CAPACITY by default.
  iree_device_size_t trace_buffer_capacity;
} iree_hal_amdgpu_queue_options_t;

// Initializes default queue options.
void iree_hal_amdgpu_queue_options_initialize(
    iree_hal_amdgpu_queue_options_t* out_options);

// Verifies queue options to ensure they meet the agent requirements.
iree_status_t iree_hal_amdgpu_queue_options_verify(
    const iree_hal_amdgpu_queue_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    hsa_agent_t gpu_agent);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_virtual_queue_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_virtual_queue_vtable_t;

// Virtual queue interface.
// Unlike most HAL resources this is not reference counted. Queues are allocated
// inline and users must first query the required size, allocate space, and then
// initialize the queue in-place. Deinitialization happens explicitly when the
// parent is deinitializing via the `deinitialize` vtable entry.
typedef struct iree_hal_amdgpu_virtual_queue_t {
  const iree_hal_amdgpu_virtual_queue_vtable_t* vtable;
} iree_hal_amdgpu_virtual_queue_t;

typedef struct iree_hal_amdgpu_virtual_queue_vtable_t {
  // Deinitializes the queue on shutdown.
  void(IREE_API_PTR* deinitialize)(iree_hal_amdgpu_virtual_queue_t* queue);

  void(IREE_API_PTR* trim)(iree_hal_amdgpu_virtual_queue_t* queue);

  iree_status_t(IREE_API_PTR* alloca)(
      iree_hal_amdgpu_virtual_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
      iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
      iree_hal_buffer_t** IREE_RESTRICT out_buffer);

  iree_status_t(IREE_API_PTR* dealloca)(
      iree_hal_amdgpu_virtual_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags);

  iree_status_t(IREE_API_PTR* fill)(
      iree_hal_amdgpu_virtual_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, uint64_t pattern_bits,
      iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

  iree_status_t(IREE_API_PTR* update)(
      iree_hal_amdgpu_virtual_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      const void* source_buffer, iree_host_size_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_update_flags_t flags);

  iree_status_t(IREE_API_PTR* copy)(
      iree_hal_amdgpu_virtual_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_copy_flags_t flags);

  // NULL if not implemented and emulation should be used.
  // TODO(benvanik): when all queue implementations support native I/O we should
  // drop the emulation (it's bad).
  iree_status_t(IREE_API_PTR* read)(
      iree_hal_amdgpu_virtual_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_file_t* source_file, uint64_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, iree_hal_read_flags_t flags);

  // NULL if not implemented and emulation should be used.
  // TODO(benvanik): when all queue implementations support native I/O we should
  // drop the emulation (it's bad).
  iree_status_t(IREE_API_PTR* write)(
      iree_hal_amdgpu_virtual_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
      iree_hal_file_t* target_file, uint64_t target_offset,
      iree_device_size_t length, iree_hal_write_flags_t flags);

  iree_status_t(IREE_API_PTR* execute)(
      iree_hal_amdgpu_virtual_queue_t* queue,
      const iree_hal_semaphore_list_t wait_semaphore_list,
      const iree_hal_semaphore_list_t signal_semaphore_list,
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table,
      iree_hal_execute_flags_t flags);

  iree_status_t(IREE_API_PTR* flush)(iree_hal_amdgpu_virtual_queue_t* queue);
} iree_hal_amdgpu_virtual_queue_vtable_t;

#endif  // IREE_HAL_DRIVERS_AMDGPU_VIRTUAL_QUEUE_H_
