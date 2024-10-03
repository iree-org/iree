// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_QUEUE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/scheduler.h"
#include "iree/hal/drivers/amdgpu/trace_buffer.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;

typedef struct iree_hal_resource_set_t iree_hal_resource_set_t;

typedef struct iree_hal_amdgpu_device_allocator_t
    iree_hal_amdgpu_device_allocator_t;

typedef struct iree_hal_amdgpu_block_allocators_t
    iree_hal_amdgpu_block_allocators_t;
typedef struct iree_hal_amdgpu_buffer_pool_t iree_hal_amdgpu_buffer_pool_t;
typedef struct iree_hal_amdgpu_host_worker_t iree_hal_amdgpu_host_worker_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_options_t
//===----------------------------------------------------------------------===//

// Power-of-two number of hardware execution queues per HAL queue.
// Currently only 1 is supported but to execute multiple operations
// simultaneously we need multiple and balancing in the scheduler.
#define IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_COUNT (1)

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

// Controls queue behavior.
typedef uint32_t iree_hal_amdgpu_queue_flags_t;
enum iree_hal_amdgpu_queue_flag_bits_e {
  IREE_HAL_AMDGPU_QUEUE_FLAG_NONE = 0u,
  // Enable tracing of dispatches (when device tracing is enabled).
  IREE_HAL_AMDGPU_QUEUE_FLAG_TRACE_EXECUTION = 1u << 0,
};

// Options used to construct a queue.
// These may vary per queue within a physical device or across physical devices.
// Flags and modes apply only to work scheduled on the queue.
typedef struct iree_hal_amdgpu_queue_options_t {
  // Flags controlling queue behavior.
  iree_hal_amdgpu_queue_flags_t flags;
  // Scheduling behavior.
  iree_hal_amdgpu_device_queue_scheduling_mode_t mode;
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
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_t
//===----------------------------------------------------------------------===//

// Pointers hoisted from the scheduler to avoid the need to indirect.
// Often on the host we are just setting one or two fields and don't want to
// access potentially device-local memory over the bus just to get a pointer.
typedef struct iree_hal_amdgpu_queue_scheduler_ptrs_t {
  // Mailbox used for accepting incoming queue entries.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_mailbox_t* mailbox;
  // A bitmask of iree_hal_amdgpu_device_queue_tick_action_set_t values
  // indicating work that needs to be performed on the next tick.
  IREE_AMDGPU_DEVICE_PTR iree_amdgpu_scoped_atomic_uint64_t* tick_action_set;
  // Queue entries that are in the active/issued state.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_active_set_t* active_set;
  // Immutable kernargs used to launch scheduler kernels on the control queue.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_scheduler_kernargs_t*
      control_kernargs;
} iree_hal_amdgpu_queue_scheduler_ptrs_t;

// A logical hardware scheduler queue mapping to a managed
// iree_hal_amdgpu_device_queue_scheduler_t. May be backed by multiple physical
// hardware execution queues on the same physical device.
//
// Thread-safe; multiple threads may schedule work against the queue at a time.
// Most host-side state is immutable and the rest is thread-safe by design like
// hsa_queue_t.
typedef struct iree_hal_amdgpu_queue_t {
  // Flags controlling queue behavior.
  iree_hal_amdgpu_queue_flags_t flags;
  // Scheduling behavior.
  iree_hal_amdgpu_device_queue_scheduling_mode_t mode;

  // System this queue is associated with.
  iree_hal_amdgpu_system_t* system;

  // GPU agent.
  hsa_agent_t device_agent;
  // Ordinal of the GPU agent within the topology.
  iree_host_size_t device_ordinal;

  // Host worker thread used to service device library requests. May be shared
  // with other queues on the same or different devices.
  iree_hal_amdgpu_host_worker_t* host_worker;

  // Block pool used for small host allocations. Shared with other queues on the
  // same physical device.
  iree_arena_block_pool_t* host_block_pool;

  // Shared block pool-based allocators for small transient allocations.
  iree_hal_amdgpu_block_allocators_t* block_allocators;

  // Transient buffer pool used for device allocation handles.
  iree_hal_amdgpu_buffer_pool_t* buffer_pool;

  // Signal that is incremented for every new queue operation and decremented
  // when they complete. A value of 0 indicates the queue is idle.
  hsa_signal_t idle_signal;

  // Device library kernels used for populating control packets.
  iree_hal_amdgpu_device_kernels_t kernels;

  // Queue-specific trace buffer.
  // This could be shared across multiple queues on the same device but today
  // is queue-local to keep the origin and time tracking easier (at the cost
  // of more device memory).
  iree_hal_amdgpu_trace_buffer_t trace_buffer;

  // Device scheduler memory.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_queue_scheduler_t* scheduler;
  // Pointers hoisted from the scheduler to avoid the need to indirect.
  iree_hal_amdgpu_queue_scheduler_ptrs_t scheduler_ptrs;

  // Ringbuffer used for kernargs. This is exclusively written by the device
  // scheduler as it issues operations.
  iree_hal_amdgpu_vmem_ringbuffer_t kernarg_ringbuffer;

  // HSA queue used for control dispatches (scheduler ticks, command buffer CFG
  // evaluation, etc). This generally should have a higher priority than an
  // execution queue so that we can more eagerly resolve retires and
  // potentially unblock work on other agents.
  hsa_queue_t* control_queue;

  // Number of execution queues.
  iree_host_size_t execution_queue_count;

  // HSA queue used for execution (DMA operations, command buffer commands,
  // etc). The queue may be processing commands from multiple queue operations
  // concurrently.
  //
  // TODO(benvanik): support multiple execution queues for concurrently
  // executing operations.
  hsa_queue_t* execution_queues[/*execution_queue_count*/];
} iree_hal_amdgpu_queue_t;

// Returns the aligned heap size in bytes required to store the queue data
// structure. Requires that the options have been verified.
static inline iree_host_size_t iree_hal_amdgpu_queue_calculate_size(
    const iree_hal_amdgpu_queue_options_t* options) {
  return iree_host_align(
      sizeof(iree_hal_amdgpu_queue_t) +
          options->execution_queue_count * sizeof(hsa_queue_t*),
      iree_max_align_t);
}

// Initializes a HAL queue by creating HSA resources (doorbells, queues, the
// scheduler, etc). Host operations requested by the device will be serviced by
// a |host_worker| which may be shared with other queues. Requires that the
// |options| have been verified.
//
// |initialization_signal| will be incremented as asynchronous initialization
// operations are enqueued and decremented as they complete. Callers must wait
// for the completion signal to reach 0 prior to deinitializing the queue even
// if initialization fails.
//
// NOTE: if initialization fails callers must call
// iree_hal_amdgpu_queue_deinitialize after |initialization_signal| is reached.
//
// |out_queue| must reference at least iree_hal_amdgpu_queue_calculate_size of
// valid host memory.
iree_status_t iree_hal_amdgpu_queue_initialize(
    iree_hal_amdgpu_queue_options_t options, iree_hal_amdgpu_system_t* system,
    hsa_agent_t device_agent, iree_host_size_t device_ordinal,
    iree_hal_amdgpu_device_allocator_t* device_allocator,
    iree_hal_amdgpu_host_worker_t* host_worker,
    iree_arena_block_pool_t* host_block_pool,
    iree_hal_amdgpu_block_allocators_t* block_allocators,
    iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_queue_t* out_queue);

// Deinitializes a queue once all work has completed.
void iree_hal_amdgpu_queue_deinitialize(iree_hal_amdgpu_queue_t* queue);

// Releases any unused pooled resources.
void iree_hal_amdgpu_queue_trim(iree_hal_amdgpu_queue_t* queue);

// Requests that the queue retire the given |entry|.
// The device-side scheduler will be asked to retire the entry on its next tick.
// Upon return the entry may be invalidated.
void iree_hal_amdgpu_queue_request_retire(
    iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* entry);

// Retires |entry| as requested by the device scheduler.
// This must only be called on entries that have fully completed. The entry
// storage may be immediately reused.
iree_status_t iree_hal_amdgpu_queue_retire_entry(
    iree_hal_amdgpu_queue_t* queue,
    iree_hal_amdgpu_device_queue_entry_header_t* entry, bool has_signals,
    uint32_t allocation_pool, uint64_t allocation_token_bits,
    iree_hal_resource_set_t* resource_set);

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_queue_alloca(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer);

iree_status_t iree_hal_amdgpu_queue_dealloca(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags);

iree_status_t iree_hal_amdgpu_queue_fill(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern, uint8_t pattern_length,
    iree_hal_fill_flags_t flags);

iree_status_t iree_hal_amdgpu_queue_update(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags);

iree_status_t iree_hal_amdgpu_queue_copy(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags);

iree_status_t iree_hal_amdgpu_queue_read(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags);

iree_status_t iree_hal_amdgpu_queue_write(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags);

iree_status_t iree_hal_amdgpu_queue_execute(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags);

iree_status_t iree_hal_amdgpu_queue_flush(iree_hal_amdgpu_queue_t* queue);

#endif  // IREE_HAL_DRIVERS_AMDGPU_QUEUE_H_
