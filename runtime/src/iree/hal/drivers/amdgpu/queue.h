// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_QUEUE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/buffer_pool.h"
#include "iree/hal/drivers/amdgpu/device/scheduler.h"
#include "iree/hal/drivers/amdgpu/device/tracing.h"
#include "iree/hal/drivers/amdgpu/host_worker.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_queue_t {
  // Host worker thread used to service device library requests. May be shared
  // with other queues on the same or different devices.
  iree_hal_amdgpu_host_worker_t* host_worker;

  // GPU agent.
  hsa_agent_t device_agent;
  // Ordinal of the GPU agent within the topology.
  iree_host_size_t device_ordinal;

  // Transient buffer pool used for device allocation handles.
  iree_hal_amdgpu_buffer_pool_t* buffer_pool;

  hsa_queue_t* control_queue;
  hsa_queue_t* execution_queue;

  // DO NOT SUBMIT large block allocs for everything?

  iree_hal_amdgpu_device_queue_scheduler_t* scheduler;
  iree_hal_amdgpu_device_trace_buffer_t* trace_buffer;

  // DO NOT SUBMIT resource pools

  iree_hal_amdgpu_device_signal_pool_t* signal_pool;
} iree_hal_amdgpu_queue_t;

iree_status_t iree_hal_amdgpu_queue_initialize(
    iree_hal_amdgpu_host_worker_t* host_worker, hsa_agent_t device_agent,
    iree_host_size_t device_ordinal, iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    iree_allocator_t host_allocator, iree_hal_amdgpu_queue_t* out_queue);

void iree_hal_amdgpu_queue_deinitialize(iree_hal_amdgpu_queue_t* queue);

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_queue_alloca(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer);

iree_status_t iree_hal_amdgpu_queue_dealloca(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer);

iree_status_t iree_hal_amdgpu_queue_fill(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_binding_t target_ref, uint64_t pattern,
    uint8_t pattern_length);

iree_status_t iree_hal_amdgpu_queue_copy(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_binding_t source_ref, iree_hal_buffer_binding_t target_ref);

iree_status_t iree_hal_amdgpu_queue_read(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags);

iree_status_t iree_hal_amdgpu_queue_write(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags);

iree_status_t iree_hal_amdgpu_queue_execute(
    iree_hal_amdgpu_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables);

iree_status_t iree_hal_amdgpu_queue_flush(iree_hal_amdgpu_queue_t* queue);

#endif  // IREE_HAL_DRIVERS_AMDGPU_QUEUE_H_
