// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_PER_DEVICE_INFORMATION_H_
#define IREE_HAL_DRIVERS_HSA_PER_DEVICE_INFORMATION_H_

#include "iree/base/internal/synchronization.h"
#include "iree/hal/drivers/hsa/hsa_headers.h"

typedef struct iree_hal_stream_tracing_context_t
    iree_hal_stream_tracing_context_t;

// Per-device information for HSA devices.
typedef struct iree_hal_hsa_per_device_info_t {
  // The HSA agent representing the GPU device.
  hsa_agent_t agent;

  // The HSA agent representing the CPU for memory operations.
  hsa_agent_t cpu_agent;

  // The HSA queue used for kernel dispatch.
  hsa_queue_t* queue;

  // Memory pools for device-local memory.
  hsa_amd_memory_pool_t device_local_memory_pool;
  bool device_local_memory_pool_valid;

  // Memory pools for host-visible (fine-grained) memory.
  hsa_amd_memory_pool_t host_visible_memory_pool;
  bool host_visible_memory_pool_valid;

  // Memory pools for kernarg memory.
  hsa_amd_memory_pool_t kernarg_memory_pool;
  bool kernarg_memory_pool_valid;

  // Completion signal for synchronization.
  hsa_signal_t completion_signal;

  // Tracing context for this device.
  iree_hal_stream_tracing_context_t* tracing_context;

  // File transfer staging buffer info.
  struct {
    iree_hal_buffer_t* buffer;
    iree_host_size_t head;
    iree_host_size_t tail;
    iree_slim_mutex_t mutex;
    iree_notification_t notify;
  } file_transfer_staging_buffer;
} iree_hal_hsa_per_device_info_t;

typedef struct iree_hal_hsa_device_topology_t {
  iree_host_size_t count;
  iree_hal_hsa_per_device_info_t* devices;
} iree_hal_hsa_device_topology_t;

#endif  // IREE_HAL_DRIVERS_HSA_PER_DEVICE_INFORMATION_H_

