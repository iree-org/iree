// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device_queue.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_queue_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_device_queue_t {
  iree_hal_amdgpu_virtual_queue_t base;
} iree_hal_amdgpu_device_queue_t;

iree_host_size_t iree_hal_amdgpu_device_queue_calculate_size(
    const iree_hal_amdgpu_queue_options_t* options) {
  IREE_ASSERT_EQ(options->placement, IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE);
  return sizeof(iree_hal_amdgpu_device_queue_t);
}

iree_status_t iree_hal_amdgpu_device_queue_initialize(
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
  IREE_ASSERT_EQ(options.placement, IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE);
  IREE_ASSERT_ARGUMENT(host_service);
  IREE_ASSERT_ARGUMENT(host_block_pool);
  IREE_ASSERT_ARGUMENT(block_allocators);
  IREE_ASSERT_ARGUMENT(buffer_pool);
  IREE_ASSERT_ARGUMENT(out_queue);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, device_ordinal);

  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "device-side queuing not yet implemented");

  IREE_TRACE_ZONE_END(z0);
  return status;
}
