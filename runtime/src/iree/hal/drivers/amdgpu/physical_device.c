// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_physical_device_initialize(
    iree_hal_amdgpu_system_t* system, hsa_agent_t host_agent,
    hsa_agent_t device_agent, iree_host_size_t queue_count,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  IREE_ASSERT_ARGUMENT(out_physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_physical_device, 0, sizeof(*out_physical_device));

  out_physical_device->system = system;
  out_physical_device->device_agent = device_agent;
  out_physical_device->queue_count = queue_count;

  // Create the host worker thread that will handle scheduler requests.
  // Each queue on this physical device will share the same worker today but we
  // could change that if we become host-bound. In general we should not be
  // using the host during our latency-critical operations but it's possible if
  // memory pool growth/trims take awhile that we end up serializing multiple
  // device queues.
  iree_status_t status = iree_hal_amdgpu_host_worker_initialize(
      system, host_agent, host_allocator, &out_physical_device->host_worker);

  // Initialize each queue and its device-side scheduler.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < queue_count; ++i) {
      status = iree_hal_amdgpu_queue_initialize(
          device_agent, host_allocator, &out_physical_device->queues[i]);
      if (!iree_status_is_ok(status)) break;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_physical_device_deinitialize(out_physical_device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_physical_device_deinitialize(
    iree_hal_amdgpu_physical_device_t* physical_device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Deinitialize all queues and their device-side schedulers before releasing
  // any resources that may be used by them (such as the host worker).
  for (iree_host_size_t i = 0; i < physical_device->queue_count; ++i) {
    iree_hal_amdgpu_queue_deinitialize(&physical_device->queues[i]);
  }

  // Deinitialize the host worker only after all queues have fully terminated.
  iree_hal_amdgpu_host_worker_deinitialize(&physical_device->host_worker);

  memset(physical_device, 0, sizeof(*physical_device));

  IREE_TRACE_ZONE_END(z0);
}
