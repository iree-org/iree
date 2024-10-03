// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/host_worker.h"
#include "iree/hal/drivers/amdgpu/queue.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_physical_device_t {
  iree_hal_amdgpu_system_t* system;
  hsa_agent_t device_agent;
  iree_hal_amdgpu_host_worker_t host_worker;
  iree_host_size_t queue_count;
  iree_hal_amdgpu_queue_t queues[];
} iree_hal_amdgpu_physical_device_t;

static inline iree_host_size_t iree_hal_amdgpu_physical_device_calculate_size(
    iree_host_size_t queue_count) {
  return sizeof(iree_hal_amdgpu_physical_device_t) +
         queue_count * sizeof(iree_hal_amdgpu_queue_t);
}

iree_status_t iree_hal_amdgpu_physical_device_initialize(
    iree_hal_amdgpu_system_t* system, hsa_agent_t host_agent,
    hsa_agent_t device_agent, iree_host_size_t queue_count,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device);

void iree_hal_amdgpu_physical_device_deinitialize(
    iree_hal_amdgpu_physical_device_t* physical_device);

#endif  // IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_H_
