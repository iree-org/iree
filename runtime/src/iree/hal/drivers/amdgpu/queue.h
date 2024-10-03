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
#include "iree/hal/drivers/amdgpu/device/tracing.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_queue_t {
  hsa_agent_t agent;

  hsa_queue_t* control_queue;
  hsa_queue_t* execution_queue;

  // DO NOT SUBMIT large block allocs for everything?

  iree_hal_amdgpu_device_queue_scheduler_t* scheduler;
  iree_hal_amdgpu_device_trace_buffer_t* trace_buffer;

  // DO NOT SUBMIT resource pools

  iree_hal_amdgpu_device_signal_pool_t* signal_pool;
} iree_hal_amdgpu_queue_t;

iree_status_t iree_hal_amdgpu_queue_initialize(
    hsa_agent_t agent, iree_allocator_t host_allocator,
    iree_hal_amdgpu_queue_t* out_queue);

void iree_hal_amdgpu_queue_deinitialize(iree_hal_amdgpu_queue_t* queue);

#endif  // IREE_HAL_DRIVERS_AMDGPU_QUEUE_H_
