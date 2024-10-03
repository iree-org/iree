// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_WORKER_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_WORKER_H_

#include "iree/base/api.h"
#include "iree/base/internal/threading.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_worker_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_host_worker_t {
  iree_hal_amdgpu_system_t* system;
  hsa_agent_t agent;
  iree_thread_t* thread;
  hsa_queue_t* queue;
  hsa_signal_t doorbell;
} iree_hal_amdgpu_host_worker_t;

iree_status_t iree_hal_amdgpu_host_worker_initialize(
    iree_hal_amdgpu_system_t* system, hsa_agent_t agent,
    iree_allocator_t host_allocator, iree_hal_amdgpu_host_worker_t* out_worker);

void iree_hal_amdgpu_host_worker_deinitialize(
    iree_hal_amdgpu_host_worker_t* worker);

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_WORKER_H_
