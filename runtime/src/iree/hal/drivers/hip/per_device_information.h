// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_PER_DEVICE_INFORMATION_H_
#define IREE_HAL_DRIVERS_HIP_PER_DEVICE_INFORMATION_H_

#include "iree/hal/drivers/hip/dispatch_thread.h"
#include "iree/hal/drivers/hip/hip_headers.h"

typedef struct iree_hal_stream_tracing_context_t
    iree_hal_stream_tracing_context_t;
typedef struct iree_hal_hip_event_pool_t iree_hal_hip_event_pool_t;

typedef struct iree_hal_hip_per_device_info_t {
  hipCtx_t hip_context;
  hipDevice_t hip_device;
  hipStream_t hip_dispatch_stream;

  iree_hal_stream_tracing_context_t* tracing_context;

  iree_hal_hip_event_pool_t* device_event_pool;

  iree_hal_hip_dispatch_thread_t* dispatch_thread;
} iree_hal_hip_per_device_info_t;

typedef struct iree_hal_hip_device_topology_t {
  iree_host_size_t count;
  iree_hal_hip_per_device_info_t devices[/*count*/];
} iree_hal_hip_device_topology_t;

#endif  // IREE_HAL_DRIVERS_HIP_PER_DEVICE_INFORMATION_H_
