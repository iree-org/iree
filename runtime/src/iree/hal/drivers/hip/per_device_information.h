// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_PER_DEVICE_INFORMATION_H__
#define IREE_HAL_DRIVERS_HIP_PER_DEVICE_INFORMATION_H__

#include "iree/hal/drivers/hip/hip_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_stream_tracing_context_t
    iree_hal_stream_tracing_context_t;
typedef struct iree_hal_hip_event_pool_t iree_hal_hip_event_pool_t;
typedef struct iree_hal_deferred_work_queue_t iree_hal_deferred_work_queue_t;

typedef struct iree_hal_hip_per_device_information_t {
  hipCtx_t hip_context;
  hipDevice_t hip_device;
  hipStream_t hip_dispatch_stream;

  iree_hal_stream_tracing_context_t* tracing_context;

  iree_hal_hip_event_pool_t* device_event_pool;

  // A queue to order device workloads and relase to the GPU when constraints
  // are met. It buffers submissions and allocations internally before they
  // are ready. This queue couples with HAL semaphores backed by iree_event_t
  // and hipEvent_t objects.
  iree_hal_deferred_work_queue_t* work_queue;

} iree_hal_hip_per_device_information_t;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_PER_DEVICE_INFORMATION_H__
