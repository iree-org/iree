// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_PER_DEVICE_INFORMATION_H_
#define IREE_HAL_DRIVERS_HIP_PER_DEVICE_INFORMATION_H_

#include "iree/hal/drivers/hip/dispatch_thread.h"
#include "iree/hal/drivers/hip/hip_headers.h"
#include "iree/hal/drivers/hip/memory_pools.h"

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

  iree_hal_hip_memory_pools_t memory_pools;

  // Used in any place we need an event that is already signaled.
  //
  // This is to work around some hip runtime limitations.
  // The implementation of hipMallocAsync will re-use ANY previous
  // allocation that is at least as large as the requested allocation,
  // So if you alloc 300MB, free 300MB, alloc 1B, alloc 300MB, free 300MB, alloc
  // 1B, you will consume 600MB for what should only be 2 bytes of data. We can
  // work around this by calling hipEventSynchronize before we allocate memory,
  // (which unfortunately clears the internal cache), and then our allocator
  // just effectively boils down to hipMalloc/hipFree.
} iree_hal_hip_per_device_info_t;

typedef struct iree_hal_hip_device_topology_t {
  iree_host_size_t count;
  iree_hal_hip_per_device_info_t* devices;
} iree_hal_hip_device_topology_t;

#endif  // IREE_HAL_DRIVERS_HIP_PER_DEVICE_INFORMATION_H_
