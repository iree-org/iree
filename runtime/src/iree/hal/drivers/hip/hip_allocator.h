// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_HIP_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/memory_pools.h"
#include "iree/hal/drivers/hip/status_util.h"

typedef struct iree_hal_hip_device_topology_t iree_hal_hip_device_topology_t;

// Creates a HIP memory allocator.
// |device| and |stream| will be used for management operations.
// |pools| provides memory pools that may be shared across multiple allocators
// and the pointer must remain valid for the lifetime of the allocator.
iree_status_t iree_hal_hip_allocator_create(
    iree_hal_device_t* parent_device,
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    iree_hal_hip_device_topology_t topology, bool supports_memory_pools,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

bool iree_hal_hip_allocator_isa(iree_hal_allocator_t* base_value);

iree_status_t iree_hal_hip_allocator_alloc_async(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* buffer);

iree_status_t iree_hal_hip_allocator_free_async(iree_hal_allocator_t* allocator,
                                                iree_hal_buffer_t* buffer);

iree_status_t iree_hal_hip_allocator_free_sync(iree_hal_allocator_t* allocator,
                                               iree_hal_buffer_t* buffer);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_ALLOCATOR_H_
