// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;
typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_allocator_t
//===----------------------------------------------------------------------===//

// Creates a buffer allocator that allocates from HSA memory pools.
//
// This is a simple direct allocator — each allocate_buffer call maps to an
// hsa_amd_memory_pool_allocate and each deallocation to an
// hsa_amd_memory_pool_free. No pooling, no suballocation. Suitable for
// bootstrapping and testing; the async suballocator (hal/utils/) will replace
// this on the transient allocation path.
//
// |logical_device| is unretained and must outlive the allocator.
iree_status_t iree_hal_amdgpu_allocator_create(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

#endif  // IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
