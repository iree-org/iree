// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SLAB_PROVIDER_H_
#define IREE_HAL_DRIVERS_AMDGPU_SLAB_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_slab_provider_t
//===----------------------------------------------------------------------===//

// Creates a slab provider backed by an HSA memory pool on one GPU agent.
//
// The provider acquires whole slabs with hsa_amd_memory_pool_allocate(), grants
// every agent in |topology| access to the slab, and wraps slab slices as
// iree_hal_amdgpu_buffer_t views. The provider borrows |device|, |libhsa|, and
// |topology|; the owning physical/logical device must outlive the provider and
// every pool/buffer created from it.
iree_status_t iree_hal_amdgpu_slab_provider_create(
    iree_hal_device_t* device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    hsa_amd_memory_pool_t memory_pool, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SLAB_PROVIDER_H_
