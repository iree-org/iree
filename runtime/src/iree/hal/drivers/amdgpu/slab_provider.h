// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SLAB_PROVIDER_H_
#define IREE_HAL_DRIVERS_AMDGPU_SLAB_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_slab_provider_t
//===----------------------------------------------------------------------===//

// HSA memory-pool properties needed to configure slab-backed HAL pools.
typedef struct iree_hal_amdgpu_slab_provider_memory_pool_properties_t {
  // Smallest allocation-size multiple accepted by hsa_amd_memory_pool_allocate.
  iree_device_size_t allocation_granule;

  // Base-pointer alignment guaranteed by hsa_amd_memory_pool_allocate.
  iree_device_size_t allocation_alignment;

  // HAL memory type bits provided by this HSA memory pool.
  iree_hal_memory_type_t memory_type;

  // HAL buffer usage bits supported by buffers materialized from this pool.
  iree_hal_buffer_usage_t supported_usage;
} iree_hal_amdgpu_slab_provider_memory_pool_properties_t;

// Queries HSA memory-pool properties used by AMDGPU slab providers and pools.
iree_status_t iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_slab_provider_memory_pool_properties_t* out_properties);

// Creates a slab provider backed by an HSA memory pool on one GPU agent.
//
// The provider acquires whole slabs with hsa_amd_memory_pool_allocate(), grants
// every agent in |topology| access to the slab, and wraps slab slices as
// iree_hal_amdgpu_buffer_t views. |queue_affinity_mask| identifies the HAL
// queues in this physical memory domain; wrap_buffer() replaces
// IREE_HAL_QUEUE_AFFINITY_ANY with that mask and rejects explicit affinities
// outside it so placement metadata always routes PREFER_ORIGIN dealloca back
// into the provider's domain. Materialized buffer view wrappers are allocated
// from |buffer_pool|, which must be in the same physical-device lifetime domain
// as the backing HSA memory. The provider borrows |device|, |libhsa|,
// |topology|, and |buffer_pool|; the owning physical/logical device must
// outlive the provider and every pool/buffer created from it.
iree_status_t iree_hal_amdgpu_slab_provider_create(
    iree_hal_device_t* device, const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    hsa_amd_memory_pool_t memory_pool, iree_host_size_t physical_device_ordinal,
    iree_hal_queue_affinity_t queue_affinity_mask,
    iree_hal_amdgpu_buffer_pool_t* buffer_pool, iree_string_view_t trace_name,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SLAB_PROVIDER_H_
