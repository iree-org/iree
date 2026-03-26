// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"

#include "iree/hal/drivers/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_kernarg_ring_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_kernarg_ring_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t local_agent,
    hsa_amd_memory_pool_t memory_pool, uint32_t min_capacity_in_blocks,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_kernarg_ring_t* out_ring) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_ring);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)min_capacity_in_blocks);
  memset(out_ring, 0, sizeof(*out_ring));

  if (!min_capacity_in_blocks ||
      !iree_host_size_is_power_of_two(min_capacity_in_blocks)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "kernarg ring capacity must be a non-zero "
                             "power of two; got %u",
                             min_capacity_in_blocks));
  }

  // Allocate the vmem ringbuffer. The physical memory comes from the HSA
  // kernarg pool and is triple-mapped (prev/base/next) so that contiguous
  // virtual access across the wrap boundary resolves correctly. The actual
  // capacity may be larger than requested if the HSA allocation granule
  // requires rounding.
  const iree_device_size_t min_capacity_in_bytes =
      (iree_device_size_t)min_capacity_in_blocks *
      sizeof(iree_hal_amdgpu_kernarg_block_t);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdgpu_vmem_ringbuffer_initialize_with_topology(
          libhsa, local_agent, memory_pool, min_capacity_in_bytes, topology,
          IREE_HAL_AMDGPU_ACCESS_MODE_SHARED, &out_ring->ringbuffer),
      "allocating kernarg vmem ringbuffer of %" PRIdsz " bytes (%u blocks)",
      min_capacity_in_bytes, min_capacity_in_blocks);

  // Derive block-level ring parameters from the vmem ringbuffer's actual
  // capacity. The vmem layer rounds up to the allocation granule, which is
  // always a power of two. Dividing by the block size (also a power of two)
  // preserves the power-of-two property.
  const uint32_t capacity = (uint32_t)(out_ring->ringbuffer.capacity /
                                       sizeof(iree_hal_amdgpu_kernarg_block_t));
  IREE_ASSERT(capacity >= min_capacity_in_blocks);
  IREE_ASSERT(iree_host_size_is_power_of_two(capacity));

  out_ring->base =
      (iree_hal_amdgpu_kernarg_block_t*)out_ring->ringbuffer.ring_base_ptr;
  out_ring->capacity = capacity;
  out_ring->mask = capacity - 1;
  iree_atomic_store(&out_ring->write_position, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_ring->read_position, 0, iree_memory_order_relaxed);

  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)capacity);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_kernarg_ring_deinitialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_kernarg_ring_t* ring) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(ring);
  IREE_TRACE_ZONE_BEGIN(z0);

  // All in-flight work must have completed and been reclaimed. Any remaining
  // gap between write and read indicates leaked kernarg blocks.
  const uint64_t write = (uint64_t)iree_atomic_load(&ring->write_position,
                                                    iree_memory_order_relaxed);
  const uint64_t read = (uint64_t)iree_atomic_load(&ring->read_position,
                                                   iree_memory_order_relaxed);
  IREE_ASSERT(write == read,
              "kernarg ring has %" PRIu64
              " unreleased blocks at deinit (write=%" PRIu64 ", read=%" PRIu64
              ")",
              write - read, write, read);

  iree_hal_amdgpu_vmem_ringbuffer_deinitialize(libhsa, &ring->ringbuffer);
  memset(ring, 0, sizeof(*ring));

  IREE_TRACE_ZONE_END(z0);
}
