// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/queue_upload_ring.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_upload_ring_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_queue_upload_ring_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_queue_upload_ring_memory_t* memory,
    uint32_t min_capacity, iree_hal_amdgpu_queue_upload_ring_t* out_ring) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(memory);
  IREE_ASSERT_ARGUMENT(out_ring);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, min_capacity);
  memset(out_ring, 0, sizeof(*out_ring));

  IREE_ASSERT(memory->memory_pool.handle,
              "queue upload ring memory descriptor must provide a memory pool");
  IREE_ASSERT(memory->access_agents,
              "queue upload ring memory descriptor must provide access agents");
  IREE_ASSERT(memory->access_agent_count > 0 &&
                  memory->access_agent_count <= UINT32_MAX,
              "queue upload ring access agent count must fit in HSA's "
              "uint32_t");
  if (!min_capacity || !iree_host_size_is_power_of_two(min_capacity)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "queue upload ring capacity must be a non-zero "
                             "power of two; got %u",
                             min_capacity));
  }
  IREE_ASSERT(memory->publication.mode ==
                  IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_NONE ||
              memory->publication.mode ==
                  IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH);
  IREE_ASSERT(memory->publication.mode !=
                  IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH ||
              memory->publication.hdp_mem_flush_control);

  size_t alloc_granule = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_amd_memory_pool_get_info(
          IREE_LIBHSA(libhsa), memory->memory_pool,
          HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &alloc_granule),
      "querying HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE for queue "
      "upload ring allocation");
  if (IREE_UNLIKELY(!alloc_granule ||
                    !iree_host_size_is_power_of_two(alloc_granule))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "queue upload memory pool allocation granule must "
                             "be a non-zero power of two (got %zu)",
                             alloc_granule));
  }

  uint32_t capacity = min_capacity;
  while ((uint64_t)capacity < alloc_granule && capacity <= UINT32_MAX / 2) {
    capacity <<= 1;
  }
  IREE_ASSERT(capacity >= min_capacity);
  IREE_ASSERT(iree_host_size_is_power_of_two(capacity));
  if (IREE_UNLIKELY(!iree_host_size_has_alignment(capacity, alloc_granule))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "queue upload ring capacity %u bytes is not "
                             "aligned to pool allocation granule %zu",
                             capacity, alloc_granule));
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_amd_memory_pool_allocate(
          IREE_LIBHSA(libhsa), memory->memory_pool, capacity,
          HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&out_ring->base),
      "allocating queue upload ring of %u bytes", capacity);
  iree_status_t status = iree_hsa_amd_agents_allow_access(
      IREE_LIBHSA(libhsa), (uint32_t)memory->access_agent_count,
      memory->access_agents, /*flags=*/NULL, out_ring->base);
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(status, iree_hsa_amd_memory_pool_free(
                                          IREE_LIBHSA(libhsa), out_ring->base));
    memset(out_ring, 0, sizeof(*out_ring));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, status,
        "making queue upload ring allocation visible to %" PRIhsz " HSA agents",
        memory->access_agent_count);
  }

  out_ring->device_base = (uint64_t)(uintptr_t)out_ring->base;
  out_ring->capacity = capacity;
  out_ring->mask = capacity - 1;
  out_ring->publication = memory->publication;
  iree_atomic_store(&out_ring->write_position, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_ring->read_position, 0, iree_memory_order_relaxed);

  // Fault in the host mapping on the initialization path instead of the first
  // submitter that writes into the ring.
  out_ring->base[0] = 0;

  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, capacity);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_queue_upload_ring_deinitialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_queue_upload_ring_t* ring) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(ring);
  IREE_TRACE_ZONE_BEGIN(z0);

  const uint64_t write = (uint64_t)iree_atomic_load(&ring->write_position,
                                                    iree_memory_order_relaxed);
  const uint64_t read = (uint64_t)iree_atomic_load(&ring->read_position,
                                                   iree_memory_order_relaxed);
  IREE_ASSERT(write == read,
              "queue upload ring has %" PRIu64
              " unreleased bytes at deinit (write=%" PRIu64 ", read=%" PRIu64
              ")",
              write - read, write, read);

  if (ring->base) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(libhsa, ring->base));
  }
  memset(ring, 0, sizeof(*ring));

  IREE_TRACE_ZONE_END(z0);
}
