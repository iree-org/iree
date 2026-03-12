// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/util/queue_header_pool.h"

#include "iree/async/region.h"

// Region that bundles the allocator for self-contained cleanup.
typedef struct iree_hal_remote_pool_region_t {
  iree_async_region_t base;
  iree_allocator_t host_allocator;
} iree_hal_remote_pool_region_t;

static void iree_hal_remote_pool_region_destroy(iree_async_region_t* region) {
  iree_hal_remote_pool_region_t* pool_region =
      (iree_hal_remote_pool_region_t*)region;
  iree_allocator_t host_allocator = pool_region->host_allocator;
  iree_allocator_free(host_allocator, pool_region->base.base_ptr);
  iree_allocator_free(host_allocator, pool_region);
}

iree_status_t iree_hal_remote_create_queue_header_pool(
    iree_host_size_t buffer_count, iree_host_size_t buffer_size,
    iree_allocator_t host_allocator, iree_async_buffer_pool_t** out_pool) {
  *out_pool = NULL;

  iree_hal_remote_pool_region_t* pool_region = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*pool_region), (void**)&pool_region));

  // Allocate the backing memory with overflow-checked sizing.
  uint8_t* memory = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, buffer_count, buffer_size, (void**)&memory);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, pool_region);
    return status;
  }

  memset(pool_region, 0, sizeof(*pool_region));
  iree_atomic_ref_count_init(&pool_region->base.ref_count);
  pool_region->base.destroy_fn = iree_hal_remote_pool_region_destroy;
  pool_region->base.base_ptr = memory;
  pool_region->base.length = buffer_count * buffer_size;
  pool_region->base.buffer_size = buffer_size;
  pool_region->base.buffer_count = buffer_count;
  pool_region->host_allocator = host_allocator;

  status = iree_async_buffer_pool_allocate(&pool_region->base, host_allocator,
                                           out_pool);
  // Release our ref -- pool retains the region. On failure, both refs are
  // released and the region self-destructs.
  iree_async_region_release(&pool_region->base);
  return status;
}
