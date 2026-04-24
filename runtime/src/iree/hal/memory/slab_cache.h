// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_SLAB_CACHE_H_
#define IREE_HAL_MEMORY_SLAB_CACHE_H_

#include "iree/base/api.h"
#include "iree/base/threading/affinity.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Options for creating a slab cache.
typedef struct iree_hal_slab_cache_options_t {
  // Size of each pre-acquired slab. All slabs in the cache are this size.
  // Requests for larger slabs bypass the cache and go directly to the inner
  // provider (no background preparation, no caching).
  iree_device_size_t slab_size;

  // Target number of ready slabs to maintain in the freelist. The background
  // thread refills to this count when the freelist drops below it.
  uint32_t target_count;

  // Maximum number of slabs to cache. Released slabs are pushed to the
  // freelist until this limit; excess is returned to the inner provider.
  // Prevents unbounded memory accumulation during low-utilization periods.
  uint32_t max_count;
} iree_hal_slab_cache_options_t;

// Creates a slab provider that wraps |inner_provider| with a background
// pre-acquisition and pre-fault cache.
//
// The returned provider maintains a freelist of pre-acquired, pre-faulted
// slabs refilled by a background thread. acquire_slab() pops from the
// freelist (fast path: nanoseconds) or waits for the thread (slow path).
// release_slab() pushes back to the freelist for reuse (up to max_count)
// or releases to the inner provider.
//
// trim() releases cached slabs according to the provided flags.
// query_stats() reports cache hit/miss rates, EMA reuse interval, and
// prefault time alongside the inner provider's statistics.
//
// The background thread is NUMA-pinned to |thread_affinity| so that
// prefault first-touch allocates pages on the correct NUMA node. Pass a
// zero-initialized affinity for no pinning.
//
// |inner_provider| is retained for the lifetime of the returned provider.
iree_status_t iree_hal_slab_cache_create(
    iree_hal_slab_cache_options_t options,
    iree_hal_slab_provider_t* inner_provider,
    iree_thread_affinity_t thread_affinity, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_SLAB_CACHE_H_
