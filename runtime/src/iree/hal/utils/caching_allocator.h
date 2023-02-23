// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_CACHING_ALLOCATOR_H_
#define IREE_HAL_UTILS_CACHING_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A HAL buffer allocator that caches allocations instead of returning them to
// the underlying device allocator.
//
// Allocation limits can be independently tuned per heap they originate from to
// enable heavier caching of more expensive allocations such as mappable
// device-local and host-visible buffers on devices with discrete memory.
// Pools are scanned in-order to allow for prioritization.
//
// Thread-safe: the allocator can be shared across multiple user-level devices
// manipulated from multiple threads.
typedef struct iree_hal_caching_allocator_t iree_hal_caching_allocator_t;

// Parameters used to configure an iree_hal_caching_allocator_t pool.
// These cannot be changed once the allocator has been created.
typedef struct iree_hal_caching_allocator_pool_params_t {
  // Underlying allocator heap that services allocation requests for the pool.
  //
  // Additional flags may be added on top of what the underlying heap supports
  // such as IREE_HAL_MEMORY_TYPE_TRANSIENT to limit a pool to only working with
  // transient buffers.
  iree_hal_allocator_memory_heap_t heap;

  // Maximum size of an allocation in bytes; larger allocations will be sent
  // directly through to the underlying allocator.
  iree_device_size_t max_allocation_size;

  // Maximum total size of all allocations made from the pool that will be
  // retained. After this limit is reached allocation requests will be sent
  // directly through to the underlying allocator.
  iree_device_size_t max_allocation_capacity;

  // Maximum number of free allocations that will be tracked.
  // This is used to allocate storage for the free list and should be reasonably
  // bounded (~64-1024).
  iree_host_size_t max_free_allocation_count;
} iree_hal_caching_allocator_pool_params_t;

// Initializes |out_params| to the default values using |heap| for storage.
void iree_hal_caching_allocator_pool_params_initialize(
    iree_hal_allocator_memory_heap_t heap,
    iree_hal_caching_allocator_pool_params_t* out_params);

// Creates an allocator that caches allocations using |device_allocator| for
// serving requests.
//
// All allocations will be cached until the allocator is trimmed and in highly
// dynamic programs this can easily exceed available memory. Prefer using
// explicit pools per heap with maximum sizes for safer behavior and better
// tuning (limit caching to expensive heaps, etc).
//
// Buffer import and export and other operations that the caching allocator
// cannot service will be directed to the underlying |device_allocator|.
//
// Thread-safe: internal synchronization of caching allocator data structures
// allows multiple threads to allocate and free buffers.
iree_status_t iree_hal_caching_allocator_create_unbounded(
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

// Creates an allocator that caches allocations using |device_allocator| for
// serving requests. Each caching allocator can have one or more pools backed by
// different underlying allocator heaps. Any allocation requests that cannot be
// serviced by the defined pools will route down to the underlying allocator.
//
// Allocations from a pool over the max_allocation_size will be routed to the
// |device_allocator| directly as will buffer import and export and other
// operations that the caching allocator cannot service.
//
// Thread-safe: internal synchronization of caching allocator data structures
// allows multiple threads to allocate and free buffers.
iree_status_t iree_hal_caching_allocator_create_with_pools(
    iree_host_size_t pool_count,
    const iree_hal_caching_allocator_pool_params_t* pool_params,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

// Creates a caching allocator with the given key-value |config_pairs|.
// When no |config_pairs| are provided the caching allocator will be created as
// unbounded, retaining all allocations of all sizes in all heaps. If pairs are
// provided then each specifies a pool in the allocator that maps to a heap
// based on the heap key as parsed by iree_hal_select_heap. Multiple pools may
// share the same heap but with different limits, for example allowing at most
// one device local allocation greater than 100MB to be retained while 10 less
// than 100MB can be retained. Wildcards can be used to indicate max values or
// defaults.
//
// Expected form:
//   heap_key=max_allocation_size;max_allocation_capacity;max_free_allocation_count
// Example:
//   device_local=1gib;1gib;8
//   host_local=*;*;32
iree_status_t iree_hal_caching_allocator_create_from_spec(
    iree_string_view_t config_pairs, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_CACHING_ALLOCATOR_H_
