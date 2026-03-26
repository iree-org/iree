// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_POOL_SET_H_
#define IREE_HAL_POOL_SET_H_

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_pool_set_t
//===----------------------------------------------------------------------===//

// A lightweight routing utility that maps buffer parameters to pools.
//
// Pool sets are NOT HAL resources — they are initialize/deinitialize structs
// with no ref counting. They hold retained references to pools but do not own
// them exclusively. A pool can be registered in multiple pool sets. When all
// pool sets (and all buffers) release their refs to a pool, the pool is
// destroyed.
//
// Pool sets are the scoping mechanism for multi-model/multi-library isolation:
// different parts of the system get different pool sets, potentially sharing
// some pools. A multi-model server creates per-model pool sets with model-
// specific KV cache pools but shared general-purpose pools. A library accepts
// a pool set (or individual pools) from its caller. A simple CLI app uses the
// device's default pool set.
//
// Selection algorithm:
//   1. For each registered pool (in registration order for stability):
//      a. Check memory type compatibility: pool provides at least the
//         required type bits.
//      b. Check usage compatibility: pool supports at least the required
//         usage bits.
//      c. Check size: allocation_size is within the pool's min/max range.
//   2. Among compatible pools, return the one with the highest priority.
//   3. If no pool matches, return NULL.
//
// This is O(N) where N is the number of registered pools — typically 3-8,
// faster than a hash lookup.
typedef struct iree_hal_pool_set_t {
  iree_host_size_t entry_count;
  iree_host_size_t entry_capacity;
  struct iree_hal_pool_set_entry_t* entries;
  iree_allocator_t host_allocator;
} iree_hal_pool_set_t;

// Initializes a pool set. Must be paired with iree_hal_pool_set_deinitialize().
// |initial_capacity| is the number of entries to pre-allocate (grows on
// demand). 8 is a reasonable default for most configurations.
iree_status_t iree_hal_pool_set_initialize(iree_host_size_t initial_capacity,
                                           iree_allocator_t host_allocator,
                                           iree_hal_pool_set_t* out_pool_set);

// Deinitializes a pool set, releasing all retained pool references and freeing
// the entry array.
void iree_hal_pool_set_deinitialize(iree_hal_pool_set_t* pool_set);

// Registers a pool with the pool set. The pool is retained.
//
// |priority| controls selection when multiple pools match the same request.
// Higher values win. Use 0 for catch-all pools and higher values for
// specialized pools (e.g., a block pool for KV cache at priority 10 takes
// precedence over a general TLSF pool at priority 0 when both are compatible).
//
// The pool's capabilities are queried and cached at registration time via
// iree_hal_pool_query_capabilities().
iree_status_t iree_hal_pool_set_register(iree_hal_pool_set_t* pool_set,
                                         int32_t priority,
                                         iree_hal_pool_t* pool);

// Selects the best pool for the given buffer parameters and allocation size.
//
// Returns the highest-priority pool whose capabilities are compatible with
// |params| and |allocation_size|. Returns NULL if no registered pool can
// satisfy the request.
iree_hal_pool_t* iree_hal_pool_set_select(const iree_hal_pool_set_t* pool_set,
                                          iree_hal_buffer_params_t params,
                                          iree_device_size_t allocation_size);

// Selects a pool and allocates a buffer from it.
//
// Equivalent to iree_hal_pool_set_select() + iree_hal_pool_allocate_buffer().
// Returns IREE_STATUS_NOT_FOUND if no pool matches the parameters (with a
// diagnostic message listing the requested params and all registered pools'
// capabilities).
iree_status_t iree_hal_pool_set_allocate_buffer(
    iree_hal_pool_set_t* pool_set, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    const iree_async_frontier_t* requester_frontier, iree_timeout_t timeout,
    iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_POOL_SET_H_
