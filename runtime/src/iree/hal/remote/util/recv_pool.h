// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ref-counted receive buffer pool for remote HAL network I/O.
//
// Bundles a slab, proactor-registered region, and lock-free buffer pool into
// a single shareable object. The driver creates one pool and all child devices
// retain it; the last release frees the underlying memory.
//
// NUMA-aware: the slab is allocated on the specified node, and the proactor
// driving I/O is selected for that node from the proactor pool.

#ifndef IREE_HAL_REMOTE_UTIL_RECV_POOL_H_
#define IREE_HAL_REMOTE_UTIL_RECV_POOL_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_recv_pool_t iree_hal_remote_recv_pool_t;

// Creates a receive buffer pool for remote HAL network I/O.
//
// Selects the proactor for |numa_node_id| from |proactor_pool| (use
// IREE_ASYNC_AFFINITY_NUMA_NODE_ANY for no preference), allocates a slab
// on that NUMA node, registers it with the proactor, and creates a lock-free
// buffer pool over the registered region.
//
// The returned pool must be released with iree_hal_remote_recv_pool_release.
iree_status_t iree_hal_remote_recv_pool_create(
    iree_async_proactor_pool_t* proactor_pool, uint32_t numa_node_id,
    iree_allocator_t host_allocator,
    iree_hal_remote_recv_pool_t** out_recv_pool);

// Wraps pre-created components into a recv_pool. Takes ownership of all
// references (retains proactor, slab, region, buffer_pool). Used when slab
// registration must happen before the proactor's poll thread starts (bd-oqi8).
iree_status_t iree_hal_remote_recv_pool_wrap(
    iree_async_proactor_t* proactor, iree_async_slab_t* slab,
    iree_async_region_t* region, iree_async_buffer_pool_t* buffer_pool,
    iree_allocator_t host_allocator,
    iree_hal_remote_recv_pool_t** out_recv_pool);

void iree_hal_remote_recv_pool_retain(iree_hal_remote_recv_pool_t* recv_pool);
void iree_hal_remote_recv_pool_release(iree_hal_remote_recv_pool_t* recv_pool);

// Returns the proactor that drives I/O for this pool's registered buffers.
// Borrowed — valid for the lifetime of the recv_pool.
iree_async_proactor_t* iree_hal_remote_recv_pool_proactor(
    iree_hal_remote_recv_pool_t* recv_pool);

// Returns the underlying buffer pool for passing to session/channel APIs.
// Borrowed — valid for the lifetime of the recv_pool.
iree_async_buffer_pool_t* iree_hal_remote_recv_pool_buffer_pool(
    iree_hal_remote_recv_pool_t* recv_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_UTIL_RECV_POOL_H_
