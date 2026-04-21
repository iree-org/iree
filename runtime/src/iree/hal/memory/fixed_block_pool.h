// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_FIXED_BLOCK_POOL_H_
#define IREE_HAL_MEMORY_FIXED_BLOCK_POOL_H_

#include "iree/async/notification.h"
#include "iree/base/api.h"
#include "iree/hal/memory/fixed_block_allocator.h"
#include "iree/hal/memory/slab_provider.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_fixed_block_pool_t
//===----------------------------------------------------------------------===//

// Options for creating a HAL pool that wraps
// iree_hal_memory_fixed_block_allocator_t.
typedef struct iree_hal_fixed_block_pool_options_t {
  // Raw fixed-block allocator configuration. The pool acquires one slab large
  // enough to cover
  // block_allocator_options.block_count * block_allocator_options.block_size.
  iree_hal_memory_fixed_block_allocator_options_t block_allocator_options;

  // Logical byte budget for live reservations in this pool. 0 means unlimited.
  // This is checked before each reservation and can return
  // IREE_HAL_POOL_ACQUIRE_OVER_BUDGET without touching the allocator.
  iree_device_size_t budget_limit;

  // Optional named-memory trace identifier for logical reservations returned by
  // this pool. Empty uses a generic process-stable identifier.
  iree_string_view_t trace_name;
} iree_hal_fixed_block_pool_options_t;

// Creates a fixed-block HAL pool backed by one slab from |slab_provider|.
//
// The pool's reserve path is lock-free on the fast path because
// iree_hal_memory_fixed_block_allocator_t is lock-free. Non-dominated recycled
// blocks are returned as NEEDS_WAIT only when callers set
// IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER and no immediately-usable
// block is available; otherwise they are skipped and the call returns EXHAUSTED
// if no immediately-usable block remains.
// Tainted recycled blocks are never returned as NEEDS_WAIT because their death
// frontier is not precise enough to construct a queue dependency.
//
// |notification| is published on reservation release and skips platform wake
// work when no waiter is observing it.
//
// |epoch_query| is an optional host-side completion predicate used to recover
// zero-sync reuse when a requester's frontier is stale but the producer queue
// has already advanced. If epoch_query.fn is NULL, only pure frontier
// dominance enables reuse.
IREE_API_EXPORT iree_status_t iree_hal_fixed_block_pool_create(
    iree_hal_fixed_block_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification,
    iree_hal_pool_epoch_query_t epoch_query, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_FIXED_BLOCK_POOL_H_
