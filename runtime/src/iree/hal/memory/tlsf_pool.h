// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_TLSF_POOL_H_
#define IREE_HAL_MEMORY_TLSF_POOL_H_

#include "iree/async/notification.h"
#include "iree/base/api.h"
#include "iree/hal/memory/slab_provider.h"
#include "iree/hal/memory/tlsf.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_tlsf_pool_t
//===----------------------------------------------------------------------===//

// Options for creating a HAL pool that wraps iree_hal_memory_tlsf_t.
typedef struct iree_hal_tlsf_pool_options_t {
  // Raw TLSF allocator configuration for each slab. The range length is the
  // fixed slab size and the maximum single reservation served by this pool.
  // Live-pressure exhaustion grows by acquiring another slab of this size.
  // Larger requests are oversized for TLSF and should route to a direct pool.
  iree_hal_memory_tlsf_options_t tlsf_options;

  // Logical byte budget for live reservations in this pool. 0 means unlimited.
  iree_device_size_t budget_limit;

  // Optional named-memory trace identifier for logical reservations returned by
  // this pool. Empty uses a generic process-stable identifier.
  iree_string_view_t trace_name;
} iree_hal_tlsf_pool_options_t;

// Creates a growable TLSF-backed HAL pool over slabs from |slab_provider|.
//
// |notification| is published on reservation release and skips platform wake
// work when no waiter is observing it.
//
// |epoch_query| is an optional host-side completion predicate used to recover
// zero-sync reuse when a requester's frontier is stale but the producer queue
// has already advanced. If epoch_query.fn is NULL, only pure frontier
// dominance enables reuse.
//
// release_reservation() is wait-free with respect to the TLSF mutex: each
// reservation owns a release node, and release publishes that node to a
// lock-free pending stack with one CAS after copying the death frontier into
// node-local storage. reserve() drains pending releases under a per-pool mutex
// before searching TLSF.
//
// Recycled blocks whose frontiers are not dominated by the requester are
// skipped. When no immediately-usable block fits, the pool grows with another
// slab instead of returning wait-frontier reservations; queue-visible
// backpressure remains a budget/provider concern, not a hidden dependency
// between arbitrary transient allocations.
IREE_API_EXPORT iree_status_t iree_hal_tlsf_pool_create(
    iree_hal_tlsf_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification,
    iree_hal_pool_epoch_query_t epoch_query, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_TLSF_POOL_H_
