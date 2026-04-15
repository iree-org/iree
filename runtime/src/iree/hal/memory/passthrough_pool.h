// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_PASSTHROUGH_POOL_H_
#define IREE_HAL_MEMORY_PASSTHROUGH_POOL_H_

#include "iree/base/api.h"
#include "iree/hal/memory/slab_provider.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_notification_t iree_async_notification_t;

// Options for creating a pass-through HAL pool.
typedef struct iree_hal_passthrough_pool_options_t {
  // Optional named-memory trace identifier for logical reservations returned by
  // this pool. Empty uses a generic process-stable identifier.
  iree_string_view_t trace_name;
} iree_hal_passthrough_pool_options_t;

// Creates a pass-through pool that delegates every allocation directly to the
// slab provider. Each acquire_reservation() acquires a new slab and each
// release_reservation() frees it. No suballocation, no offset management, and
// no death-frontier tracking.
//
// This is the simplest possible pool — it exists to provide the default device
// pool with the same behavior as direct allocation through the current
// iree_hal_allocator_t. It proves the pool vtable dispatch chain works and
// serves as a baseline for benchmarking suballocating pool types.
//
// |slab_provider| is retained for the lifetime of the pool.
// |notification| is retained for the lifetime of the pool, published on
// release_reservation(), and skips wake work when no waiter is observing it.
// |host_allocator| is used for the pool struct and per-buffer release state.
iree_status_t iree_hal_passthrough_pool_create(
    iree_hal_passthrough_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_PASSTHROUGH_POOL_H_
