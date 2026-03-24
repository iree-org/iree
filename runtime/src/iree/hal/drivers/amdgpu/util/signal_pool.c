// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/signal_pool.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_signal_pool_t
//===----------------------------------------------------------------------===//

// Creates |count| signals and pushes them onto the free list.
// Grows the free list array if needed. Caller must hold the mutex.
static iree_status_t iree_hal_amdgpu_host_signal_pool_grow(
    iree_hal_amdgpu_host_signal_pool_t* pool, iree_host_size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)count);

  // Grow the array to hold all existing + new signals. The array capacity
  // must always be >= allocated_count so that every signal can be returned
  // to the free list simultaneously.
  iree_host_size_t required_capacity = pool->allocated_count + count;
  iree_status_t status = iree_ok_status();
  if (required_capacity > pool->free_capacity) {
    status = iree_allocator_grow_array(
        pool->host_allocator, required_capacity, sizeof(hsa_signal_t),
        &pool->free_capacity, (void**)&pool->free_signals);
  }

  // Create signals and push onto the free list.
  iree_host_size_t created_count = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    hsa_signal_t signal = {0};
    status = iree_hsa_amd_signal_create(IREE_LIBHSA(pool->libhsa),
                                        /*initial_value=*/0,
                                        /*num_consumers=*/0, /*consumers=*/NULL,
                                        /*attributes=*/0, &signal);
    if (iree_status_is_ok(status)) {
      pool->free_signals[pool->free_count + i] = signal;
      ++created_count;
    }
  }

  if (iree_status_is_ok(status)) {
    pool->free_count += created_count;
    pool->allocated_count += created_count;
  } else {
    // Destroy any signals we created before the failure.
    for (iree_host_size_t i = 0; i < created_count; ++i) {
      IREE_IGNORE_ERROR(iree_hsa_signal_destroy(
          IREE_LIBHSA(pool->libhsa), pool->free_signals[pool->free_count + i]));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_host_signal_pool_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_host_size_t initial_capacity,
    iree_host_size_t batch_size, iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_signal_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)initial_capacity);

  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->libhsa = libhsa;
  out_pool->host_allocator = host_allocator;
  out_pool->batch_size =
      batch_size ? batch_size
                 : IREE_HAL_AMDGPU_HOST_SIGNAL_POOL_BATCH_SIZE_DEFAULT;

  iree_slim_mutex_initialize(&out_pool->mutex);

  iree_status_t status = iree_ok_status();
  if (initial_capacity > 0) {
    iree_slim_mutex_lock(&out_pool->mutex);
    status = iree_hal_amdgpu_host_signal_pool_grow(out_pool, initial_capacity);
    iree_slim_mutex_unlock(&out_pool->mutex);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_signal_pool_deinitialize(out_pool);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_signal_pool_deinitialize(
    iree_hal_amdgpu_host_signal_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  // All signals must have been released back to the pool.
  IREE_ASSERT(pool->free_count == pool->allocated_count,
              "signal pool has outstanding unreleased signals");

  // Destroy all signals via the free list.
  for (iree_host_size_t i = 0; i < pool->free_count; ++i) {
    IREE_IGNORE_ERROR(iree_hsa_signal_destroy(IREE_LIBHSA(pool->libhsa),
                                              pool->free_signals[i]));
  }

  iree_allocator_free(pool->host_allocator, pool->free_signals);
  iree_slim_mutex_deinitialize(&pool->mutex);
  memset(pool, 0, sizeof(*pool));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_host_signal_pool_acquire(
    iree_hal_amdgpu_host_signal_pool_t* pool, hsa_signal_value_t initial_value,
    hsa_signal_t* out_signal) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_signal);
  *out_signal = (hsa_signal_t){0};

  iree_slim_mutex_lock(&pool->mutex);

  // Grow the pool if empty.
  iree_status_t status = iree_ok_status();
  if (pool->free_count == 0) {
    status = iree_hal_amdgpu_host_signal_pool_grow(pool, pool->batch_size);
  }

  // Pop from the free list (LIFO).
  hsa_signal_t signal = {0};
  if (iree_status_is_ok(status)) {
    signal = pool->free_signals[--pool->free_count];
  }

  iree_slim_mutex_unlock(&pool->mutex);

  // Reset the signal value outside the lock — the HSA signal lives in CPU
  // kernarg memory so this is a local RAM write with no PCIe traffic.
  if (iree_status_is_ok(status)) {
    iree_hsa_signal_store_relaxed(IREE_LIBHSA(pool->libhsa), signal,
                                  initial_value);
    *out_signal = signal;
  }
  return status;
}

void iree_hal_amdgpu_host_signal_pool_release(
    iree_hal_amdgpu_host_signal_pool_t* pool, hsa_signal_t signal) {
  IREE_ASSERT_ARGUMENT(pool);

  iree_slim_mutex_lock(&pool->mutex);

  // The free list capacity is always >= allocated_count, so this can never
  // overflow unless the signal was double-freed or came from another pool.
  IREE_ASSERT(pool->free_count < pool->allocated_count,
              "signal pool release overflow: double-free or wrong pool");

  pool->free_signals[pool->free_count++] = signal;

  iree_slim_mutex_unlock(&pool->mutex);
}
