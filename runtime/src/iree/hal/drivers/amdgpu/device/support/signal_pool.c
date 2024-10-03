// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/signal_pool.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_signal_pool_t
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_device_signal_pool_initialize(
    iree_hal_amdgpu_device_signal_pool_t* IREE_AMDGPU_RESTRICT signal_pool,
    uint32_t signal_count, iree_hsa_signal_t* IREE_AMDGPU_RESTRICT signals) {
  // Populate the entire ringbuffer with the given signals list.
  signal_pool->capacity = signal_count;
  for (uint32_t i = 0; i < signal_count; ++i) {
    signal_pool->entries[i].value = signals[i].handle;
  }

  // Reset the size of the pool to the signal count as all signals have been
  // added to it. Note that we do this last to ensure that all prior writes
  // initializing the ringbuffer have completed.
  iree_amdgpu_scoped_atomic_store(&signal_pool->size, signal_count,
                                  iree_amdgpu_memory_order_seq_cst,
                                  iree_amdgpu_memory_scope_device);
}

iree_hsa_signal_t iree_hal_amdgpu_device_signal_pool_acquire(
    iree_hal_amdgpu_device_signal_pool_t* IREE_AMDGPU_RESTRICT signal_pool,
    int64_t initial_value) {
  // Acquire the signal. Note that its value is undefined as it may be recycled
  // from previous use.
  iree_hsa_signal_t result;
  result.handle = iree_hal_amdgpu_device_ringbuffer_uint64_dequeue(signal_pool);

  // TODO(benvanik): use relaxed here? callers should be releasing after they
  // acquire the signal but we err on the side of safety.
  iree_hsa_signal_store(result, initial_value,
                        iree_amdgpu_memory_order_release);

  return result;
}

void iree_hal_amdgpu_device_signal_pool_release(
    iree_hal_amdgpu_device_signal_pool_t* IREE_AMDGPU_RESTRICT signal_pool,
    iree_hsa_signal_t signal) {
  // Note that we put the signal back without changing its value - in most cases
  // it'll be 0 as we primarily use these as binary semaphores in AQL packets
  // but it's possible for it to be any value.
  iree_hal_amdgpu_device_ringbuffer_uint64_enqueue(signal_pool, signal.handle);
}
