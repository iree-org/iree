// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_MUTEX_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_MUTEX_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_mutex_t
//===----------------------------------------------------------------------===//

// Device spin-lock mutex.
// This can run on the host as well but is optimized for device usage. Spinning
// on the host is a bad idea. Spinning on the device is _also_ a bad idea, but
// does have its uses.
//
// Note that because atomics are not guaranteed to work off-agent this is only
// to be used for intra-agent exclusion such as when multiple queues on the
// same agent are sharing a data structure.
//
// Reference: https://rigtorp.se/spinlock/
typedef iree_amdgpu_scoped_atomic_uint32_t iree_hal_amdgpu_device_mutex_t;

#define IREE_HAL_AMDGPU_DEVICE_MUTEX_UNLOCKED 0u
#define IREE_HAL_AMDGPU_DEVICE_MUTEX_LOCKED 1u

// Initializes a mutex to the unlocked state.
static inline void iree_hal_amdgpu_device_mutex_initialize(
    iree_hal_amdgpu_device_mutex_t* IREE_AMDGPU_RESTRICT out_mutex) {
  uint32_t initial_value = IREE_HAL_AMDGPU_DEVICE_MUTEX_UNLOCKED;
  IREE_AMDGPU_SCOPED_ATOMIC_INIT(out_mutex, initial_value);
}

// Spins until a lock on the mutex is acquired.
static inline void iree_hal_amdgpu_device_mutex_lock(
    iree_hal_amdgpu_device_mutex_t* IREE_AMDGPU_RESTRICT mutex) {
  for (;;) {
    // Optimistically assume the lock is free on the first try.
    uint32_t prev = IREE_HAL_AMDGPU_DEVICE_MUTEX_UNLOCKED;
    if (iree_amdgpu_scoped_atomic_compare_exchange_strong(
            mutex, &prev, IREE_HAL_AMDGPU_DEVICE_MUTEX_LOCKED,
            iree_amdgpu_memory_order_acquire, iree_amdgpu_memory_order_acquire,
            iree_amdgpu_memory_scope_system)) {
      return;
    }
    // Wait for lock to be released without generating cache misses.
    while (iree_amdgpu_scoped_atomic_load(mutex,
                                          iree_amdgpu_memory_order_relaxed,
                                          iree_amdgpu_memory_scope_system)) {
      // Yield for a bit to give the other thread a chance to unlock.
      iree_amdgpu_yield();
    }
  }
}

// Unlocks a mutex. Must be called with the lock held by the caller.
static inline void iree_hal_amdgpu_device_mutex_unlock(
    iree_hal_amdgpu_device_mutex_t* IREE_AMDGPU_RESTRICT mutex) {
  iree_amdgpu_scoped_atomic_store(mutex, IREE_HAL_AMDGPU_DEVICE_MUTEX_UNLOCKED,
                                  iree_amdgpu_memory_order_release,
                                  iree_amdgpu_memory_scope_system);
}

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_MUTEX_H_
