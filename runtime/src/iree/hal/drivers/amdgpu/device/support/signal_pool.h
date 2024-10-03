// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_SIGNAL_POOL_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_SIGNAL_POOL_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/ringbuffer.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_signal_pool_t
//===----------------------------------------------------------------------===//

// A pool of HSA signals available for transient usage by the device runtime.
// The host allocates the signals and initializes the free list with their
// opaque handles. The signals must be allocated via the HSA
// hsa_amd_signal_create API with either HSA_AMD_SIGNAL_AMD_GPU_ONLY or
// HSA_AMD_SIGNAL_IPC today but we may partition pools for each as-needed.
//
// This is implemented as a ringbuffer initially populated with all available
// signals. Signals are acquired by dequeuing them from the ringbuffer and
// released by enqueuing them. There are probably easier ways of doing this but
// the ringbuffer is useful for ordering other work as well and is known to work
// well with multiple producers and consumers that may be concurrently acquiring
// and releasing signals on device.
//
// The signal pool is only used by the device and can be allocated in
// local memory without host visibility. Signals in the pool may be used on the
// host and need to be standard HSA signals.
typedef iree_hal_amdgpu_device_ringbuffer_uint64_t
    iree_hal_amdgpu_device_signal_pool_t;

// Returns the total size in bytes of the signal pool with the given
// power-of-two capacity.
#define iree_hal_amdgpu_device_signal_pool_calculate_size(capacity) \
  iree_hal_amdgpu_device_ringbuffer_uint64_calculate_size(capacity)

//===----------------------------------------------------------------------===//
// Device-side API
//===----------------------------------------------------------------------===//

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Initializes the signal pool with the given set of HSA signals.
// The signals must remain valid for the lifetime of the pool.
void iree_hal_amdgpu_device_signal_pool_initialize(
    iree_hal_amdgpu_device_signal_pool_t* IREE_AMDGPU_RESTRICT signal_pool,
    uint32_t signal_count, iree_hsa_signal_t* IREE_AMDGPU_RESTRICT signals);

// Acquires a binary signal with an initial value as specified.
// If the pool is exhausted the returned signal will have a 0 value handle and
// callers should check with `iree_hsa_signal_is_null(signal)`.
iree_hsa_signal_t iree_hal_amdgpu_device_signal_pool_acquire(
    iree_hal_amdgpu_device_signal_pool_t* IREE_AMDGPU_RESTRICT signal_pool,
    int64_t initial_value);

// Returns a signal to the pool.
// Only signals acquired from the pool may be released.
void iree_hal_amdgpu_device_signal_pool_release(
    iree_hal_amdgpu_device_signal_pool_t* IREE_AMDGPU_RESTRICT signal_pool,
    iree_hsa_signal_t signal);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_SIGNAL_POOL_H_
