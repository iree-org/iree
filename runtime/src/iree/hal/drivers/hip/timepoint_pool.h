// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_TIMEPOINT_POOL_H_
#define IREE_HAL_DRIVERS_HIP_TIMEPOINT_POOL_H_

#include "iree/base/api.h"
#include "iree/base/internal/event_pool.h"
#include "iree/hal/drivers/hip/event_pool.h"
#include "iree/hal/utils/semaphore_base.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_hip_timepoint_t
//===----------------------------------------------------------------------===//

// Forward declaration of the timepoint pool.
typedef struct iree_hal_hip_timepoint_pool_t iree_hal_hip_timepoint_pool_t;

// An enum to identify the timepoint kind in iree_hal_hip_timepoint_t objects.
typedef enum iree_hal_hip_timepoint_kind_e {
  // None; for uninitialized timepoint objects.
  IREE_HAL_HIP_TIMEPOINT_KIND_NONE = 0,
  // A timepoint waited by the host.
  IREE_HAL_HIP_TIMEPOINT_KIND_HOST_WAIT,
  // A timepoint signaled by the device.
  IREE_HAL_HIP_TIMEPOINT_KIND_DEVICE_SIGNAL,
  // A timepoint waited by the device.
  IREE_HAL_HIP_TIMEPOINT_KIND_DEVICE_WAIT,
} iree_hal_hip_timepoint_kind_t;

// An object that wraps a host iree_event_t or device iree_hal_hip_event_t to
// represent wait/signal of a timepoint on a timeline.
//
// iree_hal_hip_timepoint_t objects cannot be directly created; it should be
// acquired from the timeline pool and released back to the pool once done.
//
// Thread-compatible; a timepoint is typically only accessed by one thread.
typedef struct iree_hal_hip_timepoint_t {
  // Base timepoint structure providing intrusive linked list pointers and
  // timepoint callback mechanisms.
  iree_hal_semaphore_timepoint_t base;

  // The allocator used to create the timepoint.
  iree_allocator_t host_allocator;

  // The timepoint pool that owns this timepoint.
  iree_hal_hip_timepoint_pool_t* pool;

  iree_hal_hip_timepoint_kind_t kind;
  union {
    iree_event_t host_wait;
    iree_hal_hip_event_t* device_signal;
    // The device event to wait. NULL means no device event available to wait
    // for this timepoint at the moment.
    iree_hal_hip_event_t* device_wait;
  } timepoint;
} iree_hal_hip_timepoint_t;

//===----------------------------------------------------------------------===//
// iree_hal_hip_timepoint_pool_t
//===----------------------------------------------------------------------===//

// A simple pool of iree_hal_hip_timepoint_t objects to recycle.
//
// Thread-safe; multiple threads may acquire and release timepoints from the
// pool.
typedef struct iree_hal_hip_timepoint_pool_t iree_hal_hip_timepoint_pool_t;

// Allocates a new timepoint pool with up to |available_capacity| timepoints.
//
// Extra timepoint requests beyond the capability are directly created and
// destroyed without pooling.
iree_status_t iree_hal_hip_timepoint_pool_allocate(
    iree_event_pool_t* host_event_pool,
    iree_hal_hip_event_pool_t* device_event_pool,
    iree_host_size_t available_capacity, iree_allocator_t host_allocator,
    iree_hal_hip_timepoint_pool_t** out_timepoint_pool);

// Deallocates a timepoint pool and destroys all timepoints.
//
// All timepoints that were acquired from the pool must have already been
// released back to it prior to deallocation.
void iree_hal_hip_timepoint_pool_free(
    iree_hal_hip_timepoint_pool_t* timepoint_pool);

// Acquires one or more timepoints from the timepoint pool.
//
// |out_timepoints| are owned by the caller and must be kept live until the
// timepoints have been reached, or cancelled by the caller.
iree_status_t iree_hal_hip_timepoint_pool_acquire_host_wait(
    iree_hal_hip_timepoint_pool_t* timepoint_pool,
    iree_host_size_t timepoint_count,
    iree_hal_hip_timepoint_t** out_timepoints);
iree_status_t iree_hal_hip_timepoint_pool_acquire_device_signal(
    iree_hal_hip_timepoint_pool_t* timepoint_pool,
    iree_host_size_t timepoint_count,
    iree_hal_hip_timepoint_t** out_timepoints);
iree_status_t iree_hal_hip_timepoint_pool_acquire_device_wait(
    iree_hal_hip_timepoint_pool_t* timepoint_pool,
    iree_host_size_t timepoint_count,
    iree_hal_hip_timepoint_t** out_timepoints);

// Releases one or more timepoints back to the timepoint pool.
void iree_hal_hip_timepoint_pool_release(
    iree_hal_hip_timepoint_pool_t* timepoint_pool,
    iree_host_size_t timepoint_count, iree_hal_hip_timepoint_t** timepoints);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_TIMEPOINT_POOL_H_
