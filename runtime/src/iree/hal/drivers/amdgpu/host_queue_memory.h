// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_MEMORY_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_MEMORY_H_

#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amdgpu_alloca_reservation_readiness_e {
  // The reservation can be materialized and submitted immediately.
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY = 0,
  // The reservation must wait for a pool death frontier before materialization.
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT = 1,
  // The pool is exhausted or over budget and needs a release notification
  // retry.
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION = 2,
} iree_hal_amdgpu_alloca_reservation_readiness_t;

typedef struct iree_hal_amdgpu_alloca_reservation_t {
  // Scheduler action required before the reservation can be submitted.
  iree_hal_amdgpu_alloca_reservation_readiness_t readiness;
  // Pool acquisition result that produced |reservation|.
  iree_hal_pool_acquire_result_t acquire_result;
  // Pool-owned byte range reserved for this alloca operation.
  iree_hal_pool_reservation_t reservation;
  // Borrowed metadata returned with |reservation|.
  iree_hal_pool_acquire_info_t acquire_info;
  // Queue wait resolution to use when publishing the alloca signal.
  iree_hal_amdgpu_wait_resolution_t wait_resolution;
} iree_hal_amdgpu_alloca_reservation_t;

// Resolves the allocation pool, validates/canonicalizes the request, and
// creates the transient wrapper returned from queue_alloca.
iree_status_t iree_hal_amdgpu_host_queue_prepare_alloca_wrapper(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* pool,
    iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_alloca_flags_t flags, iree_hal_pool_t** out_allocation_pool,
    iree_hal_buffer_t** out_buffer);

// Attempts to reserve bytes from |allocation_pool| and classifies the result
// as immediate, death-frontier-waitable, or notification-retry-required.
iree_status_t iree_hal_amdgpu_host_queue_acquire_alloca_reservation(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_alloca_reservation_t* out_reservation);

// Materializes a ready reservation, stages it on |buffer|, and submits the
// queue barrier that commits the transient buffer on completion.
iree_status_t iree_hal_amdgpu_host_queue_submit_alloca_reservation(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_alloca_reservation_t* alloca_reservation,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

// Submits the queue barrier that decommits a transient buffer on completion.
iree_status_t iree_hal_amdgpu_host_queue_submit_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_MEMORY_H_
