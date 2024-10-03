// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/allocator.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_allocator_t
//===----------------------------------------------------------------------===//

bool iree_hal_amdgpu_device_allocator_alloca(
    iree_hal_amdgpu_device_allocator_t* IREE_AMDGPU_RESTRICT allocator,
    uint64_t scheduler_queue_entry, uint32_t pool_id, uint32_t min_alignment,
    uint64_t allocation_size,
    iree_hal_amdgpu_device_allocation_handle_t* IREE_AMDGPU_RESTRICT
        out_handle) {
  //  // if post to host for POOL_GROW
  // need to wait until completion then retire and reschedule
  // host method for doing all that?
  // iree_hal_amdgpu_device_queue_scheduler_retire_from_host
  // if (!async) { retire inline }
  // IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_GROW no barrier
  // IREE_HAL_AMDGPU_DEVICE_HOST_CALL_RETIRE_ENTRY w/ barrier

  // DO NOT SUBMIT
  // make pool part of the request? don't choose pool on device
  // host has to enumerate and select
  // DISADVANTAGE: device-side enqueue needs to have the same logic anyway
  iree_hal_amdgpu_device_allocator_pool_t* pool = NULL;
  uint64_t allocation_offset = 0;

  iree_hal_amdgpu_device_host_post(
      allocator->host, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_GROW,
      /*return_address=*/(uint64_t)out_handle,
      /*arg0=*/0,  // POOL DO NOT SUBMIT
      /*arg1=*/0,  // BLOCK DO NOT SUBMIT
      /*arg2=*/(allocation_size << 8) | (uint64_t)min_alignment,
      /*arg3=*/allocation_offset,
      /*completion_signal=*/iree_hsa_signal_null());

  iree_hal_amdgpu_device_host_post(
      allocator->host, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_RETIRE_ENTRY,
      /*return_address=*/0,
      /*arg0=*/scheduler_queue_entry,
      /*arg1=*/0,
      /*arg2=*/0,
      /*arg3=*/0,
      /*completion_signal=*/iree_hsa_signal_null());

  return false;
}

bool iree_hal_amdgpu_device_allocator_dealloca(
    iree_hal_amdgpu_device_allocator_t* IREE_AMDGPU_RESTRICT allocator,
    uint64_t scheduler_queue_entry, uint32_t pool_id,
    iree_hal_amdgpu_device_allocation_handle_t* IREE_AMDGPU_RESTRICT handle) {
  // IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_TRIM no barrier
  // IREE_HAL_AMDGPU_DEVICE_HOST_CALL_RETIRE_ENTRY w/ barrier
  iree_hal_amdgpu_device_host_post(
      allocator->host, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_TRIM,
      /*return_address=*/0,
      /*arg0=*/0,  // POOL DO NOT SUBMIT
      /*arg1=*/0,  // block? DO NOT SUBMIT
      /*arg2=*/0,
      /*arg3=*/0,
      /*completion_signal=*/iree_hsa_signal_null());

  iree_hal_amdgpu_device_host_post(
      allocator->host, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_RETIRE_ENTRY,
      /*return_address=*/0,
      /*arg0=*/scheduler_queue_entry,
      /*arg1=*/0,
      /*arg2=*/0,
      /*arg3=*/0,
      /*completion_signal=*/iree_hsa_signal_null());

  return false;
}
