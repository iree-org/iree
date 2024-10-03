// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_ALLOCATOR_H_

#include "iree/hal/drivers/amdgpu/device/buffer.h"
#include "iree/hal/drivers/amdgpu/device/host.h"
#include "iree/hal/drivers/amdgpu/device/support/common.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_device_allocator_pool_t {
  // Name used when tracing events related to the pool.
  // This must be globally unique and is decided by the host when the device
  // allocator is initialized.
  iree_hal_amdgpu_trace_string_literal_ptr_t trace_name;
} iree_hal_amdgpu_device_allocator_pool_t;

// DO NOT SUBMIT document
typedef struct iree_hal_amdgpu_device_allocator_t {
  // Host that handles pool management operations (grow/trim/etc).
  iree_hal_amdgpu_device_host_t* host;

  // State used for transfer operations.
  iree_hal_amdgpu_device_buffer_transfer_state_t transfer_state;

  // DO NOT SUBMIT local pools
  // iree_hal_amdgpu_device_allocator_pool_t pools[];
} iree_hal_amdgpu_device_allocator_t;

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Returns true if the operation completed synchronously. If asynchronous the
// provided |scheduler_queue_entry| will be retired after the asynchronous
// operation completes.
bool iree_hal_amdgpu_device_allocator_alloca(
    iree_hal_amdgpu_device_allocator_t* IREE_AMDGPU_RESTRICT allocator,
    uint64_t scheduler, uint64_t scheduler_queue_entry,
    iree_hal_amdgpu_device_allocation_pool_id_t pool_id, uint32_t min_alignment,
    uint64_t allocation_size,
    iree_hal_amdgpu_device_allocation_handle_t* IREE_AMDGPU_RESTRICT
        out_handle);

// Returns true if the operation completed synchronously. If asynchronous the
// provided |scheduler_queue_entry| will be retired after the asynchronous
// operation completes.
bool iree_hal_amdgpu_device_allocator_dealloca(
    iree_hal_amdgpu_device_allocator_t* IREE_AMDGPU_RESTRICT allocator,
    uint64_t scheduler, uint64_t scheduler_queue_entry,
    iree_hal_amdgpu_device_allocation_handle_t* IREE_AMDGPU_RESTRICT handle);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_ALLOCATOR_H_
