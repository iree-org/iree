// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_BUFFER_H_
#define IREE_HAL_DRIVERS_HSA_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hsa/hsa_headers.h"

typedef enum iree_hal_hsa_buffer_type_e {
  // Device local buffer; allocated with hsa_amd_memory_pool_allocate, freed
  // with hsa_amd_memory_pool_free.
  IREE_HAL_HSA_BUFFER_TYPE_DEVICE = 0,
  // Host local buffer; allocated with hsa_amd_memory_pool_allocate on
  // fine-grained pool, freed with hsa_amd_memory_pool_free.
  IREE_HAL_HSA_BUFFER_TYPE_HOST,
  // Host local buffer; registered with hsa_amd_memory_lock, freed with
  // hsa_amd_memory_unlock.
  IREE_HAL_HSA_BUFFER_TYPE_HOST_REGISTERED,
  // Externally registered buffer whose providence is unknown.
  // Must be freed by the user.
  IREE_HAL_HSA_BUFFER_TYPE_EXTERNAL,
} iree_hal_hsa_buffer_type_t;

// Wraps an HSA allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_hsa_buffer_wrap(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_hsa_buffer_type_t buffer_type, void* device_ptr, void* host_ptr,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns the underlying HSA buffer type.
iree_hal_hsa_buffer_type_t iree_hal_hsa_buffer_type(
    const iree_hal_buffer_t* buffer);

// Returns the HSA device pointer for the given |buffer|.
void* iree_hal_hsa_buffer_device_pointer(iree_hal_buffer_t* buffer);

// Returns the HSA host pointer for the given |buffer|, if available.
void* iree_hal_hsa_buffer_host_pointer(const iree_hal_buffer_t* buffer);

#endif  // IREE_HAL_DRIVERS_HSA_BUFFER_H_

