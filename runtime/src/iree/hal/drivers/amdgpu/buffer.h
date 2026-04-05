// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_buffer_t
//===----------------------------------------------------------------------===//

// Wraps an HSA memory pool allocation in an iree_hal_buffer_t.
// The buffer owns the HSA allocation and frees it on destroy.
//
// |allocation_size| is the full size of the HSA allocation and may be larger
// than the logical |byte_length| exposed through the HAL buffer.
iree_status_t iree_hal_amdgpu_buffer_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, void* host_ptr,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns the HSA-allocated base pointer for the given |buffer|, or NULL if
// |buffer| is not an AMDGPU buffer. HSA uses unified virtual addressing so
// the returned pointer is valid for both host and GPU access.
//
// This is the entire allocated_buffer and must be offset by
// iree_hal_buffer_byte_offset and the binding offset when computing kernarg
// binding addresses. |buffer| must be the allocated buffer (not a subspan);
// callers should use iree_hal_buffer_allocated_buffer() to unwrap first.
void* iree_hal_amdgpu_buffer_device_pointer(iree_hal_buffer_t* buffer);

#endif  // IREE_HAL_DRIVERS_AMDGPU_BUFFER_H_
