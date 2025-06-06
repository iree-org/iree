// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_allocator_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): implement allocator and expose ringbuffers
// (iree_hal_amdgpu_allocator_allocate_ringbuffer, etc).

// Creates a buffer allocator used for persistent allocations and import/export.
iree_status_t iree_hal_amdgpu_allocator_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

#endif  // IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
