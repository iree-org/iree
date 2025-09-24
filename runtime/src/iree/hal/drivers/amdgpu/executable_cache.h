// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_executable_cache_t
//===----------------------------------------------------------------------===//

// Creates a no-op executable cache that does not cache at all.
// Modern HSA usage on AMD hardware is all via fully baked out ELFs and does not
// perform any on-device JITing/optimization. If in the future there's something
// more PTX/HSAIL-like we'd want to manage that here.
//
// |libhsa| and |topology| are captured by-reference and must remain valid for
// the lifetime of the cache.
iree_status_t iree_hal_amdgpu_executable_cache_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#endif  // IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_CACHE_H_
