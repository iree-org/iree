// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_NOP_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_HIP_NOP_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_headers.h"
#include "iree/hal/drivers/hip/per_device_information.h"

// Creates a no-op executable cache that does not cache at all.
// This is useful to isolate pipeline caching behavior and verify compilation
// behavior.
iree_status_t iree_hal_hip_nop_executable_cache_create(
    iree_string_view_t identifier,
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_hal_hip_device_topology_t topology, iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#endif  // IREE_HAL_DRIVERS_HIP_NOP_EXECUTABLE_CACHE_H_
