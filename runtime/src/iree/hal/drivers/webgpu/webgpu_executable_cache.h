// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/handle_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_executable_cache_t
//===----------------------------------------------------------------------===//

// Creates a no-op executable cache that creates pipelines on demand.
//
// WebGPU's createComputePipeline is synchronous from the bridge perspective
// (the browser may defer actual shader compilation internally). Caching could
// be added via createComputePipelineAsync and the proactor, but the synchronous
// path is sufficient for bring-up.
//
// |device_handle| is the bridge handle for the GPUDevice, retained for passing
// to executable creation.
iree_status_t iree_hal_webgpu_executable_cache_create(
    iree_hal_webgpu_handle_t device_handle, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_EXECUTABLE_CACHE_H_
