// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/handle_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_executable_t
//===----------------------------------------------------------------------===//

// Creates a WebGPU executable from a compiled binary blob.
//
// The executable binary contains WGSL shader sources and metadata for each
// entry point. At creation time, compute pipelines are created for each export
// via the bridge layer. The executable retains the pipeline handles and
// releases them on destruction.
//
// |device_handle| is the bridge handle for the GPUDevice used to create compute
// pipelines and their associated layouts.
iree_status_t iree_hal_webgpu_executable_create(
    iree_hal_webgpu_handle_t device_handle,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the bridge handle for the compute pipeline at |export_ordinal|.
// The caller does not take ownership — the pipeline handle remains valid for
// the lifetime of the executable.
iree_hal_webgpu_handle_t iree_hal_webgpu_executable_pipeline_handle(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal);

// Returns the bridge handle for the pipeline layout at |export_ordinal|.
// Used by the command buffer to create bind groups with the matching layout.
iree_hal_webgpu_handle_t iree_hal_webgpu_executable_pipeline_layout_handle(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal);

// Returns the bridge handle for the bind group layout at |export_ordinal|.
// Used by the command buffer to create bind groups for dispatch.
iree_hal_webgpu_handle_t iree_hal_webgpu_executable_bind_group_layout_handle(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_EXECUTABLE_H_
