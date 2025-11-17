// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_WEBGPU_EXECUTABLE_H_

#include <stdint.h>

#include "experimental/webgpu/pipeline_layout.h"
#include "experimental/webgpu/platform/webgpu.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_webgpu_entry_point_t {
  WGPUComputePipeline pipeline;
  // TODO(benvanik): inline what's needed here (WGPUPipelineLayout, binding
  // info, etc) instead so that we avoid needing to query it per dispatch from
  // the layout. The extra ~32B per entry point feels like it may be worth it to
  // avoid a guaranteed cache miss.
  iree_hal_pipeline_layout_t* layout;
} iree_hal_webgpu_entry_point_t;

iree_status_t iree_hal_webgpu_executable_create(
    WGPUDevice device, const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

const iree_hal_webgpu_entry_point_t*
iree_hal_webgpu_executable_lookup_entry_point(iree_hal_executable_t* executable,
                                              uint32_t ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_EXECUTABLE_H_
