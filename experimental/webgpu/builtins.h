// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_BUILTINS_H_
#define IREE_HAL_DRIVERS_WEBGPU_BUILTINS_H_

#include "experimental/webgpu/platform/webgpu.h"
#include "experimental/webgpu/staging_buffer.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_webgpu_builtin_fill_buffer_t {
  // groupIndex[1]
  //  binding[0]: target
  WGPUBindGroupLayout buffer_group_layout;
  WGPUComputePipeline pipeline;
} iree_hal_webgpu_builtin_fill_buffer_t;

typedef struct iree_hal_webgpu_builtins_t {
  iree_hal_webgpu_builtin_fill_buffer_t fill_buffer;
} iree_hal_webgpu_builtins_t;

iree_status_t iree_hal_webgpu_builtins_initialize(
    WGPUDevice device, iree_hal_webgpu_staging_buffer_t* staging_buffer,
    iree_hal_webgpu_builtins_t* out_builtins);

void iree_hal_webgpu_builtins_deinitialize(
    iree_hal_webgpu_builtins_t* builtins);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_BUILTINS_H_
