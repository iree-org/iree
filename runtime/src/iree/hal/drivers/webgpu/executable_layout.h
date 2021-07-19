// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_DRIVERS_WEBGPU_EXECUTABLE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/descriptor_set_layout.h"
#include "iree/hal/drivers/webgpu/platform/webgpu.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_COUNT 4
#define IREE_HAL_WEBGPU_MAX_PUSH_CONSTANT_COUNT 64
#define IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX 3

typedef struct iree_hal_webgpu_set_binding_info_t {
  iree_host_size_t set_count;
  WGPUBindGroupLayout set_layouts[IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_COUNT];
  iree_hal_webgpu_binding_mask_t
      set_masks[IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_COUNT];
} iree_hal_webgpu_set_binding_info_t;

iree_status_t iree_hal_webgpu_executable_layout_create(
    WGPUDevice device, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count, iree_allocator_t host_allocator,
    iree_hal_executable_layout_t** out_executable_layout);

WGPUPipelineLayout iree_hal_webgpu_executable_layout_handle(
    iree_hal_executable_layout_t* layout);

iree_host_size_t iree_hal_webgpu_executable_layout_push_constant_count(
    iree_hal_executable_layout_t* layout);

const iree_hal_webgpu_set_binding_info_t*
iree_hal_webgpu_executable_layout_set_binding_info(
    iree_hal_executable_layout_t* layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_EXECUTABLE_LAYOUT_H_
