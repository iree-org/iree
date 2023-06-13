// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_PIPELINE_LAYOUT_H_
#define IREE_HAL_DRIVERS_WEBGPU_PIPELINE_LAYOUT_H_

#include "experimental/webgpu/platform/webgpu.h"
#include "experimental/webgpu/staging_buffer.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_COUNT 4
#define IREE_HAL_WEBGPU_MAX_PUSH_CONSTANT_COUNT 64
#define IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX 3

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): query from runtime? almost all devices support 16+ and
// that's what our compiler is assuming.
#define IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT 8

typedef uint32_t iree_hal_webgpu_binding_mask_t;
static_assert(sizeof(iree_hal_webgpu_binding_mask_t) * 8 >=
                  IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT,
              "mask must have capacity for one bit per binding in a group");

iree_status_t iree_hal_webgpu_descriptor_set_layout_create(
    WGPUDevice device, iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

WGPUBindGroupLayout iree_hal_webgpu_descriptor_set_layout_handle(
    iree_hal_descriptor_set_layout_t* layout);

iree_hal_webgpu_binding_mask_t
iree_hal_webgpu_descriptor_set_layout_binding_mask(
    iree_hal_descriptor_set_layout_t* layout);

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_pipeline_layout_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_set_binding_info_t {
  iree_host_size_t set_count;
  WGPUBindGroupLayout set_layouts[IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_COUNT];
  iree_hal_webgpu_binding_mask_t
      set_masks[IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_COUNT];
} iree_hal_webgpu_set_binding_info_t;

iree_status_t iree_hal_webgpu_pipeline_layout_create(
    WGPUDevice device, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count,
    iree_hal_webgpu_staging_buffer_t* staging_buffer,
    iree_allocator_t host_allocator,
    iree_hal_pipeline_layout_t** out_pipeline_layout);

WGPUPipelineLayout iree_hal_webgpu_pipeline_layout_handle(
    iree_hal_pipeline_layout_t* layout);

iree_host_size_t iree_hal_webgpu_pipeline_layout_push_constant_count(
    iree_hal_pipeline_layout_t* layout);

const iree_hal_webgpu_set_binding_info_t*
iree_hal_webgpu_pipeline_layout_set_binding_info(
    iree_hal_pipeline_layout_t* layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_PIPELINE_LAYOUT_H_
