// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_DESCRIPTOR_SET_LAYOUT_H_
#define IREE_HAL_DRIVERS_WEBGPU_DESCRIPTOR_SET_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/platform/webgpu.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TODO(benvanik): query from runtime? almost all devices support 16+ and
// that's what our compiler is assuming.
#define IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT 8

typedef uint32_t iree_hal_webgpu_binding_mask_t;
static_assert(sizeof(iree_hal_webgpu_binding_mask_t) * 8 >=
                  IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT,
              "mask must have capacity for one bit per binding in a group");

iree_status_t iree_hal_webgpu_descriptor_set_layout_create(
    WGPUDevice device, iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

WGPUBindGroupLayout iree_hal_webgpu_descriptor_set_layout_handle(
    iree_hal_descriptor_set_layout_t* layout);

iree_hal_webgpu_binding_mask_t
iree_hal_webgpu_descriptor_set_layout_binding_mask(
    iree_hal_descriptor_set_layout_t* layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_DESCRIPTOR_SET_LAYOUT_H_
