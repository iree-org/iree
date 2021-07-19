// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/descriptor_set_layout.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/inline_array.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_webgpu_descriptor_set_layout_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  WGPUBindGroupLayout handle;
  iree_hal_webgpu_binding_mask_t binding_mask;
} iree_hal_webgpu_descriptor_set_layout_t;

extern const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_webgpu_descriptor_set_layout_vtable;

static iree_hal_webgpu_descriptor_set_layout_t*
iree_hal_webgpu_descriptor_set_layout_cast(
    iree_hal_descriptor_set_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_webgpu_descriptor_set_layout_vtable);
  return (iree_hal_webgpu_descriptor_set_layout_t*)base_value;
}

iree_status_t iree_hal_webgpu_descriptor_set_layout_create(
    WGPUDevice device, iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  *out_descriptor_set_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_inline_array(WGPUBindGroupLayoutEntry, entries, binding_count,
                    host_allocator);
  iree_hal_webgpu_binding_mask_t binding_mask = 0;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (bindings[i].binding >=
        IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT) {
      iree_inline_array_deinitialize(entries);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "bindings must be in the range of 0-%d; binding "
                              "%zu is has ordinal %d",
                              IREE_HAL_WEBGPU_MAX_DESCRIPTOR_SET_BINDING_COUNT,
                              i, bindings[i].binding);
    }
    binding_mask |= 1u << bindings[i].binding;
    WGPUBufferBindingType binding_type = WGPUBufferBindingType_Undefined;
    bool has_dynamic_offset = false;
    switch (bindings[i].type) {
      case IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        binding_type = WGPUBufferBindingType_Storage;
        break;
      case IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        binding_type = WGPUBufferBindingType_Storage;
        has_dynamic_offset = true;
        break;
      case IREE_HAL_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        binding_type = WGPUBufferBindingType_Uniform;
        break;
      case IREE_HAL_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        binding_type = WGPUBufferBindingType_Uniform;
        has_dynamic_offset = true;
        break;
    }
    *iree_inline_array_at(entries, i) = (WGPUBindGroupLayoutEntry){
        .nextInChain = NULL,
        .binding = bindings[i].binding,
        .visibility = WGPUShaderStage_Compute,
        .buffer =
            {
                .nextInChain = NULL,
                .type = binding_type,
                .hasDynamicOffset = has_dynamic_offset,
                .minBindingSize = 0,
            },
    };
  }
  const WGPUBindGroupLayoutDescriptor descriptor = {
      .nextInChain = NULL,
      .label = NULL,
      .entryCount = (uint32_t)binding_count,
      .entries = iree_inline_array_at(entries, 0),
  };
  WGPUBindGroupLayout handle =
      wgpuDeviceCreateBindGroupLayout(device, &descriptor);
  iree_inline_array_deinitialize(entries);
  if (!handle) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "wgpuDeviceCreateBindGroupLayout failed");
  }

  iree_hal_webgpu_descriptor_set_layout_t* descriptor_set_layout = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*descriptor_set_layout),
                            (void**)&descriptor_set_layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_webgpu_descriptor_set_layout_vtable,
                                 &descriptor_set_layout->resource);
    descriptor_set_layout->host_allocator = host_allocator;
    descriptor_set_layout->handle = handle;
    descriptor_set_layout->binding_mask = binding_mask;
    *out_descriptor_set_layout =
        (iree_hal_descriptor_set_layout_t*)descriptor_set_layout;
  } else {
    iree_wgpuBindGroupLayoutDrop(handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_descriptor_set_layout_destroy(
    iree_hal_descriptor_set_layout_t* base_descriptor_set_layout) {
  iree_hal_webgpu_descriptor_set_layout_t* descriptor_set_layout =
      iree_hal_webgpu_descriptor_set_layout_cast(base_descriptor_set_layout);
  iree_allocator_t host_allocator = descriptor_set_layout->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_wgpuBindGroupLayoutDrop(descriptor_set_layout->handle);
  iree_allocator_free(host_allocator, descriptor_set_layout);

  IREE_TRACE_ZONE_END(z0);
}

WGPUBindGroupLayout iree_hal_webgpu_descriptor_set_layout_handle(
    iree_hal_descriptor_set_layout_t* layout) {
  IREE_ASSERT_ARGUMENT(layout);
  return iree_hal_webgpu_descriptor_set_layout_cast(layout)->handle;
}

iree_hal_webgpu_binding_mask_t
iree_hal_webgpu_descriptor_set_layout_binding_mask(
    iree_hal_descriptor_set_layout_t* layout) {
  IREE_ASSERT_ARGUMENT(layout);
  return iree_hal_webgpu_descriptor_set_layout_cast(layout)->binding_mask;
}

const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_webgpu_descriptor_set_layout_vtable = {
        .destroy = iree_hal_webgpu_descriptor_set_layout_destroy,
};
