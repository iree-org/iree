// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/executable_layout.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/inline_array.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/webgpu/descriptor_set_layout.h"

typedef struct iree_hal_webgpu_executable_layout_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  WGPUPipelineLayout handle;
  iree_host_size_t push_constant_count;
  iree_hal_webgpu_set_binding_info_t set_binding_info;
  iree_host_size_t set_layout_count;
  iree_hal_descriptor_set_layout_t* set_layouts[];
} iree_hal_webgpu_executable_layout_t;

extern const iree_hal_executable_layout_vtable_t
    iree_hal_webgpu_executable_layout_vtable;

static iree_hal_webgpu_executable_layout_t*
iree_hal_webgpu_executable_layout_cast(
    iree_hal_executable_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_executable_layout_vtable);
  return (iree_hal_webgpu_executable_layout_t*)base_value;
}

iree_status_t iree_hal_webgpu_executable_layout_create(
    WGPUDevice device, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count, iree_allocator_t host_allocator,
    iree_hal_executable_layout_t** out_executable_layout) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(!set_layout_count || set_layouts);
  IREE_ASSERT_ARGUMENT(out_executable_layout);
  *out_executable_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_inline_array(WGPUBindGroupLayout, bind_group_layouts, set_layout_count,
                    host_allocator);
  for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
    *iree_inline_array_at(bind_group_layouts, i) =
        iree_hal_webgpu_descriptor_set_layout_handle(set_layouts[i]);
  }
  const WGPUPipelineLayoutDescriptor descriptor = {
      .nextInChain = NULL,
      .label = NULL,
      .bindGroupLayoutCount = (uint32_t)set_layout_count,
      .bindGroupLayouts = iree_inline_array_at(bind_group_layouts, 0),
  };
  WGPUPipelineLayout handle =
      wgpuDeviceCreatePipelineLayout(device, &descriptor);
  iree_inline_array_deinitialize(bind_group_layouts);

  if (!handle) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "wgpuDeviceCreatePipelineLayout failed");
  }

  iree_hal_webgpu_executable_layout_t* executable_layout = NULL;
  iree_host_size_t total_size =
      sizeof(*executable_layout) +
      set_layout_count * sizeof(*executable_layout->set_layouts);
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size,
                                               (void**)&executable_layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_webgpu_executable_layout_vtable,
                                 &executable_layout->resource);
    executable_layout->host_allocator = host_allocator;
    executable_layout->handle = handle;
    executable_layout->push_constant_count = push_constant_count;

    executable_layout->set_layout_count = set_layout_count;
    executable_layout->set_binding_info.set_count = set_layout_count;
    for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
      executable_layout->set_layouts[i] = set_layouts[i];
      iree_hal_descriptor_set_layout_retain(set_layouts[i]);
      executable_layout->set_binding_info.set_layouts[i] =
          iree_hal_webgpu_descriptor_set_layout_handle(set_layouts[i]);
      executable_layout->set_binding_info.set_masks[i] =
          iree_hal_webgpu_descriptor_set_layout_binding_mask(set_layouts[i]);
    }

    *out_executable_layout = (iree_hal_executable_layout_t*)executable_layout;
  } else {
    iree_wgpuPipelineLayoutDrop(handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_executable_layout_destroy(
    iree_hal_executable_layout_t* base_executable_layout) {
  iree_hal_webgpu_executable_layout_t* executable_layout =
      iree_hal_webgpu_executable_layout_cast(base_executable_layout);
  iree_allocator_t host_allocator = executable_layout->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_wgpuPipelineLayoutDrop(executable_layout->handle);
  for (iree_host_size_t i = 0; i < executable_layout->set_layout_count; ++i) {
    iree_hal_descriptor_set_layout_release(executable_layout->set_layouts[i]);
  }
  iree_allocator_free(host_allocator, executable_layout);

  IREE_TRACE_ZONE_END(z0);
}

WGPUPipelineLayout iree_hal_webgpu_executable_layout_handle(
    iree_hal_executable_layout_t* layout) {
  IREE_ASSERT_ARGUMENT(layout);
  return iree_hal_webgpu_executable_layout_cast(layout)->handle;
}

iree_host_size_t iree_hal_webgpu_executable_layout_push_constant_count(
    iree_hal_executable_layout_t* layout) {
  IREE_ASSERT_ARGUMENT(layout);
  return iree_hal_webgpu_executable_layout_cast(layout)->push_constant_count;
}

const iree_hal_webgpu_set_binding_info_t*
iree_hal_webgpu_executable_layout_set_binding_info(
    iree_hal_executable_layout_t* base_layout) {
  IREE_ASSERT_ARGUMENT(base_layout);
  iree_hal_webgpu_executable_layout_t* layout =
      iree_hal_webgpu_executable_layout_cast(base_layout);
  return &layout->set_binding_info;
}

const iree_hal_executable_layout_vtable_t
    iree_hal_webgpu_executable_layout_vtable = {
        .destroy = iree_hal_webgpu_executable_layout_destroy,
};
