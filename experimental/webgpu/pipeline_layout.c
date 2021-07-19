// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/pipeline_layout.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/inline_array.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

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
    WGPUDevice device, iree_hal_descriptor_set_layout_flags_t flags,
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

    // TODO(benvanik): make all dynamic? this would let us reuse bind groups.
    WGPUBufferBindingType binding_type = WGPUBufferBindingType_Undefined;
    bool has_dynamic_offset = false;
    switch (bindings[i].type) {
      case IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        binding_type = WGPUBufferBindingType_Storage;
        break;
      case IREE_HAL_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        binding_type = WGPUBufferBindingType_Uniform;
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

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_pipeline_layout_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_pipeline_layout_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  WGPUPipelineLayout handle;
  iree_host_size_t push_constant_count;
  iree_hal_webgpu_set_binding_info_t set_binding_info;
  iree_host_size_t set_layout_count;
  iree_hal_descriptor_set_layout_t* set_layouts[];
} iree_hal_webgpu_pipeline_layout_t;

extern const iree_hal_pipeline_layout_vtable_t
    iree_hal_webgpu_pipeline_layout_vtable;

static iree_hal_webgpu_pipeline_layout_t* iree_hal_webgpu_pipeline_layout_cast(
    iree_hal_pipeline_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_pipeline_layout_vtable);
  return (iree_hal_webgpu_pipeline_layout_t*)base_value;
}

iree_status_t iree_hal_webgpu_pipeline_layout_create(
    WGPUDevice device, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count,
    iree_hal_webgpu_staging_buffer_t* staging_buffer,
    iree_allocator_t host_allocator,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(!set_layout_count || set_layouts);
  IREE_ASSERT_ARGUMENT(out_pipeline_layout);
  *out_pipeline_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (set_layout_count > IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "set_layout_count must be <= %d, as bind group index %d is reserved",
        IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX,
        IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX);
  }

  // Pad to IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX for push constant emulation.
  iree_host_size_t bind_group_layouts_count =
      push_constant_count > 0 ? IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX + 1
                              : set_layout_count;

  // Populate a WGPUBindGroupLayout array with the provided set layouts, then
  // set the staging buffer's bind group layout at the right index, padding
  // with an empty bind layout as needed.
  iree_inline_array(WGPUBindGroupLayout, bind_group_layouts,
                    bind_group_layouts_count, host_allocator);
  for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
    *iree_inline_array_at(bind_group_layouts, i) =
        iree_hal_webgpu_descriptor_set_layout_handle(set_layouts[i]);
  }
  for (iree_host_size_t i = set_layout_count; i < bind_group_layouts_count - 1;
       ++i) {
    *iree_inline_array_at(bind_group_layouts, i) =
        staging_buffer->empty_bind_group_layout;
  }
  if (push_constant_count > 0) {
    *iree_inline_array_at(bind_group_layouts,
                          IREE_HAL_WEBGPU_PARAMS_BIND_GROUP_INDEX) =
        staging_buffer->bind_group_layout;
  }
  const WGPUPipelineLayoutDescriptor descriptor = {
      .nextInChain = NULL,
      .label = NULL,
      .bindGroupLayoutCount = (uint32_t)bind_group_layouts_count,
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

  iree_hal_webgpu_pipeline_layout_t* pipeline_layout = NULL;
  iree_host_size_t total_size =
      sizeof(*pipeline_layout) +
      set_layout_count * sizeof(*pipeline_layout->set_layouts);
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size,
                                               (void**)&pipeline_layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_webgpu_pipeline_layout_vtable,
                                 &pipeline_layout->resource);
    pipeline_layout->host_allocator = host_allocator;
    pipeline_layout->handle = handle;
    pipeline_layout->push_constant_count = push_constant_count;

    pipeline_layout->set_layout_count = set_layout_count;
    pipeline_layout->set_binding_info.set_count = set_layout_count;
    for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
      pipeline_layout->set_layouts[i] = set_layouts[i];
      iree_hal_descriptor_set_layout_retain(set_layouts[i]);
      pipeline_layout->set_binding_info.set_layouts[i] =
          iree_hal_webgpu_descriptor_set_layout_handle(set_layouts[i]);
      pipeline_layout->set_binding_info.set_masks[i] =
          iree_hal_webgpu_descriptor_set_layout_binding_mask(set_layouts[i]);
    }
    // Note: not tracking the empty/padding layout or the staging buffer layout.

    *out_pipeline_layout = (iree_hal_pipeline_layout_t*)pipeline_layout;
  } else {
    iree_wgpuPipelineLayoutDrop(handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_pipeline_layout_destroy(
    iree_hal_pipeline_layout_t* base_pipeline_layout) {
  iree_hal_webgpu_pipeline_layout_t* pipeline_layout =
      iree_hal_webgpu_pipeline_layout_cast(base_pipeline_layout);
  iree_allocator_t host_allocator = pipeline_layout->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_wgpuPipelineLayoutDrop(pipeline_layout->handle);
  for (iree_host_size_t i = 0; i < pipeline_layout->set_layout_count; ++i) {
    iree_hal_descriptor_set_layout_release(pipeline_layout->set_layouts[i]);
  }
  iree_allocator_free(host_allocator, pipeline_layout);

  IREE_TRACE_ZONE_END(z0);
}

WGPUPipelineLayout iree_hal_webgpu_pipeline_layout_handle(
    iree_hal_pipeline_layout_t* layout) {
  IREE_ASSERT_ARGUMENT(layout);
  return iree_hal_webgpu_pipeline_layout_cast(layout)->handle;
}

iree_host_size_t iree_hal_webgpu_pipeline_layout_push_constant_count(
    iree_hal_pipeline_layout_t* layout) {
  IREE_ASSERT_ARGUMENT(layout);
  return iree_hal_webgpu_pipeline_layout_cast(layout)->push_constant_count;
}

const iree_hal_webgpu_set_binding_info_t*
iree_hal_webgpu_pipeline_layout_set_binding_info(
    iree_hal_pipeline_layout_t* base_layout) {
  IREE_ASSERT_ARGUMENT(base_layout);
  iree_hal_webgpu_pipeline_layout_t* layout =
      iree_hal_webgpu_pipeline_layout_cast(base_layout);
  return &layout->set_binding_info;
}

const iree_hal_pipeline_layout_vtable_t iree_hal_webgpu_pipeline_layout_vtable =
    {
        .destroy = iree_hal_webgpu_pipeline_layout_destroy,
};
