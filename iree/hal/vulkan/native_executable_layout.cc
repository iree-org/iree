// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/vulkan/native_executable_layout.h"

#include "iree/base/tracing.h"
#include "iree/hal/vulkan/native_descriptor_set_layout.h"
#include "iree/hal/vulkan/status_util.h"

using namespace iree::hal::vulkan;

typedef struct {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
  VkPipelineLayout handle;
  iree_host_size_t set_layout_count;
  iree_hal_descriptor_set_layout_t* set_layouts[];
} iree_hal_vulkan_native_executable_layout_t;

extern const iree_hal_executable_layout_vtable_t
    iree_hal_vulkan_native_executable_layout_vtable;

static iree_hal_vulkan_native_executable_layout_t*
iree_hal_vulkan_native_executable_layout_cast(
    iree_hal_executable_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_vulkan_native_executable_layout_vtable);
  return (iree_hal_vulkan_native_executable_layout_t*)base_value;
}

static iree_status_t iree_hal_vulkan_create_pipeline_layout(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_host_size_t push_constant_count, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    VkPipelineLayout* out_handle) {
  VkDescriptorSetLayout* set_layout_handles =
      (VkDescriptorSetLayout*)iree_alloca(set_layout_count *
                                          sizeof(VkDescriptorSetLayout));
  for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
    set_layout_handles[i] =
        iree_hal_vulkan_native_descriptor_set_layout_handle(set_layouts[i]);
  }

  VkPushConstantRange push_constant_ranges[1];
  push_constant_ranges[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_constant_ranges[0].offset = 0;
  push_constant_ranges[0].size =
      (uint32_t)(push_constant_count * sizeof(uint32_t));

  VkPipelineLayoutCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.setLayoutCount = (uint32_t)set_layout_count;
  create_info.pSetLayouts = set_layout_handles;
  create_info.pushConstantRangeCount = push_constant_count > 0 ? 1 : 0;
  create_info.pPushConstantRanges = push_constant_ranges;

  return VK_RESULT_TO_STATUS(logical_device->syms()->vkCreatePipelineLayout(
                                 *logical_device, &create_info,
                                 logical_device->allocator(), out_handle),
                             "vkCreatePipelineLayout");
}

static void iree_hal_vulkan_destroy_pipeline_layout(
    VkDeviceHandle* logical_device, VkPipelineLayout handle) {
  if (handle == VK_NULL_HANDLE) return;
  logical_device->syms()->vkDestroyPipelineLayout(*logical_device, handle,
                                                  logical_device->allocator());
}

iree_status_t iree_hal_vulkan_native_executable_layout_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_host_size_t push_constant_count, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_hal_executable_layout_t** out_executable_layout) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(!set_layout_count || set_layouts);
  IREE_ASSERT_ARGUMENT(out_executable_layout);
  *out_executable_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  VkPipelineLayout handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_create_pipeline_layout(
              logical_device, push_constant_count, set_layout_count,
              set_layouts, &handle));

  iree_hal_vulkan_native_executable_layout_t* executable_layout = NULL;
  iree_host_size_t total_size =
      sizeof(*executable_layout) +
      set_layout_count * sizeof(*executable_layout->set_layouts);
  iree_status_t status = iree_allocator_malloc(
      logical_device->host_allocator(), total_size, (void**)&executable_layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(
        &iree_hal_vulkan_native_executable_layout_vtable,
        &executable_layout->resource);
    executable_layout->logical_device = logical_device;
    executable_layout->handle = handle;
    executable_layout->set_layout_count = set_layout_count;
    for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
      executable_layout->set_layouts[i] = set_layouts[i];
      iree_hal_descriptor_set_layout_retain(set_layouts[i]);
    }
    *out_executable_layout = (iree_hal_executable_layout_t*)executable_layout;
  } else {
    iree_hal_vulkan_destroy_pipeline_layout(logical_device, handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_native_executable_layout_destroy(
    iree_hal_executable_layout_t* base_executable_layout) {
  iree_hal_vulkan_native_executable_layout_t* executable_layout =
      iree_hal_vulkan_native_executable_layout_cast(base_executable_layout);
  iree_allocator_t host_allocator =
      executable_layout->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_destroy_pipeline_layout(executable_layout->logical_device,
                                          executable_layout->handle);
  for (iree_host_size_t i = 0; i < executable_layout->set_layout_count; ++i) {
    iree_hal_descriptor_set_layout_release(executable_layout->set_layouts[i]);
  }
  iree_allocator_free(host_allocator, executable_layout);

  IREE_TRACE_ZONE_END(z0);
}

VkPipelineLayout iree_hal_vulkan_native_executable_layout_handle(
    iree_hal_executable_layout_t* base_executable_layout) {
  iree_hal_vulkan_native_executable_layout_t* executable_layout =
      iree_hal_vulkan_native_executable_layout_cast(base_executable_layout);
  return executable_layout->handle;
}

iree_host_size_t iree_hal_vulkan_native_executable_layout_set_count(
    iree_hal_executable_layout_t* base_executable_layout) {
  iree_hal_vulkan_native_executable_layout_t* executable_layout =
      iree_hal_vulkan_native_executable_layout_cast(base_executable_layout);
  return executable_layout->set_layout_count;
}

iree_hal_descriptor_set_layout_t* iree_hal_vulkan_native_executable_layout_set(
    iree_hal_executable_layout_t* base_executable_layout,
    iree_host_size_t set_index) {
  iree_hal_vulkan_native_executable_layout_t* executable_layout =
      iree_hal_vulkan_native_executable_layout_cast(base_executable_layout);
  if (IREE_UNLIKELY(set_index >= executable_layout->set_layout_count)) {
    return NULL;
  }
  return executable_layout->set_layouts[set_index];
}

const iree_hal_executable_layout_vtable_t
    iree_hal_vulkan_native_executable_layout_vtable = {
        /*.destroy=*/iree_hal_vulkan_native_executable_layout_destroy,
};
