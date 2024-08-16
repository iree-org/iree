// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/pipeline_layout.h"

#include <cstddef>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/dynamic_symbol_tables.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/extensibility_util.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

using namespace iree::hal::vulkan;

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_vulkan_descriptor_set_layout_create(
    VkDeviceHandle* logical_device, VkDescriptorSetLayoutCreateFlags flags,
    iree_host_size_t binding_count,
    const VkDescriptorSetLayoutBinding* bindings,
    iree_hal_vulkan_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  *out_descriptor_set_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_descriptor_set_layout_t* descriptor_set_layout = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(logical_device->host_allocator(),
                                sizeof(*descriptor_set_layout),
                                (void**)&descriptor_set_layout));
  iree_atomic_ref_count_init(&descriptor_set_layout->ref_count);
  descriptor_set_layout->logical_device = logical_device;
  descriptor_set_layout->handle = VK_NULL_HANDLE;

  VkDescriptorSetLayoutCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  create_info.pNext = NULL;

  create_info.flags = flags;
  if (binding_count > 0) {
    if (logical_device->enabled_extensions().push_descriptors) {
      // Note that we can *only* use push descriptor sets if we set this create
      // flag. If push descriptors aren't supported we emulate them with normal
      // descriptors so it's fine to have kPushOnly without support.
      // Also we only enable this when there are at least one binding in it.
      // (We can have dummy descriptor sets without any bindings for builtin
      // executables.)
      create_info.flags |=
          VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }
  }

  create_info.bindingCount = (uint32_t)binding_count;
  create_info.pBindings = bindings;

  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkCreateDescriptorSetLayout(
          *logical_device, &create_info, logical_device->allocator(),
          &descriptor_set_layout->handle),
      "vkCreateDescriptorSetLayout");

  if (iree_status_is_ok(status)) {
    *out_descriptor_set_layout = descriptor_set_layout;
  } else {
    iree_hal_vulkan_descriptor_set_layout_release(descriptor_set_layout);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_descriptor_set_layout_destroy(
    iree_hal_vulkan_descriptor_set_layout_t* descriptor_set_layout) {
  VkDeviceHandle* logical_device = descriptor_set_layout->logical_device;
  iree_allocator_t host_allocator = logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  if (descriptor_set_layout->handle != VK_NULL_HANDLE) {
    logical_device->syms()->vkDestroyDescriptorSetLayout(
        *logical_device, descriptor_set_layout->handle,
        logical_device->allocator());
  }

  iree_allocator_free(host_allocator, descriptor_set_layout);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_vulkan_descriptor_set_layout_retain(
    iree_hal_vulkan_descriptor_set_layout_t* descriptor_set_layout) {
  if (descriptor_set_layout) {
    iree_atomic_ref_count_inc(&descriptor_set_layout->ref_count);
  }
}

void iree_hal_vulkan_descriptor_set_layout_release(
    iree_hal_vulkan_descriptor_set_layout_t* descriptor_set_layout) {
  if (descriptor_set_layout &&
      iree_atomic_ref_count_dec(&descriptor_set_layout->ref_count) == 1) {
    iree_hal_vulkan_descriptor_set_layout_destroy(descriptor_set_layout);
  }
}

VkDescriptorSetLayout iree_hal_vulkan_descriptor_set_layout_handle(
    iree_hal_vulkan_descriptor_set_layout_t* descriptor_set_layout) {
  return descriptor_set_layout->handle;
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_pipeline_layout_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_vulkan_pipeline_layout_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_host_size_t push_constant_range_count,
    const VkPushConstantRange* push_constant_ranges,
    iree_host_size_t set_layout_count,
    iree_hal_vulkan_descriptor_set_layout_t* const* set_layouts,
    iree_hal_vulkan_pipeline_layout_t** out_pipeline_layout) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(!set_layout_count || set_layouts);
  IREE_ASSERT_ARGUMENT(out_pipeline_layout);
  *out_pipeline_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_pipeline_layout_t* pipeline_layout = NULL;
  const iree_host_size_t total_size =
      sizeof(*pipeline_layout) +
      set_layout_count * sizeof(*pipeline_layout->set_layouts);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(logical_device->host_allocator(), total_size,
                                (void**)&pipeline_layout));
  iree_atomic_ref_count_init(&pipeline_layout->ref_count);
  pipeline_layout->logical_device = logical_device;
  pipeline_layout->handle = VK_NULL_HANDLE;
  pipeline_layout->set_layout_count = set_layout_count;
  for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
    pipeline_layout->set_layouts[i] = set_layouts[i];
    iree_hal_vulkan_descriptor_set_layout_retain(set_layouts[i]);
  }

  VkDescriptorSetLayout* set_layout_handles =
      (VkDescriptorSetLayout*)iree_alloca(set_layout_count *
                                          sizeof(VkDescriptorSetLayout));
  for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
    set_layout_handles[i] = set_layouts[i]->handle;
  }

  VkPipelineLayoutCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  create_info.setLayoutCount = (uint32_t)set_layout_count;
  create_info.pSetLayouts = set_layout_handles;
  create_info.pushConstantRangeCount = (uint32_t)push_constant_range_count;
  create_info.pPushConstantRanges = push_constant_ranges;

  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkCreatePipelineLayout(
          *logical_device, &create_info, logical_device->allocator(),
          &pipeline_layout->handle),
      "vkCreatePipelineLayout");

  if (iree_status_is_ok(status)) {
    *out_pipeline_layout = pipeline_layout;
  } else {
    iree_hal_vulkan_pipeline_layout_release(pipeline_layout);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_pipeline_layout_destroy(
    iree_hal_vulkan_pipeline_layout_t* pipeline_layout) {
  VkDeviceHandle* logical_device = pipeline_layout->logical_device;
  iree_allocator_t host_allocator = logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  if (pipeline_layout->handle != VK_NULL_HANDLE) {
    logical_device->syms()->vkDestroyPipelineLayout(
        *logical_device, pipeline_layout->handle, logical_device->allocator());
  }

  for (iree_host_size_t i = 0; i < pipeline_layout->set_layout_count; ++i) {
    iree_hal_vulkan_descriptor_set_layout_release(
        pipeline_layout->set_layouts[i]);
  }

  iree_allocator_free(host_allocator, pipeline_layout);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_vulkan_pipeline_layout_retain(
    iree_hal_vulkan_pipeline_layout_t* pipeline_layout) {
  if (pipeline_layout) {
    iree_atomic_ref_count_inc(&pipeline_layout->ref_count);
  }
}

void iree_hal_vulkan_pipeline_layout_release(
    iree_hal_vulkan_pipeline_layout_t* pipeline_layout) {
  if (pipeline_layout &&
      iree_atomic_ref_count_dec(&pipeline_layout->ref_count) == 1) {
    iree_hal_vulkan_pipeline_layout_destroy(pipeline_layout);
  }
}

VkPipelineLayout iree_hal_vulkan_pipeline_layout_handle(
    iree_hal_vulkan_pipeline_layout_t* pipeline_layout) {
  return pipeline_layout->handle;
}

iree_host_size_t iree_hal_vulkan_pipeline_layout_set_count(
    iree_hal_vulkan_pipeline_layout_t* pipeline_layout) {
  return pipeline_layout->set_layout_count;
}

iree_hal_vulkan_descriptor_set_layout_t* iree_hal_vulkan_pipeline_layout_set(
    iree_hal_vulkan_pipeline_layout_t* pipeline_layout,
    iree_host_size_t set_index) {
  if (IREE_UNLIKELY(set_index >= pipeline_layout->set_layout_count)) {
    return NULL;
  }
  return pipeline_layout->set_layouts[set_index];
}
