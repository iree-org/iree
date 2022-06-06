// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/native_descriptor_set_layout.h"

#include <cstddef>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/extensibility_util.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

using namespace iree::hal::vulkan;

typedef struct iree_hal_vulkan_native_descriptor_set_layout_t {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
  VkDescriptorSetLayout handle;
} iree_hal_vulkan_native_descriptor_set_layout_t;

namespace {
extern const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_vulkan_native_descriptor_set_layout_vtable;
}  // namespace

static iree_hal_vulkan_native_descriptor_set_layout_t*
iree_hal_vulkan_native_descriptor_set_layout_cast(
    iree_hal_descriptor_set_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_vulkan_native_descriptor_set_layout_vtable);
  return (iree_hal_vulkan_native_descriptor_set_layout_t*)base_value;
}

static iree_status_t iree_hal_vulkan_create_descriptor_set_layout(
    VkDeviceHandle* logical_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    VkDescriptorSetLayout* out_handle) {
  VkDescriptorSetLayoutCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  create_info.pNext = NULL;
  create_info.flags = 0;
  if (usage_type == IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_PUSH_ONLY &&
      logical_device->enabled_extensions().push_descriptors) {
    // Note that we can *only* use push descriptor sets if we set this create
    // flag. If push descriptors aren't supported we emulate them with normal
    // descriptors so it's fine to have kPushOnly without support.
    create_info.flags |=
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  }

  VkDescriptorSetLayoutBinding* native_bindings = NULL;
  if (binding_count > 0) {
    // TODO(benvanik): avoid this allocation if possible (inline_array).
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        logical_device->host_allocator(),
        binding_count * sizeof(VkDescriptorSetLayoutBinding),
        (void**)&native_bindings));
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      VkDescriptorSetLayoutBinding* native_binding = &native_bindings[i];
      native_binding->binding = bindings[i].binding;
      native_binding->descriptorType =
          static_cast<VkDescriptorType>(bindings[i].type);
      native_binding->descriptorCount = 1;
      native_binding->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      native_binding->pImmutableSamplers = NULL;
    }
  }
  create_info.bindingCount = (uint32_t)binding_count;
  create_info.pBindings = native_bindings;

  iree_status_t status =
      VK_RESULT_TO_STATUS(logical_device->syms()->vkCreateDescriptorSetLayout(
                              *logical_device, &create_info,
                              logical_device->allocator(), out_handle),
                          "vkCreateDescriptorSetLayout");

  iree_allocator_free(logical_device->host_allocator(), native_bindings);
  return status;
}

static void iree_hal_vulkan_destroy_descriptor_set_layout(
    VkDeviceHandle* logical_device, VkDescriptorSetLayout handle) {
  if (handle == VK_NULL_HANDLE) return;
  logical_device->syms()->vkDestroyDescriptorSetLayout(
      *logical_device, handle, logical_device->allocator());
}

iree_status_t iree_hal_vulkan_native_descriptor_set_layout_create(
    VkDeviceHandle* logical_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  *out_descriptor_set_layout = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  VkDescriptorSetLayout handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_create_descriptor_set_layout(
              logical_device, usage_type, binding_count, bindings, &handle));

  iree_hal_vulkan_native_descriptor_set_layout_t* descriptor_set_layout = NULL;
  iree_status_t status = iree_allocator_malloc(logical_device->host_allocator(),
                                               sizeof(*descriptor_set_layout),
                                               (void**)&descriptor_set_layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(
        &iree_hal_vulkan_native_descriptor_set_layout_vtable,
        &descriptor_set_layout->resource);
    descriptor_set_layout->logical_device = logical_device;
    descriptor_set_layout->handle = handle;
    *out_descriptor_set_layout =
        (iree_hal_descriptor_set_layout_t*)descriptor_set_layout;
  } else {
    iree_hal_vulkan_destroy_descriptor_set_layout(logical_device, handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_native_descriptor_set_layout_destroy(
    iree_hal_descriptor_set_layout_t* base_descriptor_set_layout) {
  iree_hal_vulkan_native_descriptor_set_layout_t* descriptor_set_layout =
      iree_hal_vulkan_native_descriptor_set_layout_cast(
          base_descriptor_set_layout);
  iree_allocator_t host_allocator =
      descriptor_set_layout->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_destroy_descriptor_set_layout(
      descriptor_set_layout->logical_device, descriptor_set_layout->handle);
  iree_allocator_free(host_allocator, descriptor_set_layout);

  IREE_TRACE_ZONE_END(z0);
}

VkDescriptorSetLayout iree_hal_vulkan_native_descriptor_set_layout_handle(
    iree_hal_descriptor_set_layout_t* base_descriptor_set_layout) {
  iree_hal_vulkan_native_descriptor_set_layout_t* descriptor_set_layout =
      iree_hal_vulkan_native_descriptor_set_layout_cast(
          base_descriptor_set_layout);
  return descriptor_set_layout->handle;
}

namespace {
const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_vulkan_native_descriptor_set_layout_vtable = {
        /*.destroy=*/iree_hal_vulkan_native_descriptor_set_layout_destroy,
};
}  // namespace
