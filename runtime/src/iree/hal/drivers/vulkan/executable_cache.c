// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/executable_cache.h"

#include <string.h>

#include "iree/hal/drivers/vulkan/executable.h"

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_executable_cache_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_executable_cache_t {
  // HAL resource header.
  iree_hal_resource_t resource;

  // Host allocator used for cache lifetime.
  iree_allocator_t host_allocator;

  // Borrowed logical-device dispatch table.
  const iree_hal_vulkan_device_syms_t* syms;

  // Borrowed logical-device handle.
  VkDevice logical_device;

  // Borrowed immutable physical-device inventory.
  const iree_hal_vulkan_physical_device_snapshot_t* physical_device;

  // HAL feature bits enabled on the logical device.
  iree_hal_vulkan_features_t enabled_features;

  // Vulkan pipeline cache owned by this HAL executable cache.
  VkPipelineCache pipeline_cache;
} iree_hal_vulkan_executable_cache_t;

static const iree_hal_executable_cache_vtable_t
    iree_hal_vulkan_executable_cache_vtable;

static iree_hal_vulkan_executable_cache_t*
iree_hal_vulkan_executable_cache_cast(iree_hal_executable_cache_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_executable_cache_vtable);
  return (iree_hal_vulkan_executable_cache_t*)base_value;
}

iree_status_t iree_hal_vulkan_executable_cache_create(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, identifier.data, identifier.size);

  iree_hal_vulkan_executable_cache_t* executable_cache = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable_cache),
                                (void**)&executable_cache));
  memset(executable_cache, 0, sizeof(*executable_cache));
  iree_hal_resource_initialize(&iree_hal_vulkan_executable_cache_vtable,
                               &executable_cache->resource);
  executable_cache->host_allocator = host_allocator;
  executable_cache->syms = syms;
  executable_cache->logical_device = logical_device;
  executable_cache->physical_device = physical_device;
  executable_cache->enabled_features = enabled_features;

  VkPipelineCacheCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
  };
  iree_status_t status = iree_vkCreatePipelineCache(
      IREE_VULKAN_DEVICE(syms), logical_device, &create_info,
      /*pAllocator=*/NULL, &executable_cache->pipeline_cache);

  if (iree_status_is_ok(status)) {
    *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  } else {
    iree_allocator_free(host_allocator, executable_cache);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_vulkan_executable_cache_t* executable_cache =
      iree_hal_vulkan_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (executable_cache->pipeline_cache) {
    iree_vkDestroyPipelineCache(IREE_VULKAN_DEVICE(executable_cache->syms),
                                executable_cache->logical_device,
                                executable_cache->pipeline_cache,
                                /*pAllocator=*/NULL);
  }
  iree_allocator_free(host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_executable_cache_infer_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  (void)base_executable_cache;
  (void)caching_mode;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_vulkan_executable_infer_format(
      executable_data, executable_format_capacity, executable_format,
      out_inferred_size);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_vulkan_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  (void)caching_mode;
  iree_hal_vulkan_executable_cache_t* executable_cache =
      iree_hal_vulkan_executable_cache_cast(base_executable_cache);
  return iree_hal_vulkan_executable_format_supported(
      executable_cache->enabled_features, executable_format);
}

static iree_status_t iree_hal_vulkan_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_vulkan_executable_cache_t* executable_cache =
      iree_hal_vulkan_executable_cache_cast(base_executable_cache);
  if (!iree_hal_vulkan_executable_format_supported(
          executable_cache->enabled_features,
          executable_params->executable_format)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no Vulkan executable implementation registered for format '%.*s'",
        (int)executable_params->executable_format.size,
        executable_params->executable_format.data);
  }
  return iree_hal_vulkan_executable_create(
      executable_cache->syms, executable_cache->logical_device,
      executable_cache->physical_device, executable_cache->enabled_features,
      executable_cache->pipeline_cache, executable_params,
      executable_cache->host_allocator, out_executable);
}

static const iree_hal_executable_cache_vtable_t
    iree_hal_vulkan_executable_cache_vtable = {
        .destroy = iree_hal_vulkan_executable_cache_destroy,
        .infer_format = iree_hal_vulkan_executable_cache_infer_format,
        .can_prepare_format =
            iree_hal_vulkan_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_vulkan_executable_cache_prepare_executable,
};
