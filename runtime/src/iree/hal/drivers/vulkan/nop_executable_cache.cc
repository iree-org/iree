// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/nop_executable_cache.h"

#include <cstddef>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/dynamic_symbol_tables.h"
#include "iree/hal/drivers/vulkan/native_executable.h"

using namespace iree::hal::vulkan;

typedef struct iree_hal_vulkan_nop_executable_cache_t {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
} iree_hal_vulkan_nop_executable_cache_t;

namespace {
extern const iree_hal_executable_cache_vtable_t
    iree_hal_vulkan_nop_executable_cache_vtable;
}  // namespace

static iree_hal_vulkan_nop_executable_cache_t*
iree_hal_vulkan_nop_executable_cache_cast(
    iree_hal_executable_cache_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_vulkan_nop_executable_cache_vtable);
  return (iree_hal_vulkan_nop_executable_cache_t*)base_value;
}

iree_status_t iree_hal_vulkan_nop_executable_cache_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_nop_executable_cache_t* executable_cache = NULL;
  iree_status_t status = iree_allocator_malloc(logical_device->host_allocator(),
                                               sizeof(*executable_cache),
                                               (void**)&executable_cache);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_vulkan_nop_executable_cache_vtable,
                                 &executable_cache->resource);
    executable_cache->logical_device = logical_device;

    *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_nop_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_vulkan_nop_executable_cache_t* executable_cache =
      iree_hal_vulkan_nop_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator =
      executable_cache->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_nop_executable_cache_infer_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  iree_hal_vulkan_nop_executable_cache_t* executable_cache =
      iree_hal_vulkan_nop_executable_cache_cast(base_executable_cache);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                       "Vulkan SPIR-V size inference not yet implemented");
  (void)executable_cache;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_vulkan_nop_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  iree_hal_vulkan_nop_executable_cache_t* executable_cache =
      iree_hal_vulkan_nop_executable_cache_cast(base_executable_cache);
  if (iree_string_view_equal(executable_format,
                             iree_make_cstring_view("vulkan-spirv-fb"))) {
    return true;
  } else if (iree_string_view_equal(
                 executable_format,
                 iree_make_cstring_view("vulkan-spirv-fb-ptr"))) {
    return iree_all_bits_set(
        executable_cache->logical_device->enabled_features(),
        IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES);
  }
  return false;
}

static iree_status_t iree_hal_vulkan_nop_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  if (!iree_hal_vulkan_nop_executable_cache_can_prepare_format(
          base_executable_cache, executable_params->caching_mode,
          executable_params->executable_format)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no Vulkan executable implementation registered "
                            "for the given executable format '%.*s'",
                            (int)executable_params->executable_format.size,
                            executable_params->executable_format.data);
  }
  iree_hal_vulkan_nop_executable_cache_t* executable_cache =
      iree_hal_vulkan_nop_executable_cache_cast(base_executable_cache);
  return iree_hal_vulkan_native_executable_create(
      executable_cache->logical_device,
      /*pipeline_cache=*/VK_NULL_HANDLE, executable_params, out_executable);
}

namespace {
const iree_hal_executable_cache_vtable_t
    iree_hal_vulkan_nop_executable_cache_vtable = {
        /*.destroy=*/iree_hal_vulkan_nop_executable_cache_destroy,
        /*.infer_format=*/iree_hal_vulkan_nop_executable_cache_infer_format,
        /*.can_prepare_format=*/
        iree_hal_vulkan_nop_executable_cache_can_prepare_format,
        /*.prepare_executable=*/
        iree_hal_vulkan_nop_executable_cache_prepare_executable,
};
}  // namespace
