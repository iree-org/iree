// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/api.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/drivers/vulkan/syms.h"

IREE_API_EXPORT iree_status_t iree_hal_vulkan_query_extensibility_set(
    iree_hal_vulkan_features_t requested_features,
    iree_hal_vulkan_extensibility_set_t set, iree_host_size_t string_capacity,
    iree_host_size_t* out_string_count, const char** out_string_values) {
  IREE_ASSERT_ARGUMENT(out_string_count);
  *out_string_count = 0;

  if (set >= IREE_HAL_VULKAN_EXTENSIBILITY_SET_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid Vulkan extensibility set %u",
                            (uint32_t)set);
  }

  (void)requested_features;
  (void)string_capacity;
  (void)out_string_values;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_syms_create(
    void* vkGetInstanceProcAddr_fn, iree_allocator_t host_allocator,
    iree_hal_vulkan_syms_t** out_syms) {
  IREE_ASSERT_ARGUMENT(vkGetInstanceProcAddr_fn);
  IREE_ASSERT_ARGUMENT(out_syms);
  *out_syms = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_syms_t* syms = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*syms), (void**)&syms));
  memset(syms, 0, sizeof(*syms));
  iree_atomic_ref_count_init(&syms->ref_count);
  syms->host_allocator = host_allocator;

  iree_status_t status = iree_hal_vulkan_libvulkan_initialize_from_loader(
      (PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr_fn, host_allocator,
      &syms->libvulkan);

  if (iree_status_is_ok(status)) {
    *out_syms = syms;
  } else {
    iree_allocator_free(host_allocator, syms);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_syms_create_from_system_loader(
    iree_allocator_t host_allocator, iree_hal_vulkan_syms_t** out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  *out_syms = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_syms_t* syms = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*syms), (void**)&syms));
  memset(syms, 0, sizeof(*syms));
  iree_atomic_ref_count_init(&syms->ref_count);
  syms->host_allocator = host_allocator;

  iree_status_t status = iree_hal_vulkan_libvulkan_initialize(
      IREE_HAL_VULKAN_LIBVULKAN_FLAG_NONE, iree_string_view_list_empty(),
      host_allocator, &syms->libvulkan);

  if (iree_status_is_ok(status)) {
    *out_syms = syms;
  } else {
    iree_allocator_free(host_allocator, syms);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_vulkan_syms_retain(iree_hal_vulkan_syms_t* syms) {
  IREE_ASSERT_ARGUMENT(syms);
  if (syms) {
    iree_atomic_ref_count_inc(&syms->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_vulkan_syms_release(
    iree_hal_vulkan_syms_t* syms) {
  IREE_ASSERT_ARGUMENT(syms);
  if (syms && iree_atomic_ref_count_dec(&syms->ref_count) == 1) {
    iree_allocator_t host_allocator = syms->host_allocator;
    iree_hal_vulkan_libvulkan_deinitialize(&syms->libvulkan);
    iree_allocator_free(host_allocator, syms);
  }
}

IREE_API_EXPORT void iree_hal_vulkan_device_options_initialize(
    iree_hal_vulkan_device_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_wrap_device(
    iree_string_view_t identifier,
    const iree_hal_vulkan_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    const iree_hal_vulkan_syms_t* instance_syms, VkInstance instance,
    VkPhysicalDevice physical_device, VkDevice logical_device,
    const iree_hal_vulkan_queue_set_t* compute_queue_set,
    const iree_hal_vulkan_queue_set_t* transfer_queue_set,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;

  (void)identifier;
  (void)options;
  (void)create_params;
  (void)instance_syms;
  (void)instance;
  (void)physical_device;
  (void)logical_device;
  (void)compute_queue_set;
  (void)transfer_queue_set;
  (void)host_allocator;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "wrapping external VkDevice handles is not implemented in the Vulkan "
      "HAL rewrite scaffold");
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_allocated_buffer_handle(
    iree_hal_buffer_t* allocated_buffer, VkDeviceMemory* out_memory,
    VkBuffer* out_handle) {
  IREE_ASSERT_ARGUMENT(allocated_buffer);
  IREE_ASSERT_ARGUMENT(out_memory);
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_memory = (VkDeviceMemory)0;
  *out_handle = (VkBuffer)0;
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "buffer is not backed by the Vulkan HAL rewrite scaffold");
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_semaphore_handle(
    iree_hal_semaphore_t* semaphore, VkSemaphore* out_handle) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = (VkSemaphore)0;
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "semaphore is not backed by the Vulkan HAL rewrite scaffold");
}
