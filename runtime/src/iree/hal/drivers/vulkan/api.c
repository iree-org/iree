// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/api.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/drivers/vulkan/syms.h"

#if !defined(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)
#define IREE_HAL_VULKAN_KHR_PORTABILITY_SUBSET_EXTENSION_NAME \
  "VK_KHR_portability_subset"
#else
#define IREE_HAL_VULKAN_KHR_PORTABILITY_SUBSET_EXTENSION_NAME \
  VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#endif  // !VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME

static bool iree_hal_vulkan_extension_name_list_contains(
    iree_host_size_t extension_count, const char* const* extension_names,
    const char* extension_name) {
  for (iree_host_size_t i = 0; i < extension_count; ++i) {
    if (extension_names[i] != NULL &&
        strcmp(extension_name, extension_names[i]) == 0) {
      return true;
    }
  }
  return false;
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_extensions_from_names(
    iree_host_size_t extension_count, const char* const* extension_names,
    iree_hal_vulkan_device_extensions_t* out_extensions) {
  IREE_ASSERT_ARGUMENT(out_extensions);
  *out_extensions = IREE_HAL_VULKAN_DEVICE_EXTENSION_NONE;
  if (extension_count > 0 && extension_names == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan extension name list has count %" PRIhsz
                            " but no value storage",
                            extension_count);
  }
  iree_hal_vulkan_device_extensions_t extensions =
      IREE_HAL_VULKAN_DEVICE_EXTENSION_NONE;
  if (iree_hal_vulkan_extension_name_list_contains(
          extension_count, extension_names,
          IREE_HAL_VULKAN_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
    extensions |= IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PORTABILITY_SUBSET;
  }
  if (iree_hal_vulkan_extension_name_list_contains(
          extension_count, extension_names,
          VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME)) {
    extensions |= IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY;
  }
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  if (iree_hal_vulkan_extension_name_list_contains(
          extension_count, extension_names,
          VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME)) {
    extensions |= IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_WIN32;
  }
#else
  if (iree_hal_vulkan_extension_name_list_contains(
          extension_count, extension_names,
          VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) {
    extensions |= IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_FD;
  }
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)
  if (iree_hal_vulkan_extension_name_list_contains(
          extension_count, extension_names,
          VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
    extensions |= IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_EXTERNAL_MEMORY_HOST;
  }
  if (iree_hal_vulkan_extension_name_list_contains(
          extension_count, extension_names,
          VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
    extensions |= IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS;
  }
  if (iree_hal_vulkan_extension_name_list_contains(
          extension_count, extension_names,
          VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
    extensions |= IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PUSH_DESCRIPTOR;
  }
  *out_extensions = extensions;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_add_extensibility_string(
    const char* value, iree_host_size_t string_capacity,
    iree_host_size_t* out_string_count, const char** out_string_values) {
  if (out_string_values != NULL && *out_string_count >= string_capacity) {
    *out_string_count = *out_string_count + 1;
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  if (out_string_values != NULL) {
    out_string_values[*out_string_count] = value;
  }
  *out_string_count = *out_string_count + 1;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_query_extensibility_set(
    iree_hal_vulkan_request_flags_t request_flags,
    iree_hal_vulkan_extensibility_set_t set, iree_host_size_t string_capacity,
    iree_host_size_t* out_string_count, const char** out_string_values) {
  IREE_ASSERT_ARGUMENT(out_string_count);
  *out_string_count = 0;

  if (set >= IREE_HAL_VULKAN_EXTENSIBILITY_SET_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid Vulkan extensibility set %u",
                            (uint32_t)set);
  }
  if (iree_any_bit_set(request_flags,
                       ~IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unrecognized Vulkan request flag bits 0x%08x",
        request_flags & ~IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED);
  }

  iree_status_t status = iree_ok_status();
#define IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(target_set, value)      \
  if (set == (target_set)) {                                             \
    iree_status_t add_status = iree_hal_vulkan_add_extensibility_string( \
        (value), string_capacity, out_string_count, out_string_values);  \
    if (!iree_status_is_ok(add_status) && iree_status_is_ok(status)) {   \
      status = add_status;                                               \
    }                                                                    \
  }

  switch (set) {
    case IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_REQUIRED:
      if (iree_any_bit_set(request_flags,
                           IREE_HAL_VULKAN_REQUEST_FLAG_VALIDATION_LAYERS)) {
        IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
            IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_REQUIRED,
            "VK_LAYER_KHRONOS_validation");
      }
      break;
    case IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL:
      break;
    case IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_REQUIRED:
      if (iree_any_bit_set(request_flags,
                           IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS)) {
        IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
            IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_REQUIRED,
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      }
      break;
    case IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_OPTIONAL:
#if defined(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
      IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
          IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_OPTIONAL,
          VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif  // VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
      break;
    case IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED:
      break;
    case IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL:
      IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
          IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          IREE_HAL_VULKAN_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
      IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
          IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
      IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
          IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
      IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
          IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)
      IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
          IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
      IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
          IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
      IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING(
          IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
      break;
    case IREE_HAL_VULKAN_EXTENSIBILITY_SET_COUNT:
      break;
  }
#undef IREE_HAL_VULKAN_ADD_EXTENSIBILITY_STRING

  return status;
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

IREE_API_EXPORT iree_status_t iree_hal_vulkan_allocated_buffer_handle(
    iree_hal_buffer_t* allocated_buffer, VkDeviceMemory* out_memory,
    VkBuffer* out_handle) {
  IREE_ASSERT_ARGUMENT(allocated_buffer);
  if (iree_hal_vulkan_buffer_isa(
          iree_hal_buffer_allocated_buffer(allocated_buffer))) {
    return iree_hal_vulkan_buffer_handle(allocated_buffer, out_memory,
                                         out_handle);
  }
  return iree_hal_vulkan_sparse_buffer_handle(allocated_buffer, out_memory,
                                              out_handle);
}
