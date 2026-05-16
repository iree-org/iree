// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/debug_utils.h"

#include <string.h>

static bool iree_hal_vulkan_debug_utils_has_object_name_symbols(
    const iree_hal_vulkan_device_syms_t* syms) {
#if IREE_HAL_VULKAN_LIBVULKAN_STATIC
  (void)syms;
  return true;
#else
  return syms->vkSetDebugUtilsObjectNameEXT != NULL;
#endif  // IREE_HAL_VULKAN_LIBVULKAN_STATIC
}

static bool iree_hal_vulkan_debug_utils_has_command_label_symbols(
    const iree_hal_vulkan_device_syms_t* syms) {
#if IREE_HAL_VULKAN_LIBVULKAN_STATIC
  (void)syms;
  return true;
#else
  return syms->vkCmdBeginDebugUtilsLabelEXT != NULL &&
         syms->vkCmdEndDebugUtilsLabelEXT != NULL &&
         syms->vkCmdInsertDebugUtilsLabelEXT != NULL;
#endif  // IREE_HAL_VULKAN_LIBVULKAN_STATIC
}

iree_status_t iree_hal_vulkan_debug_utils_initialize(
    iree_hal_vulkan_request_flags_t request_flags,
    const iree_hal_vulkan_device_syms_t* syms,
    iree_hal_vulkan_debug_utils_t* out_debug_utils) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(out_debug_utils);
  memset(out_debug_utils, 0, sizeof(*out_debug_utils));

  const iree_hal_vulkan_request_flags_t unknown_request_flags =
      request_flags & ~IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED;
  if (unknown_request_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized Vulkan request flag bits 0x%08x",
                            unknown_request_flags);
  }
  if (!iree_any_bit_set(request_flags,
                        IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS)) {
    return iree_ok_status();
  }

  if (!iree_hal_vulkan_debug_utils_has_object_name_symbols(syms)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan debug utils requested but vkSetDebugUtilsObjectNameEXT is not "
        "loaded");
  }
  if (!iree_hal_vulkan_debug_utils_has_command_label_symbols(syms)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan debug utils requested but command label entry points are not "
        "loaded");
  }

  out_debug_utils->flags = IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_OBJECT_NAMES |
                           IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_COMMAND_LABELS;
  return iree_ok_status();
}

bool iree_hal_vulkan_debug_utils_has(
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    iree_hal_vulkan_debug_utils_flags_t required_flags) {
  return iree_all_bits_set(debug_utils->flags, required_flags);
}

static iree_status_t iree_hal_vulkan_debug_utils_append_queue_role(
    iree_string_builder_t* builder, iree_host_size_t* role_count,
    iree_string_view_t role) {
  if (*role_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("+")));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, role));
  *role_count = *role_count + 1;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_debug_utils_set_object_name(
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkObjectType object_type, uint64_t object_handle, iree_string_view_t name,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(debug_utils);
  IREE_ASSERT_ARGUMENT(syms);
  if (!iree_hal_vulkan_debug_utils_has(
          debug_utils, IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_OBJECT_NAMES)) {
    return iree_ok_status();
  }

  iree_string_builder_t name_builder;
  iree_string_builder_initialize(host_allocator, &name_builder);
  iree_status_t status = iree_string_builder_append_string(
      &name_builder, iree_string_view_is_empty(name) ? IREE_SV("") : name);
  if (iree_status_is_ok(status)) {
    const VkDebugUtilsObjectNameInfoEXT name_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .objectType = object_type,
        .objectHandle = object_handle,
        .pObjectName = iree_string_builder_buffer(&name_builder),
    };
    status = iree_vkSetDebugUtilsObjectNameEXT(IREE_VULKAN_DEVICE(syms),
                                               logical_device, &name_info);
  }
  iree_string_builder_deinitialize(&name_builder);
  return status;
}

iree_status_t iree_hal_vulkan_debug_utils_set_queue_name(
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkQueue queue, iree_hal_vulkan_debug_utils_queue_role_flags_t role_flags,
    uint32_t queue_family_index, uint32_t queue_index,
    iree_string_view_t identifier, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(debug_utils);
  IREE_ASSERT_ARGUMENT(syms);
  if (!queue ||
      !iree_hal_vulkan_debug_utils_has(
          debug_utils, IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_OBJECT_NAMES)) {
    return iree_ok_status();
  }
  const iree_hal_vulkan_debug_utils_queue_role_flags_t unknown_role_flags =
      role_flags & ~IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_ALL_RECOGNIZED;
  if (unknown_role_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized Vulkan queue debug role bits 0x%08x",
                            unknown_role_flags);
  }
  if (role_flags == IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan queue debug name requires a queue role");
  }

  iree_string_builder_t name_builder;
  iree_string_builder_initialize(host_allocator, &name_builder);
  iree_status_t status =
      iree_string_builder_append_string(&name_builder, identifier);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&name_builder, IREE_SV("/"));
  }

  iree_host_size_t role_count = 0;
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(role_flags,
                       IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_COMPUTE)) {
    status = iree_hal_vulkan_debug_utils_append_queue_role(
        &name_builder, &role_count, IREE_SV("compute"));
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(role_flags,
                       IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_TRANSFER)) {
    status = iree_hal_vulkan_debug_utils_append_queue_role(
        &name_builder, &role_count, IREE_SV("transfer"));
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(role_flags,
                       IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_SPARSE_BINDING)) {
    status = iree_hal_vulkan_debug_utils_append_queue_role(
        &name_builder, &role_count, IREE_SV("sparse-binding"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(&name_builder, "-queue[%u:%u]",
                                               queue_family_index, queue_index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_debug_utils_set_object_name(
        debug_utils, syms, logical_device, VK_OBJECT_TYPE_QUEUE,
        (uint64_t)(uintptr_t)queue, iree_string_builder_view(&name_builder),
        host_allocator);
  }
  iree_string_builder_deinitialize(&name_builder);
  return status;
}

static void iree_hal_vulkan_debug_utils_label_color(
    iree_hal_label_color_t label_color, float out_color[4]) {
  out_color[0] = (float)label_color.r / 255.0f;
  out_color[1] = (float)label_color.g / 255.0f;
  out_color[2] = (float)label_color.b / 255.0f;
  out_color[3] = (float)label_color.a / 255.0f;
}

void iree_hal_vulkan_debug_utils_begin_command_label(
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    const iree_hal_vulkan_device_syms_t* syms, VkCommandBuffer command_buffer,
    const char* label, iree_hal_label_color_t label_color) {
  IREE_ASSERT_ARGUMENT(debug_utils);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (!iree_hal_vulkan_debug_utils_has(
          debug_utils, IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_COMMAND_LABELS)) {
    return;
  }

  VkDebugUtilsLabelEXT label_info = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
      .pLabelName = label ? label : "",
  };
  iree_hal_vulkan_debug_utils_label_color(label_color, label_info.color);
  iree_vkCmdBeginDebugUtilsLabelEXT(IREE_VULKAN_DEVICE(syms), command_buffer,
                                    &label_info);
}

void iree_hal_vulkan_debug_utils_end_command_label(
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    const iree_hal_vulkan_device_syms_t* syms, VkCommandBuffer command_buffer) {
  IREE_ASSERT_ARGUMENT(debug_utils);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (!iree_hal_vulkan_debug_utils_has(
          debug_utils, IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_COMMAND_LABELS)) {
    return;
  }

  iree_vkCmdEndDebugUtilsLabelEXT(IREE_VULKAN_DEVICE(syms), command_buffer);
}
