// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/debug_reporter.h"

#include <atomic>
#include <cstddef>
#include <cstdio>

#include "iree/base/assert.h"
#include "iree/hal/drivers/vulkan/status_util.h"

struct iree_hal_vulkan_debug_reporter_t {
  iree_allocator_t host_allocator;
  VkInstance instance;
  iree::hal::vulkan::DynamicSymbols* syms;
  int32_t min_verbosity;
  bool check_errors;
  const VkAllocationCallbacks* allocation_callbacks;
  VkDebugUtilsMessengerEXT messenger;

  std::atomic<bool> did_error{false};
};

// NOTE: |user_data| may be nullptr if we are being called during instance
// creation. Otherwise it is a pointer to the DebugReporter instance.
//
// NOTE: this callback must be thread safe and must be careful not to reach too
// far outside of the call - it is called in-context from arbitrary threads with
// some amount of Vulkan state on the stack. Assume that creating or deleting
// Vulkan objects, issuing most Vulkan commands, etc are off-limits.
static VKAPI_ATTR VkBool32 VKAPI_CALL
iree_hal_vulkan_debug_utils_message_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
  iree_hal_vulkan_debug_reporter_t* reporter =
      (iree_hal_vulkan_debug_reporter_t*)user_data;

  // Filter messages based on the verbosity setting.
  // 0=none, 1=errors, 2=warnings, 3=info, 4=debug
  char severity_char = '?';
  int32_t message_verbosity = 0;
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    severity_char = '!';
    message_verbosity = 1;
  } else if (message_severity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    severity_char = 'w';
    message_verbosity = 2;
  } else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    severity_char = 'i';
    message_verbosity = 3;
  } else {
    severity_char = 'v';
    message_verbosity = 4;
  }
  if (message_verbosity < reporter->min_verbosity) {
    fprintf(stderr, "[VULKAN] %c %s\n", severity_char, callback_data->pMessage);
    if (reporter->check_errors) {
      reporter->did_error = true;
    }
  }

  return VK_FALSE;  // VK_TRUE is reserved for future use.
}

// Populates |create_info| with an instance-agnostic callback.
// This can be used during instance creation by chaining the |create_info| to
// VkInstanceCreateInfo::pNext.
//
// Only use if VK_EXT_debug_utils is present.
static void iree_hal_vulkan_debug_reporter_populate_create_info(
    VkDebugUtilsMessengerCreateInfoEXT* out_create_info) {
  out_create_info->sType =
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  out_create_info->pNext = nullptr;
  out_create_info->flags = 0;

  // TODO(benvanik): only enable the severities that logging has enabled.
  out_create_info->messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

  // TODO(benvanik): allow filtering by category as a flag.
  out_create_info->messageType =
      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

  out_create_info->pfnUserCallback =
      iree_hal_vulkan_debug_utils_message_callback;
  out_create_info->pUserData = nullptr;
}

iree_status_t iree_hal_vulkan_debug_reporter_allocate(
    VkInstance instance, iree::hal::vulkan::DynamicSymbols* syms,
    int32_t min_verbosity, bool check_errors,
    const VkAllocationCallbacks* allocation_callbacks,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_debug_reporter_t** out_reporter) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(out_reporter);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate our struct first as we need to pass the pointer to the userdata
  // of the messager instance when we create it.
  iree_hal_vulkan_debug_reporter_t* reporter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*reporter),
                                (void**)&reporter));
  reporter->host_allocator = host_allocator;
  reporter->instance = instance;
  reporter->syms = syms;
  reporter->min_verbosity = min_verbosity;
  reporter->check_errors = check_errors;
  reporter->allocation_callbacks = allocation_callbacks;

  VkDebugUtilsMessengerCreateInfoEXT create_info;
  iree_hal_vulkan_debug_reporter_populate_create_info(&create_info);
  create_info.pUserData = reporter;
  iree_status_t status = VK_RESULT_TO_STATUS(
      syms->vkCreateDebugUtilsMessengerEXT(
          instance, &create_info, allocation_callbacks, &reporter->messenger),
      "vkCreateDebugUtilsMessengerEXT");

  if (iree_status_is_ok(status)) {
    *out_reporter = reporter;
  } else {
    iree_hal_vulkan_debug_reporter_free(reporter);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_vulkan_debug_reporter_free(
    iree_hal_vulkan_debug_reporter_t* reporter) {
  if (!reporter) return;
  iree_allocator_t host_allocator = reporter->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (reporter->messenger != VK_NULL_HANDLE) {
    reporter->syms->vkDestroyDebugUtilsMessengerEXT(
        reporter->instance, reporter->messenger,
        reporter->allocation_callbacks);
  }
  if (reporter->check_errors && reporter->did_error) {
    IREE_ASSERT(0);
  }
  iree_allocator_free(host_allocator, reporter);

  IREE_TRACE_ZONE_END(z0);
}
