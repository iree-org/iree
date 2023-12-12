// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/debug_reporter.h"

#include <cstddef>
#include <cstdio>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/vulkan/status_util.h"

struct iree_hal_vulkan_debug_reporter_t {
  iree_allocator_t host_allocator;
  VkInstance instance;
  iree::hal::vulkan::DynamicSymbols* syms;
  int32_t min_verbosity;
  bool check_errors;
  const VkAllocationCallbacks* allocation_callbacks;
  VkDebugUtilsMessengerEXT messenger;

  // If |check_errors| is true, this will be set to a status code when a
  // message above |min_verbosity| is reported. Only the first status code will
  // be tracked. Messages can be sent from any thread and after any API call,
  // so components can lazily check this status to see if any errors occured.
  iree_atomic_intptr_t error_status;
};

static void iree_hal_vulkan_debug_reporter_try_set_status(
    iree_hal_vulkan_debug_reporter_t* reporter, iree_status_t new_status) {
  if (IREE_UNLIKELY(iree_status_is_ok(new_status))) return;

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "Vulkan debug error: ");
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(new_status)));

  iree_status_t old_status = iree_ok_status();
  if (!iree_atomic_compare_exchange_strong_intptr(
          &reporter->error_status, (intptr_t*)&old_status, (intptr_t)new_status,
          iree_memory_order_acq_rel,
          iree_memory_order_relaxed /* old_status is unused */)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(new_status);
  }

  IREE_TRACE_ZONE_END(z0);
}

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
      iree_hal_vulkan_debug_reporter_try_set_status(
          reporter,
          iree_make_status(IREE_STATUS_INTERNAL, "Vulkan debug message:\n%s",
                           callback_data->pMessage));
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
  iree_allocator_free(host_allocator, reporter);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_vulkan_debug_reporter_has_error(
    iree_hal_vulkan_debug_reporter_t* reporter) {
  if (!reporter->check_errors) return false;
  return iree_atomic_load_intptr(&reporter->error_status,
                                 iree_memory_order_acquire) != 0;
}

iree_status_t iree_hal_vulkan_debug_reporter_consume_status(
    iree_hal_vulkan_debug_reporter_t* reporter) {
  iree_status_t old_status = iree_ok_status();
  iree_status_t new_status = iree_ok_status();
  while (!iree_atomic_compare_exchange_strong_intptr(
      &reporter->error_status, (intptr_t*)&old_status, (intptr_t)new_status,
      iree_memory_order_acq_rel,
      iree_memory_order_acquire /* old_status is actually used */)) {
    // Previous status was not OK; we have it now though and can try again.
    new_status = iree_status_from_code(iree_status_code(old_status));
  }
  // If old_status is not iree_ok_status then it was obtained through the
  // comparison-failed mode of the above compare_exchange, which loaded it with
  // iree_memory_order_acquire. This guarantees that if we are returning a
  // failure status to the caller then all past memory operations are already
  // visible, such as any information attached to that failure status.
  return old_status;
}
