// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/vulkan/debug_reporter.h"

#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

// NOTE: |user_data| may be nullptr if we are being called during instance
// creation. Otherwise it is a pointer to the DebugReporter instance.

// NOTE: this callback must be thread safe and must be careful not to reach too
// far outside of the call - it is called in-context from arbitrary threads with
// some amount of Vulkan state on the stack. Assume that creating or deleting
// Vulkan objects, issuing most Vulkan commands, etc are off-limits.

VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsMessageCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    LOG(ERROR) << callback_data->pMessage;
  } else {
    VLOG(1) << callback_data->pMessage;
  }

  return VK_FALSE;  // VK_TRUE is reserved for future use.
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugReportCallback(
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT object_type,
    uint64_t object, size_t location, int32_t message_code,
    const char* layer_prefix, const char* message, void* user_data) {
  VLOG(1) << message;

  return VK_FALSE;  // VK_TRUE is reserved for future use.
}

}  // namespace

// static
void DebugReporter::PopulateStaticCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT* create_info) {
  create_info->sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info->pNext = nullptr;
  create_info->flags = 0;

  // TODO(benvanik): only enable the severities that logging has enabled.
  create_info->messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

  // TODO(benvanik): allow filtering by category as a flag.
  create_info->messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

  create_info->pfnUserCallback = DebugUtilsMessageCallback;
  create_info->pUserData = nullptr;
}

// static
void DebugReporter::PopulateStaticCreateInfo(
    VkDebugReportCallbackCreateInfoEXT* create_info) {
  create_info->sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
  create_info->pNext = nullptr;
  create_info->flags = 0;

  // TODO(benvanik): only enable the severities that logging has enabled.
  create_info->flags |=
      VK_DEBUG_REPORT_INFORMATION_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT |
      VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
      VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_DEBUG_BIT_EXT;

  create_info->pfnCallback = DebugReportCallback;
  create_info->pUserData = nullptr;
}

// static
StatusOr<std::unique_ptr<DebugReporter>>
DebugReporter::CreateDebugUtilsMessenger(
    VkInstance instance, const ref_ptr<DynamicSymbols>& syms,
    const VkAllocationCallbacks* allocation_callbacks) {
  IREE_TRACE_SCOPE0("DebugReporter::CreateDebugUtilsMessenger");

  auto debug_reporter = std::unique_ptr<DebugReporter>(
      new DebugReporter(instance, syms, allocation_callbacks));

  VkDebugUtilsMessengerCreateInfoEXT create_info;
  PopulateStaticCreateInfo(&create_info);
  create_info.pUserData = debug_reporter.get();

  VK_RETURN_IF_ERROR(syms->vkCreateDebugUtilsMessengerEXT(
      instance, &create_info, allocation_callbacks,
      &debug_reporter->messenger_));

  return debug_reporter;
}

// static
StatusOr<std::unique_ptr<DebugReporter>>
DebugReporter::CreateDebugReportCallback(
    VkInstance instance, const ref_ptr<DynamicSymbols>& syms,
    const VkAllocationCallbacks* allocation_callbacks) {
  IREE_TRACE_SCOPE0("DebugReporter::CreateDebugReportCallback");

  auto debug_reporter = std::unique_ptr<DebugReporter>(
      new DebugReporter(instance, syms, allocation_callbacks));

  VkDebugReportCallbackCreateInfoEXT create_info;
  PopulateStaticCreateInfo(&create_info);
  create_info.pUserData = debug_reporter.get();

  VK_RETURN_IF_ERROR(syms->vkCreateDebugReportCallbackEXT(
      instance, &create_info, allocation_callbacks,
      &debug_reporter->callback_));

  return debug_reporter;
}

DebugReporter::DebugReporter(VkInstance instance,
                             const ref_ptr<DynamicSymbols>& syms,
                             const VkAllocationCallbacks* allocation_callbacks)
    : instance_(instance),
      syms_(add_ref(syms)),
      allocation_callbacks_(allocation_callbacks) {}

DebugReporter::~DebugReporter() {
  IREE_TRACE_SCOPE0("DebugReporter::dtor");
  if (messenger_ != VK_NULL_HANDLE) {
    syms_->vkDestroyDebugUtilsMessengerEXT(instance_, messenger_,
                                           allocation_callbacks_);
  }
  if (callback_ != VK_NULL_HANDLE) {
    syms_->vkDestroyDebugReportCallbackEXT(instance_, callback_,
                                           allocation_callbacks_);
  }
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
