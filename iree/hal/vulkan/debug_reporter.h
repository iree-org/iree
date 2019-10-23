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

#ifndef IREE_HAL_VULKAN_DEBUG_REPORTER_H_
#define IREE_HAL_VULKAN_DEBUG_REPORTER_H_

#include <vulkan/vulkan.h>

#include "iree/base/status.h"
#include "iree/hal/vulkan/dynamic_symbols.h"

namespace iree {
namespace hal {
namespace vulkan {

// A debug reporter that works with the VK_EXT_debug_utils extension.
// One reporter should be created per VkInstance to receive callbacks from the
// API and route them to our logging systems. In general VK_EXT_debug_utils
// should be preferred if available as it provides a much cleaner interface and
// more plug-points than VK_EXT_debug_report.
//
// Since creating a reporter requires a VkInstance it's not possible to report
// on messages during instance creation. To work around this it's possible to
// pass a *CreateInfo struct to vkCreateInstance as part of the
// VkInstanceCreateInfo::pNext chain. The callback will only be used this way
// during the creation call after which users can create the real
// instance-specific reporter.
class DebugReporter final {
 public:
  // Populates |create_info| with an instance-agnostic callback.
  // This can be used during instance creation by chaining the |create_info| to
  // VkInstanceCreateInfo::pNext.
  //
  // Only use if VK_EXT_debug_utils is present.
  static void PopulateStaticCreateInfo(
      VkDebugUtilsMessengerCreateInfoEXT* create_info);

  // Populates |create_info| with an instance-agnostic callback.
  // This can be used during instance creation by chaining the |create_info| to
  // VkInstanceCreateInfo::pNext.
  //
  // Only use if VK_EXT_debug_report is present.
  static void PopulateStaticCreateInfo(
      VkDebugReportCallbackCreateInfoEXT* create_info);

  // Creates a debug messenger for the given Vulkan |instance| with
  // VK_EXT_debug_utils enabled.
  static StatusOr<std::unique_ptr<DebugReporter>> CreateDebugUtilsMessenger(
      VkInstance instance, const ref_ptr<DynamicSymbols>& syms,
      const VkAllocationCallbacks* allocation_callbacks);

  // Creates a debug report callback for the given Vulkan |instance| with
  // VK_EXT_debug_report enabled.
  static StatusOr<std::unique_ptr<DebugReporter>> CreateDebugReportCallback(
      VkInstance instance, const ref_ptr<DynamicSymbols>& syms,
      const VkAllocationCallbacks* allocation_callbacks);

  ~DebugReporter();

 private:
  DebugReporter(VkInstance instance, const ref_ptr<DynamicSymbols>& syms,
                const VkAllocationCallbacks* allocation_callbacks);

  VkInstance instance_ = VK_NULL_HANDLE;
  ref_ptr<DynamicSymbols> syms_;
  const VkAllocationCallbacks* allocation_callbacks_ = nullptr;

  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkDebugReportCallbackEXT callback_ = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_DEBUG_REPORTER_H_
