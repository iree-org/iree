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

#include <memory>

#include "absl/flags/flag.h"
#include "iree/base/init.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/driver_registry.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/vulkan_driver.h"

ABSL_FLAG(bool, vulkan_validation_layers, true,
          "Enables standard Vulkan validation layers.");
ABSL_FLAG(bool, vulkan_debug_utils, true,
          "Enables VK_EXT_debug_utils, records markers, and logs errors.");
ABSL_FLAG(bool, vulkan_debug_report, false,
          "Enables VK_EXT_debug_report and logs errors.");
ABSL_FLAG(bool, vulkan_push_descriptors, true,
          "Enables use of vkCmdPushDescriptorSetKHR, if available.");

namespace iree {
namespace hal {
namespace vulkan {
namespace {

StatusOr<ref_ptr<Driver>> CreateVulkanDriver() {
  IREE_TRACE_SCOPE0("CreateVulkanDriver");

  // Load the Vulkan library. This will fail if the library cannot be found or
  // does not have the expected functions.
  ASSIGN_OR_RETURN(auto syms, DynamicSymbols::CreateFromSystemLoader());

  // Setup driver options from flags. We do this here as we want to enable other
  // consumers that may not be using modules/command line flags to be able to
  // set their options however they want.
  VulkanDriver::Options options;

  // TODO: validation layers have bugs when using VK_EXT_debug_report, so if the
  // user requested that we force them off with a warning. Prefer using
  // VK_EXT_debug_utils when available.
  if (absl::GetFlag(FLAGS_vulkan_debug_report) &&
      absl::GetFlag(FLAGS_vulkan_validation_layers)) {
    LOG(WARNING) << "VK_EXT_debug_report has issues with modern validation "
                    "layers; disabling validation";
    absl::SetFlag(&FLAGS_vulkan_validation_layers, false);
  }

  // REQUIRED: these are required extensions that must be present for IREE to
  // work (such as those relied upon by SPIR-V kernels, etc).
  options.device_extensibility.required_extensions.push_back(
      VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
  // Multiple extensions depend on VK_KHR_get_physical_device_properties2.
  // This extension was deprecated in Vulkan 1.1 as its functionality was
  // promoted to core, so we list it as optional even though we require it.
  options.instance_extensibility.optional_extensions.push_back(
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  // Timeline semaphore support is optional and will be emulated if necessary.
  options.device_extensibility.optional_extensions.push_back(
      VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  // Polyfill layer - enable if present (instead of our custom emulation).
  options.instance_extensibility.optional_layers.push_back(
      "VK_LAYER_KHRONOS_timeline_semaphore");

  if (absl::GetFlag(FLAGS_vulkan_validation_layers)) {
    options.instance_extensibility.optional_layers.push_back(
        "VK_LAYER_LUNARG_standard_validation");
  }

  if (absl::GetFlag(FLAGS_vulkan_debug_report)) {
    options.instance_extensibility.optional_extensions.push_back(
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
  }
  if (absl::GetFlag(FLAGS_vulkan_debug_utils)) {
    options.instance_extensibility.optional_extensions.push_back(
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  if (absl::GetFlag(FLAGS_vulkan_push_descriptors)) {
    options.device_extensibility.optional_extensions.push_back(
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  }

  // Create the driver and VkInstance.
  ASSIGN_OR_RETURN(auto driver, VulkanDriver::Create(options, std::move(syms)));

  return driver;
}

}  // namespace
}  // namespace vulkan
}  // namespace hal
}  // namespace iree

IREE_REGISTER_MODULE_INITIALIZER(iree_hal_vulkan_driver, {
  QCHECK_OK(::iree::hal::DriverRegistry::shared_registry()->Register(
      "vulkan", ::iree::hal::vulkan::CreateVulkanDriver));
});
IREE_REGISTER_MODULE_INITIALIZER_SEQUENCE(iree_hal, iree_hal_vulkan_driver);
