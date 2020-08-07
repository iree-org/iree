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

// Utilities for working with layers and extensions.

#ifndef IREE_HAL_VULKAN_EXTENSIBILITY_UTIL_H_
#define IREE_HAL_VULKAN_EXTENSIBILITY_UTIL_H_

#include <vulkan/vulkan.h>

#include <vector>

#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/hal/vulkan/dynamic_symbols.h"

namespace iree {
namespace hal {
namespace vulkan {

// Describes required and optional extensibility points.
struct ExtensibilitySpec {
  // A list of required and optional layers.
  std::vector<const char*> required_layers;
  std::vector<const char*> optional_layers;

  // A list of required and optional extensions.
  // Prefer using the _EXTENSION_NAME macros to make tracking easier (such as
  // 'VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME').
  std::vector<const char*> required_extensions;
  std::vector<const char*> optional_extensions;
};

// Returns a list of layer names available for instances.
// Fails if any required_layers are unavailable.
StatusOr<std::vector<const char*>> MatchAvailableInstanceLayers(
    const ExtensibilitySpec& extensibility_spec, const DynamicSymbols& syms);

// Returns a list of extension names available for instances.
// Fails if any required_extensions are unavailable.
StatusOr<std::vector<const char*>> MatchAvailableInstanceExtensions(
    const ExtensibilitySpec& extensibility_spec, const DynamicSymbols& syms);

// Returns a list of extension names available for the given |physical_device|.
// Fails if any required_extensions are unavailable.
StatusOr<std::vector<const char*>> MatchAvailableDeviceExtensions(
    VkPhysicalDevice physical_device,
    const ExtensibilitySpec& extensibility_spec, const DynamicSymbols& syms);

// Bits for enabled instance extensions.
// We must use this to query support instead of just detecting symbol names as
// ICDs will resolve the functions sometimes even if they don't support the
// extension (or we didn't ask for it to be enabled).
struct InstanceExtensions {
  // VK_EXT_debug_report is enabled and a callback is regsitered.
  // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/chap44.html#VK_EXT_debug_report
  bool debug_report : 1;

  // VK_EXT_debug_utils is enabled and a debug messenger is registered.
  // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/chap44.html#VK_EXT_debug_utils
  bool debug_utils : 1;
};

// Returns a bitfield with all of the provided extension names.
InstanceExtensions PopulateEnabledInstanceExtensions(
    absl::Span<const char* const> extension_names);

// Bits for enabled device extensions.
// We must use this to query support instead of just detecting symbol names as
// ICDs will resolve the functions sometimes even if they don't support the
// extension (or we didn't ask for it to be enabled).
struct DeviceExtensions {
  // VK_KHR_push_descriptor is enabled and vkCmdPushDescriptorSetKHR is valid.
  bool push_descriptors : 1;
};

// Returns a bitfield with all of the provided extension names.
DeviceExtensions PopulateEnabledDeviceExtensions(
    absl::Span<const char* const> extension_names);

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_EXTENSIBILITY_UTIL_H_
