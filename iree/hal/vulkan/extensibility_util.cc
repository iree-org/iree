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

#include "iree/hal/vulkan/extensibility_util.h"

#include "iree/base/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

namespace {

StatusOr<std::vector<const char*>> MatchAvailableLayers(
    absl::Span<const char* const> required_layers,
    absl::Span<const char* const> optional_layers,
    absl::Span<const VkLayerProperties> properties) {
  IREE_TRACE_SCOPE0("MatchAvailableLayers");

  std::vector<const char*> enabled_layers;
  enabled_layers.reserve(required_layers.size() + optional_layers.size());

  for (const char* layer_name : required_layers) {
    bool found = false;
    for (const auto& layer_properties : properties) {
      if (std::strcmp(layer_name, layer_properties.layerName) == 0) {
        VLOG(1) << "Enabling required layer: " << layer_name;
        found = true;
        enabled_layers.push_back(layer_name);
        break;
      }
    }
    if (!found) {
      return UnavailableErrorBuilder(IREE_LOC)
             << "Required layer " << layer_name << " not available";
    }
  }

  for (const char* layer_name : optional_layers) {
    bool found = false;
    for (const auto& layer_properties : properties) {
      if (std::strcmp(layer_name, layer_properties.layerName) == 0) {
        VLOG(1) << "Enabling optional layer: " << layer_name;
        found = true;
        enabled_layers.push_back(layer_name);
        break;
      }
    }
    if (!found) {
      VLOG(1) << "Optional layer " << layer_name << " not available";
    }
  }

  return enabled_layers;
}

StatusOr<std::vector<const char*>> MatchAvailableExtensions(
    absl::Span<const char* const> required_extensions,
    absl::Span<const char* const> optional_extensions,
    absl::Span<const VkExtensionProperties> properties) {
  IREE_TRACE_SCOPE0("MatchAvailableExtensions");

  std::vector<const char*> enabled_extensions;
  enabled_extensions.reserve(required_extensions.size() +
                             optional_extensions.size());

  for (const char* extension_name : required_extensions) {
    bool found = false;
    for (const auto& extension_properties : properties) {
      if (std::strcmp(extension_name, extension_properties.extensionName) ==
          0) {
        VLOG(1) << "Enabling required extension: " << extension_name;
        found = true;
        enabled_extensions.push_back(extension_name);
        break;
      }
    }
    if (!found) {
      return UnavailableErrorBuilder(IREE_LOC)
             << "Required extension " << extension_name << " not available";
    }
  }

  for (const char* extension_name : optional_extensions) {
    bool found = false;
    for (const auto& extension_properties : properties) {
      if (std::strcmp(extension_name, extension_properties.extensionName) ==
          0) {
        VLOG(1) << "Enabling optional extension: " << extension_name;
        found = true;
        enabled_extensions.push_back(extension_name);
        break;
      }
    }
    if (!found) {
      VLOG(1) << "Optional extension " << extension_name << " not available";
    }
  }

  return enabled_extensions;
}

}  // namespace

StatusOr<std::vector<const char*>> MatchAvailableInstanceLayers(
    const ExtensibilitySpec& extensibility_spec, const DynamicSymbols& syms) {
  uint32_t layer_property_count = 0;
  VK_RETURN_IF_ERROR(
      syms.vkEnumerateInstanceLayerProperties(&layer_property_count, nullptr));
  std::vector<VkLayerProperties> layer_properties(layer_property_count);
  VK_RETURN_IF_ERROR(syms.vkEnumerateInstanceLayerProperties(
      &layer_property_count, layer_properties.data()));
  IREE_ASSIGN_OR_RETURN(auto enabled_layers,
                        MatchAvailableLayers(extensibility_spec.required_layers,
                                             extensibility_spec.optional_layers,
                                             layer_properties),
                        _ << "Unable to find all required instance layers");
  return enabled_layers;
}

StatusOr<std::vector<const char*>> MatchAvailableInstanceExtensions(
    const ExtensibilitySpec& extensibility_spec, const DynamicSymbols& syms) {
  uint32_t extension_property_count = 0;
  // Warning: leak checks remain disabled if an error is returned.
  IREE_DISABLE_LEAK_CHECKS();
  VK_RETURN_IF_ERROR(syms.vkEnumerateInstanceExtensionProperties(
      nullptr, &extension_property_count, nullptr));
  std::vector<VkExtensionProperties> extension_properties(
      extension_property_count);
  VK_RETURN_IF_ERROR(syms.vkEnumerateInstanceExtensionProperties(
      nullptr, &extension_property_count, extension_properties.data()));
  IREE_ASSIGN_OR_RETURN(
      auto enabled_extensions,
      MatchAvailableExtensions(extensibility_spec.required_extensions,
                               extensibility_spec.optional_extensions,
                               extension_properties),
      _ << "Unable to find all required instance extensions");
  IREE_ENABLE_LEAK_CHECKS();
  return enabled_extensions;
}

StatusOr<std::vector<const char*>> MatchAvailableDeviceExtensions(
    VkPhysicalDevice physical_device,
    const ExtensibilitySpec& extensibility_spec, const DynamicSymbols& syms) {
  uint32_t extension_property_count = 0;
  VK_RETURN_IF_ERROR(syms.vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &extension_property_count, nullptr));
  std::vector<VkExtensionProperties> extension_properties(
      extension_property_count);
  VK_RETURN_IF_ERROR(syms.vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &extension_property_count,
      extension_properties.data()));
  IREE_ASSIGN_OR_RETURN(
      auto enabled_extensions,
      MatchAvailableExtensions(extensibility_spec.required_extensions,
                               extensibility_spec.optional_extensions,
                               extension_properties),
      _ << "Unable to find all required device extensions");
  return enabled_extensions;
}

InstanceExtensions PopulateEnabledInstanceExtensions(
    absl::Span<const char* const> extension_names) {
  InstanceExtensions extensions = {0};
  for (const char* extension_name : extension_names) {
    if (std::strcmp(extension_name, VK_EXT_DEBUG_REPORT_EXTENSION_NAME) == 0) {
      extensions.debug_report = true;
    } else if (std::strcmp(extension_name, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) ==
               0) {
      extensions.debug_utils = true;
    }
  }
  return extensions;
}

DeviceExtensions PopulateEnabledDeviceExtensions(
    absl::Span<const char* const> extension_names) {
  DeviceExtensions extensions = {0};
  for (const char* extension_name : extension_names) {
    if (std::strcmp(extension_name, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) ==
        0) {
      extensions.push_descriptors = true;
    }
  }
  return extensions;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
