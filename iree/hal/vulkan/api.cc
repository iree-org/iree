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

#include "iree/hal/vulkan/api.h"

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/vulkan_device.h"
#include "iree/hal/vulkan/vulkan_driver.h"

namespace iree {
namespace hal {
namespace vulkan {

//===----------------------------------------------------------------------===//
// iree::hal::vulkan::DynamicSymbols
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_syms_create(
    void* vkGetInstanceProcAddr_fn, iree_hal_vulkan_syms_t** out_syms) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_syms_create");
  IREE_ASSERT_ARGUMENT(out_syms);
  *out_syms = nullptr;

  IREE_ASSIGN_OR_RETURN(
      auto syms, DynamicSymbols::Create([&vkGetInstanceProcAddr_fn](
                                            const char* function_name) {
        // Only resolve vkGetInstanceProcAddr, rely on syms->LoadFromInstance()
        // and/or syms->LoadFromDevice() for further loading.
        std::string fn = "vkGetInstanceProcAddr";
        if (strncmp(function_name, fn.data(), fn.size()) == 0) {
          return reinterpret_cast<PFN_vkVoidFunction>(vkGetInstanceProcAddr_fn);
        }
        return reinterpret_cast<PFN_vkVoidFunction>(NULL);
      }));

  *out_syms = reinterpret_cast<iree_hal_vulkan_syms_t*>(syms.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_syms_create_from_system_loader(
    iree_hal_vulkan_syms_t** out_syms) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_syms_create_from_system_loader");
  IREE_ASSERT_ARGUMENT(out_syms);
  *out_syms = nullptr;

  IREE_ASSIGN_OR_RETURN(auto syms, DynamicSymbols::CreateFromSystemLoader());
  *out_syms = reinterpret_cast<iree_hal_vulkan_syms_t*>(syms.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_vulkan_syms_release(iree_hal_vulkan_syms_t* syms) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_syms_release");
  IREE_ASSERT_ARGUMENT(syms);
  auto* handle = reinterpret_cast<DynamicSymbols*>(syms);
  handle->ReleaseReference();
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::vulkan Extensibility Util
//===----------------------------------------------------------------------===//

namespace {

ExtensibilitySpec GetInstanceExtensibilitySpec(
    const iree_hal_vulkan_features_t& features) {
  ExtensibilitySpec spec;

  // Multiple extensions depend on VK_KHR_get_physical_device_properties2.
  // This extension was deprecated in Vulkan 1.1 as its functionality was
  // promoted to core, so we list it as optional even though we require it.
  spec.optional_extensions.push_back(
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  if (features & IREE_HAL_VULKAN_ENABLE_VALIDATION_LAYERS) {
    spec.optional_layers.push_back("VK_LAYER_LUNARG_standard_validation");
  }

  if (features & IREE_HAL_VULKAN_ENABLE_DEBUG_UTILS) {
    spec.optional_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  // Polyfill layer - enable if present.
  spec.optional_layers.push_back("VK_LAYER_KHRONOS_timeline_semaphore");

  return spec;
}

ExtensibilitySpec GetDeviceExtensibilitySpec(
    const iree_hal_vulkan_features_t& features) {
  ExtensibilitySpec spec;

  // REQUIRED: these are required extensions that must be present for IREE to
  // work (such as those relied upon by SPIR-V kernels, etc).
  spec.required_extensions.push_back(
      VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
  // Timeline semaphore support is required.
  spec.required_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);

  if (features & IREE_HAL_VULKAN_ENABLE_PUSH_DESCRIPTORS) {
    spec.optional_extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  }

  return spec;
}

}  // namespace

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_get_extensions(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features, iree_host_size_t extensions_capacity,
    const char** out_extensions, iree_host_size_t* out_extensions_count) {
  IREE_ASSERT_ARGUMENT(out_extensions_count);
  *out_extensions_count = 0;

  bool is_instance = extensibility_set & IREE_HAL_VULKAN_INSTANCE_BIT;
  bool is_required = extensibility_set & IREE_HAL_VULKAN_REQUIRED_BIT;

  ExtensibilitySpec spec = is_instance ? GetInstanceExtensibilitySpec(features)
                                       : GetDeviceExtensibilitySpec(features);
  *out_extensions_count = is_required ? spec.required_extensions.size()
                                      : spec.optional_extensions.size();

  // Return early if only querying number of extensions in this configuration.
  if (!out_extensions) {
    return iree_ok_status();
  }

  if (extensions_capacity < *out_extensions_count) {
    // Not an error; just a size query.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  const std::vector<const char*>& extensions =
      is_required ? spec.required_extensions : spec.optional_extensions;
  for (int i = 0; i < extensions.size(); ++i) {
    out_extensions[i] = extensions[i];
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_get_layers(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features, iree_host_size_t layers_capacity,
    const char** out_layers, iree_host_size_t* out_layers_count) {
  IREE_ASSERT_ARGUMENT(out_layers_count);
  *out_layers_count = 0;

  // Device layers are deprecated and unsupported here.
  if (!(extensibility_set & IREE_HAL_VULKAN_INSTANCE_BIT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device layers are deprecated in Vulkan");
  }

  bool is_required = extensibility_set & IREE_HAL_VULKAN_REQUIRED_BIT;

  ExtensibilitySpec spec = GetInstanceExtensibilitySpec(features);
  *out_layers_count =
      is_required ? spec.required_layers.size() : spec.optional_layers.size();

  // Return early if only querying number of layers in this configuration.
  if (!out_layers) {
    return iree_ok_status();
  }

  if (layers_capacity < *out_layers_count) {
    // Not an error; just a size query.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  const std::vector<const char*>& layers =
      is_required ? spec.required_layers : spec.optional_layers;
  for (int i = 0; i < layers.size(); ++i) {
    out_layers[i] = layers[i];
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::vulkan::VulkanDriver
//===----------------------------------------------------------------------===//

namespace {

VulkanDriver::Options ConvertDriverOptions(
    iree_hal_vulkan_driver_options_t options) {
  VulkanDriver::Options driver_options;
  driver_options.api_version = options.api_version;
  driver_options.instance_extensibility =
      GetInstanceExtensibilitySpec(options.features);
  driver_options.device_extensibility =
      GetDeviceExtensibilitySpec(options.features);
  return driver_options;
}

}  // namespace

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_driver_create(
    iree_hal_vulkan_driver_options_t options, iree_hal_vulkan_syms_t* syms,
    iree_hal_driver_t** out_driver) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_driver_create");
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = nullptr;

  IREE_ASSIGN_OR_RETURN(
      auto driver,
      VulkanDriver::Create(ConvertDriverOptions(options),
                           add_ref(reinterpret_cast<DynamicSymbols*>(syms))));
  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_create_using_instance(
    iree_hal_vulkan_driver_options_t options, iree_hal_vulkan_syms_t* syms,
    VkInstance instance, iree_hal_driver_t** out_driver) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_driver_create_using_instance");
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = nullptr;

  IREE_ASSIGN_OR_RETURN(
      auto driver,
      VulkanDriver::CreateUsingInstance(
          ConvertDriverOptions(options),
          add_ref(reinterpret_cast<DynamicSymbols*>(syms)), instance));
  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_create_default_device(iree_hal_driver_t* driver,
                                             iree_hal_device_t** out_device) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_driver_create_default_device");
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  auto* handle = reinterpret_cast<VulkanDriver*>(driver);

  LOG(INFO) << "Enumerating available Vulkan devices...";
  IREE_ASSIGN_OR_RETURN(auto available_devices,
                        handle->EnumerateAvailableDevices());
  for (const auto& device_info : available_devices) {
    LOG(INFO) << "  Device: " << device_info.name();
  }
  LOG(INFO) << "Creating default device...";
  IREE_ASSIGN_OR_RETURN(auto device, handle->CreateDefaultDevice());
  LOG(INFO) << "Successfully created device '" << device->info().name() << "'";

  *out_device = reinterpret_cast<iree_hal_device_t*>(device.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_driver_wrap_device(
    iree_hal_driver_t* driver, VkPhysicalDevice physical_device,
    VkDevice logical_device, iree_hal_vulkan_queue_set_t compute_queue_set,
    iree_hal_vulkan_queue_set_t transfer_queue_set,
    iree_hal_device_t** out_device) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_driver_create_device");
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  auto* handle = reinterpret_cast<VulkanDriver*>(driver);

  LOG(INFO) << "Creating VulkanDevice...";
  QueueSet compute_qs;
  compute_qs.queue_family_index = compute_queue_set.queue_family_index;
  compute_qs.queue_indices = compute_queue_set.queue_indices;
  QueueSet transfer_qs;
  transfer_qs.queue_family_index = transfer_queue_set.queue_family_index;
  transfer_qs.queue_indices = transfer_queue_set.queue_indices;
  IREE_ASSIGN_OR_RETURN(auto device,
                        handle->WrapDevice(physical_device, logical_device,
                                           compute_qs, transfer_qs));
  LOG(INFO) << "Successfully created device '" << device->info().name() << "'";

  *out_device = reinterpret_cast<iree_hal_device_t*>(device.release());

  return iree_ok_status();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
