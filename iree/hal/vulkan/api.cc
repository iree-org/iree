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
#include "iree/base/api_util.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/queues_info.h"
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
  if (!out_syms) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_syms = nullptr;

  IREE_API_ASSIGN_OR_RETURN(
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
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_syms_create_from_system_loader(
    iree_hal_vulkan_syms_t** out_syms) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_syms_create_from_system_loader");
  if (!out_syms) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_syms = nullptr;

  IREE_API_ASSIGN_OR_RETURN(auto syms,
                            DynamicSymbols::CreateFromSystemLoader());
  *out_syms = reinterpret_cast<iree_hal_vulkan_syms_t*>(syms.release());
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_vulkan_syms_release(iree_hal_vulkan_syms_t* syms) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_syms_release");
  auto* handle = reinterpret_cast<DynamicSymbols*>(syms);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::vulkan Extensibility Util
//===----------------------------------------------------------------------===//

namespace {

ExtensibilitySpec GetInstanceExtensibilitySpec(
    const iree_hal_vulkan_extensibility_options_t& options) {
  ExtensibilitySpec spec;

  if (options & IREE_HAL_VULKAN_ENABLE_DEBUG_UTILS) {
    spec.optional_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  if (options & IREE_HAL_VULKAN_ENABLE_PUSH_DESCRIPTORS) {
    spec.optional_extensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  }

  return spec;
}

ExtensibilitySpec GetDeviceExtensibilitySpec(
    const iree_hal_vulkan_extensibility_options_t& options) {
  ExtensibilitySpec spec;

  // REQUIRED: these are required extensions that must be present for IREE to
  // work (such as those relied upon by SPIR-V kernels, etc).
  spec.required_extensions.push_back(
      VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);

  if (options & IREE_HAL_VULKAN_ENABLE_PUSH_DESCRIPTORS) {
    spec.optional_extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  }

  return spec;
}

}  // namespace

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_get_required_instance_extensions(
    iree_hal_vulkan_extensibility_options_t options, uint32_t* count,
    const char** out_extensions) {
  if (!count) return IREE_STATUS_INVALID_ARGUMENT;

  ExtensibilitySpec spec = GetInstanceExtensibilitySpec(options);
  *count = spec.required_extensions.size();
  if (!out_extensions) return IREE_STATUS_OK;

  for (int i = 0; i < spec.required_extensions.size(); ++i) {
    out_extensions[i] = spec.required_extensions[i];
  }
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_get_optional_instance_extensions(
    iree_hal_vulkan_extensibility_options_t options, uint32_t* count,
    const char** out_extensions) {
  if (!count) return IREE_STATUS_INVALID_ARGUMENT;

  ExtensibilitySpec spec = GetInstanceExtensibilitySpec(options);
  *count = spec.optional_extensions.size();
  if (!out_extensions) return IREE_STATUS_OK;

  for (int i = 0; i < spec.optional_extensions.size(); ++i) {
    out_extensions[i] = spec.optional_extensions[i];
  }
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_get_required_device_extensions(
    iree_hal_vulkan_extensibility_options_t options, uint32_t* count,
    const char** out_extensions) {
  if (!count) return IREE_STATUS_INVALID_ARGUMENT;

  ExtensibilitySpec spec = GetDeviceExtensibilitySpec(options);
  *count = spec.required_extensions.size();
  if (!out_extensions) return IREE_STATUS_OK;

  for (int i = 0; i < spec.required_extensions.size(); ++i) {
    out_extensions[i] = spec.required_extensions[i];
  }
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_get_optional_device_extensions(
    iree_hal_vulkan_extensibility_options_t options, uint32_t* count,
    const char** out_extensions) {
  if (!count) return IREE_STATUS_INVALID_ARGUMENT;

  ExtensibilitySpec spec = GetDeviceExtensibilitySpec(options);
  *count = spec.optional_extensions.size();
  if (!out_extensions) return IREE_STATUS_OK;

  for (int i = 0; i < spec.optional_extensions.size(); ++i) {
    out_extensions[i] = spec.optional_extensions[i];
  }
  return IREE_STATUS_OK;
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
      GetInstanceExtensibilitySpec(options.extensibility_options);
  driver_options.device_extensibility =
      GetDeviceExtensibilitySpec(options.extensibility_options);
  return driver_options;
}

}  // namespace

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_driver_create(
    iree_hal_vulkan_driver_options_t options, iree_hal_vulkan_syms_t* syms,
    iree_hal_driver_t** out_driver) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_driver_create");
  if (!out_driver) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_driver = nullptr;

  IREE_API_ASSIGN_OR_RETURN(
      auto driver,
      VulkanDriver::Create(ConvertDriverOptions(options),
                           add_ref(reinterpret_cast<DynamicSymbols*>(syms))));
  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver.release());
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_create_using_instance(
    iree_hal_vulkan_driver_options_t options, iree_hal_vulkan_syms_t* syms,
    VkInstance instance, iree_hal_driver_t** out_driver) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_driver_create_using_instance");
  if (!out_driver) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_driver = nullptr;

  IREE_API_ASSIGN_OR_RETURN(
      auto driver,
      VulkanDriver::CreateUsingInstance(
          ConvertDriverOptions(options),
          add_ref(reinterpret_cast<DynamicSymbols*>(syms)), instance));
  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver.release());
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_create_default_device(iree_hal_driver_t* driver,
                                             iree_hal_device_t** out_device) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_driver_create_default_device");
  if (!out_device) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_device = nullptr;

  auto* handle = reinterpret_cast<VulkanDriver*>(driver);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  LOG(INFO) << "Enumerating available Vulkan devices...";
  IREE_API_ASSIGN_OR_RETURN(auto available_devices,
                            handle->EnumerateAvailableDevices());
  for (const auto& device_info : available_devices) {
    LOG(INFO) << "  Device: " << device_info.name();
  }
  LOG(INFO) << "Creating default device...";
  IREE_API_ASSIGN_OR_RETURN(auto device, handle->CreateDefaultDevice());
  LOG(INFO) << "Successfully created device '" << device->info().name() << "'";

  *out_device = reinterpret_cast<iree_hal_device_t*>(device.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_create_device(
    iree_hal_driver_t* driver, VkPhysicalDevice physical_device,
    VkDevice logical_device, iree_hal_vulkan_queues_info_t* compute_queues_info,
    iree_hal_vulkan_queues_info_t* transfer_queues_info,
    iree_hal_device_t** out_device) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_driver_create_device");
  if (!out_device) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_device = nullptr;

  auto* handle = reinterpret_cast<VulkanDriver*>(driver);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  LOG(INFO) << "Creating VulkanDevice...";
  IREE_API_ASSIGN_OR_RETURN(
      auto device,
      handle->CreateDevice(
          physical_device, logical_device,
          reinterpret_cast<const QueuesInfo*>(compute_queues_info),
          reinterpret_cast<const QueuesInfo*>(transfer_queues_info)));
  LOG(INFO) << "Successfully created device '" << device->info().name() << "'";

  *out_device = reinterpret_cast<iree_hal_device_t*>(device.release());

  return IREE_STATUS_OK;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
