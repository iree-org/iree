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
// iree::hal::vulkan::VulkanDriver
//===----------------------------------------------------------------------===//

VulkanDriver::Options ConvertDriverOptions(
    iree_hal_vulkan_driver_options_t options) {
  VulkanDriver::Options driver_options;
  driver_options.api_version = options.api_version;

  // TODO: validation layers have bugs when using VK_EXT_debug_report, so if the
  // user requested that we force them off with a warning. Prefer using
  // VK_EXT_debug_utils when available.
  if ((options.extensibility_options & IREE_HAL_VULKAN_ENABLE_DEBUG_REPORT) &&
      (options.extensibility_options &
       IREE_HAL_VULKAN_ENABLE_VALIDATION_LAYERS)) {
    LOG(WARNING) << "VK_EXT_debug_report has issues with modern validation "
                    "layers; disabling validation";
    options.extensibility_options =
        static_cast<iree_hal_vulkan_extensibility_options_t>(
            options.extensibility_options &
            ~IREE_HAL_VULKAN_ENABLE_VALIDATION_LAYERS);
  }

  // REQUIRED: these are required extensions that must be present for IREE to
  // work (such as those relied upon by SPIR-V kernels, etc).
  driver_options.device_extensibility.required_extensions.push_back(
      VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);

  if (options.extensibility_options &
      IREE_HAL_VULKAN_ENABLE_VALIDATION_LAYERS) {
    driver_options.instance_extensibility.optional_layers.push_back(
        "VK_LAYER_LUNARG_standard_validation");
  }

  if (options.extensibility_options & IREE_HAL_VULKAN_ENABLE_DEBUG_REPORT) {
    driver_options.instance_extensibility.optional_extensions.push_back(
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
  }
  if (options.extensibility_options & IREE_HAL_VULKAN_ENABLE_DEBUG_UTILS) {
    driver_options.instance_extensibility.optional_extensions.push_back(
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  if (options.extensibility_options & IREE_HAL_VULKAN_ENABLE_PUSH_DESCRIPTORS) {
    driver_options.instance_extensibility.optional_extensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    driver_options.device_extensibility.optional_extensions.push_back(
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  }

  return driver_options;
}

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

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
