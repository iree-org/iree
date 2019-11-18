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

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_VULKAN_API_H_
#define IREE_HAL_VULKAN_API_H_

#include <vulkan/vulkan.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Bitfield that defines Vulkan extensibility options to enable.
typedef enum {
  // Enables standard Vulkan validation layers.
  IREE_HAL_VULKAN_ENABLE_VALIDATION_LAYERS = 0,

  // Enables VK_EXT_debug_utils, records markers, and logs errors.
  IREE_HAL_VULKAN_ENABLE_DEBUG_UTILS = 1 << 0,

  // Enables VK_EXT_debug_report and logs errors.
  IREE_HAL_VULKAN_ENABLE_DEBUG_REPORT = 1 << 1,

  // Enables use of vkCmdPushDescriptorSetKHR, if available.
  IREE_HAL_VULKAN_ENABLE_PUSH_DESCRIPTORS = 1 << 2,
} iree_hal_vulkan_extensibility_options_t;

// Vulkan driver creation options.
typedef struct {
  // Vulkan version that will be requested.
  // Driver creation will fail if the required version is not available.
  uint32_t api_version = VK_API_VERSION_1_0;

  // Vulkan options to request.
  iree_hal_vulkan_extensibility_options_t extensibility_options;
} iree_hal_vulkan_driver_options_t;

typedef struct iree_hal_vulkan_syms iree_hal_vulkan_syms_t;

//===----------------------------------------------------------------------===//
// iree::hal::vulkan::DynamicSymbols
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Loads Vulkan functions by invoking |vkGetInstanceProcAddr|.
//
// |vkGetInstanceProcAddr| can be obtained in whatever way suites the calling
// application, such as via `dlsym` or `GetProcAddress` when dynamically
// loading Vulkan, or `reinterpret_cast<void*>(&vkGetInstanceProcAddr)` when
// statically linking Vulkan.
//
// |out_syms| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_syms_create(
    void* vkGetInstanceProcAddr_fn, iree_hal_vulkan_syms_t** out_syms);

// Loads Vulkan functions from the Vulkan loader.
// This will look for a Vulkan loader on the system (like libvulkan.so) and
// dlsym the functions from that.
//
// |out_syms| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_syms_create_from_system_loader(
    iree_hal_vulkan_syms_t** out_syms);

// Releases the given |syms| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_syms_release(iree_hal_vulkan_syms_t* syms);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::hal::vulkan::VulkanDriver
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Creates a Vulkan HAL driver that manages its own VkInstance.
//
// |out_driver| must be released by the caller (see |iree_hal_driver_release|).
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_driver_create(
    iree_hal_vulkan_driver_options_t options, iree_hal_vulkan_syms_t* syms,
    iree_hal_driver_t** out_driver);

// Creates a Vulkan HAL driver that shares an existing VkInstance.
//
// An IREE_STATUS_INVALID_ARGUMENT error will be returned if the requested
// |options| are not compatible with the existing |instance|.
//
// |instance| must remain valid for the life of |out_driver| and |out_driver|
// itself must be released by the caller (see |iree_hal_driver_release|).
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_create_using_instance(
    iree_hal_vulkan_driver_options_t options, iree_hal_vulkan_syms_t* syms,
    VkInstance instance, iree_hal_driver_t** out_driver);

// Creates the default Vulkan HAL device using |driver| that manages its own
// VkPhysicalDevice/VkDevice.
//
// |out_device| must be released by the caller (see |iree_hal_device_release|).
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_create_default_device(iree_hal_driver_t* driver,
                                             iree_hal_device_t** out_device);

#endif  // IREE_API_NO_PROTOTYPES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_API_H_
