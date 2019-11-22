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

// Bitfield that defines Vulkan extensibility options to enable, if available.
typedef enum {
  // Enables VK_EXT_debug_utils, records markers, and logs errors.
  IREE_HAL_VULKAN_ENABLE_DEBUG_UTILS = 1 << 0,

  // Enables use of vkCmdPushDescriptorSetKHR.
  IREE_HAL_VULKAN_ENABLE_PUSH_DESCRIPTORS = 1 << 1,
} iree_hal_vulkan_extensibility_options_t;

// Vulkan driver creation options.
typedef struct {
  // Vulkan version that will be requested.
  // Driver creation will fail if the required version is not available.
  uint32_t api_version = VK_API_VERSION_1_0;

  // Vulkan options to request.
  iree_hal_vulkan_extensibility_options_t extensibility_options;
} iree_hal_vulkan_driver_options_t;

// A set of queues within a specific queue family on a VkDevice.
typedef struct {
  // The index of a particular queue family on a VkPhysicalDevice, as described
  // by vkGetPhysicalDeviceQueueFamilyProperties.
  uint32_t queue_family_index;

  // Bitfield of queue indices within the queue family at |queue_family_index|.
  uint64_t queue_indices;
} iree_hal_vulkan_queue_set_t;

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
// iree::hal::vulkan Extensibility Util
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Gets the names of the Vulkan instance extensions needed to create a driver
// with |iree_hal_vulkan_driver_create_using_instance| and |options|.
//
// |out_count| is a pointer to an integer related to the number of extensions.
// |out_extensions| is either nullptr or a pointer to an array to be filled with
// the required Vulkan driver extensions.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_get_required_instance_extensions(
    iree_hal_vulkan_extensibility_options_t options, uint32_t* count,
    const char** out_extensions);

// Gets the names of optional Vulkan instance extensions requested for creating
// a driver with |iree_hal_vulkan_driver_create_using_instance| and |options|.
//
// |out_count| is a pointer to an integer related to the number of extensions.
// |out_extensions| is either nullptr or a pointer to an array to be filled with
// the optional Vulkan instance extensions.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_get_optional_instance_extensions(
    iree_hal_vulkan_extensibility_options_t options, uint32_t* count,
    const char** out_extensions);

// Gets the names of the Vulkan device extensions needed to create a device
// with |iree_hal_vulkan_driver_create_device| and |options|.
//
// |out_count| is a pointer to an integer related to the number of extensions.
// |out_extensions| is either nullptr or a pointer to an array to be filled with
// the required Vulkan device extensions.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_get_required_device_extensions(
    iree_hal_vulkan_extensibility_options_t options, uint32_t* count,
    const char** out_extensions);

// Gets the names of optional the Vulkan device extensions requested for
// creating a device with |iree_hal_vulkan_driver_create_device| and |options|.
//
// |out_count| is a pointer to an integer related to the number of extensions.
// |out_extensions| is either nullptr or a pointer to an array to be filled with
// the optional Vulkan device extensions.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_get_optional_device_extensions(
    iree_hal_vulkan_extensibility_options_t options, uint32_t* count,
    const char** out_extensions);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::hal::vulkan::VulkanDriver
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// TODO(scotttodd): Allow applications to provide their own allocators here

// Creates a Vulkan HAL driver that manages its own VkInstance.
//
// |out_driver| must be released by the caller (see |iree_hal_driver_release|).
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_driver_create(
    iree_hal_vulkan_driver_options_t options, iree_hal_vulkan_syms_t* syms,
    iree_hal_driver_t** out_driver);

// Creates a Vulkan HAL driver that shares an existing VkInstance.
//
// |instance| is expected to have been created with all extensions returned by
// |iree_hal_vulkan_get_required_instance_extensions| using |options| enabled.
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

// Creates a Vulkan HAL device using |driver| that shares an existing VkDevice.
//
// HAL devices created in this way may share Vulkan resources and synchronize
// within the same physical VkPhysicalDevice and logical VkDevice directly.
//
// |logical_device| is expected to have been created with all extensions
// returned by |iree_hal_vulkan_get_required_device_extensions| using the
// options provided during driver creation.
//
// The device will schedule commands against the queues in
// |compute_queue_set| and (if set) |transfer_queue_set|.
//
// Applications may choose how these queues are created and selected in order
// to control how commands submitted by this device are prioritized and
// scheduled. For example, a low priority queue could be provided to one IREE
// device for background processing or a high priority queue could be provided
// for latency-sensitive processing.
//
// Dedicated compute queues (no graphics capabilities) are preferred within
// |compute_queue_set|, if they are available.
// Similarly, dedicated transfer queues (no compute or graphics) are preferred
// within |transfer_queue_set|.
// The queues may be the same.
//
// |out_device| must be released by the caller (see |iree_hal_device_release|).
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_create_device(
    iree_hal_driver_t* driver, VkPhysicalDevice physical_device,
    VkDevice logical_device, iree_hal_vulkan_queue_set_t compute_queue_set,
    iree_hal_vulkan_queue_set_t transfer_queue_set,
    iree_hal_device_t** out_device);

#endif  // IREE_API_NO_PROTOTYPES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_API_H_
