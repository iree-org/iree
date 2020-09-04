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

// Describes the type of a set of Vulkan extensions.
typedef enum {
  IREE_HAL_VULKAN_REQUIRED_BIT = 1 << 0,
  IREE_HAL_VULKAN_INSTANCE_BIT = 1 << 1,

  // A set of required instance extension names.
  IREE_HAL_VULKAN_INSTANCE_REQUIRED =
      IREE_HAL_VULKAN_INSTANCE_BIT | IREE_HAL_VULKAN_REQUIRED_BIT,
  // A set of optional instance extension names.
  IREE_HAL_VULKAN_INSTANCE_OPTIONAL = IREE_HAL_VULKAN_INSTANCE_BIT,
  // A set of required device extension names.
  IREE_HAL_VULKAN_DEVICE_REQUIRED = IREE_HAL_VULKAN_REQUIRED_BIT,
  // A set of optional device extension names.
  IREE_HAL_VULKAN_DEVICE_OPTIONAL = 0,
} iree_hal_vulkan_extensibility_set_t;

// Bitfield that defines sets of Vulkan features.
typedef enum {
  // Use VK_LAYER_LUNARG_standard_validation.
  IREE_HAL_VULKAN_ENABLE_VALIDATION_LAYERS = 1 << 0,

  // Use VK_EXT_debug_utils, record markers, and log errors.
  IREE_HAL_VULKAN_ENABLE_DEBUG_UTILS = 1 << 1,

  // Use vkCmdPushDescriptorSetKHR.
  IREE_HAL_VULKAN_ENABLE_PUSH_DESCRIPTORS = 1 << 2,
} iree_hal_vulkan_features_t;

// Vulkan driver creation options.
typedef struct {
  // Vulkan version that will be requested, e.g. `VK_API_VERSION_1_0`.
  // Driver creation will fail if the required version is not available.
  uint32_t api_version;

  // Vulkan features to request.
  iree_hal_vulkan_features_t features;
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

//===----------------------------------------------------------------------===//
// iree::hal::vulkan Extensibility Util
//===----------------------------------------------------------------------===//

// Gets the names of the Vulkan extensions used for a given set of |features|.
//
// Instance extensions should be enabled on VkInstances passed to
// |iree_hal_vulkan_driver_create_using_instance| and device extensions should
// be enabled on VkDevices passed to |iree_hal_vulkan_driver_wrap_device|.
//
// |extensions_capacity| defines the number of elements available in
// |out_extensions| and |out_extensions_count| will be set with the actual
// number of extensions returned. If |extensions_capacity| is too small
// IREE_STATUS_OUT_OF_RANGE will be returned with the required capacity in
// |out_extensions_count|. To only query the required capacity |out_extensions|
// may be passed as nullptr.
//
// Extension string lifetime is tied to the loader shared object or instance,
// depending on where they came from.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_get_extensions(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features, iree_host_size_t extensions_capacity,
    const char** out_extensions, iree_host_size_t* out_extensions_count);

// Gets the names of the Vulkan layers used for a given set of |features|.
//
// Instance layers should be enabled on VkInstances passed to
// |iree_hal_vulkan_driver_create_using_instance|. Device layers are deprecated
// and unsupported here.
//
// |layers_capacity| defines the number of elements available in |out_layers|
// and |out_layers_count| will be set with the actual number of layers returned.
// If |layers_capacity| is too small IREE_STATUS_OUT_OF_RANGE will be returned
// with the required capacity in |out_layers_count|. To only query the required
// capacity |out_layers| may be passed as nullptr.
//
// Layer string lifetime is tied to the loader shared object or instance,
// depending on where they came from.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_get_layers(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features, iree_host_size_t layers_capacity,
    const char** out_layers, iree_host_size_t* out_layers_count);

//===----------------------------------------------------------------------===//
// iree::hal::vulkan::VulkanDriver
//===----------------------------------------------------------------------===//

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
// |iree_hal_vulkan_get_extensions| and IREE_HAL_VULKAN_INSTANCE_REQUIRED using
// |options| enabled.
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

// Creates a Vulkan HAL device using |driver| that wraps an existing VkDevice.
//
// HAL devices created in this way may share Vulkan resources and synchronize
// within the same physical VkPhysicalDevice and logical VkDevice directly.
//
// |logical_device| is expected to have been created with all extensions
// returned by |iree_hal_vulkan_get_extensions| and
// IREE_HAL_VULKAN_DEVICE_REQUIRED using the features provided during driver
// creation.
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
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_driver_wrap_device(
    iree_hal_driver_t* driver, VkPhysicalDevice physical_device,
    VkDevice logical_device, iree_hal_vulkan_queue_set_t compute_queue_set,
    iree_hal_vulkan_queue_set_t transfer_queue_set,
    iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_API_H_
