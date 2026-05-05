// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_PHYSICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_VULKAN_PHYSICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#if !defined(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)
#define IREE_HAL_VULKAN_KHR_PORTABILITY_SUBSET_EXTENSION_NAME \
  "VK_KHR_portability_subset"
#else
#define IREE_HAL_VULKAN_KHR_PORTABILITY_SUBSET_EXTENSION_NAME \
  VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#endif  // !VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_instance_t
//===----------------------------------------------------------------------===//

// Vulkan instance wrapper used by driver-created enumeration/device paths.
typedef struct iree_hal_vulkan_instance_t {
  // Vulkan instance handle.
  VkInstance handle;

  // Instance API version requested during creation.
  uint32_t api_version;

  // Instance-level Vulkan dispatch table.
  iree_hal_vulkan_instance_syms_t syms;
} iree_hal_vulkan_instance_t;

typedef uint32_t iree_hal_vulkan_time_domain_flags_t;

typedef enum iree_hal_vulkan_time_domain_flag_bits_e {
  // No calibrated timestamp domains are available.
  IREE_HAL_VULKAN_TIME_DOMAIN_NONE = 0u,

  // VK_TIME_DOMAIN_DEVICE_EXT.
  IREE_HAL_VULKAN_TIME_DOMAIN_DEVICE = 1u << 0,

  // VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT.
  IREE_HAL_VULKAN_TIME_DOMAIN_CLOCK_MONOTONIC = 1u << 1,

  // VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT.
  IREE_HAL_VULKAN_TIME_DOMAIN_CLOCK_MONOTONIC_RAW = 1u << 2,

  // VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT.
  IREE_HAL_VULKAN_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER = 1u << 3,
} iree_hal_vulkan_time_domain_flag_bits_t;

// Creates a driver-owned Vulkan instance for enumeration or device creation.
iree_status_t iree_hal_vulkan_instance_initialize(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_vulkan_instance_t* out_instance);

// Destroys a driver-owned Vulkan instance.
void iree_hal_vulkan_instance_deinitialize(
    iree_hal_vulkan_instance_t* instance);

// Enumerates physical device handles visible to |instance|.
iree_status_t iree_hal_vulkan_enumerate_physical_device_handles(
    const iree_hal_vulkan_instance_t* instance, iree_allocator_t host_allocator,
    uint32_t* out_physical_device_count,
    VkPhysicalDevice** out_physical_devices);

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_physical_device_snapshot_t
//===----------------------------------------------------------------------===//

// Immutable snapshot of physical-device properties needed for policy checks.
typedef struct iree_hal_vulkan_physical_device_snapshot_t {
  // Physical device handle owned by the instance.
  VkPhysicalDevice handle;

  // Physical-device ordinal from vkEnumeratePhysicalDevices.
  uint32_t ordinal;

  // Base and extended device properties.
  VkPhysicalDeviceProperties2 properties2;

  // Vulkan 1.1 property set including maxMemoryAllocationSize.
  VkPhysicalDeviceVulkan11Properties properties11;

  // VK_EXT_external_memory_host properties, if the extension is available.
  VkPhysicalDeviceExternalMemoryHostPropertiesEXT
      external_memory_host_properties;

  // Stable identity properties.
  VkPhysicalDeviceIDProperties id_properties;

  // Driver properties.
  VkPhysicalDeviceDriverProperties driver_properties;

  // Subgroup operation properties.
  VkPhysicalDeviceSubgroupProperties subgroup_properties;

  // Subgroup size control properties.
  VkPhysicalDeviceSubgroupSizeControlProperties
      subgroup_size_control_properties;

  // Base and extended feature set.
  VkPhysicalDeviceFeatures2 features2;

  // Vulkan 1.2 feature set.
  VkPhysicalDeviceVulkan12Features features12;

  // Vulkan 1.3 feature set.
  VkPhysicalDeviceVulkan13Features features13;

  // Memory heap and type properties.
  VkPhysicalDeviceMemoryProperties2 memory_properties2;

  // Queue-family count.
  uint32_t queue_family_count;

  // Queue-family property list.
  VkQueueFamilyProperties2* queue_families;

  // Device extension count.
  uint32_t extension_count;

  // Device extension list.
  VkExtensionProperties* extensions;

  // Recognized device extension bits available on this physical device.
  iree_hal_vulkan_device_extensions_t available_extensions;

  // VK_EXT_calibrated_timestamps domains supported by this physical device.
  iree_hal_vulkan_time_domain_flags_t calibrated_timestamp_time_domains;
} iree_hal_vulkan_physical_device_snapshot_t;

// Captures properties, features, memory, queue, and extension data for a
// physical device.
iree_status_t iree_hal_vulkan_physical_device_snapshot_initialize(
    const iree_hal_vulkan_instance_t* instance, VkPhysicalDevice handle,
    uint32_t ordinal, iree_allocator_t host_allocator,
    iree_hal_vulkan_physical_device_snapshot_t* out_snapshot);

// Releases storage held by |snapshot|.
void iree_hal_vulkan_physical_device_snapshot_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_vulkan_physical_device_snapshot_t* snapshot);

// Returns true if |snapshot| exposes a queue family with compute capability.
bool iree_hal_vulkan_physical_device_has_compute_queue(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot);

// Returns true if |snapshot| satisfies the current Vulkan rewrite baseline.
bool iree_hal_vulkan_physical_device_supports_baseline(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot);

// Returns true if |snapshot| reports all |extension_bits|.
bool iree_hal_vulkan_physical_device_has_extension(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_device_extensions_t extension_bits);

// Appends the stable HAL device path for |snapshot| to |builder|.
iree_status_t iree_hal_vulkan_append_device_path(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_string_builder_t* builder, iree_string_view_t* out_view);

// Enumerates physical Vulkan devices visible to the loader.
iree_status_t iree_hal_vulkan_query_available_physical_devices(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* options,
    iree_allocator_t host_allocator, iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos);

// Appends detailed physical-device inventory for |device_id|.
iree_status_t iree_hal_vulkan_dump_physical_device_info(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* options,
    iree_hal_device_id_t device_id, iree_allocator_t host_allocator,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_PHYSICAL_DEVICE_H_
