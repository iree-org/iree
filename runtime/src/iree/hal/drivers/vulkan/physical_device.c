// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define VK_NO_PROTOTYPES
#include "iree/hal/drivers/vulkan/physical_device.h"

#include <stddef.h>
#include <string.h>

#include "vulkan/vulkan.h"

//===----------------------------------------------------------------------===//
// VkResult interop
//===----------------------------------------------------------------------===//

static iree_status_code_t iree_hal_vulkan_status_code(VkResult result) {
  switch (result) {
    default:
      return IREE_STATUS_UNKNOWN;
    case VK_SUCCESS:
      return IREE_STATUS_OK;
    case VK_NOT_READY:
    case VK_TIMEOUT:
    case VK_EVENT_SET:
    case VK_EVENT_RESET:
    case VK_INCOMPLETE:
      return IREE_STATUS_CANCELLED;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    case VK_ERROR_TOO_MANY_OBJECTS:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
    case VK_ERROR_INITIALIZATION_FAILED:
      return IREE_STATUS_FAILED_PRECONDITION;
    case VK_ERROR_DEVICE_LOST:
      return IREE_STATUS_DATA_LOSS;
    case VK_ERROR_MEMORY_MAP_FAILED:
      return IREE_STATUS_INTERNAL;
    case VK_ERROR_LAYER_NOT_PRESENT:
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return IREE_STATUS_NOT_FOUND;
    case VK_ERROR_FEATURE_NOT_PRESENT:
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return IREE_STATUS_UNAVAILABLE;
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return IREE_STATUS_INCOMPATIBLE;
    case VK_ERROR_FRAGMENTED_POOL:
    case VK_ERROR_FRAGMENTATION:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
    case VK_ERROR_UNKNOWN:
      return IREE_STATUS_UNKNOWN;
  }
}

static const char* iree_hal_vulkan_result_string(VkResult result) {
  switch (result) {
    default:
      return "VK_RESULT_UNKNOWN";
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_EVENT_SET:
      return "VK_EVENT_SET";
    case VK_EVENT_RESET:
      return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
      return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
      return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
      return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_UNKNOWN:
      return "VK_ERROR_UNKNOWN";
    case VK_ERROR_FRAGMENTATION:
      return "VK_ERROR_FRAGMENTATION";
  }
}

static iree_status_t iree_hal_vulkan_status_from_result(VkResult result,
                                                        const char* symbol) {
  if (result == VK_SUCCESS) return iree_ok_status();
  return iree_make_status(iree_hal_vulkan_status_code(result), "%s failed: %s",
                          symbol, iree_hal_vulkan_result_string(result));
}

//===----------------------------------------------------------------------===//
// Shared formatting utilities
//===----------------------------------------------------------------------===//

static void iree_string_builder_intern(iree_string_builder_t* builder,
                                       iree_host_size_t* intern_offset,
                                       iree_string_view_t* out_view) {
  const iree_host_size_t old_offset = *intern_offset;
  const iree_host_size_t new_offset = iree_string_builder_size(builder);
  const iree_host_size_t length = new_offset - old_offset;
  *intern_offset = new_offset;
  if (out_view != NULL) {
    *out_view = iree_make_string_view(
        iree_string_builder_buffer(builder) + old_offset, length);
  }
}

static const char* iree_hal_vulkan_device_type_string(
    VkPhysicalDeviceType device_type) {
  switch (device_type) {
    default:
      return "unknown";
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
      return "other";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "integrated-gpu";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "discrete-gpu";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "virtual-gpu";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "cpu";
  }
}

static iree_status_t iree_hal_vulkan_append_api_version(
    iree_string_builder_t* builder, uint32_t api_version) {
  return iree_string_builder_append_format(
      builder, "%u.%u.%u", VK_API_VERSION_MAJOR(api_version),
      VK_API_VERSION_MINOR(api_version), VK_API_VERSION_PATCH(api_version));
}

static iree_status_t iree_hal_vulkan_append_uuid(
    iree_string_builder_t* builder, const uint8_t uuid[VK_UUID_SIZE]) {
  static const char kHexDigits[] = "0123456789abcdef";
  char buffer[VK_UUID_SIZE * 2];
  for (iree_host_size_t i = 0; i < VK_UUID_SIZE; ++i) {
    buffer[i * 2 + 0] = kHexDigits[(uuid[i] >> 4) & 0xF];
    buffer[i * 2 + 1] = kHexDigits[uuid[i] & 0xF];
  }
  return iree_string_builder_append_string(
      builder, iree_make_string_view(buffer, sizeof(buffer)));
}

static const char* iree_hal_vulkan_bool_string(VkBool32 value) {
  return value ? "yes" : "no";
}

//===----------------------------------------------------------------------===//
// Instance creation and dispatch
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_instance_t {
  // Vulkan instance handle.
  VkInstance handle;

  // Instance API version requested during creation.
  uint32_t api_version;

  // Destroys the instance.
  PFN_vkDestroyInstance vkDestroyInstance;

  // Enumerates visible physical devices.
  PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;

  // Queries physical-device properties with pNext support.
  PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2;

  // Queries physical-device features with pNext support.
  PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2;

  // Queries physical-device memory properties with pNext support.
  PFN_vkGetPhysicalDeviceMemoryProperties2 vkGetPhysicalDeviceMemoryProperties2;

  // Queries queue-family properties with pNext support.
  PFN_vkGetPhysicalDeviceQueueFamilyProperties2
      vkGetPhysicalDeviceQueueFamilyProperties2;

  // Enumerates device extensions reported by a physical device.
  PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties;
} iree_hal_vulkan_instance_t;

static iree_status_t iree_hal_vulkan_lookup_global(
    const iree_hal_vulkan_libvulkan_t* libvulkan, const char* symbol,
    PFN_vkVoidFunction* out_fn) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(out_fn);
  *out_fn = libvulkan->vkGetInstanceProcAddr(VK_NULL_HANDLE, symbol);
  if (!*out_fn) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Vulkan loader symbol '%s' not found", symbol);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_lookup_instance(
    const iree_hal_vulkan_libvulkan_t* libvulkan, VkInstance instance,
    const char* symbol, PFN_vkVoidFunction* out_fn) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(out_fn);
  *out_fn = libvulkan->vkGetInstanceProcAddr(instance, symbol);
  if (!*out_fn) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Vulkan instance symbol '%s' not found", symbol);
  }
  return iree_ok_status();
}

static bool iree_hal_vulkan_extension_list_contains(
    uint32_t extension_count, const VkExtensionProperties* extensions,
    const char* extension_name) {
  for (uint32_t i = 0; i < extension_count; ++i) {
    if (strcmp(extension_name, extensions[i].extensionName) == 0) return true;
  }
  return false;
}

static bool iree_hal_vulkan_layer_list_contains(uint32_t layer_count,
                                                const VkLayerProperties* layers,
                                                const char* layer_name) {
  for (uint32_t i = 0; i < layer_count; ++i) {
    if (strcmp(layer_name, layers[i].layerName) == 0) return true;
  }
  return false;
}

static iree_status_t iree_hal_vulkan_enumerate_instance_extensions(
    PFN_vkEnumerateInstanceExtensionProperties
        vkEnumerateInstanceExtensionProperties,
    iree_allocator_t host_allocator, uint32_t* out_extension_count,
    VkExtensionProperties** out_extensions) {
  IREE_ASSERT_ARGUMENT(out_extension_count);
  IREE_ASSERT_ARGUMENT(out_extensions);
  *out_extension_count = 0;
  *out_extensions = NULL;

  uint32_t extension_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_status_from_result(
      vkEnumerateInstanceExtensionProperties(/*pLayerName=*/NULL,
                                             &extension_count, NULL),
      "vkEnumerateInstanceExtensionProperties"));
  if (!extension_count) return iree_ok_status();

  VkExtensionProperties* extensions = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, extension_count * sizeof(*extensions),
      (void**)&extensions));

  iree_status_t status = iree_hal_vulkan_status_from_result(
      vkEnumerateInstanceExtensionProperties(/*pLayerName=*/NULL,
                                             &extension_count, extensions),
      "vkEnumerateInstanceExtensionProperties");
  if (iree_status_is_ok(status)) {
    *out_extension_count = extension_count;
    *out_extensions = extensions;
  } else {
    iree_allocator_free(host_allocator, extensions);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_enumerate_instance_layers(
    PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties,
    iree_allocator_t host_allocator, uint32_t* out_layer_count,
    VkLayerProperties** out_layers) {
  IREE_ASSERT_ARGUMENT(out_layer_count);
  IREE_ASSERT_ARGUMENT(out_layers);
  *out_layer_count = 0;
  *out_layers = NULL;

  uint32_t layer_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_status_from_result(
      vkEnumerateInstanceLayerProperties(&layer_count, NULL),
      "vkEnumerateInstanceLayerProperties"));
  if (!layer_count) return iree_ok_status();

  VkLayerProperties* layers = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, layer_count * sizeof(*layers), (void**)&layers));

  iree_status_t status = iree_hal_vulkan_status_from_result(
      vkEnumerateInstanceLayerProperties(&layer_count, layers),
      "vkEnumerateInstanceLayerProperties");
  if (iree_status_is_ok(status)) {
    *out_layer_count = layer_count;
    *out_layers = layers;
  } else {
    iree_allocator_free(host_allocator, layers);
  }
  return status;
}

static uint32_t iree_hal_vulkan_resolve_api_version(
    const iree_hal_vulkan_driver_options_t* options) {
  return options->api_version ? options->api_version : VK_API_VERSION_1_3;
}

static iree_status_t iree_hal_vulkan_instance_initialize(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_vulkan_instance_t* out_instance) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_instance);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_instance, 0, sizeof(*out_instance));

  PFN_vkCreateInstance vkCreateInstance = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_vulkan_lookup_global(libvulkan, "vkCreateInstance",
                                    (PFN_vkVoidFunction*)&vkCreateInstance));
  PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = NULL;
  vkEnumerateInstanceVersion =
      (PFN_vkEnumerateInstanceVersion)libvulkan->vkGetInstanceProcAddr(
          VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
  PFN_vkEnumerateInstanceExtensionProperties
      vkEnumerateInstanceExtensionProperties = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_lookup_global(
              libvulkan, "vkEnumerateInstanceExtensionProperties",
              (PFN_vkVoidFunction*)&vkEnumerateInstanceExtensionProperties));
  PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties =
      NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_lookup_global(
              libvulkan, "vkEnumerateInstanceLayerProperties",
              (PFN_vkVoidFunction*)&vkEnumerateInstanceLayerProperties));

  uint32_t loader_api_version = VK_API_VERSION_1_0;
  if (vkEnumerateInstanceVersion) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_vulkan_status_from_result(
                vkEnumerateInstanceVersion(&loader_api_version),
                "vkEnumerateInstanceVersion"));
  }
  const uint32_t requested_api_version =
      iree_hal_vulkan_resolve_api_version(options);
  if (loader_api_version < requested_api_version) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan loader API version %u.%u.%u is below requested %u.%u.%u",
        VK_API_VERSION_MAJOR(loader_api_version),
        VK_API_VERSION_MINOR(loader_api_version),
        VK_API_VERSION_PATCH(loader_api_version),
        VK_API_VERSION_MAJOR(requested_api_version),
        VK_API_VERSION_MINOR(requested_api_version),
        VK_API_VERSION_PATCH(requested_api_version));
  }

  uint32_t available_extension_count = 0;
  VkExtensionProperties* available_extensions = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_enumerate_instance_extensions(
              vkEnumerateInstanceExtensionProperties, host_allocator,
              &available_extension_count, &available_extensions));

  uint32_t available_layer_count = 0;
  VkLayerProperties* available_layers = NULL;
  iree_status_t status = iree_hal_vulkan_enumerate_instance_layers(
      vkEnumerateInstanceLayerProperties, host_allocator,
      &available_layer_count, &available_layers);

  const char* enabled_extensions[8] = {0};
  uint32_t enabled_extension_count = 0;
  const char* enabled_layers[4] = {0};
  uint32_t enabled_layer_count = 0;
  VkInstanceCreateFlags instance_flags = 0;

  if (iree_status_is_ok(status) &&
      iree_hal_vulkan_extension_list_contains(
          available_extension_count, available_extensions,
          VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    enabled_extensions[enabled_extension_count++] =
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(options->requested_features,
                       IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS)) {
    if (iree_hal_vulkan_extension_list_contains(
            available_extension_count, available_extensions,
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
      enabled_extensions[enabled_extension_count++] =
          VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "requested Vulkan debug utils extension is not "
                                "available");
    }
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(options->requested_features,
                       IREE_HAL_VULKAN_FEATURE_ENABLE_VALIDATION_LAYERS)) {
    if (iree_hal_vulkan_layer_list_contains(available_layer_count,
                                            available_layers,
                                            "VK_LAYER_KHRONOS_validation")) {
      enabled_layers[enabled_layer_count++] = "VK_LAYER_KHRONOS_validation";
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "requested Vulkan validation layer is not "
                                "available");
    }
  }

  VkApplicationInfo application_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "IREE",
      .applicationVersion = 1,
      .pEngineName = "IREE",
      .engineVersion = 1,
      .apiVersion = requested_api_version,
  };
  VkInstanceCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .flags = instance_flags,
      .pApplicationInfo = &application_info,
      .enabledLayerCount = enabled_layer_count,
      .ppEnabledLayerNames = enabled_layers,
      .enabledExtensionCount = enabled_extension_count,
      .ppEnabledExtensionNames = enabled_extensions,
  };

  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_status_from_result(
        vkCreateInstance(&create_info, /*pAllocator=*/NULL,
                         &out_instance->handle),
        "vkCreateInstance");
  }

  iree_allocator_free(host_allocator, available_layers);
  iree_allocator_free(host_allocator, available_extensions);

  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_lookup_instance(
        libvulkan, out_instance->handle, "vkDestroyInstance",
        (PFN_vkVoidFunction*)&out_instance->vkDestroyInstance);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_lookup_instance(
        libvulkan, out_instance->handle, "vkEnumeratePhysicalDevices",
        (PFN_vkVoidFunction*)&out_instance->vkEnumeratePhysicalDevices);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_lookup_instance(
        libvulkan, out_instance->handle, "vkGetPhysicalDeviceProperties2",
        (PFN_vkVoidFunction*)&out_instance->vkGetPhysicalDeviceProperties2);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_lookup_instance(
        libvulkan, out_instance->handle, "vkGetPhysicalDeviceFeatures2",
        (PFN_vkVoidFunction*)&out_instance->vkGetPhysicalDeviceFeatures2);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_lookup_instance(
        libvulkan, out_instance->handle, "vkGetPhysicalDeviceMemoryProperties2",
        (PFN_vkVoidFunction*)&out_instance
            ->vkGetPhysicalDeviceMemoryProperties2);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_lookup_instance(
        libvulkan, out_instance->handle,
        "vkGetPhysicalDeviceQueueFamilyProperties2",
        (PFN_vkVoidFunction*)&out_instance
            ->vkGetPhysicalDeviceQueueFamilyProperties2);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_lookup_instance(
        libvulkan, out_instance->handle, "vkEnumerateDeviceExtensionProperties",
        (PFN_vkVoidFunction*)&out_instance
            ->vkEnumerateDeviceExtensionProperties);
  }
  if (iree_status_is_ok(status)) {
    out_instance->api_version = requested_api_version;
  }

  if (!iree_status_is_ok(status) && out_instance->handle) {
    PFN_vkDestroyInstance vkDestroyInstance =
        (PFN_vkDestroyInstance)libvulkan->vkGetInstanceProcAddr(
            out_instance->handle, "vkDestroyInstance");
    if (vkDestroyInstance) {
      vkDestroyInstance(out_instance->handle, /*pAllocator=*/NULL);
    }
    memset(out_instance, 0, sizeof(*out_instance));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_instance_deinitialize(
    iree_hal_vulkan_instance_t* instance) {
  IREE_ASSERT_ARGUMENT(instance);
  if (instance->handle && instance->vkDestroyInstance) {
    instance->vkDestroyInstance(instance->handle, /*pAllocator=*/NULL);
  }
  memset(instance, 0, sizeof(*instance));
}

//===----------------------------------------------------------------------===//
// Physical device snapshots
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_physical_device_snapshot_t {
  // Physical device handle owned by the instance.
  VkPhysicalDevice handle;

  // Physical-device ordinal from vkEnumeratePhysicalDevices.
  uint32_t ordinal;

  // Base and extended device properties.
  VkPhysicalDeviceProperties2 properties2;

  // Stable identity properties.
  VkPhysicalDeviceIDProperties id_properties;

  // Driver properties.
  VkPhysicalDeviceDriverProperties driver_properties;

  // Subgroup operation properties.
  VkPhysicalDeviceSubgroupProperties subgroup_properties;

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
} iree_hal_vulkan_physical_device_snapshot_t;

static iree_status_t iree_hal_vulkan_physical_device_snapshot_initialize(
    const iree_hal_vulkan_instance_t* instance, VkPhysicalDevice handle,
    uint32_t ordinal, iree_allocator_t host_allocator,
    iree_hal_vulkan_physical_device_snapshot_t* out_snapshot) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_snapshot);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_snapshot, 0, sizeof(*out_snapshot));
  out_snapshot->handle = handle;
  out_snapshot->ordinal = ordinal;

  out_snapshot->subgroup_properties = (VkPhysicalDeviceSubgroupProperties){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
  };
  out_snapshot->driver_properties = (VkPhysicalDeviceDriverProperties){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
      .pNext = &out_snapshot->subgroup_properties,
  };
  out_snapshot->id_properties = (VkPhysicalDeviceIDProperties){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
      .pNext = &out_snapshot->driver_properties,
  };
  out_snapshot->properties2 = (VkPhysicalDeviceProperties2){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &out_snapshot->id_properties,
  };
  instance->vkGetPhysicalDeviceProperties2(handle, &out_snapshot->properties2);

  out_snapshot->features13 = (VkPhysicalDeviceVulkan13Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };
  out_snapshot->features12 = (VkPhysicalDeviceVulkan12Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &out_snapshot->features13,
  };
  out_snapshot->features2 = (VkPhysicalDeviceFeatures2){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &out_snapshot->features12,
  };
  instance->vkGetPhysicalDeviceFeatures2(handle, &out_snapshot->features2);

  out_snapshot->memory_properties2 = (VkPhysicalDeviceMemoryProperties2){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
  };
  instance->vkGetPhysicalDeviceMemoryProperties2(
      handle, &out_snapshot->memory_properties2);

  instance->vkGetPhysicalDeviceQueueFamilyProperties2(
      handle, &out_snapshot->queue_family_count, NULL);
  iree_status_t status = iree_ok_status();
  if (out_snapshot->queue_family_count) {
    status = iree_allocator_malloc(host_allocator,
                                   out_snapshot->queue_family_count *
                                       sizeof(out_snapshot->queue_families[0]),
                                   (void**)&out_snapshot->queue_families);
    for (uint32_t i = 0;
         i < out_snapshot->queue_family_count && iree_status_is_ok(status);
         ++i) {
      out_snapshot->queue_families[i] = (VkQueueFamilyProperties2){
          .sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
      };
    }
    if (iree_status_is_ok(status)) {
      instance->vkGetPhysicalDeviceQueueFamilyProperties2(
          handle, &out_snapshot->queue_family_count,
          out_snapshot->queue_families);
    }
  }

  if (iree_status_is_ok(status)) {
    VkResult result = instance->vkEnumerateDeviceExtensionProperties(
        handle, /*pLayerName=*/NULL, &out_snapshot->extension_count, NULL);
    status = iree_hal_vulkan_status_from_result(
        result, "vkEnumerateDeviceExtensionProperties");
  }
  if (iree_status_is_ok(status) && out_snapshot->extension_count) {
    status = iree_allocator_malloc(
        host_allocator,
        out_snapshot->extension_count * sizeof(out_snapshot->extensions[0]),
        (void**)&out_snapshot->extensions);
    if (iree_status_is_ok(status)) {
      VkResult result = instance->vkEnumerateDeviceExtensionProperties(
          handle, /*pLayerName=*/NULL, &out_snapshot->extension_count,
          out_snapshot->extensions);
      status = iree_hal_vulkan_status_from_result(
          result, "vkEnumerateDeviceExtensionProperties");
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, out_snapshot->extensions);
    iree_allocator_free(host_allocator, out_snapshot->queue_families);
    memset(out_snapshot, 0, sizeof(*out_snapshot));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_physical_device_snapshot_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_vulkan_physical_device_snapshot_t* snapshot) {
  IREE_ASSERT_ARGUMENT(snapshot);
  iree_allocator_free(host_allocator, snapshot->extensions);
  iree_allocator_free(host_allocator, snapshot->queue_families);
  memset(snapshot, 0, sizeof(*snapshot));
}

static bool iree_hal_vulkan_physical_device_has_compute_queue(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot) {
  for (uint32_t i = 0; i < snapshot->queue_family_count; ++i) {
    if (iree_all_bits_set(
            snapshot->queue_families[i].queueFamilyProperties.queueFlags,
            VK_QUEUE_COMPUTE_BIT)) {
      return true;
    }
  }
  return false;
}

static bool iree_hal_vulkan_physical_device_supports_baseline(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot) {
  const VkPhysicalDeviceProperties* properties =
      &snapshot->properties2.properties;
  return properties->apiVersion >= VK_API_VERSION_1_3 &&
         snapshot->features12.bufferDeviceAddress &&
         snapshot->features12.timelineSemaphore &&
         snapshot->features12.scalarBlockLayout &&
         snapshot->features13.synchronization2 &&
         iree_hal_vulkan_physical_device_has_compute_queue(snapshot);
}

static iree_status_t iree_hal_vulkan_append_device_path(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_string_builder_t* builder, iree_string_view_t* out_view) {
  iree_host_size_t intern_offset = iree_string_builder_size(builder);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("GPU-")));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_append_uuid(builder, snapshot->id_properties.deviceUUID));
  iree_string_builder_intern(builder, &intern_offset, out_view);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_append_device_name(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_string_builder_t* builder, iree_string_view_t* out_view) {
  iree_host_size_t intern_offset = iree_string_builder_size(builder);
  const VkPhysicalDeviceProperties* properties =
      &snapshot->properties2.properties;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, properties->deviceName));
  if (snapshot->driver_properties.driverName[0]) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " (%s", snapshot->driver_properties.driverName));
    if (snapshot->driver_properties.driverInfo[0]) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " %s", snapshot->driver_properties.driverInfo));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV(")")));
  }
  if (!iree_hal_vulkan_physical_device_supports_baseline(snapshot)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV(" [below baseline]")));
  }
  iree_string_builder_intern(builder, &intern_offset, out_view);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_populate_device_info(
    const iree_hal_vulkan_instance_t* instance, VkPhysicalDevice handle,
    uint32_t ordinal, iree_allocator_t host_allocator,
    iree_hal_device_info_t* device_info, iree_string_builder_t* builder) {
  iree_hal_vulkan_physical_device_snapshot_t snapshot;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_physical_device_snapshot_initialize(
      instance, handle, ordinal, host_allocator, &snapshot));

  iree_status_t status = iree_ok_status();
  if (device_info) {
    device_info->device_id = (iree_hal_device_id_t)(ordinal + 1);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_append_device_path(
        &snapshot, builder, device_info ? &device_info->path : NULL);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_append_device_name(
        &snapshot, builder, device_info ? &device_info->name : NULL);
  }

  iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                        &snapshot);
  return status;
}

static iree_status_t iree_hal_vulkan_enumerate_physical_device_handles(
    const iree_hal_vulkan_instance_t* instance, iree_allocator_t host_allocator,
    uint32_t* out_physical_device_count,
    VkPhysicalDevice** out_physical_devices) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_physical_device_count);
  IREE_ASSERT_ARGUMENT(out_physical_devices);
  *out_physical_device_count = 0;
  *out_physical_devices = NULL;

  uint32_t physical_device_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_status_from_result(
      instance->vkEnumeratePhysicalDevices(instance->handle,
                                           &physical_device_count, NULL),
      "vkEnumeratePhysicalDevices"));
  if (!physical_device_count) return iree_ok_status();

  VkPhysicalDevice* physical_devices = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, physical_device_count * sizeof(*physical_devices),
      (void**)&physical_devices));

  iree_status_t status = iree_hal_vulkan_status_from_result(
      instance->vkEnumeratePhysicalDevices(
          instance->handle, &physical_device_count, physical_devices),
      "vkEnumeratePhysicalDevices");
  if (iree_status_is_ok(status)) {
    *out_physical_device_count = physical_device_count;
    *out_physical_devices = physical_devices;
  } else {
    iree_allocator_free(host_allocator, physical_devices);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Public physical-device inventory API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_vulkan_query_available_physical_devices(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* options,
    iree_allocator_t host_allocator, iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  *out_device_info_count = 0;
  *out_device_infos = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_instance_t instance;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_instance_initialize(libvulkan, options,
                                              host_allocator, &instance));

  uint32_t physical_device_count = 0;
  VkPhysicalDevice* physical_devices = NULL;
  iree_status_t status = iree_hal_vulkan_enumerate_physical_device_handles(
      &instance, host_allocator, &physical_device_count, &physical_devices);
  if (iree_status_is_ok(status) && !physical_device_count) {
    iree_hal_vulkan_instance_deinitialize(&instance);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_null(), &builder);
  iree_host_size_t device_info_count = 0;
  for (uint32_t i = 0; i < physical_device_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_vulkan_populate_device_info(
        &instance, physical_devices[i], i, host_allocator, NULL, &builder);
    if (iree_status_is_ok(status)) {
      ++device_info_count;
    }
  }

  iree_hal_device_info_t* device_infos = NULL;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t string_table_size =
        iree_string_builder_size(&builder) + /*NUL*/ 1;
    const iree_host_size_t total_size =
        device_info_count * sizeof(iree_hal_device_info_t) + string_table_size;
    status = iree_allocator_malloc(host_allocator, total_size,
                                   (void**)&device_infos);
    iree_string_builder_initialize_with_storage(
        (char*)device_infos + device_info_count * sizeof(device_infos[0]),
        string_table_size, &builder);
  }

  if (iree_status_is_ok(status)) {
    device_info_count = 0;
    for (uint32_t i = 0; i < physical_device_count && iree_status_is_ok(status);
         ++i) {
      status = iree_hal_vulkan_populate_device_info(
          &instance, physical_devices[i], i, host_allocator,
          &device_infos[device_info_count], &builder);
      if (iree_status_is_ok(status)) {
        ++device_info_count;
      }
    }
  }

  iree_string_builder_deinitialize(&builder);
  iree_allocator_free(host_allocator, physical_devices);
  iree_hal_vulkan_instance_deinitialize(&instance);

  if (iree_status_is_ok(status)) {
    *out_device_info_count = device_info_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_append_queue_flags(
    iree_string_builder_t* builder, VkQueueFlags flags) {
  bool emitted = false;
  VkQueueFlags emitted_flags = 0;
  enum {
    // Video and newer extension queue bits may be reported by drivers even when
    // the build's Vulkan headers hide their enum names behind feature guards.
    IREE_HAL_VULKAN_QUEUE_VIDEO_DECODE_BIT_KHR = 0x00000020u,
    IREE_HAL_VULKAN_QUEUE_VIDEO_ENCODE_BIT_KHR = 0x00000040u,
    IREE_HAL_VULKAN_QUEUE_OPTICAL_FLOW_BIT_NV = 0x00000100u,
    IREE_HAL_VULKAN_QUEUE_DATA_GRAPH_BIT_ARM = 0x00000400u,
  };
#define IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(flag, name)           \
  if (iree_all_bits_set(flags, flag)) {                         \
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(     \
        builder, emitted ? IREE_SV("|" name) : IREE_SV(name))); \
    emitted = true;                                             \
    emitted_flags |= (flag);                                    \
  }
  IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(VK_QUEUE_GRAPHICS_BIT, "graphics");
  IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(VK_QUEUE_COMPUTE_BIT, "compute");
  IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(VK_QUEUE_TRANSFER_BIT, "transfer");
  IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(VK_QUEUE_SPARSE_BINDING_BIT, "sparse");
  IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(VK_QUEUE_PROTECTED_BIT, "protected");
  IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(IREE_HAL_VULKAN_QUEUE_VIDEO_DECODE_BIT_KHR,
                                    "video-decode");
  IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(IREE_HAL_VULKAN_QUEUE_VIDEO_ENCODE_BIT_KHR,
                                    "video-encode");
  IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(IREE_HAL_VULKAN_QUEUE_OPTICAL_FLOW_BIT_NV,
                                    "optical-flow");
  IREE_HAL_VULKAN_APPEND_QUEUE_FLAG(IREE_HAL_VULKAN_QUEUE_DATA_GRAPH_BIT_ARM,
                                    "data-graph");
#undef IREE_HAL_VULKAN_APPEND_QUEUE_FLAG
  const VkQueueFlags unknown_flags = flags & ~emitted_flags;
  if (unknown_flags) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%sunknown(0x%08x)", emitted ? "|" : "", unknown_flags));
    emitted = true;
  }
  if (!emitted) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("none")));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_append_baseline_report(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_string_builder_t* builder) {
  const VkPhysicalDeviceProperties* properties =
      &snapshot->properties2.properties;
  const bool supported =
      iree_hal_vulkan_physical_device_supports_baseline(snapshot);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "feature_tier[vulkan-1.3-bda]: %s\n",
      supported ? "supported" : "unsupported"));
  if (supported) return iree_ok_status();

  if (properties->apiVersion < VK_API_VERSION_1_3) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("  missing: Vulkan API 1.3\n")));
  }
  if (!snapshot->features12.bufferDeviceAddress) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("  missing: bufferDeviceAddress\n")));
  }
  if (!snapshot->features12.timelineSemaphore) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("  missing: timelineSemaphore\n")));
  }
  if (!snapshot->features12.scalarBlockLayout) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("  missing: scalarBlockLayout\n")));
  }
  if (!snapshot->features13.synchronization2) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("  missing: synchronization2\n")));
  }
  if (!iree_hal_vulkan_physical_device_has_compute_queue(snapshot)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("  missing: compute queue family\n")));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_dump_physical_device_info(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* options,
    iree_hal_device_id_t device_id, iree_allocator_t host_allocator,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_instance_t instance;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_instance_initialize(libvulkan, options,
                                              host_allocator, &instance));

  uint32_t physical_device_count = 0;
  VkPhysicalDevice* physical_devices = NULL;
  iree_status_t status = iree_hal_vulkan_enumerate_physical_device_handles(
      &instance, host_allocator, &physical_device_count, &physical_devices);

  const uint32_t ordinal = (uint32_t)(device_id - 1);
  if (iree_status_is_ok(status) && (device_id == IREE_HAL_DEVICE_ID_DEFAULT ||
                                    ordinal >= physical_device_count)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan device id %" PRIu64
                              " out of range; driver has %u devices",
                              (uint64_t)device_id, physical_device_count);
  }

  iree_hal_vulkan_physical_device_snapshot_t snapshot;
  memset(&snapshot, 0, sizeof(snapshot));
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_physical_device_snapshot_initialize(
        &instance, physical_devices[ordinal], ordinal, host_allocator,
        &snapshot);
  }

  if (iree_status_is_ok(status)) {
    const VkPhysicalDeviceProperties* properties =
        &snapshot.properties2.properties;
#define IREE_HAL_VULKAN_APPEND(expr) \
  if (iree_status_is_ok(status)) {   \
    status = (expr);                 \
  }
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder, "name: %s\n", properties->deviceName));
    IREE_HAL_VULKAN_APPEND(
        iree_string_builder_append_string(builder, IREE_SV("path: GPU-")));
    IREE_HAL_VULKAN_APPEND(iree_hal_vulkan_append_uuid(
        builder, snapshot.id_properties.deviceUUID));
    IREE_HAL_VULKAN_APPEND(
        iree_string_builder_append_string(builder, IREE_SV("\napi: ")));
    IREE_HAL_VULKAN_APPEND(
        iree_hal_vulkan_append_api_version(builder, properties->apiVersion));
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder, "\ntype: %s\nvendor_id: 0x%04x\ndevice_id: 0x%04x\n",
        iree_hal_vulkan_device_type_string(properties->deviceType),
        properties->vendorID, properties->deviceID));
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder, "driver: %s\n", snapshot.driver_properties.driverName));
    if (snapshot.driver_properties.driverInfo[0]) {
      IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
          builder, "driver_info: %s\n", snapshot.driver_properties.driverInfo));
    }
    IREE_HAL_VULKAN_APPEND(
        iree_string_builder_append_string(builder, IREE_SV("device_uuid: ")));
    IREE_HAL_VULKAN_APPEND(iree_hal_vulkan_append_uuid(
        builder, snapshot.id_properties.deviceUUID));
    IREE_HAL_VULKAN_APPEND(
        iree_string_builder_append_string(builder, IREE_SV("\ndriver_uuid: ")));
    IREE_HAL_VULKAN_APPEND(iree_hal_vulkan_append_uuid(
        builder, snapshot.id_properties.driverUUID));
    IREE_HAL_VULKAN_APPEND(
        iree_string_builder_append_string(builder, IREE_SV("\n")));

    IREE_HAL_VULKAN_APPEND(
        iree_hal_vulkan_append_baseline_report(&snapshot, builder));
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder,
        "features: bufferDeviceAddress=%s timelineSemaphore=%s "
        "scalarBlockLayout=%s synchronization2=%s shaderInt8=%s "
        "shaderFloat16=%s shaderIntegerDotProduct=%s\n",
        iree_hal_vulkan_bool_string(snapshot.features12.bufferDeviceAddress),
        iree_hal_vulkan_bool_string(snapshot.features12.timelineSemaphore),
        iree_hal_vulkan_bool_string(snapshot.features12.scalarBlockLayout),
        iree_hal_vulkan_bool_string(snapshot.features13.synchronization2),
        iree_hal_vulkan_bool_string(snapshot.features12.shaderInt8),
        iree_hal_vulkan_bool_string(snapshot.features12.shaderFloat16),
        iree_hal_vulkan_bool_string(
            snapshot.features13.shaderIntegerDotProduct)));
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder, "sparse: binding=%s residencyBuffer=%s residencyAliased=%s\n",
        iree_hal_vulkan_bool_string(snapshot.features2.features.sparseBinding),
        iree_hal_vulkan_bool_string(
            snapshot.features2.features.sparseResidencyBuffer),
        iree_hal_vulkan_bool_string(
            snapshot.features2.features.sparseResidencyAliased)));
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder,
        "subgroup: size=%u operations=0x%08x quadOperationsInAllStages=%s\n",
        snapshot.subgroup_properties.subgroupSize,
        snapshot.subgroup_properties.supportedOperations,
        iree_hal_vulkan_bool_string(
            snapshot.subgroup_properties.quadOperationsInAllStages)));

    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder, "queue_families: %u\n", snapshot.queue_family_count));
    for (uint32_t i = 0;
         i < snapshot.queue_family_count && iree_status_is_ok(status); ++i) {
      const VkQueueFamilyProperties* queue_family =
          &snapshot.queue_families[i].queueFamilyProperties;
      IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
          builder, "  [%u] count=%u flags=", i, queue_family->queueCount));
      IREE_HAL_VULKAN_APPEND(iree_hal_vulkan_append_queue_flags(
          builder, queue_family->queueFlags));
      IREE_HAL_VULKAN_APPEND(
          iree_string_builder_append_format(builder, " timestampValidBits=%u\n",
                                            queue_family->timestampValidBits));
    }

    const VkPhysicalDeviceMemoryProperties* memory_properties =
        &snapshot.memory_properties2.memoryProperties;
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder, "memory_heaps: %u\n", memory_properties->memoryHeapCount));
    for (uint32_t i = 0;
         i < memory_properties->memoryHeapCount && iree_status_is_ok(status);
         ++i) {
      const VkMemoryHeap* heap = &memory_properties->memoryHeaps[i];
      IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
          builder, "  [%u] size=%" PRIu64 " flags=0x%08x\n", i,
          (uint64_t)heap->size, heap->flags));
    }
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder, "memory_types: %u\n", memory_properties->memoryTypeCount));
    for (uint32_t i = 0;
         i < memory_properties->memoryTypeCount && iree_status_is_ok(status);
         ++i) {
      const VkMemoryType* memory_type = &memory_properties->memoryTypes[i];
      IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
          builder, "  [%u] heap=%u flags=0x%08x\n", i, memory_type->heapIndex,
          memory_type->propertyFlags));
    }

    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder, "device_extensions: %u\n", snapshot.extension_count));
    for (uint32_t i = 0;
         i < snapshot.extension_count && iree_status_is_ok(status); ++i) {
      IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
          builder, "  %s\n", snapshot.extensions[i].extensionName));
    }
#undef IREE_HAL_VULKAN_APPEND
  }

  iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                        &snapshot);
  iree_allocator_free(host_allocator, physical_devices);
  iree_hal_vulkan_instance_deinitialize(&instance);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
