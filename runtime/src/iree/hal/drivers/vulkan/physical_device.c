// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/physical_device.h"

#include <stddef.h>
#include <string.h>

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
// Instance creation
//===----------------------------------------------------------------------===//

static bool iree_hal_vulkan_extension_list_contains(
    uint32_t extension_count, const VkExtensionProperties* extensions,
    const char* extension_name) {
  for (uint32_t i = 0; i < extension_count; ++i) {
    if (strcmp(extension_name, extensions[i].extensionName) == 0) return true;
  }
  return false;
}

static iree_hal_vulkan_device_extensions_t
iree_hal_vulkan_available_device_extensions_from_list(
    uint32_t extension_count, const VkExtensionProperties* extensions) {
  iree_hal_vulkan_device_extensions_t available_extensions =
      IREE_HAL_VULKAN_DEVICE_EXTENSION_NONE;
  if (iree_hal_vulkan_extension_list_contains(
          extension_count, extensions,
          IREE_HAL_VULKAN_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
    available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PORTABILITY_SUBSET;
  }
  if (iree_hal_vulkan_extension_list_contains(
          extension_count, extensions, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME)) {
    available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY;
  }
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  if (iree_hal_vulkan_extension_list_contains(
          extension_count, extensions,
          VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME)) {
    available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_WIN32;
  }
#else
  if (iree_hal_vulkan_extension_list_contains(
          extension_count, extensions,
          VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) {
    available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_FD;
  }
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)
  if (iree_hal_vulkan_extension_list_contains(
          extension_count, extensions,
          VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
    available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_EXTERNAL_MEMORY_HOST;
  }
  return available_extensions;
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
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_allocator_t host_allocator, uint32_t* out_extension_count,
    VkExtensionProperties** out_extensions) {
  IREE_ASSERT_ARGUMENT(out_extension_count);
  IREE_ASSERT_ARGUMENT(out_extensions);
  *out_extension_count = 0;
  *out_extensions = NULL;

  uint32_t extension_count = 0;
  IREE_RETURN_IF_ERROR(iree_vkEnumerateInstanceExtensionProperties(
      IREE_LIBVULKAN(libvulkan), /*pLayerName=*/NULL, &extension_count, NULL));
  if (!extension_count) return iree_ok_status();

  VkExtensionProperties* extensions = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, extension_count * sizeof(*extensions),
      (void**)&extensions));

  iree_status_t status = iree_vkEnumerateInstanceExtensionProperties(
      IREE_LIBVULKAN(libvulkan), /*pLayerName=*/NULL, &extension_count,
      extensions);
  if (iree_status_is_ok(status)) {
    *out_extension_count = extension_count;
    *out_extensions = extensions;
  } else {
    iree_allocator_free(host_allocator, extensions);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_enumerate_instance_layers(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_allocator_t host_allocator, uint32_t* out_layer_count,
    VkLayerProperties** out_layers) {
  IREE_ASSERT_ARGUMENT(out_layer_count);
  IREE_ASSERT_ARGUMENT(out_layers);
  *out_layer_count = 0;
  *out_layers = NULL;

  uint32_t layer_count = 0;
  IREE_RETURN_IF_ERROR(iree_vkEnumerateInstanceLayerProperties(
      IREE_LIBVULKAN(libvulkan), &layer_count, NULL));
  if (!layer_count) return iree_ok_status();

  VkLayerProperties* layers = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, layer_count * sizeof(*layers), (void**)&layers));

  iree_status_t status = iree_vkEnumerateInstanceLayerProperties(
      IREE_LIBVULKAN(libvulkan), &layer_count, layers);
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

iree_status_t iree_hal_vulkan_instance_initialize(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_vulkan_instance_t* out_instance) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_instance);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_instance, 0, sizeof(*out_instance));

  uint32_t loader_api_version = VK_API_VERSION_1_0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vkEnumerateInstanceVersion(IREE_LIBVULKAN(libvulkan),
                                          &loader_api_version));
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
              libvulkan, host_allocator, &available_extension_count,
              &available_extensions));

  uint32_t available_layer_count = 0;
  VkLayerProperties* available_layers = NULL;
  iree_status_t status = iree_hal_vulkan_enumerate_instance_layers(
      libvulkan, host_allocator, &available_layer_count, &available_layers);

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
    status = iree_vkCreateInstance(IREE_LIBVULKAN(libvulkan), &create_info,
                                   /*pAllocator=*/NULL, &out_instance->handle);
  }

  iree_allocator_free(host_allocator, available_layers);
  iree_allocator_free(host_allocator, available_extensions);

  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_libvulkan_load_instance_syms(
        libvulkan, out_instance->handle, &out_instance->syms);
  }
  if (iree_status_is_ok(status)) {
    out_instance->api_version = requested_api_version;
  }

  if (!iree_status_is_ok(status) && out_instance->handle) {
    iree_vkDestroyInstance(IREE_VULKAN_INSTANCE(&out_instance->syms),
                           out_instance->handle, /*pAllocator=*/NULL);
    memset(out_instance, 0, sizeof(*out_instance));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_vulkan_instance_deinitialize(
    iree_hal_vulkan_instance_t* instance) {
  IREE_ASSERT_ARGUMENT(instance);
  if (instance->handle) {
    iree_vkDestroyInstance(IREE_VULKAN_INSTANCE(&instance->syms),
                           instance->handle, /*pAllocator=*/NULL);
  }
  memset(instance, 0, sizeof(*instance));
}

//===----------------------------------------------------------------------===//
// Physical device snapshots
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_vulkan_physical_device_snapshot_initialize(
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
      .pNext = &out_snapshot->subgroup_size_control_properties,
  };
  out_snapshot->subgroup_size_control_properties =
      (VkPhysicalDeviceSubgroupSizeControlProperties){
          .sType =
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
      };
  out_snapshot->driver_properties = (VkPhysicalDeviceDriverProperties){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
      .pNext = &out_snapshot->subgroup_properties,
  };
  out_snapshot->id_properties = (VkPhysicalDeviceIDProperties){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
      .pNext = &out_snapshot->driver_properties,
  };
  out_snapshot->properties11 = (VkPhysicalDeviceVulkan11Properties){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
      .pNext = &out_snapshot->id_properties,
  };
  out_snapshot->properties2 = (VkPhysicalDeviceProperties2){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &out_snapshot->properties11,
  };
  iree_vkGetPhysicalDeviceProperties2(IREE_VULKAN_INSTANCE(&instance->syms),
                                      handle, &out_snapshot->properties2);

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
  iree_vkGetPhysicalDeviceFeatures2(IREE_VULKAN_INSTANCE(&instance->syms),
                                    handle, &out_snapshot->features2);

  out_snapshot->memory_properties2 = (VkPhysicalDeviceMemoryProperties2){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
  };
  iree_vkGetPhysicalDeviceMemoryProperties2(
      IREE_VULKAN_INSTANCE(&instance->syms), handle,
      &out_snapshot->memory_properties2);

  iree_vkGetPhysicalDeviceQueueFamilyProperties2(
      IREE_VULKAN_INSTANCE(&instance->syms), handle,
      &out_snapshot->queue_family_count, NULL);
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
      iree_vkGetPhysicalDeviceQueueFamilyProperties2(
          IREE_VULKAN_INSTANCE(&instance->syms), handle,
          &out_snapshot->queue_family_count, out_snapshot->queue_families);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_vkEnumerateDeviceExtensionProperties(
        IREE_VULKAN_INSTANCE(&instance->syms), handle, /*pLayerName=*/NULL,
        &out_snapshot->extension_count, NULL);
  }
  if (iree_status_is_ok(status) && out_snapshot->extension_count) {
    status = iree_allocator_malloc(
        host_allocator,
        out_snapshot->extension_count * sizeof(out_snapshot->extensions[0]),
        (void**)&out_snapshot->extensions);
    if (iree_status_is_ok(status)) {
      status = iree_vkEnumerateDeviceExtensionProperties(
          IREE_VULKAN_INSTANCE(&instance->syms), handle, /*pLayerName=*/NULL,
          &out_snapshot->extension_count, out_snapshot->extensions);
    }
  }
  if (iree_status_is_ok(status)) {
    out_snapshot->available_extensions =
        iree_hal_vulkan_available_device_extensions_from_list(
            out_snapshot->extension_count, out_snapshot->extensions);
  }

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, out_snapshot->extensions);
    iree_allocator_free(host_allocator, out_snapshot->queue_families);
    memset(out_snapshot, 0, sizeof(*out_snapshot));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_vulkan_physical_device_snapshot_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_vulkan_physical_device_snapshot_t* snapshot) {
  IREE_ASSERT_ARGUMENT(snapshot);
  iree_allocator_free(host_allocator, snapshot->extensions);
  iree_allocator_free(host_allocator, snapshot->queue_families);
  memset(snapshot, 0, sizeof(*snapshot));
}

bool iree_hal_vulkan_physical_device_has_compute_queue(
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

bool iree_hal_vulkan_physical_device_supports_baseline(
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

bool iree_hal_vulkan_physical_device_has_extension(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_device_extensions_t extension_bits) {
  return iree_all_bits_set(snapshot->available_extensions, extension_bits);
}

iree_status_t iree_hal_vulkan_append_device_path(
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

iree_status_t iree_hal_vulkan_enumerate_physical_device_handles(
    const iree_hal_vulkan_instance_t* instance, iree_allocator_t host_allocator,
    uint32_t* out_physical_device_count,
    VkPhysicalDevice** out_physical_devices) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_physical_device_count);
  IREE_ASSERT_ARGUMENT(out_physical_devices);
  *out_physical_device_count = 0;
  *out_physical_devices = NULL;

  uint32_t physical_device_count = 0;
  IREE_RETURN_IF_ERROR(iree_vkEnumeratePhysicalDevices(
      IREE_VULKAN_INSTANCE(&instance->syms), instance->handle,
      &physical_device_count, NULL));
  if (!physical_device_count) return iree_ok_status();

  VkPhysicalDevice* physical_devices = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, physical_device_count * sizeof(*physical_devices),
      (void**)&physical_devices));

  iree_status_t status = iree_vkEnumeratePhysicalDevices(
      IREE_VULKAN_INSTANCE(&instance->syms), instance->handle,
      &physical_device_count, physical_devices);
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
        "shaderFloat16=%s shaderIntegerDotProduct=%s "
        "subgroupSizeControl=%s\n",
        iree_hal_vulkan_bool_string(snapshot.features12.bufferDeviceAddress),
        iree_hal_vulkan_bool_string(snapshot.features12.timelineSemaphore),
        iree_hal_vulkan_bool_string(snapshot.features12.scalarBlockLayout),
        iree_hal_vulkan_bool_string(snapshot.features13.synchronization2),
        iree_hal_vulkan_bool_string(snapshot.features12.shaderInt8),
        iree_hal_vulkan_bool_string(snapshot.features12.shaderFloat16),
        iree_hal_vulkan_bool_string(
            snapshot.features13.shaderIntegerDotProduct),
        iree_hal_vulkan_bool_string(snapshot.features13.subgroupSizeControl)));
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder, "sparse: binding=%s residencyBuffer=%s residencyAliased=%s\n",
        iree_hal_vulkan_bool_string(snapshot.features2.features.sparseBinding),
        iree_hal_vulkan_bool_string(
            snapshot.features2.features.sparseResidencyBuffer),
        iree_hal_vulkan_bool_string(
            snapshot.features2.features.sparseResidencyAliased)));
    IREE_HAL_VULKAN_APPEND(iree_string_builder_append_format(
        builder,
        "subgroup: size=%u operations=0x%08x quadOperationsInAllStages=%s "
        "minSubgroupSize=%u maxSubgroupSize=%u "
        "requiredSubgroupSizeStages=0x%08x\n",
        snapshot.subgroup_properties.subgroupSize,
        snapshot.subgroup_properties.supportedOperations,
        iree_hal_vulkan_bool_string(
            snapshot.subgroup_properties.quadOperationsInAllStages),
        snapshot.subgroup_size_control_properties.minSubgroupSize,
        snapshot.subgroup_size_control_properties.maxSubgroupSize,
        snapshot.subgroup_size_control_properties.requiredSubgroupSizeStages));

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
