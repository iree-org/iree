// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/logical_device.h"

#include <string.h>

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/vulkan/allocator.h"
#include "iree/hal/drivers/vulkan/physical_device.h"
#include "iree/hal/drivers/vulkan/semaphore.h"

//===----------------------------------------------------------------------===//
// Physical-device selection
//===----------------------------------------------------------------------===//

typedef enum iree_hal_vulkan_physical_device_selector_mode_e {
  IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT = 0,
  IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID = 1,
  IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH = 2,
} iree_hal_vulkan_physical_device_selector_mode_t;

typedef struct iree_hal_vulkan_physical_device_selector_t {
  // Selection mode used when walking visible physical devices.
  iree_hal_vulkan_physical_device_selector_mode_t mode;

  // HAL device id to match when mode is ID.
  iree_hal_device_id_t device_id;

  // HAL device path to match when mode is PATH.
  iree_string_view_t device_path;
} iree_hal_vulkan_physical_device_selector_t;

static iree_status_t iree_hal_vulkan_device_options_parse(
    iree_hal_vulkan_device_options_t* options, iree_string_pair_list_t params) {
  IREE_ASSERT_ARGUMENT(options);
  if (!params.count) return iree_ok_status();
  const iree_string_pair_t* first_param = &params.pairs[0];
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "Vulkan logical device options do not support key/value parameter '%.*s'",
      (int)first_param->key.size, first_param->key.data);
}

static bool iree_hal_vulkan_selector_matches(
    const iree_hal_vulkan_physical_device_selector_t* selector,
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_status_t* out_status) {
  *out_status = iree_ok_status();
  switch (selector->mode) {
    case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT:
      return iree_hal_vulkan_physical_device_supports_baseline(snapshot);
    case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID:
      return selector->device_id ==
             (iree_hal_device_id_t)(snapshot->ordinal + 1);
    case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH: {
      char path_storage[64] = {0};
      iree_string_builder_t builder;
      iree_string_builder_initialize_with_storage(
          path_storage, sizeof(path_storage), &builder);
      iree_string_view_t candidate_path = iree_string_view_empty();
      *out_status = iree_hal_vulkan_append_device_path(snapshot, &builder,
                                                       &candidate_path);
      const bool matches =
          iree_status_is_ok(*out_status) &&
          iree_string_view_equal(candidate_path, selector->device_path);
      iree_string_builder_deinitialize(&builder);
      return matches;
    }
    default:
      *out_status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                     "invalid Vulkan selector mode %u",
                                     (uint32_t)selector->mode);
      return false;
  }
}

static iree_status_t iree_hal_vulkan_select_physical_device(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_physical_device_selector_t* selector,
    iree_allocator_t host_allocator, iree_hal_vulkan_instance_t* out_instance,
    iree_hal_vulkan_physical_device_snapshot_t* out_snapshot) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(driver_options);
  IREE_ASSERT_ARGUMENT(selector);
  IREE_ASSERT_ARGUMENT(out_instance);
  IREE_ASSERT_ARGUMENT(out_snapshot);
  memset(out_instance, 0, sizeof(*out_instance));
  memset(out_snapshot, 0, sizeof(*out_snapshot));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_vulkan_instance_initialize(
      libvulkan, driver_options, host_allocator, out_instance);

  uint32_t physical_device_count = 0;
  VkPhysicalDevice* physical_devices = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_enumerate_physical_device_handles(
        out_instance, host_allocator, &physical_device_count,
        &physical_devices);
  }

  if (iree_status_is_ok(status) && !physical_device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "Vulkan driver has no physical devices");
  }

  if (iree_status_is_ok(status) &&
      selector->mode == IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID) {
    const uint32_t ordinal = (uint32_t)(selector->device_id - 1);
    if (selector->device_id == IREE_HAL_DEVICE_ID_DEFAULT ||
        ordinal >= physical_device_count) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan device id %" PRIu64 " out of range; driver has %u devices",
          (uint64_t)selector->device_id, physical_device_count);
    }
  }

  bool selected = false;
  for (uint32_t i = 0;
       i < physical_device_count && iree_status_is_ok(status) && !selected;
       ++i) {
    iree_hal_vulkan_physical_device_snapshot_t snapshot;
    status = iree_hal_vulkan_physical_device_snapshot_initialize(
        out_instance, physical_devices[i], i, host_allocator, &snapshot);
    if (!iree_status_is_ok(status)) break;

    iree_status_t match_status = iree_ok_status();
    const bool matches =
        iree_hal_vulkan_selector_matches(selector, &snapshot, &match_status);
    if (!iree_status_is_ok(match_status)) {
      status = match_status;
    } else if (matches) {
      if (iree_hal_vulkan_physical_device_supports_baseline(&snapshot)) {
        *out_snapshot = snapshot;
        memset(&snapshot, 0, sizeof(snapshot));
        selected = true;
      } else {
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "Vulkan physical device %u does not satisfy the rewrite baseline",
            i);
      }
    }
    iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                          &snapshot);
  }

  if (iree_status_is_ok(status) && !selected) {
    switch (selector->mode) {
      case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT:
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "no Vulkan physical device satisfies the rewrite baseline");
        break;
      case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID:
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "Vulkan device id %" PRIu64 " not found",
                                  (uint64_t)selector->device_id);
        break;
      case IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH:
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND, "Vulkan device path '%.*s' not found",
            (int)selector->device_path.size, selector->device_path.data);
        break;
      default:
        break;
    }
  }

  iree_allocator_free(host_allocator, physical_devices);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                          out_snapshot);
    iree_hal_vulkan_instance_deinitialize(out_instance);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Queue selection
//===----------------------------------------------------------------------===//

#define IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY UINT32_MAX

typedef struct iree_hal_vulkan_selected_queue_t {
  // Queue family index from the selected physical device.
  uint32_t family_index;

  // Queue index within the queue family.
  uint32_t queue_index;

  // Vulkan queue handle borrowed from the logical device.
  VkQueue handle;

  // Queue family capability flags cached from the physical snapshot.
  VkQueueFlags flags;

  // HAL queue affinity bit that maps to this queue.
  iree_hal_queue_affinity_t affinity;
} iree_hal_vulkan_selected_queue_t;

typedef struct iree_hal_vulkan_selected_queues_t {
  // Compute queue used for dispatch-capable operations.
  iree_hal_vulkan_selected_queue_t compute;

  // Transfer queue used for copy/fill/update-capable operations.
  iree_hal_vulkan_selected_queue_t transfer;

  // Count of distinct HAL queues exposed by this logical device.
  iree_host_size_t queue_count;
} iree_hal_vulkan_selected_queues_t;

static bool iree_hal_vulkan_queue_family_has(
    const VkQueueFamilyProperties* queue_family, VkQueueFlags required_flags) {
  return queue_family->queueCount > 0 &&
         iree_all_bits_set(queue_family->queueFlags, required_flags);
}

static uint32_t iree_hal_vulkan_select_compute_queue_family(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* options) {
  const bool prefer_dedicated = iree_any_bit_set(
      options->flags, IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE);
  uint32_t fallback_family_index = IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY;
  for (uint32_t i = 0; i < snapshot->queue_family_count; ++i) {
    const VkQueueFamilyProperties* queue_family =
        &snapshot->queue_families[i].queueFamilyProperties;
    if (!iree_hal_vulkan_queue_family_has(queue_family, VK_QUEUE_COMPUTE_BIT)) {
      continue;
    }
    const bool has_graphics =
        iree_any_bit_set(queue_family->queueFlags, VK_QUEUE_GRAPHICS_BIT);
    if (!prefer_dedicated || !has_graphics) return i;
    if (fallback_family_index == IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY) {
      fallback_family_index = i;
    }
  }
  return fallback_family_index;
}

static uint32_t iree_hal_vulkan_select_transfer_queue_family(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    uint32_t compute_family_index) {
  uint32_t fallback_family_index = IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY;
  for (uint32_t i = 0; i < snapshot->queue_family_count; ++i) {
    const VkQueueFamilyProperties* queue_family =
        &snapshot->queue_families[i].queueFamilyProperties;
    if (!iree_hal_vulkan_queue_family_has(queue_family,
                                          VK_QUEUE_TRANSFER_BIT)) {
      continue;
    }
    const bool has_graphics =
        iree_any_bit_set(queue_family->queueFlags, VK_QUEUE_GRAPHICS_BIT);
    const bool has_compute =
        iree_any_bit_set(queue_family->queueFlags, VK_QUEUE_COMPUTE_BIT);
    if (!has_graphics && !has_compute) return i;
    if (fallback_family_index == IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY) {
      fallback_family_index = i;
    }
  }
  return fallback_family_index == IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY
             ? compute_family_index
             : fallback_family_index;
}

static iree_status_t iree_hal_vulkan_select_queues(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* options,
    iree_hal_vulkan_selected_queues_t* out_queues) {
  memset(out_queues, 0, sizeof(*out_queues));
  const uint32_t compute_family_index =
      iree_hal_vulkan_select_compute_queue_family(snapshot, options);
  if (compute_family_index == IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "Vulkan physical device has no compute queue");
  }
  const uint32_t transfer_family_index =
      iree_hal_vulkan_select_transfer_queue_family(snapshot,
                                                   compute_family_index);

  const VkQueueFamilyProperties* compute_family =
      &snapshot->queue_families[compute_family_index].queueFamilyProperties;
  const VkQueueFamilyProperties* transfer_family =
      &snapshot->queue_families[transfer_family_index].queueFamilyProperties;
  uint32_t transfer_queue_index = 0;
  if (transfer_family_index == compute_family_index &&
      transfer_family->queueCount > 1) {
    transfer_queue_index = 1;
  }

  out_queues->compute = (iree_hal_vulkan_selected_queue_t){
      .family_index = compute_family_index,
      .queue_index = 0,
      .flags = compute_family->queueFlags,
      .affinity = 1ull << 0,
  };
  out_queues->transfer = (iree_hal_vulkan_selected_queue_t){
      .family_index = transfer_family_index,
      .queue_index = transfer_queue_index,
      .flags = transfer_family->queueFlags,
      .affinity = transfer_queue_index == 0 &&
                          transfer_family_index == compute_family_index
                      ? 1ull << 0
                      : 1ull << 1,
  };
  out_queues->queue_count =
      out_queues->transfer.affinity == (1ull << 0) ? 1 : 2;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_affinity_normalize(
    iree_hal_queue_affinity_t supported_affinity,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_queue_affinity_t* out_normalized_affinity) {
  iree_hal_queue_affinity_t normalized_affinity =
      iree_hal_queue_affinity_is_any(requested_affinity) ? supported_affinity
                                                         : requested_affinity;
  iree_hal_queue_affinity_and_into(normalized_affinity, supported_affinity);
  if (iree_hal_queue_affinity_is_empty(normalized_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid Vulkan queue affinity bits specified");
  }
  *out_normalized_affinity = normalized_affinity;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Device extension/feature policy
//===----------------------------------------------------------------------===//

static void iree_hal_vulkan_enable_extension_if_available(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_vulkan_device_extensions_t extension_bit,
    const char* extension_name, const char** enabled_extensions,
    uint32_t* enabled_extension_count,
    iree_hal_vulkan_device_extensions_t* out_enabled_extensions) {
  if (iree_hal_vulkan_physical_device_has_extension(snapshot, extension_bit)) {
    enabled_extensions[*enabled_extension_count] = extension_name;
    *enabled_extension_count = *enabled_extension_count + 1;
    *out_enabled_extensions |= extension_bit;
  }
}

static iree_status_t iree_hal_vulkan_create_logical_device_handle(
    const iree_hal_vulkan_instance_t* instance,
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* device_options,
    iree_hal_vulkan_features_t requested_features,
    iree_hal_vulkan_selected_queues_t* selected_queues,
    iree_hal_vulkan_features_t* out_enabled_features,
    iree_hal_vulkan_device_extensions_t* out_enabled_extensions,
    VkDevice* out_logical_device) {
  IREE_ASSERT_ARGUMENT(out_enabled_features);
  IREE_ASSERT_ARGUMENT(out_enabled_extensions);
  IREE_ASSERT_ARGUMENT(out_logical_device);
  *out_enabled_features = IREE_HAL_VULKAN_FEATURE_NONE;
  *out_enabled_extensions = IREE_HAL_VULKAN_DEVICE_EXTENSION_NONE;
  *out_logical_device = VK_NULL_HANDLE;

  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_select_queues(snapshot, device_options, selected_queues));

  const char* enabled_extensions[8] = {0};
  uint32_t enabled_extension_count = 0;
  iree_hal_vulkan_enable_extension_if_available(
      snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PORTABILITY_SUBSET,
      IREE_HAL_VULKAN_KHR_PORTABILITY_SUBSET_EXTENSION_NAME, enabled_extensions,
      &enabled_extension_count, out_enabled_extensions);
  iree_hal_vulkan_enable_extension_if_available(
      snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY,
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, enabled_extensions,
      &enabled_extension_count, out_enabled_extensions);
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  iree_hal_vulkan_enable_extension_if_available(
      snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_WIN32,
      VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, enabled_extensions,
      &enabled_extension_count, out_enabled_extensions);
#else
  iree_hal_vulkan_enable_extension_if_available(
      snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_FD,
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, enabled_extensions,
      &enabled_extension_count, out_enabled_extensions);
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)
  iree_hal_vulkan_enable_extension_if_available(
      snapshot, IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_EXTERNAL_MEMORY_HOST,
      VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME, enabled_extensions,
      &enabled_extension_count, out_enabled_extensions);

  float queue_priorities[2] = {1.0f, 1.0f};
  VkDeviceQueueCreateInfo queue_create_infos[2] = {0};
  uint32_t queue_create_info_count = 0;
  if (selected_queues->compute.family_index ==
      selected_queues->transfer.family_index) {
    queue_create_infos[queue_create_info_count++] = (VkDeviceQueueCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = selected_queues->compute.family_index,
        .queueCount = selected_queues->transfer.queue_index + 1,
        .pQueuePriorities = queue_priorities,
    };
  } else {
    queue_create_infos[queue_create_info_count++] = (VkDeviceQueueCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = selected_queues->compute.family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priorities[0],
    };
    queue_create_infos[queue_create_info_count++] = (VkDeviceQueueCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = selected_queues->transfer.family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priorities[1],
    };
  }

  VkPhysicalDeviceVulkan13Features enabled_features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE,
      .shaderIntegerDotProduct = snapshot->features13.shaderIntegerDotProduct,
  };
  VkPhysicalDeviceVulkan12Features enabled_features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &enabled_features13,
      .storageBuffer8BitAccess = snapshot->features12.storageBuffer8BitAccess,
      .uniformAndStorageBuffer8BitAccess =
          snapshot->features12.uniformAndStorageBuffer8BitAccess,
      .storagePushConstant8 = snapshot->features12.storagePushConstant8,
      .shaderFloat16 = snapshot->features12.shaderFloat16,
      .shaderInt8 = snapshot->features12.shaderInt8,
      .scalarBlockLayout = VK_TRUE,
      .timelineSemaphore = VK_TRUE,
      .bufferDeviceAddress = VK_TRUE,
      .vulkanMemoryModel = snapshot->features12.vulkanMemoryModel,
      .vulkanMemoryModelDeviceScope =
          snapshot->features12.vulkanMemoryModelDeviceScope,
  };
  VkPhysicalDeviceFeatures2 enabled_features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &enabled_features12,
      .features =
          {
              .robustBufferAccess = VK_FALSE,
              .shaderFloat64 = snapshot->features2.features.shaderFloat64,
              .shaderInt64 = snapshot->features2.features.shaderInt64,
              .shaderInt16 = snapshot->features2.features.shaderInt16,
              .sparseBinding = VK_FALSE,
              .sparseResidencyBuffer = VK_FALSE,
              .sparseResidencyAliased = VK_FALSE,
          },
  };

  iree_hal_vulkan_features_t enabled_features =
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES;
  if (iree_any_bit_set(requested_features,
                       IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS)) {
    if (!snapshot->features2.features.robustBufferAccess) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "requested Vulkan robustBufferAccess is not available");
    }
    enabled_features2.features.robustBufferAccess = VK_TRUE;
    enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS;
  }
  if (iree_any_bit_set(requested_features,
                       IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING)) {
    if (!snapshot->features2.features.sparseBinding) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "requested Vulkan sparseBinding is not available");
    }
    enabled_features2.features.sparseBinding = VK_TRUE;
    enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING;
  }
  if (iree_any_bit_set(
          requested_features,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED)) {
    if (!snapshot->features2.features.sparseResidencyBuffer ||
        !snapshot->features2.features.sparseResidencyAliased) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "requested Vulkan sparse residency aliasing is not available");
    }
    enabled_features2.features.sparseResidencyBuffer = VK_TRUE;
    enabled_features2.features.sparseResidencyAliased = VK_TRUE;
    enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED;
  }

  VkDeviceCreateInfo device_create_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled_features2,
      .queueCreateInfoCount = queue_create_info_count,
      .pQueueCreateInfos = queue_create_infos,
      .enabledExtensionCount = enabled_extension_count,
      .ppEnabledExtensionNames = enabled_extensions,
  };
  IREE_RETURN_IF_ERROR(iree_vkCreateDevice(
      IREE_VULKAN_INSTANCE(&instance->syms), snapshot->handle,
      &device_create_info, /*pAllocator=*/NULL, out_logical_device));

  *out_enabled_features = enabled_features;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_logical_device_t
//===----------------------------------------------------------------------===//

struct iree_hal_vulkan_logical_device_t {
  // HAL resource header.
  iree_hal_resource_t resource;

  // Host allocator used for logical-device-owned host allocations.
  iree_allocator_t host_allocator;

  // Stable device identifier stored inline after this struct.
  iree_string_view_t identifier;

  // Retained Vulkan loader keeping resolved entry-point code live.
  iree_hal_vulkan_libvulkan_t libvulkan;

  // Proactor pool retained from create_params for async host waits.
  iree_async_proactor_pool_t* proactor_pool;

  // Proactor borrowed from the pool for device-local async operations.
  iree_async_proactor_t* proactor;

  // Driver-owned Vulkan instance and instance dispatch table.
  iree_hal_vulkan_instance_t instance;

  // Immutable physical-device inventory used for capability decisions.
  iree_hal_vulkan_physical_device_snapshot_t physical_device;

  // Vulkan logical device handle.
  VkDevice logical_device;

  // True when destruction should call vkDestroyDevice.
  bool owns_logical_device;

  // Device-level Vulkan dispatch table.
  iree_hal_vulkan_device_syms_t syms;

  // HAL feature bits enabled on the logical device.
  iree_hal_vulkan_features_t enabled_features;

  // Recognized Vulkan device extension bits enabled on the logical device.
  iree_hal_vulkan_device_extensions_t enabled_extensions;

  // Selected compute-capable queue.
  iree_hal_vulkan_selected_queue_t compute_queue;

  // Selected transfer-capable queue.
  iree_hal_vulkan_selected_queue_t transfer_queue;

  // Count of distinct HAL queues exposed through queue affinity.
  iree_host_size_t queue_count;

  // Mask of valid queue affinity bits for this logical device.
  iree_hal_queue_affinity_t queue_affinity_mask;

  // Logical allocator.
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Shared frontier tracker retained after topology assignment.
  iree_async_frontier_tracker_t* frontier_tracker;

  // Topology-assigned axis registered with the frontier tracker.
  iree_async_axis_t axis;

  // Topology information if this device is part of a multi-device topology.
  iree_hal_device_topology_info_t topology_info;

  // + trailing identifier string storage.
};

static const iree_hal_device_vtable_t iree_hal_vulkan_logical_device_vtable;

static iree_hal_vulkan_logical_device_t* iree_hal_vulkan_logical_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_logical_device_vtable);
  return (iree_hal_vulkan_logical_device_t*)base_value;
}

static iree_status_t iree_hal_vulkan_unimplemented(
    iree_string_view_t operation) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Vulkan %.*s is not implemented in the rewrite HAL",
                          (int)operation.size, operation.data);
}

static void iree_hal_vulkan_logical_device_clear_topology_info(
    iree_hal_vulkan_logical_device_t* device) {
  if (device->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        device->frontier_tracker, device->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(device->frontier_tracker);
    device->frontier_tracker = NULL;
    device->axis = 0;
  }
  memset(&device->topology_info, 0, sizeof(device->topology_info));
}

static void iree_hal_vulkan_logical_device_destroy(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_logical_device_clear_topology_info(device);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_hal_allocator_release(device->device_allocator);
  iree_async_proactor_pool_release(device->proactor_pool);
  if (device->logical_device && device->owns_logical_device) {
    iree_vkDestroyDevice(IREE_VULKAN_DEVICE(&device->syms),
                         device->logical_device, /*pAllocator=*/NULL);
  }
  iree_hal_vulkan_physical_device_snapshot_deinitialize(
      host_allocator, &device->physical_device);
  iree_hal_vulkan_instance_deinitialize(&device->instance);
  iree_hal_vulkan_libvulkan_deinitialize(&device->libvulkan);
  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_vulkan_logical_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_vulkan_logical_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_vulkan_logical_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_vulkan_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_vulkan_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_vulkan_logical_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_vulkan_logical_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }
  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = 0;
    return iree_ok_status();
  }
  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = (int64_t)device->queue_count;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = 1;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("vulkan.device"))) {
    if (iree_string_view_equal(key, IREE_SV("api_version"))) {
      *out_value = device->physical_device.properties2.properties.apiVersion;
      return iree_ok_status();
    } else if (iree_string_view_equal(key, IREE_SV("subgroup_size"))) {
      *out_value = device->physical_device.subgroup_properties.subgroupSize;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_vulkan_logical_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  memset(out_capabilities, 0, sizeof(*out_capabilities));

  memcpy(out_capabilities->physical_device_uuid,
         device->physical_device.id_properties.deviceUUID,
         sizeof(out_capabilities->physical_device_uuid));
  out_capabilities->has_physical_device_uuid = true;
  out_capabilities->driver_device_handle =
      (uintptr_t)device->physical_device.handle;
  out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_TIMELINE_SEMAPHORES |
                             IREE_HAL_DEVICE_CAPABILITY_ATOMIC_SCOPE_DEVICE;
  if (iree_all_bits_set(
          device->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_FD)) {
    out_capabilities->buffer_export_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD;
    out_capabilities->buffer_import_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD;
  }
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_vulkan_logical_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_vulkan_logical_device_refine_topology_edge(
    iree_hal_device_t* source_device, iree_hal_device_t* target_device,
    iree_hal_topology_edge_t* edge) {
  (void)source_device;
  (void)target_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_logical_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  if (!topology_info) {
    iree_hal_vulkan_logical_device_clear_topology_info(device);
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t* frontier_tracker =
      topology_info->frontier.tracker;
  iree_async_axis_t axis = topology_info->frontier.base_axis;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL));
  device->topology_info = *topology_info;
  device->frontier_tracker = frontier_tracker;
  device->axis = axis;
  iree_async_frontier_tracker_retain(device->frontier_tracker);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_logical_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  (void)base_device;
  (void)queue_affinity;
  (void)params;
  *out_channel = NULL;
  return iree_hal_vulkan_unimplemented(IREE_SV("collective channels"));
}

static iree_status_t iree_hal_vulkan_logical_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  (void)base_device;
  (void)mode;
  (void)command_categories;
  (void)queue_affinity;
  (void)binding_capacity;
  *out_command_buffer = NULL;
  return iree_hal_vulkan_unimplemented(IREE_SV("command buffers"));
}

static iree_status_t iree_hal_vulkan_logical_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  (void)base_device;
  (void)queue_affinity;
  (void)flags;
  *out_event = NULL;
  return iree_hal_vulkan_unimplemented(IREE_SV("events"));
}

static iree_status_t iree_hal_vulkan_logical_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  (void)base_device;
  (void)identifier;
  *out_executable_cache = NULL;
  return iree_hal_vulkan_unimplemented(IREE_SV("executable caches"));
}

static iree_status_t iree_hal_vulkan_logical_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  (void)base_device;
  (void)queue_affinity;
  (void)access;
  (void)handle;
  (void)flags;
  *out_file = NULL;
  return iree_hal_vulkan_unimplemented(IREE_SV("file import"));
}

static iree_status_t iree_hal_vulkan_logical_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_vulkan_logical_device_t* device =
      iree_hal_vulkan_logical_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_affinity_normalize(
      device->queue_affinity_mask, queue_affinity, &queue_affinity));
  return iree_hal_vulkan_semaphore_create(
      &device->syms, device->logical_device, device->proactor, queue_affinity,
      initial_value, flags, device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_vulkan_logical_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  (void)base_device;
  if (iree_hal_vulkan_semaphore_isa(semaphore)) {
    return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
  }
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_vulkan_logical_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  (void)base_device;
  (void)queue_affinity;
  (void)out_backend;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue pool backend"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)pool;
  (void)params;
  (void)allocation_size;
  (void)flags;
  *out_buffer = NULL;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue alloca"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)buffer;
  (void)flags;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue dealloca"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)target_buffer;
  (void)target_offset;
  (void)length;
  (void)pattern;
  (void)pattern_length;
  (void)flags;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue fill"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_buffer;
  (void)source_offset;
  (void)target_buffer;
  (void)target_offset;
  (void)length;
  (void)flags;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue update"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_buffer;
  (void)source_offset;
  (void)target_buffer;
  (void)target_offset;
  (void)length;
  (void)flags;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue copy"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_file;
  (void)source_offset;
  (void)target_buffer;
  (void)target_offset;
  (void)length;
  (void)flags;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue read"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)source_buffer;
  (void)source_offset;
  (void)target_file;
  (void)target_offset;
  (void)length;
  (void)flags;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue write"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)call;
  (void)args;
  (void)flags;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue host calls"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)executable;
  (void)export_ordinal;
  (void)config;
  (void)constants;
  (void)bindings;
  (void)flags;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue dispatch"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  (void)base_device;
  (void)queue_affinity;
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)command_buffer;
  (void)binding_table;
  (void)flags;
  return iree_hal_vulkan_unimplemented(IREE_SV("queue execute"));
}

static iree_status_t iree_hal_vulkan_logical_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  (void)base_device;
  (void)queue_affinity;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_logical_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  (void)base_device;
  (void)options;
  return iree_hal_vulkan_unimplemented(IREE_SV("device profiling"));
}

static iree_status_t iree_hal_vulkan_logical_device_profiling_flush(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_hal_vulkan_unimplemented(IREE_SV("device profiling"));
}

static iree_status_t iree_hal_vulkan_logical_device_profiling_end(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_hal_vulkan_unimplemented(IREE_SV("device profiling"));
}

static iree_status_t iree_hal_vulkan_logical_device_external_capture_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_external_capture_options_t* options) {
  (void)base_device;
  (void)options;
  return iree_hal_vulkan_unimplemented(IREE_SV("external capture"));
}

static iree_status_t iree_hal_vulkan_logical_device_external_capture_end(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_hal_vulkan_unimplemented(IREE_SV("external capture"));
}

static iree_status_t iree_hal_vulkan_logical_device_create_from_selection(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_device_options_t* device_options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_vulkan_instance_t* instance,
    iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t total_size = sizeof(iree_hal_vulkan_logical_device_t);
  if (!iree_host_size_checked_add(total_size, identifier.size, &total_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan logical device allocation overflow");
  }

  iree_hal_vulkan_logical_device_t* device = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_vulkan_logical_device_vtable,
                               &device->resource);
  device->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + sizeof(*device));

  iree_status_t status =
      iree_hal_vulkan_libvulkan_copy(libvulkan, &device->libvulkan);
  if (iree_status_is_ok(status)) {
    device->instance = *instance;
    memset(instance, 0, sizeof(*instance));
    device->physical_device = *snapshot;
    memset(snapshot, 0, sizeof(*snapshot));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create(
        &device->physical_device, host_allocator, &device->device_allocator);
  }
  if (iree_status_is_ok(status)) {
    device->proactor_pool = create_params->proactor_pool;
    iree_async_proactor_pool_retain(device->proactor_pool);
    status = iree_async_proactor_pool_get(device->proactor_pool, 0,
                                          &device->proactor);
  }

  iree_hal_vulkan_selected_queues_t selected_queues;
  memset(&selected_queues, 0, sizeof(selected_queues));
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_logical_device_handle(
        &device->instance, &device->physical_device, device_options,
        driver_options->requested_features, &selected_queues,
        &device->enabled_features, &device->enabled_extensions,
        &device->logical_device);
  }
  if (iree_status_is_ok(status)) {
    device->owns_logical_device = true;
    status = iree_hal_vulkan_libvulkan_load_device_syms(
        &device->instance.syms, device->logical_device, &device->syms);
  }
  if (iree_status_is_ok(status)) {
    VkDeviceQueueInfo2 queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
        .queueFamilyIndex = selected_queues.compute.family_index,
        .queueIndex = selected_queues.compute.queue_index,
    };
    iree_vkGetDeviceQueue2(IREE_VULKAN_DEVICE(&device->syms),
                           device->logical_device, &queue_info,
                           &selected_queues.compute.handle);
    queue_info.queueFamilyIndex = selected_queues.transfer.family_index;
    queue_info.queueIndex = selected_queues.transfer.queue_index;
    iree_vkGetDeviceQueue2(IREE_VULKAN_DEVICE(&device->syms),
                           device->logical_device, &queue_info,
                           &selected_queues.transfer.handle);

    device->compute_queue = selected_queues.compute;
    device->transfer_queue = selected_queues.transfer;
    device->queue_count = selected_queues.queue_count;
    device->queue_affinity_mask = (1ull << device->queue_count) - 1;
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_logical_device_create_with_selector(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_physical_device_selector_t* selector,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver_options);
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(selector);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!create_params->proactor_pool) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan logical device creation requires a proactor pool");
  }

  iree_hal_vulkan_device_options_t device_options =
      driver_options->device_options;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_device_options_parse(&device_options,
                                               (iree_string_pair_list_t){
                                                   .count = param_count,
                                                   .pairs = params,
                                               }));

  iree_hal_vulkan_instance_t instance;
  iree_hal_vulkan_physical_device_snapshot_t snapshot;
  iree_status_t status = iree_hal_vulkan_select_physical_device(
      libvulkan, driver_options, selector, host_allocator, &instance,
      &snapshot);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_logical_device_create_from_selection(
        identifier, driver_options, libvulkan, &device_options, create_params,
        host_allocator, &instance, &snapshot, out_device);
  }
  iree_hal_vulkan_physical_device_snapshot_deinitialize(host_allocator,
                                                        &snapshot);
  iree_hal_vulkan_instance_deinitialize(&instance);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_logical_device_create_by_id(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_hal_device_id_t device_id, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  const iree_hal_vulkan_physical_device_selector_t selector = {
      .mode = device_id == IREE_HAL_DEVICE_ID_DEFAULT
                  ? IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT
                  : IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID,
      .device_id = device_id,
  };
  return iree_hal_vulkan_logical_device_create_with_selector(
      identifier, driver_options, libvulkan, &selector, param_count, params,
      create_params, host_allocator, out_device);
}

iree_status_t iree_hal_vulkan_logical_device_create_by_path(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  const iree_hal_vulkan_physical_device_selector_t selector = {
      .mode = IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH,
      .device_path = device_path,
  };
  return iree_hal_vulkan_logical_device_create_with_selector(
      identifier, driver_options, libvulkan, &selector, param_count, params,
      create_params, host_allocator, out_device);
}

static const iree_hal_device_vtable_t iree_hal_vulkan_logical_device_vtable = {
    .destroy = iree_hal_vulkan_logical_device_destroy,
    .id = iree_hal_vulkan_logical_device_id,
    .host_allocator = iree_hal_vulkan_logical_device_host_allocator,
    .device_allocator = iree_hal_vulkan_logical_device_allocator,
    .replace_device_allocator = iree_hal_vulkan_replace_device_allocator,
    .replace_channel_provider = iree_hal_vulkan_replace_channel_provider,
    .trim = iree_hal_vulkan_logical_device_trim,
    .query_i64 = iree_hal_vulkan_logical_device_query_i64,
    .query_capabilities = iree_hal_vulkan_logical_device_query_capabilities,
    .topology_info = iree_hal_vulkan_logical_device_topology_info,
    .refine_topology_edge = iree_hal_vulkan_logical_device_refine_topology_edge,
    .assign_topology_info = iree_hal_vulkan_logical_device_assign_topology_info,
    .create_channel = iree_hal_vulkan_logical_device_create_channel,
    .create_command_buffer =
        iree_hal_vulkan_logical_device_create_command_buffer,
    .create_event = iree_hal_vulkan_logical_device_create_event,
    .create_executable_cache =
        iree_hal_vulkan_logical_device_create_executable_cache,
    .import_file = iree_hal_vulkan_logical_device_import_file,
    .create_semaphore = iree_hal_vulkan_logical_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_vulkan_logical_device_query_semaphore_compatibility,
    .query_queue_pool_backend =
        iree_hal_vulkan_logical_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_vulkan_logical_device_queue_alloca,
    .queue_dealloca = iree_hal_vulkan_logical_device_queue_dealloca,
    .queue_fill = iree_hal_vulkan_logical_device_queue_fill,
    .queue_update = iree_hal_vulkan_logical_device_queue_update,
    .queue_copy = iree_hal_vulkan_logical_device_queue_copy,
    .queue_read = iree_hal_vulkan_logical_device_queue_read,
    .queue_write = iree_hal_vulkan_logical_device_queue_write,
    .queue_host_call = iree_hal_vulkan_logical_device_queue_host_call,
    .queue_dispatch = iree_hal_vulkan_logical_device_queue_dispatch,
    .queue_execute = iree_hal_vulkan_logical_device_queue_execute,
    .queue_flush = iree_hal_vulkan_logical_device_queue_flush,
    .profiling_begin = iree_hal_vulkan_logical_device_profiling_begin,
    .profiling_flush = iree_hal_vulkan_logical_device_profiling_flush,
    .profiling_end = iree_hal_vulkan_logical_device_profiling_end,
    .external_capture_begin =
        iree_hal_vulkan_logical_device_external_capture_begin,
    .external_capture_end = iree_hal_vulkan_logical_device_external_capture_end,
};
