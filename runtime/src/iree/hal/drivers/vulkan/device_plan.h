// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DEVICE_PLAN_H_
#define IREE_HAL_DRIVERS_VULKAN_DEVICE_PLAN_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/physical_device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Invalid Vulkan queue family index used for absent optional queue roles.
#define IREE_HAL_VULKAN_QUEUE_FAMILY_INVALID UINT32_MAX

// Default count of public HAL queue affinity lanes exposed by Vulkan devices.
#define IREE_HAL_VULKAN_DEFAULT_QUEUE_COUNT 2

// Maximum live Vulkan queue lanes needed for the current queue role model.
#define IREE_HAL_VULKAN_MAX_QUEUE_LANES 3

// Maximum recognized device extension names enabled during VkDevice creation.
#define IREE_HAL_VULKAN_MAX_DEVICE_EXTENSION_NAMES 8

// Maximum queue-family create infos needed for current queue role assignment.
#define IREE_HAL_VULKAN_MAX_QUEUE_CREATE_INFOS 3

// Selected physical Vulkan queue identity and queue-family capabilities for
// one logical queue role.
typedef struct iree_hal_vulkan_queue_selection_t {
  // Queue family index from the selected physical device.
  uint32_t family_index;

  // Queue index within the selected family.
  uint32_t queue_index;

  // Queue family capability flags cached from the physical snapshot.
  VkQueueFlags flags;

  // Valid timestamp bits reported by the selected queue family.
  uint32_t timestamp_valid_bits;

  // Current public HAL queue affinity bit that maps to this role.
  iree_hal_queue_affinity_t affinity;
} iree_hal_vulkan_queue_selection_t;

// Queue role assignment for a logical device plan.
typedef struct iree_hal_vulkan_queue_assignment_t {
  // Compute queue used for dispatch-capable operations.
  iree_hal_vulkan_queue_selection_t compute;

  // Transfer queue used for copy/fill/update-capable operations.
  iree_hal_vulkan_queue_selection_t transfer;

  // Internal queue used for sparse memory binding operations.
  iree_hal_vulkan_queue_selection_t sparse_binding;

  // Count of distinct HAL queues exposed through queue affinity.
  iree_host_size_t queue_count;
} iree_hal_vulkan_queue_assignment_t;

// Planned logical-device policy decisions that can be computed without a live
// VkDevice handle.
typedef struct iree_hal_vulkan_device_plan_t {
  // Queue role assignment selected for the logical device.
  iree_hal_vulkan_queue_assignment_t queue_assignment;

  // Non-device-feature behavior requested for the logical device.
  iree_hal_vulkan_request_flags_t request_flags;

  // HAL feature bits enabled or supplied on the logical device.
  iree_hal_vulkan_features_t enabled_features;

  // Recognized Vulkan device extension bits enabled on the logical device.
  iree_hal_vulkan_device_extensions_t enabled_extensions;

  // Executable dispatch ABI bits enabled for the logical device.
  iree_hal_vulkan_dispatch_abis_t enabled_dispatch_abis;

  // Vulkan extension names enabled during driver-owned VkDevice creation.
  const char*
      enabled_extension_names[IREE_HAL_VULKAN_MAX_DEVICE_EXTENSION_NAMES];

  // Count of valid entries in enabled_extension_names.
  uint32_t enabled_extension_count;

  // Queue priorities referenced by queue_create_infos.
  float queue_priorities[IREE_HAL_VULKAN_MAX_QUEUE_CREATE_INFOS];

  // Queue-family create infos for driver-owned VkDevice creation.
  VkDeviceQueueCreateInfo
      queue_create_infos[IREE_HAL_VULKAN_MAX_QUEUE_CREATE_INFOS];

  // Queue-priority start index for each entry in queue_create_infos.
  uint32_t queue_priority_offsets[IREE_HAL_VULKAN_MAX_QUEUE_CREATE_INFOS];

  // Count of valid entries in queue_create_infos.
  uint32_t queue_create_info_count;

  // Vulkan 1.3 features enabled during driver-owned VkDevice creation.
  VkPhysicalDeviceVulkan13Features enabled_features13;

  // Vulkan 1.2 features enabled during driver-owned VkDevice creation.
  VkPhysicalDeviceVulkan12Features enabled_features12;

  // Base features enabled during driver-owned VkDevice creation.
  VkPhysicalDeviceFeatures2 enabled_features2;
} iree_hal_vulkan_device_plan_t;

// Returns true if both queue selections refer to the same physical queue.
bool iree_hal_vulkan_queue_selection_is_same(
    const iree_hal_vulkan_queue_selection_t* lhs,
    const iree_hal_vulkan_queue_selection_t* rhs);

// Returns true if the queue assignment has a sparse-binding queue role.
bool iree_hal_vulkan_queue_assignment_has_sparse_binding(
    const iree_hal_vulkan_queue_assignment_t* queue_assignment);

// Normalizes a public HAL queue affinity against the supported queue mask.
iree_status_t iree_hal_vulkan_queue_affinity_normalize(
    iree_hal_queue_affinity_t supported_affinity,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_queue_affinity_t* out_normalized_affinity);

// Initializes |out_plan| for a driver-owned VkDevice created from |snapshot|.
iree_status_t iree_hal_vulkan_device_plan_initialize_for_create(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* device_options,
    iree_hal_vulkan_request_flags_t request_flags,
    iree_hal_vulkan_features_t requested_features,
    iree_hal_vulkan_device_plan_t* out_plan);

// Initializes |out_plan| for a wrapped external VkDevice.
iree_status_t iree_hal_vulkan_device_plan_initialize_for_wrap(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* device_options,
    const iree_hal_vulkan_external_device_params_t* external_device_params,
    iree_hal_vulkan_device_plan_t* out_plan);

// Produces a VkDeviceCreateInfo view over |plan| for driver-owned device
// creation.
//
// The returned pointers reference storage inside |plan|. Call this after the
// plan is in its final storage location; the function refreshes internal pNext
// and queue-priority pointers before filling |out_create_info|.
void iree_hal_vulkan_device_plan_make_create_info(
    iree_hal_vulkan_device_plan_t* plan, VkDeviceCreateInfo* out_create_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_DEVICE_PLAN_H_
