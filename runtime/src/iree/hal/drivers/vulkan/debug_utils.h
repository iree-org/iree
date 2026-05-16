// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DEBUG_UTILS_H_
#define IREE_HAL_DRIVERS_VULKAN_DEBUG_UTILS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_vulkan_debug_utils_flag_bits_e {
  // No VK_EXT_debug_utils families are enabled.
  IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_NONE = 0u,

  // vkSetDebugUtilsObjectNameEXT is usable for named Vulkan objects.
  IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_OBJECT_NAMES = 1u << 0,

  // vkCmdBegin/End/InsertDebugUtilsLabelEXT are usable during command replay.
  IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_COMMAND_LABELS = 1u << 1,
} iree_hal_vulkan_debug_utils_flag_bits_t;

typedef uint32_t iree_hal_vulkan_debug_utils_flags_t;

typedef enum iree_hal_vulkan_debug_utils_queue_role_bits_e {
  // No queue roles.
  IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_NONE = 0u,

  // Queue is used for compute-capable submissions.
  IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_COMPUTE = 1u << 0,

  // Queue is used for transfer-capable submissions.
  IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_TRANSFER = 1u << 1,

  // Queue is used for sparse memory binding submissions.
  IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_SPARSE_BINDING = 1u << 2,

  // Recognized queue role bits.
  IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_ALL_RECOGNIZED =
      IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_COMPUTE |
      IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_TRANSFER |
      IREE_HAL_VULKAN_DEBUG_UTILS_QUEUE_ROLE_SPARSE_BINDING,
} iree_hal_vulkan_debug_utils_queue_role_bits_t;

typedef uint32_t iree_hal_vulkan_debug_utils_queue_role_flags_t;

// Resolved VK_EXT_debug_utils capabilities for a logical device.
typedef struct iree_hal_vulkan_debug_utils_t {
  // Enabled debug-utils entry-point families.
  iree_hal_vulkan_debug_utils_flags_t flags;
} iree_hal_vulkan_debug_utils_t;

// Initializes |out_debug_utils| from requested behavior and loaded device
// symbols.
iree_status_t iree_hal_vulkan_debug_utils_initialize(
    iree_hal_vulkan_request_flags_t request_flags,
    const iree_hal_vulkan_device_syms_t* syms,
    iree_hal_vulkan_debug_utils_t* out_debug_utils);

// Returns true if |debug_utils| exposes every bit in |required_flags|.
bool iree_hal_vulkan_debug_utils_has(
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    iree_hal_vulkan_debug_utils_flags_t required_flags);

// Sets a Vulkan object name when object names are available.
iree_status_t iree_hal_vulkan_debug_utils_set_object_name(
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkObjectType object_type, uint64_t object_handle, iree_string_view_t name,
    iree_allocator_t host_allocator);

// Sets a Vulkan queue object name derived from |identifier|, queue roles, and
// the selected physical queue coordinates.
iree_status_t iree_hal_vulkan_debug_utils_set_queue_name(
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkQueue queue, iree_hal_vulkan_debug_utils_queue_role_flags_t role_flags,
    uint32_t queue_family_index, uint32_t queue_index,
    iree_string_view_t identifier, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_DEBUG_UTILS_H_
