// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_STATUS_UTIL_H_
#define IREE_HAL_DRIVERS_VULKAN_STATUS_UTIL_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/drivers/vulkan/vulkan_headers.h"
// clang-format on

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts a VkResult to an iree_status_t.
//
// Usage:
//   iree_status_t status = VK_RESULT_TO_STATUS(vkDoThing(...));
#define VK_RESULT_TO_STATUS(expr, ...) \
  iree_hal_vulkan_result_to_status((expr), __FILE__, __LINE__)

// IREE_RETURN_IF_ERROR but implicitly converts the VkResult return value to
// a Status.
//
// Usage:
//   VK_RETURN_IF_ERROR(vkDoThing(...), "message");
#define VK_RETURN_IF_ERROR(expr, ...) \
  IREE_RETURN_IF_ERROR(               \
      iree_hal_vulkan_result_to_status(expr, __FILE__, __LINE__), __VA_ARGS__)

// IREE_CHECK_OK but implicitly converts the VkResults return value to a
// Status and checks that it is OkStatus.
//
// Usage:
//   VK_CHECK_OK(vkDoThing(...));
#define VK_CHECK_OK(expr) \
  IREE_CHECK_OK(iree_hal_vulkan_result_to_status(expr, __FILE__, __LINE__))

// Converts a VkResult to a Status object.
//
// Vulkan considers the following as "success codes" and users should ensure
// they first check the result prior to converting:
//
// - VK_SUCCESS        -> OkStatus()
// - VK_NOT_READY      -> OkStatus()
// - VK_TIMEOUT        -> OkStatus()
// - VK_EVENT_SET      -> OkStatus()
// - VK_EVENT_RESET    -> OkStatus()
// - VK_INCOMPLETE     -> OkStatus()
// - VK_SUBOPTIMAL_KHR -> OkStatus()
//
// The rest are considered as "error codes":
//
// - VK_ERROR_OUT_OF_HOST_MEMORY          -> ResourceExhaustedError("VK...")
// - VK_ERROR_OUT_OF_DEVICE_MEMORY        -> ResourceExhaustedError("VK...")
// - VK_ERROR_INITIALIZATION_FAILED       -> InternalError("VK...")
// - VK_ERROR_DEVICE_LOST                 -> InternalError("VK...")
// - VK_ERROR_MEMORY_MAP_FAILED           -> InternalError("VK...")
// - VK_ERROR_LAYER_NOT_PRESENT           -> NotFoundError("VK...")
// - VK_ERROR_EXTENSION_NOT_PRESENT       -> NotFoundError("VK...")
// - VK_ERROR_FEATURE_NOT_PRESENT         -> NotFoundError("VK...")
// - VK_ERROR_INCOMPATIBLE_DRIVER         -> FailedPreconditionError("VK...")
// - VK_ERROR_TOO_MANY_OBJECTS            -> ResourceExhaustedError("VK...")
// - VK_ERROR_FORMAT_NOT_SUPPORTED        -> UnimplementedError("VK...")
// - VK_ERROR_FRAGMENTED_POOL             -> ResourceExhaustedError("VK...")
// - VK_ERROR_OUT_OF_POOL_MEMORY          -> ResourceExhaustedError("VK...")
// - VK_ERROR_INVALID_EXTERNAL_HANDLE     -> InvalidArgumentError("VK...")
// - VK_ERROR_SURFACE_LOST_KHR            -> InternalError("VK...")
// - VK_ERROR_NATIVE_WINDOW_IN_USE_KHR    -> InternalError("VK...")
// - VK_ERROR_OUT_OF_DATE_KHR             -> InternalError("VK...")
// - VK_ERROR_INCOMPATIBLE_DISPLAY_KHR    -> InternalError("VK...")
// - VK_ERROR_VALIDATION_FAILED_EXT       -> InternalError("VK...")
// - VK_ERROR_INVALID_SHADER_NV           -> InternalError("VK...")
// - VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT  -> InternalError
// - VK_ERROR_FRAGMENTATION_EXT           -> ResourceExhaustedError("VK...")
// - VK_ERROR_NOT_PERMITTED_EXT           -> PermissionDeniedError("VK...")
// - VK_ERROR_INVALID_DEVICE_ADDRESS_EXT  -> OutOfRangeError("VK...")
// - VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT -> InternalError("VK...")
iree_status_t iree_hal_vulkan_result_to_status(VkResult result,
                                               const char* file, uint32_t line);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_STATUS_UTIL_H_
