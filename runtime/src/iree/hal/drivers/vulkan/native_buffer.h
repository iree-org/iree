// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_NATIVE_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_NATIVE_BUFFER_H_

// clang-format off: must be included before all other headers.
#include "iree/hal/drivers/vulkan/vulkan_headers.h"  // IWYU pragma: export
// clang-format on

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef void(IREE_API_PTR* iree_hal_vulkan_native_buffer_release_fn_t)(
    void* user_data, iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkDeviceMemory device_memory, VkBuffer handle);

// A callback issued when a buffer is released.
typedef struct {
  // Callback function pointer.
  iree_hal_vulkan_native_buffer_release_fn_t fn;
  // User data passed to the callback function. Unowned.
  void* user_data;
} iree_hal_vulkan_native_buffer_release_callback_t;

// Wraps a Vulkan |buffer| bound to device |device_memory| for exposure into the
// HAL. The provided callback is made when the buffer is destroyed to allow the
// caller to clean up as appropriate.
iree_status_t iree_hal_vulkan_native_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkDeviceMemory device_memory, VkBuffer handle,
    iree_hal_vulkan_native_buffer_release_callback_t internal_release_callback,
    iree_hal_buffer_release_callback_t user_release_callback,
    iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_NATIVE_BUFFER_H_
