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

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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
