// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_PHYSICAL_DEVICE_SELECTION_H_
#define IREE_HAL_DRIVERS_VULKAN_PHYSICAL_DEVICE_SELECTION_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/physical_device.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Physical-device selection
//===----------------------------------------------------------------------===//

typedef enum iree_hal_vulkan_physical_device_selector_mode_e {
  IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_DEFAULT = 0,
  IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_ID = 1,
  IREE_HAL_VULKAN_PHYSICAL_DEVICE_SELECTOR_PATH = 2,
} iree_hal_vulkan_physical_device_selector_mode_t;

// Physical-device selector used by driver-owned logical device creation.
typedef struct iree_hal_vulkan_physical_device_selector_t {
  // Selection mode used when walking visible physical devices.
  iree_hal_vulkan_physical_device_selector_mode_t mode;

  // HAL device id to match when mode is ID.
  iree_hal_device_id_t device_id;

  // HAL device path to match when mode is PATH.
  iree_string_view_t device_path;
} iree_hal_vulkan_physical_device_selector_t;

// Returns whether |snapshot| satisfies |selector|.
//
// DEFAULT selectors only match devices satisfying the Vulkan HAL baseline.
// ID/PATH selectors match device identity only; callers that require baseline
// support must validate that separately so diagnostics can distinguish "not
// found" from "found but unsupported".
iree_status_t iree_hal_vulkan_physical_device_selector_match(
    const iree_hal_vulkan_physical_device_selector_t* selector,
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    bool* out_matches);

// Creates a driver-owned instance and selects one visible physical device.
//
// On success |out_instance| and |out_snapshot| are initialized and owned by the
// caller. On failure both outputs are deinitialized/zeroed.
iree_status_t iree_hal_vulkan_physical_device_select(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    const iree_hal_vulkan_driver_options_t* driver_options,
    const iree_hal_vulkan_physical_device_selector_t* selector,
    iree_allocator_t host_allocator, iree_hal_vulkan_instance_t* out_instance,
    iree_hal_vulkan_physical_device_snapshot_t* out_snapshot);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_PHYSICAL_DEVICE_SELECTION_H_
