// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DEVICE_OPTIONS_H_
#define IREE_HAL_DRIVERS_VULKAN_DEVICE_OPTIONS_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Verifies that |options| contains only recognized, internally consistent
// device option values.
iree_status_t iree_hal_vulkan_device_options_verify(
    const iree_hal_vulkan_device_options_t* options);

// Applies string-pair device options to |options| and verifies the result.
iree_status_t iree_hal_vulkan_device_options_parse(
    iree_hal_vulkan_device_options_t* options, iree_string_pair_list_t params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_DEVICE_OPTIONS_H_
