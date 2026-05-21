// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_WEBGPU_DEVICE_H_
#define IREE_HAL_DRIVERS_WEBGPU_WEBGPU_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/handle_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_device_t
//===----------------------------------------------------------------------===//

// Flags controlling WebGPU device behavior.
enum iree_hal_webgpu_device_flag_bits_e {
  IREE_HAL_WEBGPU_DEVICE_FLAG_NONE = 0u,
  // The device owns the GPUDevice bridge handle and will destroy it when the
  // HAL device is released. Used by the driver factory path where the driver
  // created the GPUDevice via request_adapter → adapter_request_device.
  IREE_HAL_WEBGPU_DEVICE_FLAG_OWNS_DEVICE_HANDLE = 1u << 0,
};
typedef uint32_t iree_hal_webgpu_device_flags_t;

// Creates a WebGPU HAL device. See api.h for full documentation.
iree_status_t iree_hal_webgpu_device_create(
    iree_string_view_t identifier, iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_device_flags_t flags,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_WEBGPU_DEVICE_H_
