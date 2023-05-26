// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_DRIVERS_WEBGPU_API_H_
#define IREE_HAL_DRIVERS_WEBGPU_API_H_

#include "experimental/webgpu/platform/webgpu.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_device_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): replace with flag list (easier to version).
enum iree_hal_webgpu_device_flag_bits_t {
  IREE_HAL_WEBGPU_DEVICE_RESERVED = 0u,
};
typedef uint32_t iree_hal_webgpu_device_flags_t;

typedef struct iree_hal_webgpu_device_options_t {
  // Flags controlling device behavior.
  iree_hal_webgpu_device_flags_t flags;

  // Size of the per-queue uniform staging buffer.
  // Larger buffer sizes will result in fewer flushes in large command buffers.
  iree_device_size_t queue_uniform_buffer_size;
} iree_hal_webgpu_device_options_t;

IREE_API_EXPORT void iree_hal_webgpu_device_options_initialize(
    iree_hal_webgpu_device_options_t* out_options);

IREE_API_EXPORT iree_status_t iree_hal_webgpu_wrap_device(
    iree_string_view_t identifier,
    const iree_hal_webgpu_device_options_t* options, WGPUDevice handle,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_driver_t
//===----------------------------------------------------------------------===//

typedef enum iree_hal_webgpu_driver_backend_e {
  IREE_HAL_WEBGPU_DRIVER_BACKEND_ANY = 0u,
  IREE_HAL_WEBGPU_DRIVER_BACKEND_D3D12 = 1u,
  IREE_HAL_WEBGPU_DRIVER_BACKEND_METAL = 2u,
  IREE_HAL_WEBGPU_DRIVER_BACKEND_VULKAN = 3u,
} iree_hal_webgpu_driver_backend_t;

typedef enum iree_hal_webgpu_driver_log_level_e {
  IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_OFF = 0u,
  IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_ERROR = 1u,
  IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_WARNING = 2u,
  IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_INFO = 3u,
  IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_DEBUG = 4u,
  IREE_HAL_WEBGPU_DRIVER_LOG_LEVEL_TRACE = 5u,
} iree_hal_webgpu_driver_log_level_t;

// WebGPU native driver creation options.
typedef struct iree_hal_webgpu_driver_options_t {
  // Logging level for messages logged to stderr. Disabled by default.
  iree_hal_webgpu_driver_log_level_t log_level;

  // Preferred backend - ignored if backend is not available.
  iree_hal_webgpu_driver_backend_t backend_preference;

  // TODO(benvanik): remove this single setting - it would be nice instead to
  // pass a list to force device enumeration/matrix expansion or omit entirely
  // to have auto-discovered options based on capabilities. Right now this
  // forces all devices - even if from different vendors - to have the same
  // options.
  // Options to use for all devices created by the driver.
  iree_hal_webgpu_device_options_t device_options;

  // Controls adapter selection when multiple exist in the system having
  // different power characteristics (such as integrated vs discrete GPUs).
  WGPUPowerPreference power_preference;
} iree_hal_webgpu_driver_options_t;

IREE_API_EXPORT void iree_hal_webgpu_driver_options_initialize(
    iree_hal_webgpu_driver_options_t* out_options);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_API_H_
