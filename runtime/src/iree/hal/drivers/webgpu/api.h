// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_API_H_
#define IREE_HAL_DRIVERS_WEBGPU_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/webgpu/webgpu_device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_device_t
//===----------------------------------------------------------------------===//

// Wraps a user-provided GPUDevice as an IREE HAL device.
//
// This is the primary production API for JS embeddings. The caller provides a
// GPUDevice they already own (from navigator.gpu or node-dawn) and receives a
// fully initialized HAL device. No driver registration or factory is needed —
// link only the webgpu_device target.
//
// The caller retains ownership of |device_handle| — IREE will not destroy it.
// The caller must ensure the GPUDevice outlives the returned HAL device.
//
// Internally creates an inline-mode proactor for async I/O. For worker-mode
// deployments with an externally managed proactor pool, use
// iree_hal_webgpu_device_create() directly.
//
// |out_device| must be released by the caller (see iree_hal_device_release).
IREE_API_EXPORT iree_status_t iree_hal_webgpu_device_wrap(
    iree_string_view_t identifier, iree_hal_webgpu_handle_t device_handle,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_driver_t
//===----------------------------------------------------------------------===//

// Parameters for configuring an iree_hal_webgpu_driver_t.
// Must be initialized with iree_hal_webgpu_driver_options_initialize prior to
// use.
typedef struct iree_hal_webgpu_driver_options_t {
  int reserved;
} iree_hal_webgpu_driver_options_t;

// Initializes the given |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_webgpu_driver_options_initialize(
    iree_hal_webgpu_driver_options_t* out_options);

// Creates a WebGPU HAL driver from which devices can be enumerated and created.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. When compiling programs with the IREE
// compiler this must match the value used by IREE::HAL::TargetDevice.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_webgpu_driver_create(
    iree_string_view_t identifier,
    const iree_hal_webgpu_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_WEBGPU_API_H_
