// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_CUDA_API_H_
#define IREE_HAL_CUDA_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parameters configuring an iree_hal_cuda_device_t.
// Must be initialized with iree_hal_cuda_device_params_initialize prior to use.
typedef struct iree_hal_cuda_device_params_t {
  // Number of queues exposed on the device.
  // Each queue acts as a separate synchronization scope where all work executes
  // concurrently unless prohibited by semaphores.
  iree_host_size_t queue_count;

  // Total size of each block in the device shared block pool.
  // Larger sizes will lower overhead and ensure the heap isn't hit for
  // transient allocations while also increasing memory consumption.
  iree_host_size_t arena_block_size;

  // Switch for using deferred command buffer or default graph command buffer
  bool use_deferred_submission;
} iree_hal_cuda_device_params_t;

// Initializes |out_params| to default values.
void iree_hal_cuda_device_params_initialize(
    iree_hal_cuda_device_params_t* out_params);

//===----------------------------------------------------------------------===//
// iree_hal_cuda_driver_t
//===----------------------------------------------------------------------===//

// CUDA driver creation options.
typedef struct iree_hal_cuda_driver_options_t {
  // Index of the default CUDA device to use within the list of available
  // devices.
  int default_device_index;
} iree_hal_cuda_driver_options_t;

IREE_API_EXPORT void iree_hal_cuda_driver_options_initialize(
    iree_hal_cuda_driver_options_t* out_options);

// Creates a CUDA HAL driver that manage its own CUcontext.
//
// |out_driver| must be released by the caller (see |iree_hal_driver_release|).
IREE_API_EXPORT iree_status_t iree_hal_cuda_driver_create(
    iree_string_view_t identifier,
    const iree_hal_cuda_device_params_t* default_params,
    const iree_hal_cuda_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

// TODO(thomasraoux): Support importing a CUcontext from app.

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_API_H_
