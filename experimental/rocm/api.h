// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_ROCM_API_H_
#define IREE_HAL_ROCM_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_rocm_device_t
//===----------------------------------------------------------------------===//

// Defines how command buffers are recorded and executed.
typedef enum iree_hal_rocm_command_buffer_mode_e {
  // Command buffers are recorded into ROCM null stream.
  IREE_HAL_ROCM_COMMAND_BUFFER_MODE_DIRECT = 0,
  // Command buffers are directly issued against ROCM stream.
  IREE_HAL_ROCM_COMMAND_BUFFER_MODE_STREAM = 1,
  // Command buffers are recorded into ROCM graphs.
  IREE_HAL_ROCM_COMMAND_BUFFER_MODE_GRAPH = 2,
} iree_hal_rocm_command_buffer_mode_t;

// Parameters configuring an iree_hal_rocm_device_t.
// Must be initialized with iree_hal_rocm_device_params_initialize prior to use.
typedef struct iree_hal_rocm_device_params_t {

  // Total size of each block in the device shared block pool.
  // Larger sizes will lower overhead and ensure the heap isn't hit for
  // transient allocations while also increasing memory consumption.
  iree_host_size_t arena_block_size;

  // Specifies how command buffers are recorded and executed.
  iree_hal_rocm_command_buffer_mode_t command_buffer_mode;

  // Enables tracing of command buffers when IREE tracing is enabled.
  // May take advantage of additional extensions for more accurate timing or
  // hardware-specific performance counters.
  //
  // NOTE: tracing has a non-trivial overhead and will skew the timing of
  // submissions and introduce false barriers between dispatches. Use this to
  // identify slow dispatches and refine from there; be wary of whole-program
  // tracing with this enabled.
  bool stream_tracing;

} iree_hal_rocm_device_params_t;

// Initializes |out_params| to default values.
IREE_API_EXPORT void iree_hal_rocm_device_params_initialize(
    iree_hal_rocm_device_params_t* out_params);

//===----------------------------------------------------------------------===//
// iree_hal_rocm_driver_t
//===----------------------------------------------------------------------===//

// ROCM driver creation options.
typedef struct iree_hal_rocm_driver_options_t {
  // Index of the default ROCM device to use within the list of available
  // devices.
  int default_device_index;
} iree_hal_rocm_driver_options_t;

IREE_API_EXPORT void iree_hal_rocm_driver_options_initialize(
    iree_hal_rocm_driver_options_t *out_options);

// Creates a ROCM HAL driver that manage its own hipcontext.
//
// |out_driver| must be released by the caller (see |iree_hal_driver_release|).
IREE_API_EXPORT iree_status_t iree_hal_rocm_driver_create(
    iree_string_view_t identifier,
    const iree_hal_rocm_device_params_t* default_params,
    const iree_hal_rocm_driver_options_t *options,
    iree_allocator_t host_allocator, iree_hal_driver_t **out_driver);

// TODO(thomasraoux): Support importing a CUcontext from app.

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_API_H_
