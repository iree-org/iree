// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_DRIVERS_CUDA_API_H_
#define IREE_HAL_DRIVERS_CUDA_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_cuda_device_t
//===----------------------------------------------------------------------===//

// Defines how command buffers are recorded and executed.
typedef enum iree_hal_cuda_command_buffer_mode_e {
  // Command buffers are recorded into CUDA graphs.
  IREE_HAL_CUDA_COMMAND_BUFFER_MODE_GRAPH = 0,
  // Command buffers are directly issued against a CUDA stream.
  IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM = 1,
} iree_hal_cuda_command_buffer_mode_t;

// ncclUniqueId exposed without exporting the NCCL headers.
typedef struct {
  char data[128];
} iree_hal_cuda_nccl_id_t;

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

  // Specifies how command buffers are recorded and executed.
  iree_hal_cuda_command_buffer_mode_t command_buffer_mode;

  // Allow executing command buffers against CUDA streams as they are recorded.
  // Only command buffers produced by the compiler that have the
  // IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION bit set will use this.
  bool allow_inline_execution;

  // Enables tracing of command buffers when IREE tracing is enabled.
  // May take advantage of additional extensions for more accurate timing or
  // hardware-specific performance counters.
  //
  // NOTE: tracing has a non-trivial overhead and will skew the timing of
  // submissions and introduce false barriers between dispatches. Use this to
  // identify slow dispatches and refine from there; be wary of whole-program
  // tracing with this enabled.
  bool stream_tracing;

  // Unowned provider of default channel configuration and creation.
  // Must remain valid for the lifetime of the driver/device.
  iree_hal_channel_provider_t channel_provider;
} iree_hal_cuda_device_params_t;

// Initializes |out_params| to default values.
IREE_API_EXPORT void iree_hal_cuda_device_params_initialize(
    iree_hal_cuda_device_params_t* out_params);

// Calls ncclGetUniqueId and returns the resulting unique ID.
// Only valid if NCCL is initialized by having set a channel provider when the
// device was configured.
IREE_API_EXPORT iree_status_t iree_hal_cuda_nccl_get_unique_id(
    iree_hal_device_t* device, iree_hal_cuda_nccl_id_t* out_id);

// Default implementation of the collective channel provider that uses MPI.
// Hosting layers would want to use their own implementation to exchange IDs.
IREE_API_EXPORT iree_status_t iree_hal_cuda_nccl_query_group_params(
    void* self, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_byte_span_t id_storage,
    iree_hal_channel_params_t* out_params);

//===----------------------------------------------------------------------===//
// iree_hal_cuda_driver_t
//===----------------------------------------------------------------------===//

// CUDA driver creation options.
typedef struct iree_hal_cuda_driver_options_t {
  // Index of the default CUDA device to use within the list of available
  // devices.
  int default_device_index;
  // The rank in the default collective channel.
  int default_rank;
  // The count of the default collective channel.
  int default_count;
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

#endif  // IREE_HAL_DRIVERS_CUDA_API_H_
