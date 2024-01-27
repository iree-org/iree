// Copyright 2023 The IREE Authors
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

// How command buffers are recorded and executed.
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

// Parameters defining a CUmemoryPool.
typedef struct iree_hal_cuda_memory_pool_params_t {
  // Minimum number of bytes to keep in the pool when trimming with
  // iree_hal_device_trim.
  uint64_t minimum_capacity;
  // Soft maximum number of bytes to keep in the pool.
  // When more than this is allocated the extra will be freed at the next
  // device synchronization in order to remain under the threshold.
  uint64_t release_threshold;
  // TODO: per-device access permissions array.
} iree_hal_cuda_memory_pool_params_t;

// Parameters for each CUmemoryPool used for queue-ordered allocations.
typedef struct iree_hal_cuda_memory_pooling_params_t {
  // Used exclusively for DEVICE_LOCAL allocations.
  iree_hal_cuda_memory_pool_params_t device_local;
  // Used for any host-visible/host-local memory types.
  iree_hal_cuda_memory_pool_params_t other;
} iree_hal_cuda_memory_pooling_params_t;

// Parameters configuring an iree_hal_cuda_device_t.
// Must be initialized with iree_hal_cuda_device_params_initialize prior to
// use.
typedef struct iree_hal_cuda_device_params_t {
  // Number of queues exposed on the device.
  // Each queue acts as a separate synchronization scope where all work executes
  // concurrently unless prohibited by semaphores.
  iree_host_size_t queue_count;

  // Total size of each block in the device shared block pool.
  // Larger sizes will lower overhead and ensure the heap isn't hit for
  // transient allocations while also increasing memory consumption.
  iree_host_size_t arena_block_size;

  // The host and device event pool capacity.
  // The CUDA driver implements semaphore with host and device events. This
  // parameter controls the size of those pools. Larger values would make
  // creating semaphore values quicker, though with increased memory
  // consumption.
  iree_host_size_t event_pool_capacity;

  // Specifies how command buffers are recorded and executed.
  iree_hal_cuda_command_buffer_mode_t command_buffer_mode;

  // Enables tracing of command buffers when IREE tracing is enabled.
  // May take advantage of additional extensions for more accurate timing or
  // hardware-specific performance counters.
  //
  // NOTE: tracing has a non-trivial overhead and will skew the timing of
  // submissions and introduce false barriers between dispatches. Use this to
  // identify slow dispatches and refine from there; be wary of whole-program
  // tracing with this enabled.
  bool stream_tracing;

  // Whether to use async allocations even if reported as available by the
  // device. Defaults to true when the device supports it.
  bool async_allocations;

  // Parameters for each CUmemoryPool used for queue-ordered allocations.
  iree_hal_cuda_memory_pooling_params_t memory_pools;
} iree_hal_cuda_device_params_t;

// Initializes |out_params| to default values.
IREE_API_EXPORT void iree_hal_cuda_device_params_initialize(
    iree_hal_cuda_device_params_t* out_params);

//===----------------------------------------------------------------------===//
// iree_hal_cuda_driver_t
//===----------------------------------------------------------------------===//

// CUDA HAL driver creation options.
typedef struct iree_hal_cuda_driver_options_t {
  // The index of the default CUDA device to use within the list of available
  // devices.
  int default_device_index;
} iree_hal_cuda_driver_options_t;

// Initializes the given |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_cuda_driver_options_initialize(
    iree_hal_cuda_driver_options_t* out_options);

// Creates a CUDA HAL driver with the given |options|, from which CUDA devices
// can be enumerated and created with specific parameters.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_cuda_driver_create(
    iree_string_view_t identifier,
    const iree_hal_cuda_driver_options_t* options,
    const iree_hal_cuda_device_params_t* default_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_API_H_
