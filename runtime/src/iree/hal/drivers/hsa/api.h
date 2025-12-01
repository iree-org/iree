// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_DRIVERS_HSA_API_H_
#define IREE_HAL_DRIVERS_HSA_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_hsa_device_t
//===----------------------------------------------------------------------===//

// Parameters defining memory pool configuration.
typedef struct iree_hal_hsa_memory_pool_params_t {
  // Minimum number of bytes to keep in the pool when trimming with
  // iree_hal_device_trim.
  uint64_t minimum_capacity;
  // Soft maximum number of bytes to keep in the pool.
  uint64_t release_threshold;
} iree_hal_hsa_memory_pool_params_t;

// Parameters for memory pooling.
typedef struct iree_hal_hsa_memory_pooling_params_t {
  // Used exclusively for DEVICE_LOCAL allocations.
  iree_hal_hsa_memory_pool_params_t device_local;
  // Used for any host-visible/host-local memory types.
  iree_hal_hsa_memory_pool_params_t other;
} iree_hal_hsa_memory_pooling_params_t;

// Parameters configuring an iree_hal_hsa_device_t.
// Must be initialized with iree_hal_hsa_device_params_initialize prior to use.
typedef struct iree_hal_hsa_device_params_t {
  // Number of queues exposed on the device.
  iree_host_size_t queue_count;

  // Total size of each block in the device shared block pool.
  iree_host_size_t arena_block_size;

  // The host and device event pool capacity.
  iree_host_size_t event_pool_capacity;

  // Controls the verbosity of command buffers tracing when IREE tracing is
  // enabled.
  int32_t stream_tracing;

  // Whether to use async allocations if supported by the device.
  bool async_allocations;

  // The reserved buffer size for asynchronous file transfers.
  iree_device_size_t file_transfer_buffer_size;

  // The maximum chunk size for any single asynchronous file transfer.
  iree_device_size_t file_transfer_chunk_size;

  // Parameters for memory pooling.
  iree_hal_hsa_memory_pooling_params_t memory_pools;

  // Allow executing command buffers inline when possible.
  bool allow_inline_execution;
} iree_hal_hsa_device_params_t;

// Initializes |out_params| to default values.
IREE_API_EXPORT void iree_hal_hsa_device_params_initialize(
    iree_hal_hsa_device_params_t* out_params);

//===----------------------------------------------------------------------===//
// iree_hal_hsa_driver_t
//===----------------------------------------------------------------------===//

// HSA HAL driver creation options.
typedef struct iree_hal_hsa_driver_options_t {
  // The index of the default HSA device to use within the list of available
  // devices.
  int32_t default_device_index;

  // List of paths to guide searching for the dynamic libhsa-runtime64.so,
  // which contains the backing HSA runtime library.
  iree_string_view_t* hsa_lib_search_paths;
  iree_host_size_t hsa_lib_search_path_count;
} iree_hal_hsa_driver_options_t;

// Initializes the given |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_hsa_driver_options_initialize(
    iree_hal_hsa_driver_options_t* out_options);

// Creates an HSA HAL driver with the given |options|, from which HSA devices
// can be enumerated and created with specific parameters.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_hsa_driver_create(
    iree_string_view_t identifier, const iree_hal_hsa_driver_options_t* options,
    const iree_hal_hsa_device_params_t* default_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HSA_API_H_

