// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_API_H_
#define IREE_HAL_DRIVERS_AMDGPU_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

// Exported for API interop:
#include "iree/hal/drivers/amdgpu/util/libhsa.h"    // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/util/topology.h"  // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_logical_device_t
//===----------------------------------------------------------------------===//

// Parameters configuring an iree_hal_amdgpu_logical_device_t.
// Must be initialized with iree_hal_amdgpu_logical_device_options_initialize
// prior to use.
typedef struct iree_hal_amdgpu_logical_device_options_t {
  // Size of a block in each host block pool.
  struct {
    struct {
      // Size in bytes of a small host block. Must be a power of two.
      iree_host_size_t block_size;
    } small;
    struct {
      // Size in bytes of a large host block. Must be a power of two.
      iree_host_size_t block_size;
    } large;
  } host_block_pools;

  // Size of a block in each device block pool.
  struct {
    struct {
      // Size in bytes of a small device block. Must be a power of two.
      iree_device_size_t block_size;
      // Initial small block pool block allocation count.
      iree_host_size_t initial_capacity;
    } small;
    struct {
      // Size in bytes of a large device block. Must be a power of two.
      iree_device_size_t block_size;
      // Initial large block pool block allocation count.
      iree_host_size_t initial_capacity;
    } large;
  } device_block_pools;

  // Preallocates a reasonable number of resources in pools to reduce initial
  // execution latency.
  uint64_t preallocate_pools : 1;

  // Enables dispatch-level tracing (if device instrumentation is compiled in).
  uint64_t trace_execution : 1;

  // Forces queues to run one entry at a time instead of overlapping or
  // aggressively scheduling queue entries out-of-order.
  uint64_t exclusive_execution : 1;

  // Uses HSA_WAIT_STATE_ACTIVE for up to duration before switching to
  // HSA_WAIT_STATE_BLOCKED. Above zero this will increase CPU usage in cases
  // where the waits are long and decrease latency in cases where the waits are
  // short.
  //
  // TODO(benvanik): add as a value to device wait semaphores instead.
  iree_duration_t wait_active_for_ns;
} iree_hal_amdgpu_logical_device_options_t;

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_amdgpu_logical_device_options_initialize(
    iree_hal_amdgpu_logical_device_options_t* out_options);

// Parses |params| and updates |options|.
// String views may be set to reference strings in the original parameters and
// the caller must ensure the options does not outlive the storage.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_options_parse(
    iree_hal_amdgpu_logical_device_options_t* options,
    iree_string_pair_list_t params);

// Creates a AMDGPU HAL device with the given |options| and |topology|.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by `IREE::HAL::TargetDevice`.
//
// |options|, |libhsa|, and |topology| will be cloned into the device and need
// not live beyond the call.
//
// |out_device| must be released by the caller (see iree_hal_device_release).
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_driver_t
//===----------------------------------------------------------------------===//

// Parameters for configuring an iree_hal_amdgpu_driver_t.
// Must be initialized with iree_hal_amdgpu_driver_options_initialize prior to
// use.
typedef struct iree_hal_amdgpu_driver_options_t {
  // Search paths (directories or files) for finding the HSA runtime shared
  // library.
  iree_string_view_list_t libhsa_search_paths;

  // Default device options when none are provided during device creation.
  iree_hal_amdgpu_logical_device_options_t default_device_options;
} iree_hal_amdgpu_driver_options_t;

// Initializes the given |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_amdgpu_driver_options_initialize(
    iree_hal_amdgpu_driver_options_t* out_options);

// Parses |params| and updates |options|.
// String views may be set to reference strings in the original parameters and
// the caller must ensure the options does not outlive the storage.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_driver_options_parse(
    iree_hal_amdgpu_driver_options_t* options, iree_string_pair_list_t params);

// Creates a AMDGPU HAL driver with the given |options|, from which AMDGPU
// devices can be enumerated and created with specific parameters.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by IREE::HAL::TargetDevice.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_driver_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_API_H_
