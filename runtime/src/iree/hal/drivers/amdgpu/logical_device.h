// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_LOGICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDGPU_LOGICAL_DEVICE_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_async_proactor_pool_t iree_async_proactor_pool_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_epoch_signal_table_t
    iree_hal_amdgpu_epoch_signal_table_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;
typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_block_pools_t
//===----------------------------------------------------------------------===//

// Block pools for host memory blocks of various sizes.
typedef struct iree_hal_amdgpu_host_block_pools_t {
  // Used for small allocations of around 1-4KB.
  iree_arena_block_pool_t small;
  // Used for large page-sized allocations of 32-64kB.
  iree_arena_block_pool_t large;
} iree_hal_amdgpu_host_block_pools_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_logical_device_t
//===----------------------------------------------------------------------===//

// A logical HAL device composed of one or more physical devices.
// Each physical device may have one or more HAL queues that map to one or more
// hardware queues.
//
// All physical devices that make up a logical device are expected to be the
// same today. It's possible to support heterogeneous devices but the
// implementation does not currently handle taking the minimum capabilities and
// limits.
typedef struct iree_hal_amdgpu_logical_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Proactor pool retained from create_params; provides async I/O proactors.
  iree_async_proactor_pool_t* proactor_pool;
  // Proactor borrowed from the pool for this device's async operations.
  iree_async_proactor_t* proactor;

  // Shared frontier tracker for cross-device causal ordering.
  // Borrowed from the session — valid as long as the session is alive.
  // NULL if frontier-based fast paths are not enabled.
  iree_async_frontier_tracker_t* frontier_tracker;

  // This device's base axis and logical-device epoch counter for frontier
  // tracking. Host queues derive per-queue axes from this base axis and publish
  // completion epochs through the shared epoch-signal table.
  iree_async_axis_t axis;
  iree_atomic_int64_t epoch;

  iree_string_view_t identifier;

  // Block pools for host memory blocks of various sizes.
  iree_hal_amdgpu_host_block_pools_t host_block_pools;

  // HSA system instantiated from the user-provided topology.
  // This retains our fixed resources (like the device library) on the subset of
  // the agents available in HSA that are represented as physical devices.
  iree_hal_amdgpu_system_t* system;

  // Shared epoch-signal table used by all host queues on this logical device
  // for local cross-queue barrier emission. Owned by the logical device and
  // deregistered by each host queue before this table is freed.
  iree_hal_amdgpu_epoch_signal_table_t* host_queue_epoch_table;

  // Mask indicating which queue affinities are valid.
  iree_hal_queue_affinity_t queue_affinity_mask;

  // Logical allocator.
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Sticky logical device-global error flag.
  // Asynchronous errors from subsystems get routed back to this as our "device
  // loss" trigger.
  iree_atomic_intptr_t failure_status;

  iree_hal_device_topology_info_t topology_info;

  // Count of physical devices.
  iree_host_size_t physical_device_count;
  // One or more physical devices backing the logical device.
  iree_hal_amdgpu_physical_device_t*
      physical_devices[/*physical_device_count*/];

  // + trailing identifier string storage
} iree_hal_amdgpu_logical_device_t;

// Creates a AMDGPU logical HAL device with the given |options| and |topology|.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by `IREE::HAL::TargetDevice`.
//
// |options|, |libhsa|, and |topology| will be cloned into the device and need
// not live beyond the call.
//
// |out_logical_device| must be released by the caller (see
// iree_hal_device_release).
iree_status_t iree_hal_amdgpu_logical_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#endif  // IREE_HAL_DRIVERS_AMDGPU_LOGICAL_DEVICE_H_
