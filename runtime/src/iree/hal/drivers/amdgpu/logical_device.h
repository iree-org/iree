// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_LOGICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDGPU_LOGICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/buffer_pool.h"
#include "iree/hal/drivers/amdgpu/command_buffer.h"
#include "iree/hal/drivers/amdgpu/semaphore_pool.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;
typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

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
  iree_string_view_t identifier;

  // Block pools for host memory blocks of various sizes.
  iree_hal_amdgpu_host_block_pools_t host_block_pools;

  // HSA system instantiated from the user-provided topology.
  // This retains our fixed resources (like the device library) on the subset of
  // the agents available in HSA that are represented as physical devices.
  iree_hal_amdgpu_system_t* system;

  // Mask indicating which queue affinities are valid.
  iree_hal_queue_affinity_t queue_affinity_mask;

  // Logical allocator.
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Growable pool of HAL semaphores and their matching device allocations.
  // Semaphores can be used on any CPU and GPU agent in the system.
  iree_hal_amdgpu_semaphore_pool_t semaphore_pool;

  // Growable pool of transient buffers and their matching device handles.
  // Allocation handles can be used on any CPU and GPU agent in the system.
  iree_hal_amdgpu_buffer_pool_t buffer_pool;

  // Sticky logical device-global error flag.
  // Asynchronous errors from subsystems get routed back to this as our "device
  // loss" trigger.
  iree_atomic_intptr_t failure_status;

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
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

#endif  // IREE_HAL_DRIVERS_AMDGPU_LOGICAL_DEVICE_H_
