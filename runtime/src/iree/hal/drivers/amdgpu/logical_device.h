// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_LOGICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDGPU_LOGICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_logical_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_logical_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;

  // HSA system instantiated from the user-provided topology.
  // This retains our fixed resources (like the device library) on the subset of
  // the agents available in HSA that are represented as physical devices.
  iree_hal_amdgpu_system_t system;

  // Logical allocator.
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Count of physical devices.
  iree_host_size_t physical_device_count;
  // One or more physical devices backing the logical device.
  iree_hal_amdgpu_physical_device_t* physical_devices[];

  // + trailing identifier string storage
} iree_hal_amdgpu_logical_device_t;

// Creates a AMDGPU logical HAL device with the given |params| and |topology|.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by `IREE::HAL::TargetDevice`.
//
// |out_logical_device| must be released by the caller (see
// iree_hal_device_release).
iree_status_t iree_hal_amdgpu_logical_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_amdgpu_logical_device_t** out_logical_device);

#endif  // IREE_HAL_DRIVERS_AMDGPU_LOGICAL_DEVICE_H_
