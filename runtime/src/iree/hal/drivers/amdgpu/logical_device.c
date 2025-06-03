// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/logical_device.h"

#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/util/affinity.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_logical_device_options_t
//===----------------------------------------------------------------------===//

// Power-of-two size for the shared host small block pool in bytes.
// Used for small host-side transients/wrappers of device-side resources.
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_DEFAULT_SMALL_HOST_BLOCK_SIZE (8 * 1024)

// Minimum size of a small host block (some structures require at least this
// much memory).
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_SMALL_HOST_BLOCK_SIZE (4 * 1024)

// Power-of-two size for the shared host large block pool in bytes.
// Used for resource tracking and command buffer recording.
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_DEFAULT_LARGE_HOST_BLOCK_SIZE (64 * 1024)

// Minimum size of a large host block (some structures require at least this
// much memory).
#define IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_LARGE_HOST_BLOCK_SIZE (64 * 1024)

IREE_API_EXPORT void iree_hal_amdgpu_logical_device_options_initialize(
    iree_hal_amdgpu_logical_device_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));

  // TODO(benvanik): set defaults based on compiler configuration. Flags should
  // not be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.

  out_options->host_block_pools.small.block_size =
      IREE_HAL_AMDGPU_LOGICAL_DEVICE_DEFAULT_SMALL_HOST_BLOCK_SIZE;
  out_options->host_block_pools.large.block_size =
      IREE_HAL_AMDGPU_LOGICAL_DEVICE_DEFAULT_LARGE_HOST_BLOCK_SIZE;

  out_options->device_block_pools.small.block_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_SIZE_DEFAULT;
  out_options->device_block_pools.small.initial_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT;
  out_options->device_block_pools.large.block_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_SIZE_DEFAULT;
  out_options->device_block_pools.large.initial_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT;

  out_options->preallocate_pools = 1;
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_options_parse(
    iree_hal_amdgpu_logical_device_options_t* options,
    iree_string_pair_list_t params) {
  IREE_ASSERT_ARGUMENT(options);
  if (!params.count) return iree_ok_status();  // no-op
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): parameters.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_options_verify(
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): verify that the parameters are within expected ranges and
  // any requested features are supported.

  if (options->host_block_pools.small.block_size <
          IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_SMALL_HOST_BLOCK_SIZE ||
      !iree_host_size_is_power_of_two(
          options->host_block_pools.small.block_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "small host block pool size invalid, expected a "
        "power-of-two greater than %d and got %" PRIhsz,
        IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_SMALL_HOST_BLOCK_SIZE,
        options->host_block_pools.small.block_size);
  }
  if (options->host_block_pools.large.block_size <
          IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_LARGE_HOST_BLOCK_SIZE ||
      !iree_host_size_is_power_of_two(
          options->host_block_pools.large.block_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "large host block pool size invalid, expected a "
        "power-of-two greater than %d and got %" PRIhsz,
        IREE_HAL_AMDGPU_LOGICAL_DEVICE_MIN_LARGE_HOST_BLOCK_SIZE,
        options->host_block_pools.large.block_size);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_logical_device_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): implement iree_hal_amdgpu_logical_device_t.

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_logical_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "not yet implemented");
}
