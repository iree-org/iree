// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device.h"

#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_options_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_SMALL_DEVICE_BLOCK_SIZE (1 * 1024)
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_LARGE_DEVICE_BLOCK_SIZE (64 * 1024)

// Not currently configurable but could be:
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_SMALL_PAGE_SIZE 128
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_LARGE_PAGE_SIZE 4096
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_SMALL_PAGE_SIZE 128
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_LARGE_PAGE_SIZE 4096

void iree_hal_amdgpu_physical_device_options_initialize(
    iree_hal_amdgpu_physical_device_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));

  out_options->device_block_pools.small.block_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_SIZE_DEFAULT;
  out_options->device_block_pools.large.min_blocks_per_allocation =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT;
  out_options->device_block_pools.small.initial_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT;

  out_options->device_block_pools.large.block_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_SIZE_DEFAULT;
  out_options->device_block_pools.large.min_blocks_per_allocation =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT;
  out_options->device_block_pools.large.initial_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_INITIAL_CAPACITY_DEFAULT;

  out_options->host_block_pool_size =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_HOST_BLOCK_SIZE_DEFAULT;
}

iree_status_t iree_hal_amdgpu_physical_device_options_verify(
    const iree_hal_amdgpu_physical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    hsa_agent_t gpu_agent) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(libhsa);

  // Verify pool sizes.
  if (options->device_block_pools.small.block_size <
          IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_SMALL_DEVICE_BLOCK_SIZE ||
      !iree_host_size_is_power_of_two(
          options->device_block_pools.small.block_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "small device block pool size invalid, expected a "
        "power-of-two greater than %d and got %" PRIhsz,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_SMALL_DEVICE_BLOCK_SIZE,
        options->device_block_pools.small.block_size);
  }
  if (options->device_block_pools.large.block_size <
          IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_LARGE_DEVICE_BLOCK_SIZE ||
      !iree_host_size_is_power_of_two(
          options->device_block_pools.large.block_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "large device block pool size invalid, expected a "
        "power-of-two greater than %d and got %" PRIhsz,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_LARGE_DEVICE_BLOCK_SIZE_DEFAULT,
        options->device_block_pools.large.block_size);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_t
//===----------------------------------------------------------------------===//

iree_host_size_t iree_hal_amdgpu_physical_device_calculate_size(
    const iree_hal_amdgpu_physical_device_options_t* options) {
  return iree_host_align(sizeof(iree_hal_amdgpu_physical_device_t),
                         iree_max_align_t);
}

iree_status_t iree_hal_amdgpu_physical_device_initialize(
    iree_hal_amdgpu_system_t* system,
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_host_size_t host_ordinal,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_host_size_t device_ordinal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  IREE_ASSERT_ARGUMENT(out_physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;

  hsa_agent_t device_agent = system->topology.gpu_agents[device_ordinal];

  // Zeroing allows for deinitialization to happen midway through initialization
  // if something fails.
  memset(out_physical_device, 0, sizeof(*out_physical_device));

  out_physical_device->device_agent = device_agent;
  out_physical_device->device_ordinal = device_ordinal;

  // Initialize the per-device host block pool.
  // This should be pinned to the host NUMA node associated with the devices but
  // today we rely on the OS to migrate pages as needed.
  iree_arena_block_pool_initialize(options->host_block_pool_size,
                                   host_allocator,
                                   &out_physical_device->fine_host_block_pool);

  // Find the device memory pools and create block pools/allocators.
  hsa_amd_memory_pool_t coarse_block_memory_pool = {0};
  hsa_amd_memory_pool_t fine_block_memory_pool = {0};
  iree_status_t status = iree_hal_amdgpu_find_coarse_global_memory_pool(
      libhsa, device_agent, &coarse_block_memory_pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_find_fine_global_memory_pool(
        libhsa, device_agent, &fine_block_memory_pool);
  }
  if (iree_status_is_ok(status) && options->host_block_pool_initial_capacity) {
    status = iree_arena_block_pool_preallocate(
        &out_physical_device->fine_host_block_pool,
        options->host_block_pool_initial_capacity);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_pool_initialize(
        libhsa, options->device_block_pools.small, device_agent,
        coarse_block_memory_pool, host_allocator,
        &out_physical_device->coarse_block_pools.small);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_pool_initialize(
        libhsa, options->device_block_pools.large, device_agent,
        coarse_block_memory_pool, host_allocator,
        &out_physical_device->coarse_block_pools.large);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_pool_initialize(
        libhsa, options->device_block_pools.small, device_agent,
        fine_block_memory_pool, host_allocator,
        &out_physical_device->fine_block_pools.small);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_pool_initialize(
        libhsa, options->device_block_pools.large, device_agent,
        fine_block_memory_pool, host_allocator,
        &out_physical_device->fine_block_pools.large);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_allocator_initialize(
        &out_physical_device->coarse_block_pools.small,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_SMALL_PAGE_SIZE,
        &out_physical_device->coarse_block_allocators.small);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_allocator_initialize(
        &out_physical_device->coarse_block_pools.large,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_COARSE_BLOCK_POOL_LARGE_PAGE_SIZE,
        &out_physical_device->coarse_block_allocators.large);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_allocator_initialize(
        &out_physical_device->fine_block_pools.small,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_SMALL_PAGE_SIZE,
        &out_physical_device->fine_block_allocators.small);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_block_allocator_initialize(
        &out_physical_device->fine_block_pools.large,
        IREE_HAL_AMDGPU_PHYSICAL_DEVICE_FINE_BLOCK_POOL_LARGE_PAGE_SIZE,
        &out_physical_device->fine_block_allocators.large);
  }

  // Initialize the host signal pool.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_signal_pool_initialize(
        libhsa,
        /*initial_capacity=*/
        IREE_HAL_AMDGPU_HOST_SIGNAL_POOL_BATCH_SIZE_DEFAULT,
        /*batch_size=*/0, host_allocator,
        &out_physical_device->host_signal_pool);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_physical_device_deinitialize(out_physical_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_physical_device_deinitialize(
    iree_hal_amdgpu_physical_device_t* physical_device) {
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_host_signal_pool_deinitialize(
      &physical_device->host_signal_pool);

  iree_arena_block_pool_deinitialize(&physical_device->fine_host_block_pool);

  iree_hal_amdgpu_block_allocator_deinitialize(
      &physical_device->coarse_block_allocators.small);
  iree_hal_amdgpu_block_allocator_deinitialize(
      &physical_device->coarse_block_allocators.large);
  iree_hal_amdgpu_block_allocator_deinitialize(
      &physical_device->fine_block_allocators.small);
  iree_hal_amdgpu_block_allocator_deinitialize(
      &physical_device->fine_block_allocators.large);
  iree_hal_amdgpu_block_pool_deinitialize(
      &physical_device->coarse_block_pools.small);
  iree_hal_amdgpu_block_pool_deinitialize(
      &physical_device->coarse_block_pools.large);
  iree_hal_amdgpu_block_pool_deinitialize(
      &physical_device->fine_block_pools.small);
  iree_hal_amdgpu_block_pool_deinitialize(
      &physical_device->fine_block_pools.large);

  memset(physical_device, 0, sizeof(*physical_device));

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amdgpu_physical_device_trim(
    iree_hal_amdgpu_physical_device_t* physical_device) {
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_block_pool_trim(&physical_device->coarse_block_pools.small);
  iree_hal_amdgpu_block_pool_trim(&physical_device->coarse_block_pools.large);
  iree_hal_amdgpu_block_pool_trim(&physical_device->fine_block_pools.small);
  iree_hal_amdgpu_block_pool_trim(&physical_device->fine_block_pools.large);

  iree_arena_block_pool_trim(&physical_device->fine_host_block_pool);

  IREE_TRACE_ZONE_END(z0);
}
