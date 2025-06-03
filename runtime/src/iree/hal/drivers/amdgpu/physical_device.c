// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_options_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_SMALL_DEVICE_BLOCK_SIZE (1 * 1024)
#define IREE_HAL_AMDGPU_PHYSICAL_DEVICE_MIN_LARGE_DEVICE_BLOCK_SIZE (64 * 1024)

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

  out_options->queue_count =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_QUEUE_COUNT;
  // TODO(benvanik): implement queues.
  // iree_hal_amdgpu_queue_options_initialize(&out_options->queue_options);
}

iree_status_t iree_hal_amdgpu_physical_device_options_verify(
    const iree_hal_amdgpu_physical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(libhsa);

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

  // Verify each queue - if used - is valid.
  if (options->queue_count > 0 && options->queue_count <= 64) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "a physical device may only have 1-64 HAL queues");
  }
  // TODO(benvanik): implement queues.
  // IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_options_verify(
  //     &options->queue_options, libhsa, agent));

  // Verify that the total hardware queues required is less than the total
  // available on the device. Some of those queues may be in use at the time we
  // are running and so we may fail to allocate them all even if this reports
  // OK.
  uint32_t max_queue_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), agent,
                              HSA_AGENT_INFO_QUEUES_MAX, &max_queue_count),
      "querying HSA_AGENT_INFO_QUEUES_MAX");
  const iree_host_size_t device_queue_count = options->queue_count;
  // TODO(benvanik): implement queues:
  // * options->queue_options.execution_queue_count;
  if (device_queue_count > max_queue_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "maximum hardware queue count exceeded; device reports %u available "
        "queues (at maximum) and %" PRIhsz " were requested",
        max_queue_count, device_queue_count);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): implement iree_hal_amdgpu_physical_device_t.
