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
  out_options->device_block_pools.small.min_blocks_per_allocation =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_SMALL_DEVICE_BLOCKS_PER_ALLOCATION_DEFAULT;
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

  out_options->host_queue_count =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_QUEUE_COUNT;
  out_options->host_queue_aql_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_AQL_CAPACITY;
  out_options->host_queue_notification_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_NOTIFICATION_CAPACITY;
  out_options->host_queue_kernarg_capacity =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_HOST_QUEUE_KERNARG_CAPACITY;
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

  if (options->host_queue_count == 0 || options->host_queue_count > UINT8_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "host queue count must be in [1, %u] to fit the queue-axis encoding "
        "(got %" PRIhsz ")",
        UINT8_MAX, options->host_queue_count);
  }
  if (!iree_host_size_is_power_of_two(options->host_queue_aql_capacity) ||
      !iree_host_size_is_power_of_two(
          options->host_queue_notification_capacity) ||
      !iree_host_size_is_power_of_two(options->host_queue_kernarg_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "host queue AQL, notification, and kernarg capacities must all be "
        "powers of two (got aql=%u, notification=%u, kernarg_blocks=%u)",
        options->host_queue_aql_capacity,
        options->host_queue_notification_capacity,
        options->host_queue_kernarg_capacity);
  }
  if (options->host_queue_kernarg_capacity / 2u <
      options->host_queue_aql_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "host queue kernarg capacity must be at least 2x the AQL queue "
        "capacity because each staged kernarg block consumes one reserved "
        "AQL slot and wrap padding may skip one tail fragment (got "
        "kernarg_blocks=%u, aql_packets=%u)",
        options->host_queue_kernarg_capacity, options->host_queue_aql_capacity);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_physical_device_t
//===----------------------------------------------------------------------===//

iree_host_size_t iree_hal_amdgpu_physical_device_calculate_size(
    const iree_hal_amdgpu_physical_device_options_t* options) {
  IREE_ASSERT_ARGUMENT(options);
  return iree_host_align(
      sizeof(iree_hal_amdgpu_physical_device_t) +
          sizeof(iree_hal_amdgpu_host_queue_t) * options->host_queue_count,
      iree_max_align_t);
}

iree_status_t iree_hal_amdgpu_physical_device_initialize(
    iree_hal_amdgpu_system_t* system,
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_async_proactor_t* proactor, iree_async_axis_t base_axis,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_signal_table,
    iree_host_size_t host_ordinal,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_host_size_t device_ordinal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(epoch_signal_table);
  IREE_ASSERT_ARGUMENT(host_memory_pools);
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
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_device_library_populate_agent_kernels(
        &system->device_library, device_agent,
        &out_physical_device->device_kernels);
  }
  if (iree_status_is_ok(status)) {
    out_physical_device->buffer_transfer_context.kernels =
        &out_physical_device->device_kernels;
  }
  if (iree_status_is_ok(status)) {
    const uint8_t session_epoch = iree_async_axis_session(base_axis);
    const uint8_t machine_index = iree_async_axis_machine(base_axis);
    for (iree_host_size_t queue_ordinal = 0;
         queue_ordinal < options->host_queue_count; ++queue_ordinal) {
      iree_async_axis_t queue_axis = iree_async_axis_make_queue(
          session_epoch, machine_index, (uint8_t)device_ordinal,
          (uint8_t)queue_ordinal);
      status = iree_hal_amdgpu_host_queue_initialize(
          libhsa, proactor, device_agent, host_memory_pools->coarse_pool,
          queue_axis, epoch_signal_table,
          &out_physical_device->fine_host_block_pool,
          &out_physical_device->buffer_transfer_context, device_ordinal,
          options->host_queue_aql_capacity,
          options->host_queue_notification_capacity,
          options->host_queue_kernarg_capacity, host_allocator,
          &out_physical_device->host_queues[queue_ordinal]);
      if (!iree_status_is_ok(status)) break;
      out_physical_device->host_queue_count = queue_ordinal + 1;
    }
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

  for (iree_host_size_t i = 0; i < physical_device->host_queue_count; ++i) {
    iree_hal_amdgpu_host_queue_deinitialize(&physical_device->host_queues[i]);
  }
  physical_device->host_queue_count = 0;

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

  for (iree_host_size_t i = 0; i < physical_device->host_queue_count; ++i) {
    physical_device->host_queues[i].base.vtable->trim(
        &physical_device->host_queues[i].base);
  }

  iree_hal_amdgpu_block_pool_trim(&physical_device->coarse_block_pools.small);
  iree_hal_amdgpu_block_pool_trim(&physical_device->coarse_block_pools.large);
  iree_hal_amdgpu_block_pool_trim(&physical_device->fine_block_pools.small);
  iree_hal_amdgpu_block_pool_trim(&physical_device->fine_block_pools.large);

  iree_arena_block_pool_trim(&physical_device->fine_host_block_pool);

  IREE_TRACE_ZONE_END(z0);
}
