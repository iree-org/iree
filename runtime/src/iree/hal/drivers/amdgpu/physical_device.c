// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device.h"

#include "iree/hal/drivers/amdgpu/device_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
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

  out_options->queue_count =
      IREE_HAL_AMDGPU_PHYSICAL_DEVICE_DEFAULT_QUEUE_COUNT;
  iree_hal_amdgpu_queue_options_initialize(&out_options->queue_options);
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

  // Verify each queue - if used - is valid.
  if (options->queue_count > 0 && options->queue_count <= 64) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "a physical device may only have 1-64 HAL queues");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_queue_options_verify(&options->queue_options, libhsa,
                                           cpu_agent, gpu_agent),
      "verifying queue options");

  // Verify that the total hardware queues required is less than the total
  // available on the device. Some of those queues may be in use at the time we
  // are running and so we may fail to allocate them all even if this reports
  // OK.
  uint32_t max_queue_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), gpu_agent,
                              HSA_AGENT_INFO_QUEUES_MAX, &max_queue_count),
      "querying HSA_AGENT_INFO_QUEUES_MAX");
  const iree_host_size_t device_queue_count =
      options->queue_count * options->queue_options.execution_queue_count;
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

// Calculates the size in bytes of the storage required for a queue
// implementation based on the provided |options|.
static iree_host_size_t iree_hal_amdgpu_queue_calculate_size(
    const iree_hal_amdgpu_queue_options_t* options) {
  switch (options->placement) {
    case IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST:
      return iree_hal_amdgpu_host_queue_calculate_size(options);
    case IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE:
      return iree_hal_amdgpu_device_queue_calculate_size(options);
    default:
      IREE_ASSERT_UNREACHABLE("queue placement should have resolved earlier");
      return 0;  // invalid handled elsewhere
  }
}

// Initializes |out_queue| in-place based on |options|.
static iree_status_t iree_hal_amdgpu_queue_initialize(
    iree_hal_amdgpu_system_t* system, iree_hal_amdgpu_queue_options_t options,
    hsa_agent_t device_agent, iree_host_size_t device_ordinal,
    iree_hal_amdgpu_host_service_t* host_service,
    iree_arena_block_pool_t* host_block_pool,
    iree_hal_amdgpu_block_allocators_t* block_allocators,
    iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    iree_hal_amdgpu_error_callback_t error_callback,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_queue_t* out_queue) {
  // Note that today these mostly take the same arguments but we may need to
  // provide more for device queues (such as peer information).
  switch (options.placement) {
    case IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST:
      return iree_hal_amdgpu_host_queue_initialize(
          system, options, device_agent, device_ordinal, host_service,
          host_block_pool, block_allocators, buffer_pool, error_callback,
          initialization_signal, host_allocator, out_queue);
    case IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE:
      return iree_hal_amdgpu_device_queue_initialize(
          system, options, device_agent, device_ordinal, host_service,
          host_block_pool, block_allocators, buffer_pool, error_callback,
          initialization_signal, host_allocator, out_queue);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid queue placement %d",
                              (int)options.placement);
  }
}

iree_host_size_t iree_hal_amdgpu_physical_device_calculate_size(
    const iree_hal_amdgpu_physical_device_options_t* options) {
  const iree_host_size_t base_size =
      sizeof(iree_hal_amdgpu_physical_device_t) +
      iree_host_align(
          options->queue_count * sizeof(iree_hal_amdgpu_virtual_queue_t*),
          iree_max_align_t);
  const iree_host_size_t queue_size =
      iree_hal_amdgpu_queue_calculate_size(&options->queue_options);
  return iree_host_align(base_size + options->queue_count * queue_size,
                         iree_max_align_t);
}

iree_status_t iree_hal_amdgpu_physical_device_initialize(
    iree_hal_amdgpu_system_t* system,
    const iree_hal_amdgpu_physical_device_options_t* options,
    iree_host_size_t host_ordinal,
    const iree_hal_amdgpu_host_memory_pools_t* host_memory_pools,
    iree_host_size_t device_ordinal, iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    iree_hal_amdgpu_error_callback_t error_callback,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_physical_device_t* out_physical_device) {
  IREE_ASSERT_ARGUMENT(out_physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;

  hsa_agent_t host_agent = system->topology.cpu_agents[host_ordinal];
  hsa_agent_t device_agent = system->topology.gpu_agents[device_ordinal];

  // Zeroing allows for deinitialization to happen midway through initialization
  // if something fails.
  memset(out_physical_device, 0, sizeof(*out_physical_device));

  out_physical_device->device_agent = device_agent;
  out_physical_device->device_ordinal = device_ordinal;
  out_physical_device->queue_count = options->queue_count;

  // Setup queue pointers before anything else so that when we deinitialize they
  // are valid.
  const iree_host_size_t total_queue_size =
      iree_hal_amdgpu_queue_calculate_size(&options->queue_options);
  uint8_t* queue_base_ptr =
      (uint8_t*)out_physical_device + sizeof(*out_physical_device) +
      iree_host_align(
          options->queue_count * sizeof(out_physical_device->queues[0]),
          iree_max_align_t);
  for (iree_host_size_t i = 0; i < options->queue_count; ++i) {
    out_physical_device->queues[i] =
        (iree_hal_amdgpu_virtual_queue_t*)(queue_base_ptr +
                                           i * total_queue_size);
  }

  // Find the device pools used for blocks.
  hsa_amd_memory_pool_t coarse_block_memory_pool = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_find_coarse_global_memory_pool(
              libhsa, device_agent, &coarse_block_memory_pool));
  hsa_amd_memory_pool_t fine_block_memory_pool = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_find_fine_global_memory_pool(
              libhsa, device_agent, &fine_block_memory_pool));

  // Initialize the per-device block pool.
  // This should be pinned to the host NUMA node associated with the devices but
  // today we rely on the OS to migrate pages as needed.
  iree_arena_block_pool_initialize(options->host_block_pool_size,
                                   host_allocator,
                                   &out_physical_device->fine_host_block_pool);
  if (options->host_block_pool_initial_capacity) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_block_pool_preallocate(
                &out_physical_device->fine_host_block_pool,
                options->host_block_pool_initial_capacity));
  }

  // Create block pools and allocators used to back device-side resources.
  // Shared amongst all queues on the device.
  iree_status_t status = iree_ok_status();
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
  // TODO(benvanik): expose block allocator min_page_size values as options.
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

  // Create the host worker thread that will handle scheduler requests.
  // Each queue on this physical device will share the same worker today but we
  // could change that if we become host-bound. In general we should not be
  // using the host during our latency-critical operations but it's possible if
  // memory pool growth/trims take awhile that we end up serializing multiple
  // device queues.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_service_initialize(
        libhsa, host_ordinal, host_agent, host_memory_pools->fine_region,
        device_ordinal, error_callback, host_allocator,
        &out_physical_device->host_service);
  }

  // Initialize each queue and its device-side scheduler.
  // Note that initialization may happen asynchronously with the
  // initialization_signal. Queues may require host services to initialize and
  // this must happen after all other physical device state has completed
  // initialization.
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < options->queue_count; ++i) {
    status = iree_hal_amdgpu_queue_initialize(
        system, options->queue_options, device_agent, device_ordinal,
        &out_physical_device->host_service,
        &out_physical_device->fine_host_block_pool,
        &out_physical_device->fine_block_allocators, buffer_pool,
        error_callback, initialization_signal, host_allocator,
        out_physical_device->queues[i]);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_physical_device_deinitialize(
    iree_hal_amdgpu_physical_device_t* physical_device) {
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Deinitialize all queues and their device-side schedulers before releasing
  // any resources that may be used by them (such as the host worker).
  for (iree_host_size_t i = 0; i < physical_device->queue_count; ++i) {
    iree_hal_amdgpu_virtual_queue_t* queue = physical_device->queues[i];
    if (queue) {
      queue->vtable->deinitialize(queue);
    }
  }

  // Deinitialize the host service only after all queues have fully terminated.
  iree_hal_amdgpu_host_service_deinitialize(&physical_device->host_service);

  // Note that the host service may be using allocations from the host block
  // pool and that this must happen after it has fully terminated.
  iree_arena_block_pool_deinitialize(&physical_device->fine_host_block_pool);

  // Note that other per-device data structures may be using blocks until they
  // are deinitialized and this must be deinitialized last.
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

  // Trim queues first to release resources back to block pools.
  for (iree_host_size_t i = 0; i < physical_device->queue_count; ++i) {
    iree_hal_amdgpu_virtual_queue_t* queue = physical_device->queues[i];
    queue->vtable->trim(queue);
  }

  iree_hal_amdgpu_block_pool_trim(&physical_device->coarse_block_pools.small);
  iree_hal_amdgpu_block_pool_trim(&physical_device->coarse_block_pools.large);
  iree_hal_amdgpu_block_pool_trim(&physical_device->fine_block_pools.small);
  iree_hal_amdgpu_block_pool_trim(&physical_device->fine_block_pools.large);

  iree_arena_block_pool_trim(&physical_device->fine_host_block_pool);

  IREE_TRACE_ZONE_END(z0);
}
