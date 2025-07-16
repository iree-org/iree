// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/logical_device.h"

#include "iree/hal/drivers/amdgpu/allocator.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/channel.h"
#include "iree/hal/drivers/amdgpu/command_buffer.h"
#include "iree/hal/drivers/amdgpu/event.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/executable_cache.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/affinity.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/virtual_queue.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static iree_hal_amdgpu_device_affinity_t
iree_hal_amdgpu_device_affinity_from_queue_affinity(
    iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t per_device_queue_count) {
  iree_hal_amdgpu_device_affinity_t device_affinity = 0;
  IREE_HAL_FOR_QUEUE_AFFINITY(queue_affinity) {
    const int physical_device_ordinal = queue_ordinal / per_device_queue_count;
    iree_hal_amdgpu_device_affinity_or_into(device_affinity,
                                            1ull << physical_device_ordinal);
  }
  return device_affinity;
}
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

  out_options->queue_placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_ANY;

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

static const iree_hal_device_vtable_t iree_hal_amdgpu_logical_device_vtable;

static iree_hal_amdgpu_logical_device_t* iree_hal_amdgpu_logical_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_logical_device_vtable);
  return (iree_hal_amdgpu_logical_device_t*)base_value;
}

static void iree_hal_amdgpu_logical_device_error_handler(void* user_data,
                                                         iree_status_t status) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Display the error in trace tooling.
  IREE_TRACE({
    char buffer[1024];
    iree_host_size_t buffer_length = 0;
    if (iree_status_format(status, sizeof(buffer), buffer, &buffer_length)) {
      IREE_TRACE_MESSAGE_DYNAMIC(ERROR, buffer, buffer_length);
    }
  });

  // Set the device sticky error status (if it is not already set).
  intptr_t current_value = 0;
  if (!iree_atomic_compare_exchange_strong(
          &logical_device->failure_status, &current_value, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_logical_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  // Verify the topology is valid for a logical device.
  // This may have already been performed by the caller but doing it here
  // ensures all code paths must verify prior to creating a device.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_topology_verify(topology, libhsa),
      "verifying topology");

  // Verify the parameters prior to creating resources.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdgpu_logical_device_options_verify(options, libhsa, topology),
      "verifying logical device options");

  // Copy options relevant during construction.
  //
  // TODO(benvanik): maybe expose these on the public API? feels like too much
  // churn for too little benefit - option parsing is still possible, though.
  iree_hal_amdgpu_physical_device_options_t physical_device_options = {0};
  iree_hal_amdgpu_physical_device_options_initialize(&physical_device_options);
  physical_device_options.device_block_pools.small.block_size =
      options->device_block_pools.small.block_size;
  physical_device_options.device_block_pools.small.initial_capacity =
      options->device_block_pools.small.initial_capacity;
  physical_device_options.device_block_pools.large.block_size =
      options->device_block_pools.large.block_size;
  physical_device_options.device_block_pools.large.initial_capacity =
      options->device_block_pools.large.initial_capacity;
  physical_device_options.host_block_pool_initial_capacity =
      options->preallocate_pools ? 16 : 0;
  physical_device_options.queue_count = topology->gpu_agent_queue_count;
  if (options->trace_execution) {
    physical_device_options.queue_options.flags |=
        IREE_HAL_AMDGPU_QUEUE_FLAG_TRACE_EXECUTION;
  }
  if (options->exclusive_execution) {
    physical_device_options.queue_options.mode |=
        IREE_HAL_AMDGPU_QUEUE_SCHEDULING_MODE_EXCLUSIVE;
  } else {
    physical_device_options.queue_options.mode |=
        IREE_HAL_AMDGPU_QUEUE_SCHEDULING_MODE_WORK_CONSERVING;
  }

  // Heterogeneous GPU agents may end up with different queue placement
  // strategies. For any not explicitly specified as part of the options we
  // infer the optimal placement and cache the result for subsequent use during
  // initialization. Note that heterogeneous requires extra coordination during
  // peer communication as though a CPU<->GPU may be able to communicate OK a
  // GPU<->GPU pair may not be able to.
  IREE_ASSERT_LE(topology->gpu_agent_count, 128);
  iree_hal_amdgpu_queue_placement_t* gpu_agent_queue_placement =
      (iree_hal_amdgpu_queue_placement_t*)iree_alloca(
          topology->gpu_agent_count *
          sizeof(iree_hal_amdgpu_queue_placement_t));
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    switch (options->queue_placement) {
      case IREE_HAL_AMDGPU_QUEUE_PLACEMENT_ANY: {
        hsa_agent_t cpu_agent = topology->cpu_agents[topology->gpu_cpu_map[i]];
        hsa_agent_t gpu_agent = topology->gpu_agents[i];
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_hal_amdgpu_queue_infer_placement(
                libhsa, cpu_agent, gpu_agent, &gpu_agent_queue_placement[i]),
            "inferring optimal queue placement for GPU agent %" PRIhsz, i);
        break;
      }
      default: {
        gpu_agent_queue_placement[i] = options->queue_placement;
        break;
      }
    }
  }

  // Verify all GPU agents meet the required physical device options.
  // If they verify OK we are able to compute their total size (as each may
  // differ) used to allocate the logical device that embeds their data
  // structures.
  iree_host_size_t total_physical_device_size = 0;
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    hsa_agent_t gpu_agent = topology->gpu_agents[i];
    hsa_agent_t cpu_agent = topology->cpu_agents[topology->gpu_cpu_map[i]];
    physical_device_options.queue_options.placement =
        gpu_agent_queue_placement[i];
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdgpu_physical_device_options_verify(
            &physical_device_options, libhsa, cpu_agent, gpu_agent),
        "verifying GPU agent %" PRIhsz " meets required options", i);
    total_physical_device_size +=
        iree_hal_amdgpu_physical_device_calculate_size(
            &physical_device_options);
  }

  // Allocate the logical device and all nested physical device data structures.
  iree_hal_amdgpu_logical_device_t* logical_device = NULL;
  const iree_host_size_t total_size =
      sizeof(*logical_device) +
      iree_host_align(sizeof(logical_device->physical_devices[0]) *
                          topology->gpu_agent_count,
                      iree_max_align_t) +
      total_physical_device_size + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&logical_device));
  iree_hal_resource_initialize(&iree_hal_amdgpu_logical_device_vtable,
                               &logical_device->resource);
  iree_string_view_append_to_buffer(
      identifier, &logical_device->identifier,
      (char*)logical_device + total_size - identifier.size);
  logical_device->host_allocator = host_allocator;
  logical_device->failure_status = IREE_ATOMIC_VAR_INIT(0);

  // Setup physical device table.
  // This extra indirection is unfortunate but allows us to have dynamic queue
  // counts based on options.
  // We need to initialize this first so that any failure cleanup has a valid
  // table.
  logical_device->physical_device_count = topology->gpu_agent_count;
  uint8_t* physical_device_base =
      (uint8_t*)logical_device + sizeof(*logical_device) +
      iree_host_align(sizeof(logical_device->physical_devices[0]) *
                          topology->gpu_agent_count,
                      iree_max_align_t);
  for (iree_host_size_t i = 0, queue_index = 0;
       i < logical_device->physical_device_count; ++i) {
    logical_device->physical_devices[i] =
        (iree_hal_amdgpu_physical_device_t*)physical_device_base;
    physical_device_options.queue_options.placement =
        gpu_agent_queue_placement[i];
    physical_device_base += iree_hal_amdgpu_physical_device_calculate_size(
        &physical_device_options);
    for (iree_host_size_t j = 0; j < topology->gpu_agent_queue_count;
         ++j, ++queue_index) {
      iree_hal_queue_affinity_or_into(logical_device->queue_affinity_mask,
                                      1ull << queue_index);
    }
  }

  // Block pools used by subsequent data structures.
  iree_arena_block_pool_initialize(options->host_block_pools.small.block_size,
                                   host_allocator,
                                   &logical_device->host_block_pools.small);
  iree_arena_block_pool_initialize(options->host_block_pools.large.block_size,
                                   host_allocator,
                                   &logical_device->host_block_pools.large);

  // Instantiate system container for agents used by the logical device. Loads
  // fixed per-agent resources like the device library.
  iree_hal_amdgpu_system_options_t system_options = {
      .trace_execution = options->trace_execution,
      .exclusive_execution = options->exclusive_execution,
  };
  iree_status_t status =
      iree_hal_amdgpu_system_allocate(libhsa, topology, system_options,
                                      host_allocator, &logical_device->system);
  iree_hal_amdgpu_system_t* system = logical_device->system;

  // Signal used for asynchronous initialization.
  // Incremented by each initialization dispatch issued on any agent and
  // decremented as they complete. When this reaches 0 all physical devices have
  // completed initialization.
  hsa_signal_t initialization_signal = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(&system->libhsa), 0ull, topology->all_agent_count,
        topology->all_agents, 0, &initialization_signal);
  }

  // TODO(benvanik): pass device handles and pool configuration to the
  // allocator. Some implementations may share allocators across multiple
  // devices created from the same driver.
  // if (iree_status_is_ok(status)) {
  //   status = iree_hal_amdgpu_allocator_create(
  //       host_allocator, &logical_device->device_allocator);
  // }

  // Initialize a pool for all internal semaphores across all agents.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_semaphore_options_t semaphore_options = {
        .wait_active_for_ns = options->wait_active_for_ns,
    };
    status = iree_hal_amdgpu_semaphore_pool_initialize(
        &system->libhsa, &system->topology,
        IREE_HAL_AMDGPU_SEMAPHORE_POOL_DEFAULT_BLOCK_CAPACITY,
        semaphore_options, IREE_HAL_SEMAPHORE_FLAG_NONE, host_allocator,
        system->host_memory_pools[0].fine_pool,
        &logical_device->semaphore_pool);
  }

  // Initialize a pool for all transient buffer handles across all agents.
  //
  // TODO(benvanik): possibly make this per NUMA node/CPU agent. Devices will
  // be accessing the allocation handles and we don't want that to make multiple
  // hops if we can avoid it.
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = (iree_hal_device_t*)logical_device,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
    };
    status = iree_hal_amdgpu_buffer_pool_initialize(
        &system->libhsa, &system->topology, placement,
        IREE_HAL_AMDGPU_BUFFER_POOL_DEFAULT_BLOCK_CAPACITY, host_allocator,
        system->host_memory_pools[0].fine_pool, &logical_device->buffer_pool);
  }

  // Route asynchronous errors back to the logical device so we have a single
  // place to hold on to the error and service it back to users.
  iree_hal_amdgpu_error_callback_t error_callback = {
      .fn = iree_hal_amdgpu_logical_device_error_handler,
      .user_data = logical_device,
  };

  // Initialize physical devices for each GPU agent in the topology.
  // Their order matches the original but each may represent more than one
  // logical queue affinity bit.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t device_ordinal = 0;
         device_ordinal < logical_device->physical_device_count;
         ++device_ordinal) {
      physical_device_options.queue_options.placement =
          gpu_agent_queue_placement[device_ordinal];
      const iree_host_size_t host_ordinal =
          topology->gpu_cpu_map[device_ordinal];
      status = iree_hal_amdgpu_physical_device_initialize(
          system, &physical_device_options, host_ordinal,
          &system->host_memory_pools[host_ordinal], device_ordinal,
          &logical_device->buffer_pool, error_callback, initialization_signal,
          host_allocator, logical_device->physical_devices[device_ordinal]);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // If requested then warmup pools that we expect to grow on the first usage of
  // the backend. The first use may need more than the warmup provides here but
  // that's ok - users can warmup if they want.
  if (options->preallocate_pools) {
    if (iree_status_is_ok(status)) {
      status = iree_arena_block_pool_preallocate(
          &logical_device->host_block_pools.small, 16);
    }
    if (iree_status_is_ok(status)) {
      status = iree_arena_block_pool_preallocate(
          &logical_device->host_block_pools.large, 16);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_semaphore_pool_preallocate(
          &logical_device->semaphore_pool, 256);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_buffer_pool_preallocate(
          &logical_device->buffer_pool, 256);
    }
  }

  // Wait for all initialization that may still be in progress to complete.
  // This ensures we don't tear down data structures that may still be in use
  // on a device doing asynchronous initialization.
  if (initialization_signal.handle) {
    iree_hsa_signal_wait_scacquire(IREE_LIBHSA(libhsa), initialization_signal,
                                   HSA_SIGNAL_CONDITION_LT, 1u, UINT64_MAX,
                                   HSA_WAIT_STATE_BLOCKED);
    IREE_IGNORE_ERROR(
        iree_hsa_signal_destroy(IREE_LIBHSA(libhsa), initialization_signal));
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)logical_device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)logical_device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_logical_device_destroy(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Devices may hold allocations and need to be cleaned up first.
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    iree_hal_amdgpu_physical_device_deinitialize(
        logical_device->physical_devices[i]);
  }

  // All buffers must have been returned to the HAL device by this point.
  iree_hal_amdgpu_buffer_pool_deinitialize(&logical_device->buffer_pool);

  // All semaphores must have been returned to the HAL device by this point.
  iree_hal_amdgpu_semaphore_pool_deinitialize(&logical_device->semaphore_pool);

  iree_hal_allocator_release(logical_device->device_allocator);
  iree_hal_channel_provider_release(logical_device->channel_provider);

  // This may unload HSA; must come after all resources are released.
  iree_hal_amdgpu_system_free(logical_device->system);

  // Note that these may be used by other child data types and must be freed
  // last.
  iree_arena_block_pool_deinitialize(&logical_device->host_block_pools.small);
  iree_arena_block_pool_deinitialize(&logical_device->host_block_pools.large);

  iree_allocator_free(host_allocator, logical_device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_amdgpu_logical_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return logical_device->identifier;
}

static iree_allocator_t iree_hal_amdgpu_logical_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return logical_device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_amdgpu_logical_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return logical_device->device_allocator;
}

static void iree_hal_amdgpu_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(logical_device->device_allocator);
  logical_device->device_allocator = new_allocator;
}

static void iree_hal_amdgpu_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(logical_device->channel_provider);
  logical_device->channel_provider = new_provider;
}

static iree_status_t iree_hal_amdgpu_logical_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Release pooled resources from each physical device. These may return items
  // back to the parent logical device pools.
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    iree_hal_amdgpu_physical_device_trim(logical_device->physical_devices[i]);
  }

  // Release pooled resources that aren't required for any currently live uses.
  // May release device memory.
  iree_hal_amdgpu_buffer_pool_trim(&logical_device->buffer_pool);
  iree_hal_amdgpu_semaphore_pool_trim(&logical_device->semaphore_pool);

  // Trim the allocator pools, if any.
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_trim(logical_device->device_allocator));

  // Trim host pools.
  iree_arena_block_pool_trim(&logical_device->host_block_pools.small);
  iree_arena_block_pool_trim(&logical_device->host_block_pools.large);

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    // NOTE: this is a fuzzy match and can allow a program to work with multiple
    // device implementations.
    *out_value =
        iree_string_view_match_pattern(logical_device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  iree_hal_amdgpu_system_t* system = logical_device->system;

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    bool is_supported = false;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_format_supported(
        &system->libhsa, system->topology.gpu_agents[0], key, &is_supported,
        /*out_isa=*/NULL));
    *out_value = is_supported ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = system->topology.gpu_agent_count *
                   system->topology.gpu_agent_queue_count;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      uint32_t compute_unit_count = 0;
      IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
          IREE_LIBHSA(&system->libhsa), system->topology.gpu_agents[0],
          (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
          &compute_unit_count));
      *out_value = compute_unit_count;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_amdgpu_logical_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Mask the user-provided queue affinity to only those we have.
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  // TODO(benvanik): pass any additional resources required to create the
  // channel. The device->channel_provider can be used to get default
  // rank/count, exchange IDs, etc as needed.
  (void)logical_device;

  return iree_hal_amdgpu_channel_create(
      params, iree_hal_device_host_allocator(base_device), out_channel);
}

static iree_status_t iree_hal_amdgpu_logical_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Mask the user-provided queue affinity to only those we have.
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  // Determine the physical devices the command buffer will be uploaded to based
  // on the queues that it is declared as being executable on. A single physical
  // device may have multiple queues.
  const iree_hal_amdgpu_device_affinity_t device_affinity =
      iree_hal_amdgpu_device_affinity_from_queue_affinity(
          queue_affinity,
          logical_device->system->topology.gpu_agent_queue_count);
  IREE_ASSERT_GT(iree_hal_amdgpu_device_affinity_count(device_affinity), 0,
                 "must have at least one device");

  iree_hal_amdgpu_command_buffer_options_t options;
  iree_hal_amdgpu_command_buffer_options_initialize(
      iree_hal_device_allocator(base_device), mode, command_categories,
      queue_affinity, binding_capacity, &options);

  // TODO(benvanik): assign based on device options controlling behavior.
  options.recording_flags = IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORDING_FLAG_NONE;

  // Host block pool is shared across all command buffers to hopefully allow us
  // to reuse allocations when command buffers are short-lived.
  options.host_block_pools = &logical_device->host_block_pools;

  // Gather block pools for all of the physical devices - the command buffer
  // doesn't care about the device and only needs to know how to allocate blocks
  // for storing the command buffer data.
  const int device_count =
      iree_hal_amdgpu_device_affinity_count(device_affinity);
  iree_hal_amdgpu_block_pools_t** device_block_pools =
      iree_alloca(device_count * sizeof(iree_hal_amdgpu_block_pools_t*));
  IREE_HAL_AMDGPU_FOR_PHYSICAL_DEVICE(device_affinity) {
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[device_ordinal];
    device_block_pools[device_index] = &physical_device->coarse_block_pools;
  }
  options.device_affinity = device_affinity;
  options.device_block_pools = device_block_pools;

  return iree_hal_amdgpu_command_buffer_create(
      &options, logical_device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_amdgpu_logical_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Mask the user-provided queue affinity to only those we have.
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  // TODO(benvanik): pass any additional resources required to create the event.
  // The implementation could pool events here.
  (void)logical_device;

  return iree_hal_amdgpu_event_create(
      queue_affinity, flags, iree_hal_device_host_allocator(base_device),
      out_event);
}

static iree_status_t iree_hal_amdgpu_logical_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return iree_hal_amdgpu_executable_cache_create(
      &logical_device->system->libhsa, &logical_device->system->topology,
      identifier, iree_hal_device_host_allocator(base_device),
      out_executable_cache);
}

static iree_status_t iree_hal_amdgpu_logical_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Mask the user-provided queue affinity to only those we have.
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_amdgpu_logical_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // TODO(benvanik): support exportable semaphores based on flags. We 99.9% of
  // the time want our internal ones.

  // Acquire a semaphore from the pool.
  return iree_hal_amdgpu_semaphore_pool_acquire(
      &logical_device->semaphore_pool, initial_value, flags, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_amdgpu_logical_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  iree_hal_semaphore_compatibility_t compatibility =
      IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
  if (iree_hal_amdgpu_internal_semaphore_isa(semaphore)) {
    // Internal semaphores need to be in a pool that allows access by all. We
    // could fast path for semaphores created from this logical device and then
    // fall back to querying for pool compatibility. Today all semaphores are
    // created from HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL pools and we just
    // assume regardless of origin they are compatible.
    compatibility = IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
  } else {
    // TODO(benvanik): support external semaphores. We can wrap them and have
    // the device library post back to the host to signal them in cases where we
    // can't do so via memory operations.
    compatibility = IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
  }
  return compatibility;
}

// Resolves a queue affinity to a particular device queue.
// If the affinity specifies more than one queue we always go with the first one
// set today (so 0b110 is the same as 0b010).
//
// In the future we could load balance (distribute independently executable
// work) or chain (ensure dependent executable work ends up on the same queue
// selected earlier). That gets tricky, though, if we start submitting work
// across peer devices where we may not be able to (quickly) check such
// dependencies on the host (or have to check them on peers). For now the
// first-set will handle most cases the compiler generates beyond the
// unspecified ANY affinity where everything will end up on device 0/queue 0.
static iree_status_t iree_hal_amdgpu_logical_device_select_queue(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_amdgpu_virtual_queue_t** out_queue) {
  // Mask the user-provided queue affinity to only those we have.
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  // Find the first set bit as our default policy.
  const int logical_queue_ordinal =
      iree_hal_queue_affinity_find_first_set(queue_affinity);

  // Map queue ordinal to physical device ordinal and its local queue ordinal.
  const iree_host_size_t per_queue_count =
      logical_device->system->topology.gpu_agent_queue_count;
  const iree_host_size_t physical_device_ordinal =
      logical_queue_ordinal / per_queue_count;
  const iree_host_size_t physical_queue_ordinal =
      logical_queue_ordinal % per_queue_count;

  *out_queue = logical_device->physical_devices[physical_device_ordinal]
                   ->queues[physical_queue_ordinal];
  return iree_ok_status();
}

// GNUC #define's alloca!
// <michael> Why should I change, they're the ones that suck! </michael>
#undef alloca

static iree_status_t iree_hal_amdgpu_logical_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->alloca(queue, wait_semaphore_list,
                               signal_semaphore_list, pool, params,
                               allocation_size, flags, out_buffer);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->dealloca(queue, wait_semaphore_list,
                                 signal_semaphore_list, buffer, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  uint64_t pattern_bits = 0;
  switch (pattern_length) {
    case 1:
    case 2:
    case 4:
    case 8:
      memcpy(&pattern_bits, pattern, pattern_length);
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pattern length must be 1, 2, 4, or 8 - got %" PRIhsz,
          pattern_length);
  }
  return queue->vtable->fill(queue, wait_semaphore_list, signal_semaphore_list,
                             target_buffer, target_offset, length, pattern_bits,
                             pattern_length, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->update(
      queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->copy(queue, wait_semaphore_list, signal_semaphore_list,
                             source_buffer, source_offset, target_buffer,
                             target_offset, length, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Route to optimized queue I/O if available.
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  if (queue->vtable->read) {
    return queue->vtable->read(
        queue, wait_semaphore_list, signal_semaphore_list, source_file,
        source_offset, target_buffer, target_offset, length, flags);
  }

  // Fall back to inefficient emulated I/O.
  // TODO(benvanik): when all queue implementations support native I/O we should
  // drop the emulation (it's bad).
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Route to optimized queue I/O if available.
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  if (queue->vtable->read) {
    return queue->vtable->write(
        queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
        source_offset, target_file, target_offset, length, flags);
  }

  // Fall back to inefficient emulated I/O.
  // TODO(benvanik): when all queue implementations support native I/O we should
  // drop the emulation (it's bad).
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->execute(queue, wait_semaphore_list,
                                signal_semaphore_list, command_buffer,
                                binding_table, flags);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_virtual_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return queue->vtable->flush(queue);
}

static iree_status_t iree_hal_amdgpu_logical_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return iree_hal_amdgpu_wait_semaphores(&logical_device->system->libhsa,
                                         logical_device->semaphore_pool.options,
                                         wait_mode, semaphore_list, timeout);
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // TODO(benvanik): figure out if there's any AMD tooling calls we can make.
  (void)logical_device;
  iree_status_t status = iree_ok_status();

  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // TODO(benvanik): figure out if there's any AMD tooling calls we can make.
  (void)logical_device;
  iree_status_t status = iree_ok_status();

  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // TODO(benvanik): figure out if there's any AMD tooling calls we can make.
  (void)logical_device;
  iree_status_t status = iree_ok_status();

  return status;
}

static const iree_hal_device_vtable_t iree_hal_amdgpu_logical_device_vtable = {
    .destroy = iree_hal_amdgpu_logical_device_destroy,
    .id = iree_hal_amdgpu_logical_device_id,
    .host_allocator = iree_hal_amdgpu_logical_device_host_allocator,
    .device_allocator = iree_hal_amdgpu_logical_device_allocator,
    .replace_device_allocator = iree_hal_amdgpu_replace_device_allocator,
    .replace_channel_provider = iree_hal_amdgpu_replace_channel_provider,
    .trim = iree_hal_amdgpu_logical_device_trim,
    .query_i64 = iree_hal_amdgpu_logical_device_query_i64,
    .create_channel = iree_hal_amdgpu_logical_device_create_channel,
    .create_command_buffer =
        iree_hal_amdgpu_logical_device_create_command_buffer,
    .create_event = iree_hal_amdgpu_logical_device_create_event,
    .create_executable_cache =
        iree_hal_amdgpu_logical_device_create_executable_cache,
    .import_file = iree_hal_amdgpu_logical_device_import_file,
    .create_semaphore = iree_hal_amdgpu_logical_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_amdgpu_logical_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_amdgpu_logical_device_queue_alloca,
    .queue_dealloca = iree_hal_amdgpu_logical_device_queue_dealloca,
    .queue_fill = iree_hal_amdgpu_logical_device_queue_fill,
    .queue_update = iree_hal_amdgpu_logical_device_queue_update,
    .queue_copy = iree_hal_amdgpu_logical_device_queue_copy,
    .queue_read = iree_hal_amdgpu_logical_device_queue_read,
    .queue_write = iree_hal_amdgpu_logical_device_queue_write,
    .queue_execute = iree_hal_amdgpu_logical_device_queue_execute,
    .queue_flush = iree_hal_amdgpu_logical_device_queue_flush,
    .wait_semaphores = iree_hal_amdgpu_logical_device_wait_semaphores,
    .profiling_begin = iree_hal_amdgpu_logical_device_profiling_begin,
    .profiling_flush = iree_hal_amdgpu_logical_device_profiling_flush,
    .profiling_end = iree_hal_amdgpu_logical_device_profiling_end,
};
