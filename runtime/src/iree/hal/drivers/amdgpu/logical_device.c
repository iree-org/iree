// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/logical_device.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amdgpu/allocator.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/executable_cache.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/affinity.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/utils/file_registry.h"

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
    const iree_hal_amdgpu_topology_t* topology,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
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

  // Verify all GPU agents meet the required physical device options.
  // If they verify OK we are able to compute their total size used to allocate
  // the logical device that embeds their data
  // structures.
  iree_host_size_t total_physical_device_size = 0;
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    hsa_agent_t gpu_agent = topology->gpu_agents[i];
    hsa_agent_t cpu_agent = topology->cpu_agents[topology->gpu_cpu_map[i]];
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

  // Retain the proactor pool and acquire a proactor for this device.
  logical_device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(logical_device->proactor_pool);
  logical_device->frontier_tracker = create_params->frontier.tracker;
  logical_device->axis = create_params->frontier.base_axis;
  iree_atomic_store(&logical_device->epoch, 0, iree_memory_order_relaxed);
  if (logical_device->frontier_tracker) {
    iree_async_axis_table_add(&logical_device->frontier_tracker->axis_table,
                              logical_device->axis, /*semaphore=*/NULL);
  }
  iree_status_t status = iree_async_proactor_pool_get(
      logical_device->proactor_pool, 0, &logical_device->proactor);

  // Setup physical device table.
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
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_system_allocate(libhsa, topology, system_options,
                                             host_allocator,
                                             &logical_device->system);
  }
  iree_hal_amdgpu_system_t* system = logical_device->system;

  // Create the device allocator backed by HSA memory pools.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_allocator_create(
        logical_device, &system->libhsa, &system->topology, host_allocator,
        &logical_device->device_allocator);
  }

  // Initialize physical devices for each GPU agent in the topology.
  // Their order matches the original but each may represent more than one
  // logical queue affinity bit.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t device_ordinal = 0;
         device_ordinal < logical_device->physical_device_count;
         ++device_ordinal) {
      const iree_host_size_t host_ordinal =
          topology->gpu_cpu_map[device_ordinal];
      status = iree_hal_amdgpu_physical_device_initialize(
          system, &physical_device_options, host_ordinal,
          &system->host_memory_pools[host_ordinal], device_ordinal,
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

  iree_hal_allocator_release(logical_device->device_allocator);
  iree_hal_channel_provider_release(logical_device->channel_provider);

  // This may unload HSA; must come after all resources are released.
  iree_hal_amdgpu_system_free(logical_device->system);

  // Note that these may be used by other child data types and must be freed
  // last.
  iree_arena_block_pool_deinitialize(&logical_device->host_block_pools.small);
  iree_arena_block_pool_deinitialize(&logical_device->host_block_pools.large);

  iree_async_proactor_pool_release(logical_device->proactor_pool);

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

static iree_status_t iree_hal_amdgpu_logical_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  memset(out_capabilities, 0, sizeof(*out_capabilities));

  // For single-GPU logical devices, query the first physical device.
  // TODO(multi-gpu): for multi-GPU logical devices, aggregate capabilities from
  // all physical devices (take intersection of supported features, lowest
  // common denominator for limits, etc.).
  if (logical_device->physical_device_count == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "logical device has no physical devices (initialization incomplete)");
  }

  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];
  hsa_agent_t gpu_agent = physical_device->device_agent;
  const iree_hal_amdgpu_libhsa_t* libhsa = &logical_device->system->libhsa;

  // Query device UUID (32-byte from HSA, truncate to 16 for HAL).
  char uuid_buffer[32];
  memset(uuid_buffer, 0, sizeof(uuid_buffer));
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), gpu_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID,
      uuid_buffer));
  memcpy(out_capabilities->physical_device_uuid, uuid_buffer, 16);
  out_capabilities->has_physical_device_uuid = true;

  // Query NUMA node from HSA.
  uint32_t numa_node;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), gpu_agent, HSA_AGENT_INFO_NODE, &numa_node));
  out_capabilities->numa_node = (uint8_t)numa_node;

  // External handle types (DMA-BUF support from system info).
  if (logical_device->system->info.dmabuf_supported) {
    out_capabilities->buffer_export_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF;
    out_capabilities->buffer_import_types |=
        IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF;
  }

  // Capability flags.
  if (logical_device->system->info.svm_accessible_by_default) {
    out_capabilities->flags |= IREE_HAL_DEVICE_CAPABILITY_UNIFIED_MEMORY;
  }

  // Driver handle (HSA agent handle for same-driver refinement).
  out_capabilities->driver_device_handle = (uintptr_t)gpu_agent.handle;

  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_amdgpu_logical_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  return &logical_device->topology_info;
}

static iree_status_t iree_hal_amdgpu_logical_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  logical_device->topology_info = *topology_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU collective channels not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU command buffers not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU events not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
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
      /*proactor=*/NULL, iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_amdgpu_logical_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU semaphores not yet implemented");
}

static iree_hal_semaphore_compatibility_t
iree_hal_amdgpu_logical_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU queue operations not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU queue operations not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU queue operations not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU queue operations not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU queue operations not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU queue operations not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU queue operations not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU queue operations not yet implemented");
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU queue operations not yet implemented");
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
    .query_capabilities = iree_hal_amdgpu_logical_device_query_capabilities,
    .topology_info = iree_hal_amdgpu_logical_device_topology_info,
    .refine_topology_edge = iree_hal_amdgpu_logical_device_refine_topology_edge,
    .assign_topology_info = iree_hal_amdgpu_logical_device_assign_topology_info,
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
    .profiling_begin = iree_hal_amdgpu_logical_device_profiling_begin,
    .profiling_flush = iree_hal_amdgpu_logical_device_profiling_flush,
    .profiling_end = iree_hal_amdgpu_logical_device_profiling_end,
};
