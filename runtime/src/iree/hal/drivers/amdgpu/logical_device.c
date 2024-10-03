// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/logical_device.h"

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amdgpu/allocator.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/channel.h"
#include "iree/hal/drivers/amdgpu/command_buffer.h"
#include "iree/hal/drivers/amdgpu/event.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/executable_cache.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_options_t
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_device_options_initialize(
    iree_hal_amdgpu_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  // TODO(benvanik): set defaults based on compiler configuration. Flags should
  // not be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.
}

static iree_status_t iree_hal_amdgpu_device_options_verify(
    const iree_hal_amdgpu_device_options_t* options) {
  // TODO(benvanik): verify that the parameters are within expected ranges and
  // any requested features are supported.
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

iree_status_t iree_hal_amdgpu_logical_device_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_device_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_amdgpu_logical_device_t** out_logical_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_logical_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_logical_device = NULL;

  // Verify the parameters prior to creating resources.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_device_options_verify(options));

  iree_host_size_t physical_device_size =
      iree_hal_amdgpu_physical_device_calculate_size(
          topology->gpu_agent_queue_count);
  iree_hal_amdgpu_logical_device_t* logical_device = NULL;
  iree_host_size_t total_size =
      sizeof(*logical_device) +
      sizeof(logical_device->physical_devices[0]) * topology->gpu_agent_count +
      physical_device_size * topology->gpu_agent_count + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&logical_device));
  iree_hal_resource_initialize(&iree_hal_amdgpu_logical_device_vtable,
                               &logical_device->resource);
  iree_string_view_append_to_buffer(
      identifier, &logical_device->identifier,
      (char*)logical_device + total_size - identifier.size);
  logical_device->host_allocator = host_allocator;

  // Setup physical device table.
  // This extra indirection is unfortunate but allows us to have dynamic queue
  // counts based on options.
  // We need to initialize this first so that any failure cleanup has a valid
  // table.
  logical_device->physical_device_count = topology->gpu_agent_count;
  uint8_t* physical_device_base =
      (uint8_t*)logical_device +
      sizeof(logical_device->physical_devices[0]) * topology->gpu_agent_count;
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    logical_device->physical_devices[i] =
        (iree_hal_amdgpu_physical_device_t*)physical_device_base;
    physical_device_base += physical_device_size;
  }

  // Instantiate system container for agents used by the logical device. Loads
  // fixed per-agent resources like the device library.
  iree_hal_amdgpu_system_t* system = &logical_device->system;
  iree_status_t status = iree_hal_amdgpu_system_initialize(
      libhsa, topology, host_allocator, system);

  // TODO(benvanik): pass device handles and pool configuration to the
  // allocator. Some implementations may share allocators across multiple
  // devices created from the same driver.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_allocator_create(
        host_allocator, &logical_device->device_allocator);
  }

  // Initialize a pool for all internal semaphores across all agents.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_semaphore_pool_initialize(
        &system->libhsa, &system->topology,
        IREE_HAL_AMDGPU_SEMAPHORE_POOL_DEFAULT_BLOCK_CAPACITY,
        IREE_HAL_SEMAPHORE_FLAG_NONE, host_allocator, system->shared_fine_pool,
        &logical_device->semaphore_pool);
  }

  // Initialize a pool for all transient buffer handles across all agents.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_buffer_pool_initialize(
        &system->libhsa, &system->topology,
        IREE_HAL_AMDGPU_BUFFER_POOL_DEFAULT_BLOCK_CAPACITY, host_allocator,
        system->shared_fine_pool, &logical_device->buffer_pool);
  }

  // Initialize physical devices for each GPU agent in the topology.
  // Their order matches the original but each may represent more than one
  // logical queue affinity bit.
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    status = iree_hal_amdgpu_physical_device_initialize(
        system, topology->cpu_agents[topology->gpu_cpu_map[i]],
        topology->gpu_agents[i], i, topology->gpu_agent_queue_count,
        &logical_device->buffer_pool, host_allocator,
        logical_device->physical_devices[i]);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    *out_logical_device = logical_device;
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
  iree_hal_amdgpu_system_deinitialize(&logical_device->system);

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

  // TODO(benvanik): if the device has any cached resources that can be safely
  // dropped here (unused pools/etc). This is usually called in low-memory
  // situations or when the HAL device will not be used for awhile (device
  // entering sleep mode or a low power state, etc).

  // Release semaphore pool resources that aren't required for any currently
  // live semaphores. May release device memory.
  iree_hal_amdgpu_semaphore_pool_trim(&logical_device->semaphore_pool);

  // Trim the allocator pools, if any.
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_trim(logical_device->device_allocator));

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

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    bool is_supported = false;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_format_supported(
        &logical_device->system.libhsa,
        logical_device->system.topology.gpu_agents[0], key, &is_supported,
        /*out_isa=*/NULL));
    *out_value = is_supported ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = logical_device->system.topology.gpu_agent_count *
                   logical_device->system.topology.gpu_agent_queue_count;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      uint32_t compute_unit_count = 0;
      IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
          &logical_device->system.libhsa,
          logical_device->system.topology.gpu_agents[0],
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

  // DO NOT SUBMIT
  // need to pass in:
  //   libhsa, topology (queue -> physical device resolution?)
  // could alloca/bake out physical devices here to avoid bleeding topology
  // could just pass in hsa_agent_t device_agent and allocator?
  // may still need topology
  // could pass in bitmap of physical device ordinals? avoid queue expansion?
  // queue_affinity is just for tracking then, not authoritative
  return iree_hal_amdgpu_command_buffer_create(
      iree_hal_device_allocator(base_device), mode, command_categories,
      queue_affinity, binding_capacity, logical_device->host_allocator,
      out_command_buffer);
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
      &logical_device->system.libhsa, &logical_device->system.topology,
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

  // TODO(benvanik): if the implementation supports native file operations
  // definitely prefer that. The emulated file I/O present here as a default is
  // inefficient. The queue affinity specifies which queues may access the file
  // via read and write queue operations.
  if (iree_io_file_handle_type(handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "implementation does not support the external file type");
  }
  return iree_hal_memory_file_wrap(
      queue_affinity, access, handle, iree_hal_device_allocator(base_device),
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
  iree_hal_amdgpu_internal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_semaphore_pool_acquire(
      &logical_device->semaphore_pool, initial_value, flags, &semaphore));

  *out_semaphore = (iree_hal_semaphore_t*)semaphore;
  return iree_ok_status();
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

// Resolves a queue affinity to a particular queue.
// If the affinity specifies more than one queue we always go with the first one
// set today.
static iree_status_t iree_hal_amdgpu_logical_device_select_queue(
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_amdgpu_queue_t** out_queue) {
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

  // Map queue ordinal to physical device ordinal and its local queue ordinal
  const int per_queue_count =
      (int)logical_device->system.topology.gpu_agent_queue_count;
  const int physical_device_ordinal = logical_queue_ordinal / per_queue_count;
  const int physical_queue_ordinal = logical_queue_ordinal % per_queue_count;

  *out_queue = &logical_device->physical_devices[physical_device_ordinal]
                    ->queues[physical_queue_ordinal];
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return iree_hal_amdgpu_queue_alloca(queue, wait_semaphore_list,
                                      signal_semaphore_list, pool, params,
                                      allocation_size, out_buffer);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return iree_hal_amdgpu_queue_dealloca(queue, wait_semaphore_list,
                                        signal_semaphore_list, buffer);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Mask the user-provided queue affinity to only those we have.
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  // TODO(benvanik): redirect to queues and implement instead of emulating.
  //
  // iree_hal_amdgpu_queue_t* queue = NULL;
  // IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
  //     logical_device, queue_affinity, &queue));
  // return iree_hal_amdgpu_queue_read(
  //     queue, wait_semaphore_list, signal_semaphore_list, source_file,
  //     source_offset, target_buffer, target_offset, length, flags);

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
    iree_device_size_t length, uint32_t flags) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // Mask the user-provided queue affinity to only those we have.
  iree_hal_queue_affinity_and_into(queue_affinity,
                                   logical_device->queue_affinity_mask);
  if (iree_hal_queue_affinity_is_empty(queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }

  // TODO(benvanik): redirect to queues and implement instead of emulating.
  //
  // iree_hal_amdgpu_queue_t* queue = NULL;
  // IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
  //     logical_device, queue_affinity, &queue));
  // return iree_hal_amdgpu_queue_write(
  //     queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
  //     source_offset, target_file, target_offset, length, flags);

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
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return iree_hal_amdgpu_queue_execute(
      queue, wait_semaphore_list, signal_semaphore_list, command_buffer_count,
      command_buffers, binding_tables);
}

static iree_status_t iree_hal_amdgpu_logical_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);
  iree_hal_amdgpu_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_select_queue(
      logical_device, queue_affinity, &queue));
  return iree_hal_amdgpu_queue_flush(queue);
}

static iree_status_t iree_hal_amdgpu_logical_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // TODO(benvanik): implement multi-wait as either an ALL (AND) or ANY (OR)
  // operation. Semaphores are expected to be compatible with the device today
  // and may come from other device instances provided by the same driver or
  // have been imported by a device instance.

  // TODO(benvanik): if any semaphore has a failure status set return
  // `iree_status_from_code(IREE_STATUS_ABORTED)`. Avoid a full status as it may
  // capture a backtrace and allocate and callers are expected to follow up a
  // failed wait with a query to get the status.

  // TODO(benvanik): prefer having a fast-path for if the semaphores are
  // known-signaled in user-mode. This can usually avoid syscalls/ioctls and
  // potential context switches in polling cases.

  // TODO(benvanik): check for `iree_timeout_is_immediate(timeout)` and return
  // immediately if the condition is not satisfied before waiting with
  // `iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED)`. Prefer the raw code
  // status instead of a full status object as immediate timeouts are used when
  // polling and a full status may capture a backtrace and allocate.

  (void)logical_device;
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "semaphore multi-wait not implemented");

  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // TODO(benvanik): set implementation-defined device or global profiling
  // modes. This will be paired with a profiling_end call at some point in the
  // future. Hosting applications may periodically call profiling_flush.
  (void)logical_device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "device profiling not implemented");

  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // TODO(benvanik): flush if needed. May be no-op. Any accumulated profiling
  // information should be carried across the flush but the event can be used to
  // reclaim resources or perform other expensive bookkeeping. Benchmarks, for
  // example, are expected to call this periodically with their timing
  // suspended.
  (void)logical_device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "device profiling not implemented");

  return status;
}

static iree_status_t iree_hal_amdgpu_logical_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      iree_hal_amdgpu_logical_device_cast(base_device);

  // TODO(benvanik): unset whatever profiling_begin set, if anything. May be
  // no-op.
  (void)logical_device;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "device profiling not implemented");

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
    .queue_read = iree_hal_amdgpu_logical_device_queue_read,
    .queue_write = iree_hal_amdgpu_logical_device_queue_write,
    .queue_execute = iree_hal_amdgpu_logical_device_queue_execute,
    .queue_flush = iree_hal_amdgpu_logical_device_queue_flush,
    .wait_semaphores = iree_hal_amdgpu_logical_device_wait_semaphores,
    .profiling_begin = iree_hal_amdgpu_logical_device_profiling_begin,
    .profiling_flush = iree_hal_amdgpu_logical_device_profiling_flush,
    .profiling_end = iree_hal_amdgpu_logical_device_profiling_end,
};
