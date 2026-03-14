// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/task_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/local_task/block_command_buffer.h"
#include "iree/hal/drivers/local_task/task_event.h"
#include "iree/hal/drivers/local_task/task_queue.h"
#include "iree/hal/drivers/local_task/task_semaphore.h"
#include "iree/hal/drivers/local_task/transient_buffer.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/local_executable_cache.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/queue_emulation.h"

typedef struct iree_hal_task_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for small allocations like tasks and submissions.
  iree_arena_block_pool_t small_block_pool;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t large_block_pool;

  iree_host_size_t loader_count;
  iree_hal_executable_loader_t** loaders;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Proactor pool for async I/O. Retained for the lifetime of the device to
  // ensure proactor threads outlive all device resources (semaphores, etc.).
  iree_async_proactor_pool_t* proactor_pool;

  // Proactor selected from the pool for this device's async I/O operations.
  // Borrowed from the pool — valid as long as the pool is retained.
  iree_async_proactor_t* proactor;

  // Shared frontier tracker for cross-device causal ordering.
  // Borrowed from the session — valid as long as the session is alive.
  // NULL if frontier-based fast paths are not enabled.
  iree_async_frontier_tracker_t* frontier_tracker;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  iree_hal_device_topology_info_t topology_info;

  iree_host_size_t queue_count;
  iree_hal_task_queue_t queues[];
} iree_hal_task_device_t;

static const iree_hal_device_vtable_t iree_hal_task_device_vtable;

static iree_hal_task_device_t* iree_hal_task_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_task_device_vtable);
  return (iree_hal_task_device_t*)base_value;
}

void iree_hal_task_device_params_initialize(
    iree_hal_task_device_params_t* out_params) {
  out_params->arena_block_size = 32 * 1024;
  out_params->queue_scope_flags = IREE_TASK_SCOPE_FLAG_NONE;
}

static iree_status_t iree_hal_task_device_check_params(
    const iree_hal_task_device_params_t* params, iree_host_size_t queue_count) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  if (queue_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "must have at least one queue");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_task_device_create(
    iree_string_view_t identifier, const iree_hal_task_device_params_t* params,
    iree_host_size_t queue_count, iree_task_executor_t* const* queue_executors,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_hal_allocator_t* device_allocator,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(!queue_count || queue_executors);
  IREE_ASSERT_ARGUMENT(!loader_count || loaders);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_task_device_check_params(params, queue_count));

  iree_hal_task_device_t* device = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t loaders_offset = 0;
  iree_host_size_t identifier_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(*device), &total_size,
                             IREE_STRUCT_FIELD_ALIGNED(
                                 queue_count, iree_hal_task_queue_t, 1, NULL),
                             IREE_STRUCT_FIELD_ALIGNED(
                                 loader_count, iree_hal_executable_loader_t*, 1,
                                 &loaders_offset),
                             IREE_STRUCT_FIELD_ALIGNED(identifier.size, char, 1,
                                                       &identifier_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_task_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + identifier_offset);
  device->host_allocator = host_allocator;
  device->device_allocator = device_allocator;
  iree_hal_allocator_retain(device_allocator);

  // Retain the proactor pool. Each queue will get a NUMA-correct proactor
  // borrowed from the pool based on its executor's node assignment.
  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);
  device->frontier_tracker = create_params->frontier.tracker;

  // Select the device-level default proactor from the first queue's executor
  // NUMA node. Used for operations without specific queue affinity.
  iree_task_topology_node_id_t default_node_id =
      iree_task_executor_node_id(queue_executors[0]);
  iree_status_t status = iree_async_proactor_pool_get_for_node(
      device->proactor_pool, default_node_id, &device->proactor);

  if (iree_status_is_ok(status)) {
    iree_arena_block_pool_initialize(4096, host_allocator,
                                     &device->small_block_pool);
    iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                     &device->large_block_pool);

    device->loader_count = loader_count;
    device->loaders =
        (iree_hal_executable_loader_t**)((uint8_t*)device + loaders_offset);
    for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
      device->loaders[i] = loaders[i];
      iree_hal_executable_loader_retain(device->loaders[i]);
    }

    device->queue_count = queue_count;
    for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
      iree_hal_queue_affinity_t queue_affinity = 1ull << i;
      // Select a NUMA-correct proactor for this queue based on its executor's
      // node assignment. Falls back to the first proactor in the pool if the
      // executor's node doesn't have a dedicated proactor.
      iree_async_proactor_t* queue_proactor = NULL;
      iree_task_topology_node_id_t node_id =
          iree_task_executor_node_id(queue_executors[i]);
      status = iree_async_proactor_pool_get_for_node(device->proactor_pool,
                                                     node_id, &queue_proactor);
      if (!iree_status_is_ok(status)) break;

      // Derive per-queue axis from device base_axis by setting queue_index
      // in the ordinal bits [31:24].
      iree_async_axis_t queue_axis =
          create_params->frontier.base_axis | ((uint64_t)i << 24);

      // Register the queue's axis in the frontier tracker's axis table.
      if (device->frontier_tracker) {
        int32_t table_index = iree_async_axis_table_add(
            &device->frontier_tracker->axis_table, queue_axis,
            /*semaphore=*/NULL);
        (void)table_index;
      }

      iree_hal_task_queue_initialize(
          device->identifier, queue_affinity, params->queue_scope_flags,
          queue_executors[i], queue_proactor, device->frontier_tracker,
          queue_axis, &device->small_block_pool, &device->large_block_pool,
          device->device_allocator, &device->queues[i]);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_task_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    iree_hal_task_queue_deinitialize(&device->queues[i]);
  }

  for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
    iree_hal_executable_loader_release(device->loaders[i]);
  }

  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_async_proactor_pool_release(device->proactor_pool);

  iree_arena_block_pool_deinitialize(&device->large_block_pool);
  iree_arena_block_pool_deinitialize(&device->small_block_pool);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_task_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_task_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_task_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_task_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_task_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_task_device_trim(iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);

  // Before trimming the block pools try to trim subsystems that may be holding
  // on to blocks.
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    iree_hal_task_queue_trim(&device->queues[i]);
  }
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));

  iree_arena_block_pool_trim(&device->small_block_pool);
  iree_arena_block_pool_trim(&device->large_block_pool);

  return iree_ok_status();
}

static iree_status_t iree_hal_task_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value =
        iree_hal_query_any_executable_loader_support(
            device->loader_count, device->loaders, /*caching_mode=*/0, key)
            ? 1
            : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = (int64_t)device->queue_count;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      // NOTE: we always return the queue 0 worker count. This will be incorrect
      // if there are multiple queues with differing queue counts but that's ok.
      *out_value =
          (int64_t)iree_task_executor_worker_count(device->queues[0].executor);
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.cpu"))) {
    return iree_cpu_lookup_data_by_key(key, out_value);
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_task_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_task_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_task_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_task_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  device->topology_info = *topology_info;
  return iree_ok_status();
}

// Returns the queue index to submit work to based on the |queue_affinity|.
//
// If we wanted to have dedicated transfer queues we'd fork off based on
// command_categories. For now all queues are general purpose.
static iree_host_size_t iree_hal_task_device_select_queue(
    iree_hal_task_device_t* device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity) {
  // TODO(benvanik): evaluate if we want to obscure this mapping a bit so that
  // affinity really means "equivalent affinities map to equivalent queues" and
  // not a specific queue index.
  return queue_affinity % device->queue_count;
}

static iree_status_t iree_hal_task_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_task_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return iree_hal_block_command_buffer_create(
      iree_hal_device_allocator(base_device), mode, command_categories,
      queue_affinity, binding_capacity, &device->large_block_pool,
      device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_task_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_hal_task_event_create(queue_affinity, flags,
                                    iree_hal_device_host_allocator(base_device),
                                    out_event);
}

static iree_status_t iree_hal_task_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);

  // Sum up the total worker count across all queues so that the loaders can
  // preallocate worker-specific storage.
  iree_host_size_t total_worker_count = 0;
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    total_worker_count +=
        iree_task_executor_worker_count(device->queues[i].executor);
  }

  return iree_hal_local_executable_cache_create(
      identifier, total_worker_count, device->loader_count, device->loaders,
      iree_hal_device_host_allocator(base_device), out_executable_cache);
}

// Returns the proactor for the given queue affinity. If the affinity specifies
// a particular queue, returns that queue's NUMA-correct proactor. Otherwise
// returns the device default proactor (from queue 0's NUMA node).
static iree_async_proactor_t* iree_hal_task_device_proactor_for_affinity(
    iree_hal_task_device_t* device, iree_hal_queue_affinity_t queue_affinity) {
  if (queue_affinity != 0 && queue_affinity != IREE_HAL_QUEUE_AFFINITY_ANY) {
    iree_host_size_t queue_index =
        iree_math_count_trailing_zeros_u64(queue_affinity);
    if (queue_index < device->queue_count) {
      return device->queues[queue_index].proactor;
    }
  }
  return device->proactor;
}

static iree_status_t iree_hal_task_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_async_proactor_t* proactor =
      iree_hal_task_device_proactor_for_affinity(device, queue_affinity);
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      proactor, iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_task_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_async_proactor_t* proactor =
      iree_hal_task_device_proactor_for_affinity(device, queue_affinity);
  return iree_hal_task_semaphore_create(proactor, initial_value,
                                        device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_task_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  if (iree_hal_task_semaphore_isa(semaphore)) {
    // Fast-path for semaphores related to this device.
    // TODO(benvanik): ensure the creating devices are compatible as if
    // independent task systems are used things may not work right (ownership
    // confusion).
    return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
  }
  // For now we support all semaphore types as we only need wait sources and
  // all semaphores can be wrapped in those.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
}

static iree_status_t iree_hal_task_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  const iree_host_size_t queue_index = iree_hal_task_device_select_queue(
      device, IREE_HAL_COMMAND_CATEGORY_ANY, queue_affinity);
  iree_hal_task_queue_t* queue = &device->queues[queue_index];

  // Create the transient buffer handle (reservation). This is returned to the
  // caller immediately — the backing memory is allocated in the queue drain
  // handler when all wait semaphores have been satisfied.
  iree_hal_buffer_placement_t placement = {
      .device = base_device,
      .queue_affinity = queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  iree_hal_buffer_t* transient_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_task_transient_buffer_create(
      placement, params, allocation_size,
      iree_hal_allocator_host_allocator(iree_hal_device_allocator(base_device)),
      &transient_buffer));

  // Submit the alloca operation to the queue. The drain handler will allocate
  // the real backing memory and commit it into the transient buffer.
  iree_status_t status = iree_hal_task_queue_submit_alloca(
      queue, iree_hal_device_allocator(base_device), params, allocation_size,
      transient_buffer, wait_semaphore_list, signal_semaphore_list);
  if (iree_status_is_ok(status)) {
    *out_buffer = transient_buffer;
  } else {
    iree_hal_buffer_release(transient_buffer);
  }
  return status;
}

static iree_status_t iree_hal_task_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  const iree_host_size_t queue_index = iree_hal_task_device_select_queue(
      device, IREE_HAL_COMMAND_CATEGORY_ANY, queue_affinity);
  iree_hal_task_queue_t* queue = &device->queues[queue_index];

  if (iree_hal_task_transient_buffer_isa(buffer)) {
    // Native dealloca: decommit the transient buffer in the queue drain handler
    // after all wait semaphores have been satisfied.
    return iree_hal_task_queue_submit_dealloca(
        queue, buffer, wait_semaphore_list, signal_semaphore_list);
  }

  // Non-transient buffer (e.g. synchronous allocation): degrade to barrier.
  return iree_hal_device_queue_barrier(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_task_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ));

  // Zero-length: degenerate to barrier (just forward wait→signal).
  if (length == 0) {
    return iree_hal_device_queue_barrier(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  // Memory file fast path: route to queue_copy via the storage buffer.
  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(source_file);
  if (storage_buffer) {
    return iree_hal_device_queue_copy(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        storage_buffer, (iree_device_size_t)source_offset, target_buffer,
        target_offset, length, IREE_HAL_COPY_FLAG_NONE);
  }

  // FD file path: async proactor I/O.
  if (!iree_hal_file_async_handle(source_file)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file has no storage buffer and no async handle; cannot perform read");
  }

  // Validate range against file length.
  uint64_t file_length = iree_hal_file_length(source_file);
  if (source_offset + length > file_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "read range [%" PRIu64 ", %" PRIu64 ") exceeds file length %" PRIu64,
        source_offset, source_offset + (uint64_t)length, file_length);
  }

  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  const iree_host_size_t queue_index = iree_hal_task_device_select_queue(
      device, IREE_HAL_COMMAND_CATEGORY_ANY, queue_affinity);
  return iree_hal_task_queue_submit_read(
      &device->queues[queue_index], source_file, source_offset, target_buffer,
      target_offset, length, wait_semaphore_list, signal_semaphore_list);
}

static iree_status_t iree_hal_task_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(target_file, IREE_HAL_MEMORY_ACCESS_WRITE));

  // Zero-length: degenerate to barrier.
  if (length == 0) {
    return iree_hal_device_queue_barrier(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  // Memory file fast path: route to queue_copy via the storage buffer.
  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(target_file);
  if (storage_buffer) {
    return iree_hal_device_queue_copy(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        source_buffer, source_offset, storage_buffer,
        (iree_device_size_t)target_offset, length, IREE_HAL_COPY_FLAG_NONE);
  }

  // FD file path: async proactor I/O.
  if (!iree_hal_file_async_handle(target_file)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file has no storage buffer and no async handle; cannot perform write");
  }

  // Validate range against file length.
  uint64_t file_length = iree_hal_file_length(target_file);
  if (target_offset + length > file_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "write range [%" PRIu64 ", %" PRIu64 ") exceeds file length %" PRIu64,
        target_offset, target_offset + (uint64_t)length, file_length);
  }

  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  const iree_host_size_t queue_index = iree_hal_task_device_select_queue(
      device, IREE_HAL_COMMAND_CATEGORY_ANY, queue_affinity);
  return iree_hal_task_queue_submit_write(
      &device->queues[queue_index], source_buffer, source_offset, target_file,
      target_offset, length, wait_semaphore_list, signal_semaphore_list);
}

static iree_status_t iree_hal_task_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  const iree_host_size_t queue_index = iree_hal_task_device_select_queue(
      device, IREE_HAL_COMMAND_CATEGORY_ANY, queue_affinity);
  return iree_hal_task_queue_submit_host_call(
      &device->queues[queue_index], base_device, 1ull << queue_index,
      wait_semaphore_list, signal_semaphore_list, call, args, flags);
}

static iree_status_t iree_hal_task_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  // NOTE: today we are not discriminating queues based on command type.
  const iree_host_size_t queue_index = iree_hal_task_device_select_queue(
      device, IREE_HAL_COMMAND_CATEGORY_ANY, queue_affinity);
  if (command_buffer == NULL) {
    // Fast-path for barriers (fork/join/sequence).
    return iree_hal_task_queue_submit_barrier(&device->queues[queue_index],
                                              wait_semaphore_list,
                                              signal_semaphore_list);
  }
  iree_hal_task_submission_batch_t batch = {
      .wait_semaphores = wait_semaphore_list,
      .signal_semaphores = signal_semaphore_list,
      .command_buffer = command_buffer,
      .binding_table = binding_table,
  };
  return iree_hal_task_queue_submit_commands(&device->queues[queue_index], 1,
                                             &batch);
}

static iree_status_t iree_hal_task_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Currently unused; we flush as submissions are made.
  return iree_ok_status();
}

static iree_status_t iree_hal_task_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  // We could hook in to vendor APIs (Intel/ARM/etc) or generic perf infra:
  // https://man7.org/linux/man-pages/man2/perf_event_open.2.html
  // Capturing things like:
  //   PERF_COUNT_HW_CPU_CYCLES / PERF_COUNT_HW_INSTRUCTIONS
  //   PERF_COUNT_HW_CACHE_REFERENCES / PERF_COUNT_HW_CACHE_MISSES
  //   etc
  // TODO(benvanik): shared iree/hal/local/profiling implementation of this.
  return iree_ok_status();
}

static iree_status_t iree_hal_task_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_task_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_task_device_vtable = {
    .destroy = iree_hal_task_device_destroy,
    .id = iree_hal_task_device_id,
    .host_allocator = iree_hal_task_device_host_allocator,
    .device_allocator = iree_hal_task_device_allocator,
    .replace_device_allocator = iree_hal_task_replace_device_allocator,
    .replace_channel_provider = iree_hal_task_replace_channel_provider,
    .trim = iree_hal_task_device_trim,
    .query_i64 = iree_hal_task_device_query_i64,
    .query_capabilities = iree_hal_task_device_query_capabilities,
    .topology_info = iree_hal_task_device_topology_info,
    .refine_topology_edge = iree_hal_task_device_refine_topology_edge,
    .assign_topology_info = iree_hal_task_device_assign_topology_info,
    .create_channel = iree_hal_task_device_create_channel,
    .create_command_buffer = iree_hal_task_device_create_command_buffer,
    .create_event = iree_hal_task_device_create_event,
    .create_executable_cache = iree_hal_task_device_create_executable_cache,
    .import_file = iree_hal_task_device_import_file,
    .create_semaphore = iree_hal_task_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_task_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_task_device_queue_alloca,
    .queue_dealloca = iree_hal_task_device_queue_dealloca,
    .queue_fill = iree_hal_device_queue_emulated_fill,
    .queue_update = iree_hal_device_queue_emulated_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_task_device_queue_read,
    .queue_write = iree_hal_task_device_queue_write,
    .queue_host_call = iree_hal_task_device_queue_host_call,
    .queue_dispatch = iree_hal_device_queue_emulated_dispatch,
    .queue_execute = iree_hal_task_device_queue_execute,
    .queue_flush = iree_hal_task_device_queue_flush,
    .profiling_begin = iree_hal_task_device_profiling_begin,
    .profiling_flush = iree_hal_task_device_profiling_flush,
    .profiling_end = iree_hal_task_device_profiling_end,
};
