// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_sync/sync_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/cpu.h"
#include "iree/hal/drivers/local_sync/sync_event.h"
#include "iree/hal/drivers/local_sync/sync_semaphore.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/inline_command_buffer.h"
#include "iree/hal/local/inline_dispatch.h"
#include "iree/hal/local/local_executable_cache.h"
#include "iree/hal/local/transient_buffer.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_registry.h"

typedef struct iree_hal_sync_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;
  iree_hal_slab_provider_t* default_slab_provider;
  iree_async_notification_t* default_pool_notification;
  iree_hal_pool_t* default_pool;

  // Proactor pool for async I/O. Retained for the lifetime of the device to
  // ensure proactor threads outlive all device resources (semaphores, etc.).
  iree_async_proactor_pool_t* proactor_pool;

  // Proactor selected from the pool for this device's async I/O operations.
  // Borrowed from the pool -- valid as long as the pool is retained.
  iree_async_proactor_t* proactor;

  // Shared frontier tracker for cross-device causal ordering. Retained after
  // topology assignment and released during device destruction.
  iree_async_frontier_tracker_t* frontier_tracker;

  // This device's single queue axis and monotonic epoch counter.
  // local_sync is synchronous — advance() is called after each signal.
  iree_async_axis_t axis;
  iree_atomic_int64_t epoch;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  iree_hal_device_topology_info_t topology_info;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t large_block_pool;

  iree_host_size_t loader_count;
  iree_hal_executable_loader_t* loaders[];
} iree_hal_sync_device_t;

static const iree_hal_device_vtable_t iree_hal_sync_device_vtable;

static iree_hal_sync_device_t* iree_hal_sync_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_sync_device_vtable);
  return (iree_hal_sync_device_t*)base_value;
}

static iree_status_t iree_hal_sync_device_create_default_pool(
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_slab_provider,
    iree_async_notification_t** out_notification, iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_slab_provider);
  IREE_ASSERT_ARGUMENT(out_notification);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_slab_provider = NULL;
  *out_notification = NULL;
  *out_pool = NULL;

  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_cpu_slab_provider_create(host_allocator, &slab_provider));

  iree_async_notification_t* notification = NULL;
  iree_status_t status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification);
  if (iree_status_is_ok(status)) {
    iree_hal_passthrough_pool_options_t options = {0};
    status = iree_hal_passthrough_pool_create(
        options, slab_provider, notification, host_allocator, out_pool);
  }
  if (iree_status_is_ok(status)) {
    *out_slab_provider = slab_provider;
    *out_notification = notification;
    slab_provider = NULL;
    notification = NULL;
  }
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
  return status;
}

static iree_status_t iree_hal_sync_device_resolve_pool(
    iree_hal_sync_device_t* device, iree_hal_pool_t* pool,
    iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = pool ? pool : device->default_pool;
  return iree_ok_status();
}

static bool iree_hal_sync_device_query_pool_epoch(void* user_data,
                                                  iree_async_axis_t axis,
                                                  uint64_t epoch) {
  iree_async_frontier_tracker_t* frontier_tracker =
      (iree_async_frontier_tracker_t*)user_data;
  return iree_async_frontier_tracker_query_epoch(frontier_tracker, axis, epoch);
}

// Advances the frontier tracker epoch for the device's single queue.
// Called after each successful semaphore signal. local_sync is fully
// synchronous so advance() at signal time = completion time.
static void iree_hal_sync_device_advance_frontier(
    iree_hal_sync_device_t* device) {
  uint64_t epoch = (uint64_t)iree_atomic_fetch_add(&device->epoch, 1,
                                                   iree_memory_order_acq_rel) +
                   1;
  iree_async_frontier_tracker_advance(device->frontier_tracker, device->axis,
                                      epoch);
}

void iree_hal_sync_device_params_initialize(
    iree_hal_sync_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
}

static iree_status_t iree_hal_sync_device_check_params(
    const iree_hal_sync_device_params_t* params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_sync_device_create(
    iree_string_view_t identifier, const iree_hal_sync_device_params_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(!loader_count || loaders);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_sync_device_check_params(params));

  iree_hal_sync_device_t* device = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t identifier_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(*device), &total_size,
              IREE_STRUCT_FIELD_ALIGNED(loader_count,
                                        iree_hal_executable_loader_t*, 1, NULL),
              IREE_STRUCT_FIELD_ALIGNED(identifier.size, char, 1,
                                        &identifier_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_aligned(host_allocator, total_size,
                                        iree_alignof(iree_hal_sync_device_t), 0,
                                        (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_sync_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + identifier_offset);
  device->host_allocator = host_allocator;
  device->device_allocator = device_allocator;
  iree_hal_allocator_retain(device_allocator);

  // Retain the proactor pool and acquire a proactor for this device.
  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);
  iree_atomic_store(&device->epoch, 0, iree_memory_order_relaxed);
  iree_status_t status =
      iree_async_proactor_pool_get(device->proactor_pool, 0, &device->proactor);

  if (iree_status_is_ok(status)) {
    status = iree_hal_sync_device_create_default_pool(
        device->proactor, device->host_allocator,
        &device->default_slab_provider, &device->default_pool_notification,
        &device->default_pool);
  }

  if (iree_status_is_ok(status)) {
    iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                     &device->large_block_pool);

    device->loader_count = loader_count;
    for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
      device->loaders[i] = loaders[i];
      iree_hal_executable_loader_retain(device->loaders[i]);
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

static void iree_hal_sync_device_clear_topology_info(
    iree_hal_sync_device_t* device) {
  if (device->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        device->frontier_tracker, device->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(device->frontier_tracker);
    device->frontier_tracker = NULL;
    device->axis = 0;
  }
  memset(&device->topology_info, 0, sizeof(device->topology_info));
}

static void iree_hal_sync_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
    iree_hal_executable_loader_release(device->loaders[i]);
  }

  iree_hal_pool_release(device->default_pool);
  iree_hal_slab_provider_release(device->default_slab_provider);
  iree_async_notification_release(device->default_pool_notification);
  iree_hal_sync_device_clear_topology_info(device);
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_async_proactor_pool_release(device->proactor_pool);

  iree_arena_block_pool_deinitialize(&device->large_block_pool);

  iree_allocator_free_aligned(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_sync_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_sync_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_sync_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_sync_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_sync_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_sync_device_trim(iree_hal_device_t* base_device) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_pool_trim(device->default_pool));
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_sync_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
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
      *out_value = 1;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = 1;
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

static iree_status_t iree_hal_sync_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_sync_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_sync_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  if (!topology_info) {
    iree_hal_sync_device_clear_topology_info(device);
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t* frontier_tracker =
      topology_info->frontier.tracker;
  iree_async_axis_t axis = topology_info->frontier.base_axis;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL));
  device->topology_info = *topology_info;
  device->frontier_tracker = frontier_tracker;
  device->axis = axis;
  iree_async_frontier_tracker_retain(device->frontier_tracker);
  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_sync_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  if (iree_all_bits_set(mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
    return iree_hal_inline_command_buffer_create(
        iree_hal_device_allocator(base_device), mode, command_categories,
        queue_affinity, binding_capacity,
        iree_hal_device_host_allocator(base_device), out_command_buffer);
  } else {
    iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
    return iree_hal_deferred_command_buffer_create(
        iree_hal_device_allocator(base_device), mode, command_categories,
        queue_affinity, binding_capacity, &device->large_block_pool,
        device->host_allocator, out_command_buffer);
  }
}

static iree_status_t iree_hal_sync_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_hal_sync_event_create(queue_affinity, flags,
                                    iree_hal_device_host_allocator(base_device),
                                    out_event);
}

static iree_status_t iree_hal_sync_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return iree_hal_local_executable_cache_create(
      identifier, /*worker_capacity=*/1, device->loader_count, device->loaders,
      iree_hal_device_host_allocator(base_device), out_executable_cache);
}

static iree_status_t iree_hal_sync_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      /*proactor=*/NULL, iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_sync_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return iree_hal_sync_semaphore_create(device->proactor, queue_affinity,
                                        initial_value, flags,
                                        device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_sync_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // The synchronous submission queue handles all semaphores as if host-side.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_sync_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  out_backend->slab_provider = device->default_slab_provider;
  out_backend->notification = device->default_pool_notification;
  out_backend->epoch_query = (iree_hal_pool_epoch_query_t){
      .fn = iree_hal_sync_device_query_pool_epoch,
      .user_data = device->frontier_tracker,
  };
  return iree_ok_status();
}

// Waits for all semaphore dependencies before a queue operation body.
static inline iree_status_t iree_hal_sync_device_queue_op_begin(
    iree_hal_sync_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list) {
  return iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
}

// Signals semaphores after a queue operation body completes (or fails them on
// error) and advances the frontier tracker.
static inline iree_status_t iree_hal_sync_device_queue_op_end(
    iree_hal_sync_device_t* device,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_status_t status) {
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                            /*frontier=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_sync_device_advance_frontier(device);
  } else {
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
    iree_async_frontier_tracker_fail_axis(
        device->frontier_tracker, device->axis,
        iree_status_from_code(iree_status_code(status)));
  }
  return status;
}

static iree_status_t iree_hal_sync_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  iree_hal_pool_t* allocation_pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_resolve_pool(device, pool, &allocation_pool));
  iree_hal_buffer_params_canonicalize(&params);
  iree_hal_allocator_query_buffer_compatibility(
      device->device_allocator, params, allocation_size, &params,
      /*out_allocation_size=*/NULL);
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_queue_op_begin(device, wait_semaphore_list));

  iree_hal_buffer_t* backing_buffer = NULL;
  iree_hal_buffer_t* transient_buffer = NULL;
  iree_status_t status = iree_hal_pool_allocate_buffer(
      allocation_pool, params, allocation_size,
      /*requester_frontier=*/NULL, iree_infinite_timeout(), &backing_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_placement_t placement = {
        .device = base_device,
        .queue_affinity = queue_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
    };
    status = iree_hal_local_transient_buffer_create(
        placement, params, iree_hal_buffer_allocation_size(backing_buffer),
        iree_hal_buffer_byte_length(backing_buffer), device->host_allocator,
        &transient_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_local_transient_buffer_stage_backing(transient_buffer,
                                                  backing_buffer);
    iree_hal_local_transient_buffer_commit(transient_buffer);
  }
  iree_hal_buffer_release(backing_buffer);

  status =
      iree_hal_sync_device_queue_op_end(device, signal_semaphore_list, status);
  if (iree_status_is_ok(status)) {
    *out_buffer = transient_buffer;
  } else {
    iree_hal_buffer_release(transient_buffer);
  }
  return status;
}

static iree_status_t iree_hal_sync_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_queue_op_begin(device, wait_semaphore_list));
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  if (placement.device == base_device &&
      iree_hal_local_transient_buffer_isa(buffer)) {
    iree_hal_local_transient_buffer_decommit(buffer);
  }
  return iree_hal_sync_device_queue_op_end(device, signal_semaphore_list,
                                           iree_ok_status());
}

static iree_status_t iree_hal_sync_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_queue_op_begin(device, wait_semaphore_list));
  iree_status_t status = iree_hal_buffer_map_fill(
      target_buffer, target_offset, length, pattern, pattern_length);
  return iree_hal_sync_device_queue_op_end(device, signal_semaphore_list,
                                           status);
}

static iree_status_t iree_hal_sync_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_queue_op_begin(device, wait_semaphore_list));
  iree_status_t status = iree_hal_buffer_map_write(
      target_buffer, target_offset,
      (const uint8_t*)source_buffer + source_offset, length);
  return iree_hal_sync_device_queue_op_end(device, signal_semaphore_list,
                                           status);
}

static iree_status_t iree_hal_sync_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_queue_op_begin(device, wait_semaphore_list));
  iree_hal_buffer_mapping_t source_mapping = {{0}};
  iree_status_t status = iree_hal_buffer_map_range(
      source_buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      source_offset, length, &source_mapping);
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_write(target_buffer, target_offset,
                                       source_mapping.contents.data,
                                       source_mapping.contents.data_length);
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&source_mapping));
  }
  return iree_hal_sync_device_queue_op_end(device, signal_semaphore_list,
                                           status);
}

static iree_status_t iree_hal_sync_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ));
  if (length == 0) {
    return iree_hal_device_queue_barrier(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_EXECUTE_FLAG_NONE);
  }
  uint64_t file_length = iree_hal_file_length(source_file);
  if (file_length > 0 && source_offset + length > file_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "read range [%" PRIu64 ", %" PRIu64 ") exceeds file length %" PRIu64,
        source_offset, source_offset + (uint64_t)length, file_length);
  }
  // Memory file fast path: route to queue_copy via the storage buffer.
  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(source_file);
  if (storage_buffer) {
    return iree_hal_device_queue_copy(
        base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        storage_buffer, (iree_device_size_t)source_offset, target_buffer,
        target_offset, length, IREE_HAL_COPY_FLAG_NONE);
  }
  if (!iree_hal_file_supports_synchronous_io(source_file)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file does not support synchronous I/O");
  }
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_queue_op_begin(device, wait_semaphore_list));
  iree_status_t status = iree_hal_file_read(
      source_file, source_offset, target_buffer, target_offset, length);
  return iree_hal_sync_device_queue_op_end(device, signal_semaphore_list,
                                           status);
}

static iree_status_t iree_hal_sync_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(target_file, IREE_HAL_MEMORY_ACCESS_WRITE));
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
  if (!iree_hal_file_supports_synchronous_io(target_file)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file does not support synchronous I/O");
  }
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_queue_op_begin(device, wait_semaphore_list));
  iree_status_t status = iree_hal_file_write(
      target_file, target_offset, source_buffer, source_offset, length);
  return iree_hal_sync_device_queue_op_end(device, signal_semaphore_list,
                                           status);
}

static iree_status_t iree_hal_sync_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_queue_op_begin(device, wait_semaphore_list));
  iree_status_t status = iree_hal_local_executable_dispatch_inline(
      executable, export_ordinal, config, constants, bindings.values,
      bindings.count, flags);
  return iree_hal_sync_device_queue_op_end(device, signal_semaphore_list,
                                           status);
}

static iree_status_t iree_hal_sync_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  // Wait for all dependencies.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  // If non-blocking then immediately signal the dependencies instead of letting
  // the call do it. We don't expect this to allow more work to proceed in the
  // sync device case _on this device_ but it may on others.
  const bool is_nonblocking =
      iree_any_bit_set(flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  if (is_nonblocking) {
    // NOTE: the signals can fail in which case we never perform the call.
    // That's ok as failure to signal is considered a device-loss/death
    // situation as there's no telling what has gone wrong.
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list,
                                                        /*frontier=*/NULL));
    iree_hal_sync_device_advance_frontier(device);
  }

  // Issue the call.
  iree_hal_host_call_context_t context = {
      .device = base_device,
      .queue_affinity = queue_affinity,
      .signal_semaphore_list = is_nonblocking ? iree_hal_semaphore_list_empty()
                                              : signal_semaphore_list,
  };
  iree_status_t call_status = call.fn(call.user_data, args, &context);

  if (is_nonblocking || iree_status_is_deferred(call_status)) {
    // User callback will signal in the future (or they are fire-and-forget).
    return iree_ok_status();
  } else if (iree_status_is_ok(call_status)) {
    // Signal callback completed synchronously.
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list,
                                                        /*frontier=*/NULL));
    iree_hal_sync_device_advance_frontier(device);
    return iree_ok_status();
  } else {
    // If the call failed we need to fail all dependent semaphores to propagate
    // the error.
    if (!is_nonblocking) {
      iree_async_frontier_tracker_fail_axis(
          device->frontier_tracker, device->axis,
          iree_status_from_code(iree_status_code(call_status)));
      iree_hal_semaphore_list_fail(signal_semaphore_list, call_status);
    } else {
      iree_status_ignore(call_status);
    }
    return iree_ok_status();
  }
}

static iree_status_t iree_hal_sync_device_apply_deferred_command_buffer(
    iree_hal_sync_device_t* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  // If there were no deferred command buffers no-op this call - they've already
  // been issued.
  if (!command_buffer ||
      !iree_hal_deferred_command_buffer_isa(command_buffer)) {
    return iree_ok_status();
  }

  // Stack allocate storage for an inline command buffer we'll use to replay
  // the deferred command buffers. We want to reset it between each apply so
  // that we don't get state carrying across.
  iree_host_size_t storage_size = iree_hal_inline_command_buffer_size(
      iree_hal_command_buffer_mode(command_buffer) |
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
          // NOTE: we need to validate if a binding table is provided as
          // the bindings were not known when it was originally recorded.
          (iree_hal_buffer_binding_table_is_empty(binding_table)
               ? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
               : 0),
      /*binding_capacity=*/0);
  iree_byte_span_t storage =
      iree_make_byte_span(iree_alloca(storage_size), storage_size);

  // NOTE: we run unvalidated as inline command buffers don't support
  // binding tables and can be validated entirely while recording.
  iree_hal_command_buffer_t* inline_command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_inline_command_buffer_initialize(
      device->device_allocator,
      iree_hal_command_buffer_mode(command_buffer) |
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
          // NOTE: we need to validate if a binding table is provided as the
          // bindings were not known when it was originally recorded.
          (iree_hal_buffer_binding_table_is_empty(binding_table)
               ? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
               : 0),
      iree_hal_command_buffer_allowed_categories(command_buffer),
      IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, device->host_allocator, storage,
      &inline_command_buffer));

  iree_status_t status = iree_hal_deferred_command_buffer_apply(
      command_buffer, inline_command_buffer, binding_table);

  iree_hal_inline_command_buffer_deinitialize(inline_command_buffer);
  return status;
}

static iree_status_t iree_hal_sync_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_sync_device_queue_op_begin(device, wait_semaphore_list));
  iree_status_t status = iree_hal_sync_device_apply_deferred_command_buffer(
      device, command_buffer, binding_table);
  return iree_hal_sync_device_queue_op_end(device, signal_semaphore_list,
                                           status);
}

static iree_status_t iree_hal_sync_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Currently unused; we flush as submissions are made.
  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_profiling_begin(
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

static iree_status_t iree_hal_sync_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_sync_device_vtable = {
    .destroy = iree_hal_sync_device_destroy,
    .id = iree_hal_sync_device_id,
    .host_allocator = iree_hal_sync_device_host_allocator,
    .device_allocator = iree_hal_sync_device_allocator,
    .replace_device_allocator = iree_hal_sync_replace_device_allocator,
    .replace_channel_provider = iree_hal_sync_replace_channel_provider,
    .trim = iree_hal_sync_device_trim,
    .query_i64 = iree_hal_sync_device_query_i64,
    .query_capabilities = iree_hal_sync_device_query_capabilities,
    .topology_info = iree_hal_sync_device_topology_info,
    .refine_topology_edge = iree_hal_sync_device_refine_topology_edge,
    .assign_topology_info = iree_hal_sync_device_assign_topology_info,
    .create_channel = iree_hal_sync_device_create_channel,
    .create_command_buffer = iree_hal_sync_device_create_command_buffer,
    .create_event = iree_hal_sync_device_create_event,
    .create_executable_cache = iree_hal_sync_device_create_executable_cache,
    .import_file = iree_hal_sync_device_import_file,
    .create_semaphore = iree_hal_sync_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_sync_device_query_semaphore_compatibility,
    .query_queue_pool_backend = iree_hal_sync_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_sync_device_queue_alloca,
    .queue_dealloca = iree_hal_sync_device_queue_dealloca,
    .queue_fill = iree_hal_sync_device_queue_fill,
    .queue_update = iree_hal_sync_device_queue_update,
    .queue_copy = iree_hal_sync_device_queue_copy,
    .queue_read = iree_hal_sync_device_queue_read,
    .queue_write = iree_hal_sync_device_queue_write,
    .queue_host_call = iree_hal_sync_device_queue_host_call,
    .queue_dispatch = iree_hal_sync_device_queue_dispatch,
    .queue_execute = iree_hal_sync_device_queue_execute,
    .queue_flush = iree_hal_sync_device_queue_flush,
    .profiling_begin = iree_hal_sync_device_profiling_begin,
    .profiling_flush = iree_hal_sync_device_profiling_flush,
    .profiling_end = iree_hal_sync_device_profiling_end,
};
