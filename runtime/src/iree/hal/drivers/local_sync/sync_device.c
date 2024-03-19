// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_sync/sync_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/cpu.h"
#include "iree/hal/drivers/local_sync/sync_event.h"
#include "iree/hal/drivers/local_sync/sync_semaphore.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/inline_command_buffer.h"
#include "iree/hal/local/local_executable_cache.h"
#include "iree/hal/local/local_pipeline_layout.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"

typedef struct iree_hal_sync_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t large_block_pool;

  // Shared semaphore state used to emulate OS-level primitives. This backend
  // is intended to run on bare-metal systems where we need to perform all
  // synchronization ourselves.
  iree_hal_sync_semaphore_state_t semaphore_state;

  iree_host_size_t loader_count;
  iree_hal_executable_loader_t* loaders[];
} iree_hal_sync_device_t;

static const iree_hal_device_vtable_t iree_hal_sync_device_vtable;

static iree_hal_sync_device_t* iree_hal_sync_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_sync_device_vtable);
  return (iree_hal_sync_device_t*)base_value;
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
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(!loader_count || loaders);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_sync_device_check_params(params));

  iree_hal_sync_device_t* device = NULL;
  iree_host_size_t struct_size =
      sizeof(*device) + loader_count * sizeof(*device->loaders);
  iree_host_size_t total_size = struct_size + identifier.size;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device);
  if (iree_status_is_ok(status)) {
    memset(device, 0, total_size);
    iree_hal_resource_initialize(&iree_hal_sync_device_vtable,
                                 &device->resource);
    iree_string_view_append_to_buffer(identifier, &device->identifier,
                                      (char*)device + struct_size);
    device->host_allocator = host_allocator;
    device->device_allocator = device_allocator;
    iree_hal_allocator_retain(device_allocator);
    iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                     &device->large_block_pool);

    device->loader_count = loader_count;
    for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
      device->loaders[i] = loaders[i];
      iree_hal_executable_loader_retain(device->loaders[i]);
    }

    iree_hal_sync_semaphore_state_initialize(&device->semaphore_state);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_sync_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_sync_semaphore_state_deinitialize(&device->semaphore_state);

  for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
    iree_hal_executable_loader_release(device->loaders[i]);
  }

  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);

  iree_arena_block_pool_deinitialize(&device->large_block_pool);

  iree_allocator_free(host_allocator, device);

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
        base_device, mode, command_categories, queue_affinity, binding_capacity,
        iree_hal_device_host_allocator(base_device), out_command_buffer);
  } else {
    iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
    return iree_hal_deferred_command_buffer_create(
        base_device, mode, command_categories, binding_capacity,
        &device->large_block_pool, device->host_allocator, out_command_buffer);
  }
}

static iree_status_t iree_hal_sync_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  return iree_hal_local_descriptor_set_layout_create(
      flags, binding_count, bindings,
      iree_hal_device_host_allocator(base_device), out_descriptor_set_layout);
}

static iree_status_t iree_hal_sync_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  return iree_hal_sync_event_create(iree_hal_device_host_allocator(base_device),
                                    out_event);
}

static iree_status_t iree_hal_sync_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return iree_hal_local_executable_cache_create(
      identifier, /*worker_capacity=*/1, device->loader_count, device->loaders,
      iree_hal_device_host_allocator(base_device), out_executable_cache);
}

static iree_status_t iree_hal_sync_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
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

static iree_status_t iree_hal_sync_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  return iree_hal_local_pipeline_layout_create(
      push_constants, set_layout_count, set_layouts,
      iree_hal_device_host_allocator(base_device), out_pipeline_layout);
}

static iree_status_t iree_hal_sync_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return iree_hal_sync_semaphore_create(&device->semaphore_state, initial_value,
                                        device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_sync_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // The synchronous submission queue handles all semaphores as if host-side.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_sync_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // TODO(benvanik): queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  // TODO(benvanik): queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_apply_deferred_command_buffers(
    iree_hal_sync_device_t* device, iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  // See if there are any deferred command buffers; this saves us work in cases
  // of pure inline execution.
  bool any_deferred = false;
  for (iree_host_size_t i = 0; i < command_buffer_count && !any_deferred; ++i) {
    any_deferred = iree_hal_deferred_command_buffer_isa(command_buffers[i]);
  }
  if (!any_deferred) return iree_ok_status();

  // Stack allocate storage for an inline command buffer we'll use to replay
  // the deferred command buffers. We want to reset it between each apply so
  // that we don't get state carrying across.
  iree_byte_span_t storage =
      iree_make_byte_span(iree_alloca(iree_hal_inline_command_buffer_size()),
                          iree_hal_inline_command_buffer_size());

  // NOTE: we ignore any inline command buffers that may be passed in as they've
  // already executed during recording. The caller is probably in for a bad time
  // if they mixed the two modes together!
  for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
    iree_hal_command_buffer_t* command_buffer = command_buffers[i];
    if (iree_hal_deferred_command_buffer_isa(command_buffer)) {
      iree_hal_command_buffer_t* inline_command_buffer = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_inline_command_buffer_initialize(
          (iree_hal_device_t*)device,
          iree_hal_command_buffer_mode(command_buffer) |
              IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
          IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
          /*binding_capacity=*/0, device->host_allocator, storage,
          &inline_command_buffer));
      iree_status_t status = iree_hal_deferred_command_buffer_apply(
          command_buffer, inline_command_buffer,
          iree_hal_buffer_binding_table_empty());
      iree_hal_inline_command_buffer_deinitialize(inline_command_buffer);
      IREE_RETURN_IF_ERROR(status);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
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

static iree_status_t iree_hal_sync_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
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

static iree_status_t iree_hal_sync_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);

  // TODO(#4680): there is some better error handling here needed; we should
  // propagate failures to all signal semaphores. Today we aren't as there
  // shouldn't be any failures or if there are there's not much we'd be able to
  // do - chances are we already executed everything inline!

  // Wait for semaphores to be signaled before performing any work.
  IREE_RETURN_IF_ERROR(iree_hal_sync_semaphore_multi_wait(
      &device->semaphore_state, IREE_HAL_WAIT_MODE_ALL, wait_semaphore_list,
      iree_infinite_timeout()));

  // Run all deferred command buffers - any we could have run inline we already
  // did during recording.
  IREE_RETURN_IF_ERROR(iree_hal_sync_device_apply_deferred_command_buffers(
      device, command_buffer_count, command_buffers));

  // Signal all semaphores now that batch work has completed.
  IREE_RETURN_IF_ERROR(iree_hal_sync_semaphore_multi_signal(
      &device->semaphore_state, signal_semaphore_list));

  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Currently unused; we flush as submissions are made.
  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return iree_hal_sync_semaphore_multi_wait(&device->semaphore_state, wait_mode,
                                            semaphore_list, timeout);
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
    .create_channel = iree_hal_sync_device_create_channel,
    .create_command_buffer = iree_hal_sync_device_create_command_buffer,
    .create_descriptor_set_layout =
        iree_hal_sync_device_create_descriptor_set_layout,
    .create_event = iree_hal_sync_device_create_event,
    .create_executable_cache = iree_hal_sync_device_create_executable_cache,
    .import_file = iree_hal_sync_device_import_file,
    .create_pipeline_layout = iree_hal_sync_device_create_pipeline_layout,
    .create_semaphore = iree_hal_sync_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_sync_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_sync_device_queue_alloca,
    .queue_dealloca = iree_hal_sync_device_queue_dealloca,
    .queue_read = iree_hal_sync_device_queue_read,
    .queue_write = iree_hal_sync_device_queue_write,
    .queue_execute = iree_hal_sync_device_queue_execute,
    .queue_flush = iree_hal_sync_device_queue_flush,
    .wait_semaphores = iree_hal_sync_device_wait_semaphores,
    .profiling_begin = iree_hal_sync_device_profiling_begin,
    .profiling_flush = iree_hal_sync_device_profiling_flush,
    .profiling_end = iree_hal_sync_device_profiling_end,
};
