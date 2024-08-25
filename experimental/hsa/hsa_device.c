// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/hsa_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/hsa/dynamic_symbols.h"
#include "experimental/hsa/event_pool.h"
#include "experimental/hsa/event_semaphore.h"
#include "experimental/hsa/hsa_allocator.h"
#include "experimental/hsa/hsa_buffer.h"
#include "experimental/hsa/nop_executable_cache.h"
#include "experimental/hsa/pending_queue_actions.h"
#include "experimental/hsa/pipeline_layout.h"
#include "experimental/hsa/queue_command_buffer.h"
#include "experimental/hsa/status_util.h"
#include "experimental/hsa/timepoint_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/event_pool.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"

//===----------------------------------------------------------------------===//
// iree_hal_hsa_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hsa_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the HSA symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  const iree_hal_hsa_dynamic_symbols_t* hsa_symbols;

  // Parameters used to control device behavior.
  iree_hal_hsa_device_params_t params;

  // The hsa agent
  hsa_agent_t hsa_agent;

  // The queue where we will dispatch work
  hsa_queue_t* hsa_dispatch_queue;

  // The host allocator
  iree_allocator_t host_allocator;

  // Host/device event pools, used for backing semaphore timepoints.
  iree_event_pool_t* host_event_pool;
  iree_hal_hsa_event_pool_t* device_event_pool;
  // Timepoint pools, shared by various semaphores.
  iree_hal_hsa_timepoint_pool_t* timepoint_pool;

  // A queue to order device workloads and relase to the GPU when constraints
  // are met. It buffers submissions and allocations internally before they
  // are ready. This queue couples with HAL semaphores backed by iree_event_t
  // and hsa_signal_t objects.
  iree_hal_hsa_pending_queue_actions_t* pending_queue_actions;

  // Device allocator.
  iree_hal_allocator_t* device_allocator;
} iree_hal_hsa_device_t;

static const iree_hal_device_vtable_t iree_hal_hsa_device_vtable;

static iree_hal_hsa_device_t* iree_hal_hsa_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_device_vtable);
  return (iree_hal_hsa_device_t*)base_value;
}

static iree_hal_hsa_device_t* iree_hal_hsa_device_cast_unsafe(
    iree_hal_device_t* base_value) {
  return (iree_hal_hsa_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hsa_device_params_initialize(
    iree_hal_hsa_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->event_pool_capacity = 32;
  out_params->queue_count = 1;
  out_params->queue_tracing = false;
}

static iree_status_t iree_hal_hsa_device_check_params(
    const iree_hal_hsa_device_params_t* params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  if (params->queue_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one queue is required");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hsa_device_params_t* params, hsa_agent_t agent,
    hsa_queue_t* dispatch_queue, const iree_hal_hsa_dynamic_symbols_t* symbols,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_hsa_device_t* device = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_hsa_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + iree_sizeof_struct(*device));
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  device->hsa_symbols = symbols;
  device->params = *params;
  device->hsa_agent = agent;
  device->hsa_dispatch_queue = dispatch_queue;
  device->host_allocator = host_allocator;

  iree_status_t status = iree_hal_hsa_pending_queue_actions_create(
      symbols, &device->block_pool, host_allocator,
      &device->pending_queue_actions);

  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_allocator_create(symbols, agent, host_allocator,
                                           &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_hsa_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hsa_device_params_t* params,
    const iree_hal_hsa_dynamic_symbols_t* symbols, hsa_agent_t agent,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_hsa_device_check_params(params);

  size_t num_queue_packets = 1024;
  hsa_queue_type_t queue_type = HSA_QUEUE_TYPE_MULTI;
  void* callback = NULL;
  void* data = NULL;
  uint32_t private_segment_size = 0;
  uint32_t group_segment_size = 0;
  hsa_queue_t* dispatch_queue;

  IREE_HSA_RETURN_IF_ERROR(
      symbols,
      hsa_queue_create(agent, num_queue_packets, queue_type, callback, data,
                       private_segment_size, group_segment_size,
                       &dispatch_queue),
      "hsa_queue_create");

  status = iree_hal_hsa_device_create_internal(driver, identifier, params,
                                               agent, dispatch_queue, symbols,
                                               host_allocator, out_device);

  iree_event_pool_t* host_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_event_pool_allocate(params->event_pool_capacity,
                                      host_allocator, &host_event_pool);
  }

  iree_hal_hsa_event_pool_t* device_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_hsa_event_pool_allocate(symbols, params->event_pool_capacity,
                                         host_allocator, &device_event_pool);
  }

  iree_hal_hsa_timepoint_pool_t* timepoint_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_timepoint_pool_allocate(
        host_event_pool, device_event_pool, params->event_pool_capacity,
        host_allocator, &timepoint_pool);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_hsa_device_t* hsa_device = iree_hal_hsa_device_cast(*out_device);
    hsa_device->host_event_pool = host_event_pool;
    hsa_device->device_event_pool = device_event_pool;
    hsa_device->timepoint_pool = timepoint_pool;
  } else {
    // Release resources we have accquired after HAL device creation.
    if (timepoint_pool) iree_hal_hsa_timepoint_pool_free(timepoint_pool);
    if (device_event_pool) iree_hal_hsa_event_pool_release(device_event_pool);
    if (host_event_pool) iree_event_pool_free(host_event_pool);
    // Release other resources via the HAL device.
    iree_hal_device_release(*out_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_hsa_dynamic_symbols_t* iree_hal_hsa_device_dynamic_symbols(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast_unsafe(base_device);
  return device->hsa_symbols;
}

static void iree_hal_hsa_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Destroy the pending workload queue.
  iree_hal_hsa_pending_queue_actions_destroy(
      (iree_hal_resource_t*)device->pending_queue_actions);

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Destroy various pools for synchronization.
  if (device->timepoint_pool) {
    iree_hal_hsa_timepoint_pool_free(device->timepoint_pool);
  }
  if (device->device_event_pool) {
    iree_hal_hsa_event_pool_release(device->device_event_pool);
  }
  if (device->host_event_pool) iree_event_pool_free(device->host_event_pool);

  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_hsa_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_hsa_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_hsa_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_hsa_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_hsa_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
}

static iree_status_t iree_hal_hsa_device_trim(iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "memory pools are not supported");
}

static iree_status_t iree_hal_hsa_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("rocm-hsaco-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_hsa_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "channel not yet implemented");
}

iree_status_t iree_hal_hsa_device_create_queue_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_queue_command_buffer_create(
      base_device, device->hsa_symbols, mode, command_categories,
      binding_capacity, device->hsa_dispatch_queue, &device->block_pool,
      device->host_allocator, device->device_allocator, out_command_buffer);
}

static iree_status_t iree_hal_hsa_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);

  return iree_hal_deferred_command_buffer_create(
      iree_hal_device_allocator(base_device), mode, command_categories,
      binding_capacity, &device->block_pool,
      iree_hal_device_host_allocator(base_device), out_command_buffer);
}

static iree_status_t iree_hal_hsa_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_descriptor_set_layout_create(
      flags, binding_count, bindings, device->host_allocator,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_hsa_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet implemented");
}

static iree_status_t iree_hal_hsa_device_import_file(
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

static iree_status_t iree_hal_hsa_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_nop_executable_cache_create(
      identifier, device->hsa_symbols, device->hsa_agent,
      device->host_allocator, device->device_allocator, out_executable_cache);
}

static iree_status_t iree_hal_hsa_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_pipeline_layout_create(
      set_layout_count, set_layouts, push_constants, device->host_allocator,
      out_pipeline_layout);
}

static iree_status_t iree_hal_hsa_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_event_semaphore_create(
      initial_value, device->hsa_symbols, device->timepoint_pool,
      device->pending_queue_actions, device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_hsa_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement HSA semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_hsa_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The HSA HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  iree_status_t status =
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer);

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  return status;
}

static iree_status_t iree_hal_hsa_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The HSA HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  // Buffer will be freed when the buffer is released.

  // Only signal if not returning a synchronous error
  iree_status_t status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  return status;
}

static iree_status_t iree_hal_hsa_device_queue_read(
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

static iree_status_t iree_hal_hsa_device_queue_write(
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

static void iree_hal_hsa_device_collect_tracing_context(void* user_data) {}

static iree_status_t iree_hal_hsa_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_hsa_pending_queue_actions_enqueue_execution(
      base_device, device->hsa_dispatch_queue, device->pending_queue_actions,
      iree_hal_hsa_device_collect_tracing_context, wait_semaphore_list,
      signal_semaphore_list, command_buffer_count, command_buffers);
  if (iree_status_is_ok(status)) {
    // Try to advance the pending workload queue.
    status =
        iree_hal_hsa_pending_queue_actions_issue(device->pending_queue_actions);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  // Try to advance the pending workload queue.
  iree_status_t status =
      iree_hal_hsa_pending_queue_actions_issue(device->pending_queue_actions);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_semaphore_multi_wait(semaphore_list, wait_mode, timeout,
                                           &device->block_pool);
}

static iree_status_t iree_hal_hsa_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_hsa_device_vtable = {
    .destroy = iree_hal_hsa_device_destroy,
    .id = iree_hal_hsa_device_id,
    .host_allocator = iree_hal_hsa_device_host_allocator,
    .device_allocator = iree_hal_hsa_device_allocator,
    .replace_device_allocator = iree_hal_hsa_replace_device_allocator,
    .replace_channel_provider = iree_hal_hsa_replace_channel_provider,
    .trim = iree_hal_hsa_device_trim,
    .query_i64 = iree_hal_hsa_device_query_i64,
    .create_channel = iree_hal_hsa_device_create_channel,
    .create_command_buffer = iree_hal_hsa_device_create_command_buffer,
    .create_descriptor_set_layout =
        iree_hal_hsa_device_create_descriptor_set_layout,
    .create_event = iree_hal_hsa_device_create_event,
    .create_executable_cache = iree_hal_hsa_device_create_executable_cache,
    .import_file = iree_hal_hsa_device_import_file,
    .create_pipeline_layout = iree_hal_hsa_device_create_pipeline_layout,
    .create_semaphore = iree_hal_hsa_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_hsa_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_hsa_device_queue_alloca,
    .queue_dealloca = iree_hal_hsa_device_queue_dealloca,
    .queue_read = iree_hal_hsa_device_queue_read,
    .queue_write = iree_hal_hsa_device_queue_write,
    .queue_execute = iree_hal_hsa_device_queue_execute,
    .queue_flush = iree_hal_hsa_device_queue_flush,
    .wait_semaphores = iree_hal_hsa_device_wait_semaphores,
    .profiling_begin = iree_hal_hsa_device_profiling_begin,
    .profiling_flush = iree_hal_hsa_device_profiling_flush,
    .profiling_end = iree_hal_hsa_device_profiling_end,
};
