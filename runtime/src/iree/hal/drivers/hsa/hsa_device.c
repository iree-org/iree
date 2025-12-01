// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hsa/hsa_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/hsa/dynamic_symbols.h"
#include "iree/hal/drivers/hsa/hsa_allocator.h"
#include "iree/hal/drivers/hsa/hsa_buffer.h"
#include "iree/hal/drivers/hsa/native_executable.h"
#include "iree/hal/drivers/hsa/nop_executable_cache.h"
#include "iree/hal/drivers/hsa/per_device_information.h"
#include "iree/hal/drivers/hsa/status_util.h"
#include "iree/hal/drivers/hsa/hsa_semaphore.h"
#include "iree/hal/drivers/hsa/stream_command_buffer.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"

#define IREE_HAL_DEVICE_TRANSFER_DEFAULT_BUFFER_SIZE (128 * 1024 * 1024)
#define IREE_HAL_DEVICE_MAX_TRANSFER_DEFAULT_CHUNK_SIZE (64 * 1024 * 1024)

//===----------------------------------------------------------------------===//
// iree_hal_hsa_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hsa_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size.
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the HSA symbols.
  iree_hal_driver_t* driver;

  const iree_hal_hsa_dynamic_symbols_t* hsa_symbols;

  // Parameters used to control device behavior.
  iree_hal_hsa_device_params_t params;

  iree_allocator_t host_allocator;

  // Device memory allocator.
  iree_hal_allocator_t* device_allocator;

  // Per-device information.
  iree_hal_hsa_per_device_info_t device_info;

  // Shared semaphore state for all semaphores created by this device.
  iree_hal_hsa_semaphore_state_t semaphore_state;
} iree_hal_hsa_device_t;

static iree_hal_hsa_device_t* iree_hal_hsa_device_cast(
    iree_hal_device_t* base_value);

static const iree_hal_device_vtable_t iree_hal_hsa_device_vtable;

static iree_hal_hsa_device_t* iree_hal_hsa_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_device_vtable);
  return (iree_hal_hsa_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hsa_device_params_initialize(
    iree_hal_hsa_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->event_pool_capacity = 32;
  out_params->queue_count = 1;
  out_params->stream_tracing = 0;
  out_params->async_allocations = true;
  out_params->file_transfer_buffer_size =
      IREE_HAL_DEVICE_TRANSFER_DEFAULT_BUFFER_SIZE;
  out_params->file_transfer_chunk_size =
      IREE_HAL_DEVICE_MAX_TRANSFER_DEFAULT_CHUNK_SIZE;
  out_params->allow_inline_execution = false;
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

// Callback for finding memory pools on an agent.
typedef struct iree_hal_hsa_memory_pool_search_t {
  iree_hal_hsa_dynamic_symbols_t* symbols;
  iree_hal_hsa_per_device_info_t* device_info;
  hsa_agent_t agent;
} iree_hal_hsa_memory_pool_search_t;

static hsa_status_t iree_hal_hsa_find_memory_pools_callback(
    hsa_amd_memory_pool_t pool, void* data) {
  iree_hal_hsa_memory_pool_search_t* search =
      (iree_hal_hsa_memory_pool_search_t*)data;

  // Check if pool is valid for allocations.
  bool alloc_allowed = false;
  hsa_status_t status = search->symbols->hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
  if (status != HSA_STATUS_SUCCESS || !alloc_allowed) {
    return HSA_STATUS_SUCCESS;  // Skip this pool.
  }

  // Get pool segment type.
  hsa_amd_segment_t segment;
  status = search->symbols->hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    return HSA_STATUS_SUCCESS;  // Skip this pool.
  }

  // Get global flags for global segment.
  if (segment == HSA_AMD_SEGMENT_GLOBAL) {
    uint32_t global_flags = 0;
    status = search->symbols->hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flags);
    if (status != HSA_STATUS_SUCCESS) {
      return HSA_STATUS_SUCCESS;
    }

    // Check if this is a coarse-grained (device local) pool.
    if ((global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
        !search->device_info->device_local_memory_pool_valid) {
      search->device_info->device_local_memory_pool = pool;
      search->device_info->device_local_memory_pool_valid = true;
    }

    // Check if this is a fine-grained (host visible) pool.
    if ((global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
        !search->device_info->host_visible_memory_pool_valid) {
      search->device_info->host_visible_memory_pool = pool;
      search->device_info->host_visible_memory_pool_valid = true;
    }

    // Check if this is a kernarg pool (using KERNARG_INIT which is the newer
    // flag).
    if ((global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) &&
        !search->device_info->kernarg_memory_pool_valid) {
      search->device_info->kernarg_memory_pool = pool;
      search->device_info->kernarg_memory_pool_valid = true;
    }
  }

  return HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_hsa_device_initialize_memory_pools(
    iree_hal_hsa_device_t* device, hsa_agent_t gpu_agent,
    hsa_agent_t cpu_agent) {
  // Find memory pools on the GPU agent.
  iree_hal_hsa_memory_pool_search_t gpu_search = {
      .symbols = (iree_hal_hsa_dynamic_symbols_t*)device->hsa_symbols,
      .device_info = &device->device_info,
      .agent = gpu_agent,
  };
  IREE_HSA_RETURN_IF_ERROR(
      device->hsa_symbols,
      hsa_amd_agent_iterate_memory_pools(
          gpu_agent, iree_hal_hsa_find_memory_pools_callback, &gpu_search),
      "hsa_amd_agent_iterate_memory_pools (GPU)");

  // Find fine-grained memory pool on CPU agent if not found on GPU.
  if (!device->device_info.host_visible_memory_pool_valid) {
    iree_hal_hsa_memory_pool_search_t cpu_search = {
        .symbols = (iree_hal_hsa_dynamic_symbols_t*)device->hsa_symbols,
        .device_info = &device->device_info,
        .agent = cpu_agent,
    };
    IREE_HSA_RETURN_IF_ERROR(
        device->hsa_symbols,
        hsa_amd_agent_iterate_memory_pools(
            cpu_agent, iree_hal_hsa_find_memory_pools_callback, &cpu_search),
        "hsa_amd_agent_iterate_memory_pools (CPU)");
  }

  if (!device->device_info.device_local_memory_pool_valid) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no device local memory pool found");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_hsa_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hsa_device_params_t* params,
    const iree_hal_hsa_dynamic_symbols_t* symbols, hsa_agent_t gpu_agent,
    hsa_agent_t cpu_agent, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_device = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_hsa_device_check_params(params));

  iree_hal_hsa_device_t* device = NULL;
  const iree_host_size_t total_size =
      iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_hsa_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + iree_sizeof_struct(*device));
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  device->hsa_symbols = symbols;
  memcpy(&device->params, params, sizeof(*params));
  device->host_allocator = host_allocator;

  // Initialize per-device info.
  memset(&device->device_info, 0, sizeof(device->device_info));
  device->device_info.agent = gpu_agent;
  device->device_info.cpu_agent = cpu_agent;

  // Initialize semaphore state.
  iree_hal_hsa_semaphore_state_initialize(&device->semaphore_state);

  iree_status_t status = iree_ok_status();

  // Initialize memory pools.
  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_device_initialize_memory_pools(device, gpu_agent,
                                                         cpu_agent);
  }

  // Create HSA queue for dispatch.
  if (iree_status_is_ok(status)) {
    uint32_t queue_size = 0;
    status = IREE_HSA_CALL_TO_STATUS(
        symbols,
        hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                           &queue_size),
        "hsa_agent_get_info(QUEUE_MAX_SIZE)");
    if (iree_status_is_ok(status)) {
      // Use a reasonable default queue size.
      if (queue_size > 4096) queue_size = 4096;
      status = IREE_HSA_CALL_TO_STATUS(
          symbols,
          hsa_queue_create(gpu_agent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL,
                           NULL, UINT32_MAX, UINT32_MAX,
                           &device->device_info.queue),
          "hsa_queue_create");
    }
  }

  // Create completion signal.
  if (iree_status_is_ok(status)) {
    status = IREE_HSA_CALL_TO_STATUS(
        symbols,
        hsa_signal_create(1, 0, NULL, &device->device_info.completion_signal),
        "hsa_signal_create");
  }

  // Create device allocator.
  if (iree_status_is_ok(status)) {
    iree_hal_hsa_device_topology_t topology = {
        .count = 1,
        .devices = &device->device_info,
    };
    status = iree_hal_hsa_allocator_create((iree_hal_device_t*)device, symbols,
                                           topology, host_allocator,
                                           &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_hsa_dynamic_symbols_t* iree_hal_hsa_device_dynamic_symbols(
    iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return device->hsa_symbols;
}

static void iree_hal_hsa_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release device allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Wait for any pending work to complete before destroying resources.
  // This ensures the queue is idle before destruction.
  if (device->device_info.queue && device->device_info.completion_signal.handle) {
    // Wait for the completion signal to reach its expected value.
    // A short timeout helps avoid hanging if something went wrong.
    device->hsa_symbols->hsa_signal_wait_scacquire(
        device->device_info.completion_signal,
        HSA_SIGNAL_CONDITION_EQ, 1,
        /*timeout_hint=*/1000000000ull,  // 1 second timeout
        HSA_WAIT_STATE_BLOCKED);
  }

  // Destroy completion signal.
  if (device->device_info.completion_signal.handle) {
    IREE_HSA_IGNORE_ERROR(
        device->hsa_symbols,
        hsa_signal_destroy(device->device_info.completion_signal));
  }

  // Destroy queue.
  if (device->device_info.queue) {
    IREE_HSA_IGNORE_ERROR(device->hsa_symbols,
                          hsa_queue_destroy(device->device_info.queue));
  }

  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Deinitialize semaphore state.
  iree_hal_hsa_semaphore_state_deinitialize(&device->semaphore_state);

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
  // HSA backend does not support channels yet.
}

static iree_status_t iree_hal_hsa_device_trim(iree_hal_device_t* base_device) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_hsa_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value = iree_string_view_match_pattern(device->identifier, key) ? 1
                                                                         : 0;
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
                          "HSA collective channels not yet implemented");
}

static iree_status_t iree_hal_hsa_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);

  if (iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT) &&
      iree_all_bits_set(mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION) &&
      device->params.allow_inline_execution) {
    // Use stream command buffer for inline execution.
    return iree_hal_hsa_stream_command_buffer_create(
        device->device_allocator, device->hsa_symbols,
        device->device_info.tracing_context, mode, command_categories,
        queue_affinity, binding_capacity, &device->device_info,
        &device->block_pool, device->host_allocator, out_command_buffer);
  }

  // Default to deferred command buffer.
  return iree_hal_deferred_command_buffer_create(
      device->device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, &device->block_pool, device->host_allocator,
      out_command_buffer);
}

static iree_status_t iree_hal_hsa_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "events not yet implemented");
}

static iree_status_t iree_hal_hsa_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_nop_executable_cache_create(
      identifier, device->hsa_symbols, device->host_allocator,
      out_executable_cache);
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
      iree_hal_hsa_device_allocator(base_device), queue_affinity, access, handle,
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_hsa_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_semaphore_create(&device->semaphore_state, initial_value,
                                       device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_hsa_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_hsa_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue alloca not yet implemented");
}

static iree_status_t iree_hal_hsa_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue dealloca not yet implemented");
}

static iree_status_t iree_hal_hsa_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue read not yet implemented");
}

static iree_status_t iree_hal_hsa_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue write not yet implemented");
}

static iree_status_t iree_hal_hsa_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);

  // Wait for the wait semaphores before executing.
  IREE_RETURN_IF_ERROR(iree_hal_hsa_semaphore_multi_wait(
      &device->semaphore_state, IREE_HAL_WAIT_MODE_ALL, wait_semaphore_list,
      iree_infinite_timeout(), /*flags=*/0));

  iree_status_t status = iree_ok_status();

  if (command_buffer != NULL) {
    if (iree_hal_hsa_stream_command_buffer_isa(command_buffer)) {
      // Stream command buffer - already executed inline.
    } else if (iree_hal_deferred_command_buffer_isa(command_buffer)) {
      // Create a stream command buffer and replay the deferred commands.
      iree_hal_command_buffer_t* stream_command_buffer = NULL;
      status = iree_hal_hsa_stream_command_buffer_create(
          device->device_allocator, device->hsa_symbols,
          device->device_info.tracing_context,
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
          iree_hal_command_buffer_allowed_categories(command_buffer),
          queue_affinity, /*binding_capacity=*/0, &device->device_info,
          &device->block_pool, device->host_allocator, &stream_command_buffer);

      if (iree_status_is_ok(status)) {
        status = iree_hal_deferred_command_buffer_apply(
            command_buffer, stream_command_buffer, binding_table);
      }

      iree_hal_command_buffer_release(stream_command_buffer);
    }
  }

  // Signal the signal semaphores after execution (even if there's no command
  // buffer or execution failed).
  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_semaphore_multi_signal(&device->semaphore_state,
                                                  signal_semaphore_list);
  }

  return status;
}

static iree_status_t iree_hal_hsa_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Nothing to flush in our implementation.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags) {
  iree_hal_hsa_device_t* device = iree_hal_hsa_device_cast(base_device);
  return iree_hal_hsa_semaphore_multi_wait(&device->semaphore_state, wait_mode,
                                           semaphore_list, timeout, flags);
}

static iree_status_t iree_hal_hsa_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Profiling not yet implemented.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_profiling_flush(
    iree_hal_device_t* base_device) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_device_profiling_end(
    iree_hal_device_t* base_device) {
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
    .create_event = iree_hal_hsa_device_create_event,
    .create_executable_cache = iree_hal_hsa_device_create_executable_cache,
    .import_file = iree_hal_hsa_device_import_file,
    .create_semaphore = iree_hal_hsa_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_hsa_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_hsa_device_queue_alloca,
    .queue_dealloca = iree_hal_hsa_device_queue_dealloca,
    .queue_fill = iree_hal_device_queue_emulated_fill,
    .queue_update = iree_hal_device_queue_emulated_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_hsa_device_queue_read,
    .queue_write = iree_hal_hsa_device_queue_write,
    .queue_host_call = iree_hal_device_queue_emulated_host_call,
    .queue_dispatch = iree_hal_device_queue_emulated_dispatch,
    .queue_execute = iree_hal_hsa_device_queue_execute,
    .queue_flush = iree_hal_hsa_device_queue_flush,
    .wait_semaphores = iree_hal_hsa_device_wait_semaphores,
    .profiling_begin = iree_hal_hsa_device_profiling_begin,
    .profiling_flush = iree_hal_hsa_device_profiling_flush,
    .profiling_end = iree_hal_hsa_device_profiling_end,
};
