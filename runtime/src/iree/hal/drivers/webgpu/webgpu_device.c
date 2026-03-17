// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_device.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/webgpu/webgpu_allocator.h"
#include "iree/hal/drivers/webgpu/webgpu_builtins.h"
#include "iree/hal/drivers/webgpu/webgpu_command_buffer.h"
#include "iree/hal/drivers/webgpu/webgpu_executable_cache.h"
#include "iree/hal/drivers/webgpu/webgpu_fd_file.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"
#include "iree/hal/drivers/webgpu/webgpu_queue.h"
#include "iree/hal/drivers/webgpu/webgpu_semaphore.h"
#include "iree/hal/utils/memory_file.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Flags controlling ownership and behavior.
  iree_hal_webgpu_device_flags_t flags;

  // Bridge handle for the GPUDevice.
  iree_hal_webgpu_handle_t device_handle;

  // Built-in WGSL compute pipelines for buffer operations (fill, copy) that
  // WebGPU's command encoder does not natively support with arbitrary
  // alignment.
  iree_hal_webgpu_builtins_t builtins;

  // True if the execution context supports blocking waits (Atomics.wait).
  // On the browser main thread this is false — Atomics.wait throws TypeError.
  // On Web Workers with cross-origin isolation this is true.
  // Native (Dawn) always supports blocking.
  bool can_block;

  // Proactor pool for async I/O. Retained for the lifetime of the device to
  // ensure proactor threads outlive all device resources (semaphores, etc.).
  iree_async_proactor_pool_t* proactor_pool;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Topology information if this device is part of a multi-device topology.
  iree_hal_device_topology_info_t topology_info;

  // Today: 1 queue. Future: queue[N] with queue selection by affinity.
  iree_hal_webgpu_queue_t queue;

  // + trailing identifier string storage
} iree_hal_webgpu_device_t;

static const iree_hal_device_vtable_t iree_hal_webgpu_device_vtable;

static iree_hal_webgpu_device_t* iree_hal_webgpu_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_device_vtable);
  return (iree_hal_webgpu_device_t*)base_value;
}

iree_status_t iree_hal_webgpu_device_create(
    iree_string_view_t identifier, iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_device_flags_t flags,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  iree_hal_webgpu_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  iree_hal_resource_initialize(&iree_hal_webgpu_device_vtable,
                               &device->resource);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + total_size - identifier.size);
  device->host_allocator = host_allocator;
  device->flags = flags;

  // Store bridge handle. WebGPU has exactly one queue per device.
  device->device_handle = device_handle;
  device->can_block = iree_hal_webgpu_import_can_block() != 0;

  // Retain the proactor pool and acquire a proactor for queue initialization.
  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);

  iree_async_frontier_tracker_t* frontier_tracker =
      create_params->frontier.tracker;
  iree_async_axis_t axis = create_params->frontier.base_axis;
  if (frontier_tracker) {
    iree_async_axis_table_add(&frontier_tracker->axis_table, axis,
                              /*semaphore=*/NULL);
  }

  iree_async_proactor_t* proactor = NULL;
  iree_status_t status =
      iree_async_proactor_pool_get(device->proactor_pool, 0, &proactor);

  // Create the builtin compute pipelines for fill/copy operations.
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_webgpu_builtins_initialize(device_handle, &device->builtins);
  }

  // Initialize the queue (owns block pool, scratch builder, epoch tracking).
  if (iree_status_is_ok(status)) {
    iree_hal_webgpu_handle_t queue_handle =
        iree_hal_webgpu_import_device_get_queue(device_handle);
    status = iree_hal_webgpu_queue_initialize(
        device_handle, queue_handle, &device->builtins, proactor,
        frontier_tracker, axis, host_allocator, &device->queue);
  }

  // Create the device allocator.
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_allocator_create(device_handle, host_allocator,
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

IREE_API_EXPORT iree_status_t iree_hal_webgpu_device_wrap(
    iree_string_view_t identifier, iree_hal_webgpu_handle_t device_handle,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  // Create an inline-mode proactor pool (single node, no threads).
  iree_async_proactor_pool_t* proactor_pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_pool_create(
              /*node_count=*/1, /*node_ids=*/NULL,
              iree_async_proactor_pool_options_default(), host_allocator,
              &proactor_pool));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  // Caller retains ownership of the device handle — no OWNS_DEVICE_HANDLE.
  iree_status_t status = iree_hal_webgpu_device_create(
      identifier, device_handle, IREE_HAL_WEBGPU_DEVICE_FLAG_NONE,
      &create_params, host_allocator, out_device);

  // The device retains the pool — release our reference.
  iree_async_proactor_pool_release(proactor_pool);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(device->device_allocator);
  iree_hal_webgpu_queue_deinitialize(&device->queue);
  iree_hal_webgpu_builtins_deinitialize(&device->builtins);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_async_proactor_pool_release(device->proactor_pool);

  // Release the GPUDevice bridge handle only if we own it. When created via
  // iree_hal_webgpu_device_wrap(), the caller retains ownership and the handle
  // must outlive the HAL device.
  if (iree_all_bits_set(device->flags,
                        IREE_HAL_WEBGPU_DEVICE_FLAG_OWNS_DEVICE_HANDLE)) {
    iree_hal_webgpu_import_device_destroy(device->device_handle);
  }

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_webgpu_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_webgpu_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_webgpu_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_webgpu_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_webgpu_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_webgpu_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("webgpu-wgsl-v0")) ? 1 : 0;
    return iree_ok_status();
  }

  // WebGPU has a single queue — concurrency is always 1.
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
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_webgpu_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  // WebGPU provides no UUIDs, no external semaphore/buffer handle types, and
  // no NUMA information. The capabilities struct remains zeroed.
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_webgpu_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_webgpu_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  // WebGPU devices are isolated — no direct peer-to-peer access.
  (void)src_device;
  (void)dst_device;
  (void)edge;
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  device->topology_info = *topology_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  // WebGPU has no collective communication primitives.
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support collective channels");
}

static iree_status_t iree_hal_webgpu_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_command_buffer_create(
      device->device_handle, device->queue.queue_handle, &device->builtins,
      &device->queue.block_pool, iree_hal_device_allocator(base_device), mode,
      command_categories, queue_affinity, binding_capacity,
      device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_webgpu_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  // WebGPU commands within a single queue execute in submission order with
  // implicit memory visibility. Events (intra-command-buffer barriers) are
  // not needed — the command encoder provides implicit ordering.
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support events; commands within a "
                          "queue have implicit ordering");
}

static iree_status_t iree_hal_webgpu_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_executable_cache_create(
      device->device_handle, identifier, device->host_allocator,
      out_executable_cache);
}

static iree_status_t iree_hal_webgpu_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_io_file_handle_primitive_t primitive =
      iree_io_file_handle_primitive(handle);
  switch (primitive.type) {
    case IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION:
      // Use generic memory_file — storage_buffer() returns a HOST_LOCAL heap
      // buffer via the heap_buffer_wrap fallback (since WebGPU cannot import
      // host allocations as GPU buffers).
      return iree_hal_memory_file_wrap(
          iree_hal_device_allocator(base_device), queue_affinity, access,
          handle, iree_hal_device_host_allocator(base_device), out_file);
    case IREE_IO_FILE_HANDLE_TYPE_FD: {
      // Use WebGPU FD file — the fd is a JS file object table index, not a
      // POSIX fd. The standard fd_file uses pread/pwrite (unavailable on wasm).
      uint64_t length =
          iree_hal_webgpu_import_file_get_length((uint32_t)primitive.value.fd);
      return iree_hal_webgpu_fd_file_from_handle(
          access, handle, length, iree_hal_device_host_allocator(base_device),
          out_file);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported file handle type %d",
                              (int)primitive.type);
  }
}

static iree_status_t iree_hal_webgpu_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_semaphore_create(
      device->queue.proactor, queue_affinity, initial_value, flags,
      device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_webgpu_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  // WebGPU semaphores are software timeline semaphores. Device wait/signal
  // always works (the proactor advances the timeline on onSubmittedWorkDone
  // completion, and submission checks timeline values before queue.submit).
  // Host signal always works (just a CAS on the timeline value).
  //
  // Host WAIT requires blocking (Atomics.wait), which is only available on
  // Web Workers — not on the browser main thread where Atomics.wait throws
  // TypeError. The can_block flag is queried once at device creation via the
  // bridge and gates HOST_WAIT here so callers never attempt a blocking wait
  // in a context that can't support it.
  iree_hal_semaphore_compatibility_t compatibility =
      IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_ONLY |
      IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL;
  if (device->can_block) {
    compatibility |= IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT;
  }
  return compatibility;
}

//===----------------------------------------------------------------------===//
// Queue operation vtable wrappers
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_webgpu_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_alloca(
      &device->queue, device->device_allocator, wait_semaphore_list,
      signal_semaphore_list, pool, params, allocation_size, flags, out_buffer);
}

static iree_status_t iree_hal_webgpu_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_dealloca(&device->queue, wait_semaphore_list,
                                        signal_semaphore_list, buffer, flags);
}

static iree_status_t iree_hal_webgpu_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_fill(
      &device->queue, wait_semaphore_list, signal_semaphore_list, target_buffer,
      target_offset, length, pattern, pattern_length, flags);
}

static iree_status_t iree_hal_webgpu_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_update(
      &device->queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_webgpu_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_copy(
      &device->queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_webgpu_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_read(
      &device->queue, wait_semaphore_list, signal_semaphore_list, source_file,
      source_offset, target_buffer, target_offset, length, flags);
}

static iree_status_t iree_hal_webgpu_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_write(
      &device->queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_file, target_offset, length, flags);
}

static iree_status_t iree_hal_webgpu_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_host_call(
      &device->queue, base_device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, call, args, flags);
}

static iree_status_t iree_hal_webgpu_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_dispatch(
      &device->queue, wait_semaphore_list, signal_semaphore_list, executable,
      export_ordinal, config, constants, bindings, flags);
}

static iree_status_t iree_hal_webgpu_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_execute(&device->queue, wait_semaphore_list,
                                       signal_semaphore_list, command_buffer,
                                       binding_table, flags);
}

static iree_status_t iree_hal_webgpu_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_queue_flush(&device->queue);
}

//===----------------------------------------------------------------------===//
// Profiling (no-ops for WebGPU)
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_webgpu_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // WebGPU has no user-accessible profiling API.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_device_profiling_flush(
    iree_hal_device_t* base_device) {
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_device_profiling_end(
    iree_hal_device_t* base_device) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_webgpu_device_vtable = {
    .destroy = iree_hal_webgpu_device_destroy,
    .id = iree_hal_webgpu_device_id,
    .host_allocator = iree_hal_webgpu_device_host_allocator,
    .device_allocator = iree_hal_webgpu_device_allocator,
    .replace_device_allocator = iree_hal_webgpu_replace_device_allocator,
    .replace_channel_provider = iree_hal_webgpu_replace_channel_provider,
    .trim = iree_hal_webgpu_device_trim,
    .query_i64 = iree_hal_webgpu_device_query_i64,
    .query_capabilities = iree_hal_webgpu_device_query_capabilities,
    .topology_info = iree_hal_webgpu_device_topology_info,
    .refine_topology_edge = iree_hal_webgpu_device_refine_topology_edge,
    .assign_topology_info = iree_hal_webgpu_device_assign_topology_info,
    .create_channel = iree_hal_webgpu_device_create_channel,
    .create_command_buffer = iree_hal_webgpu_device_create_command_buffer,
    .create_event = iree_hal_webgpu_device_create_event,
    .create_executable_cache = iree_hal_webgpu_device_create_executable_cache,
    .import_file = iree_hal_webgpu_device_import_file,
    .create_semaphore = iree_hal_webgpu_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_webgpu_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_webgpu_device_queue_alloca,
    .queue_dealloca = iree_hal_webgpu_device_queue_dealloca,
    .queue_fill = iree_hal_webgpu_device_queue_fill,
    .queue_update = iree_hal_webgpu_device_queue_update,
    .queue_copy = iree_hal_webgpu_device_queue_copy,
    .queue_read = iree_hal_webgpu_device_queue_read,
    .queue_write = iree_hal_webgpu_device_queue_write,
    .queue_host_call = iree_hal_webgpu_device_queue_host_call,
    .queue_dispatch = iree_hal_webgpu_device_queue_dispatch,
    .queue_execute = iree_hal_webgpu_device_queue_execute,
    .queue_flush = iree_hal_webgpu_device_queue_flush,
    .profiling_begin = iree_hal_webgpu_device_profiling_begin,
    .profiling_flush = iree_hal_webgpu_device_profiling_flush,
    .profiling_end = iree_hal_webgpu_device_profiling_end,
};
