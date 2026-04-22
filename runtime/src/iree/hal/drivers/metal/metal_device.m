// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/metal_device.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/api.h"
#include "iree/hal/drivers/metal/builtin_executables.h"
#include "iree/hal/drivers/metal/direct_allocator.h"
#include "iree/hal/drivers/metal/direct_command_buffer.h"
#include "iree/hal/drivers/metal/nop_executable_cache.h"
#include "iree/hal/drivers/metal/shared_event.h"
#include "iree/hal/drivers/metal/staging_buffer.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_metal_device_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command buffers can
  // contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Per-queue staging buffer for parameter uploads.
  iree_hal_metal_staging_buffer_t staging_buffer;

  iree_hal_metal_device_params_t params;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Proactor pool retained from create_params; provides async I/O proactors.
  iree_async_proactor_pool_t* proactor_pool;
  // Proactor borrowed from the pool for this device's async operations.
  iree_async_proactor_t* proactor;

  // Shared frontier tracker for cross-device causal ordering. Retained after
  // topology assignment and released during device destruction.
  iree_async_frontier_tracker_t* frontier_tracker;

  // This device's axis and monotonic epoch counter for frontier tracking.
  // Metal submits through [commandBuffer commit] — advance() is called at
  // submit time because the Metal command queue is FIFO-ordered.
  iree_async_axis_t axis;
  iree_atomic_int64_t epoch;

  id<MTLDevice> device;
  // We only expose one single command queue for now. This simplifies synchronization.
  // We can relax this to support multiple queues when needed later.
  id<MTLCommandQueue> queue;
  // A command buffer descriptor used for creating command buffers to signal/wait MTLSharedEvent.
  MTLCommandBufferDescriptor* command_buffer_descriptor;

  iree_hal_metal_command_buffer_resource_reference_mode_t command_buffer_resource_reference_mode;

  // For polyfilling fill/copy/update buffers that are not directly supported by Metal APIs.
  iree_hal_metal_builtin_executable_t* builtin_executable;

  // A dispatch queue and associated event listener for running Objective-C blocks to signal
  // semaphores and wake up threads.
  dispatch_queue_t semaphore_notification_queue;
  MTLSharedEventListener* event_listener;

  // Retained Metal capture manager while an external capture range is active.
  MTLCaptureManager* capture_manager;

  iree_hal_device_topology_info_t topology_info;
} iree_hal_metal_device_t;

static const iree_hal_device_vtable_t iree_hal_metal_device_vtable;

static iree_hal_metal_device_t* iree_hal_metal_device_cast(iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_device_vtable);
  return (iree_hal_metal_device_t*)base_value;
}

static const iree_hal_metal_device_t* iree_hal_metal_device_const_cast(
    const iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_device_vtable);
  return (const iree_hal_metal_device_t*)base_value;
}

// Advances the frontier tracker epoch for the device.
// Called at submit time ([commandBuffer commit]) because the Metal command
// queue is FIFO-ordered: submission order = causal ordering.
static void iree_hal_metal_device_advance_frontier(iree_hal_metal_device_t* device) {
  uint64_t epoch =
      (uint64_t)iree_atomic_fetch_add(&device->epoch, 1, iree_memory_order_acq_rel) + 1;
  iree_async_frontier_tracker_advance(device->frontier_tracker, device->axis, epoch);
}

void iree_hal_metal_device_params_initialize(iree_hal_metal_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->queue_uniform_buffer_size = IREE_HAL_METAL_STAGING_BUFFER_DEFAULT_CAPACITY;
  out_params->command_dispatch_type = IREE_HAL_METAL_COMMAND_DISPATCH_TYPE_CONCURRENT;
  out_params->command_buffer_resource_reference_mode =
      IREE_HAL_METAL_COMMAND_BUFFER_RESOURCE_REFERENCE_MODE_UNRETAINED;
  out_params->resource_hazard_tracking_mode =
      IREE_HAL_METAL_RESOURCE_HAZARD_TRACKING_MODE_UNTRACKED;
}

const iree_hal_metal_device_params_t* iree_hal_metal_device_params(
    const iree_hal_device_t* base_device) {
  const iree_hal_metal_device_t* device = iree_hal_metal_device_const_cast(base_device);
  return &device->params;
}

static iree_status_t iree_hal_metal_device_create_internal(
    iree_string_view_t identifier, const iree_hal_metal_device_params_t* params,
    id<MTLDevice> metal_device, const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_metal_device_t* device = NULL;

  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_metal_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + iree_sizeof_struct(*device));
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator, &device->block_pool);
  device->params = *params;
  device->host_allocator = host_allocator;

  // Retain the proactor pool and acquire a proactor for this device.
  device->proactor_pool = create_params->proactor_pool;
  iree_async_proactor_pool_retain(device->proactor_pool);
  iree_atomic_store(&device->epoch, 0, iree_memory_order_relaxed);
  iree_status_t status = iree_async_proactor_pool_get(device->proactor_pool, 0, &device->proactor);
  if (!iree_status_is_ok(status)) {
    iree_hal_device_release((iree_hal_device_t*)device);
    return status;
  }

  device->device = [metal_device retain];                            // +1
  id<MTLCommandQueue> metal_queue = [metal_device newCommandQueue];  // +1
  device->queue = metal_queue;

  MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];  // +1
  descriptor.retainedReferences = params->command_buffer_resource_reference_mode ==
                                  IREE_HAL_METAL_COMMAND_BUFFER_RESOURCE_REFERENCE_MODE_RETAINED;
  descriptor.errorOptions = MTLCommandBufferErrorOptionNone;
  device->command_buffer_descriptor = descriptor;

  device->command_buffer_resource_reference_mode = params->command_buffer_resource_reference_mode;
  dispatch_queue_attr_t queue_attr = dispatch_queue_attr_make_with_qos_class(
      DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, /*relative_priority=*/0);
  device->semaphore_notification_queue = dispatch_queue_create("dev.iree.queue.metal", queue_attr);
  device->event_listener = [[MTLSharedEventListener alloc]
      initWithDispatchQueue:device->semaphore_notification_queue];  // +1
  device->capture_manager = NULL;

  status = iree_hal_metal_allocator_create((iree_hal_device_t*)device, metal_device,
#if defined(IREE_PLATFORM_MACOS)
                                           metal_queue,
#endif  // IREE_PLATFORM_MACOS
                                           params->resource_hazard_tracking_mode, host_allocator,
                                           &device->device_allocator);

  if (iree_status_is_ok(status)) {
    status = iree_hal_metal_builtin_executable_create(metal_device, host_allocator,
                                                      &device->builtin_executable);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_metal_staging_buffer_initialize(
        metal_device, params->queue_uniform_buffer_size, &device->staging_buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_metal_device_create(iree_string_view_t identifier,
                                           const iree_hal_metal_device_params_t* params,
                                           id<MTLDevice> device,
                                           const iree_hal_device_create_params_t* create_params,
                                           iree_allocator_t host_allocator,
                                           iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_metal_device_create_internal(
      identifier, params, device, create_params, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_device_clear_topology_info(iree_hal_metal_device_t* device) {
  if (device->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(device->frontier_tracker, device->axis,
                                            iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(device->frontier_tracker);
    device->frontier_tracker = NULL;
    device->axis = 0;
  }
  memset(&device->topology_info, 0, sizeof(device->topology_info));
}

static void iree_hal_metal_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  [device->event_listener release];  // -1
  dispatch_release(device->semaphore_notification_queue);

  iree_hal_metal_builtin_executable_destroy(device->builtin_executable);

  iree_hal_metal_device_clear_topology_info(device);

  iree_hal_allocator_release(device->device_allocator);
  [device->command_buffer_descriptor release];  // -1
  [device->queue release];                      // -1
  [device->device release];                     // -1

  iree_hal_metal_staging_buffer_deinitialize(&device->staging_buffer);
  iree_arena_block_pool_deinitialize(&device->block_pool);

  iree_async_proactor_pool_release(device->proactor_pool);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_metal_device_id(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_metal_device_host_allocator(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_metal_device_allocator(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_metal_replace_device_allocator(iree_hal_device_t* base_device,
                                                    iree_hal_allocator_t* new_allocator) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_metal_replace_channel_provider(iree_hal_device_t* base_device,
                                                    iree_hal_channel_provider_t* new_provider) {
  (void)base_device;
  (void)new_provider;
}

static iree_status_t iree_hal_metal_device_trim(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_metal_device_query_i64(iree_hal_device_t* base_device,
                                                     iree_string_view_t category,
                                                     iree_string_view_t key, int64_t* out_value) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value = iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, iree_make_cstring_view("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, iree_make_cstring_view("metal-msl-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "unknown device configuration key value '%.*s :: %.*s'",
                          (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_metal_device_query_capabilities(
    iree_hal_device_t* base_device, iree_hal_device_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t* iree_hal_metal_device_topology_info(
    iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_metal_device_refine_topology_edge(iree_hal_device_t* src_device,
                                                                iree_hal_device_t* dst_device,
                                                                iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_assign_topology_info(
    iree_hal_device_t* base_device, const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  if (!topology_info) {
    iree_hal_metal_device_clear_topology_info(device);
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t* frontier_tracker = topology_info->frontier.tracker;
  iree_async_axis_t axis = topology_info->frontier.base_axis;
  IREE_RETURN_IF_ERROR(
      iree_async_frontier_tracker_register_axis(frontier_tracker, axis, /*semaphore=*/NULL));
  device->topology_info = *topology_info;
  device->frontier_tracker = frontier_tracker;
  device->axis = axis;
  iree_async_frontier_tracker_retain(device->frontier_tracker);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_create_channel(iree_hal_device_t* base_device,
                                                          iree_hal_queue_affinity_t queue_affinity,
                                                          iree_hal_channel_params_t params,
                                                          iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "collectives not yet supported");
}

static iree_status_t iree_hal_metal_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t binding_capacity, iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);

  // Native Metal command buffers are not reusable so we emulate by recording into our own reusable
  // instance. This will be replayed against a Metal command buffer upon submission.
  //
  // TODO(indirect-cmd): natively support indirect command buffers in Metal via
  // MTLIndirectCommandBuffer. We could switch to exclusively using that for all modes to keep the
  // number of code paths down. MTLIndirectCommandBuffer is both reusable and has what we require
  // for argument buffer updates to pass in binding tables.
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT) || binding_capacity > 0) {
    return iree_hal_deferred_command_buffer_create(
        device->device_allocator, mode, command_categories, queue_affinity, binding_capacity,
        &device->block_pool, device->host_allocator, out_command_buffer);
  }

  return iree_hal_metal_direct_command_buffer_create(
      base_device, mode, command_categories, binding_capacity,
      device->command_buffer_resource_reference_mode, device->queue, &device->block_pool,
      &device->staging_buffer, device->builtin_executable, device->host_allocator,
      out_command_buffer);
}

static iree_status_t iree_hal_metal_device_create_event(iree_hal_device_t* base_device,
                                                        iree_hal_queue_affinity_t queue_affinity,
                                                        iree_hal_event_flags_t flags,
                                                        iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_metal_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_nop_executable_cache_create(device->device, identifier,
                                                    device->host_allocator, out_executable_cache);
}

static iree_status_t iree_hal_metal_device_import_file(iree_hal_device_t* base_device,
                                                       iree_hal_queue_affinity_t queue_affinity,
                                                       iree_hal_memory_access_t access,
                                                       iree_io_file_handle_t* handle,
                                                       iree_hal_external_file_flags_t flags,
                                                       iree_hal_file_t** out_file) {
  return iree_hal_file_from_handle(
      iree_hal_device_allocator(base_device), queue_affinity, access, handle,
      /*proactor=*/NULL, iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_metal_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_shared_event_create(device->proactor, device->device, initial_value,
                                            device->event_listener, device->host_allocator,
                                            out_semaphore);
}

static iree_hal_semaphore_compatibility_t iree_hal_metal_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  if (iree_hal_metal_shared_event_isa(semaphore)) {
    // Fast-path for semaphores related to this device.
    // TODO(benvanik): ensure the creating devices are compatible in cases where
    // multiple devices are used.
    return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
  }
  // TODO(benvanik): semaphore APIs for querying allowed export formats. We
  // can check device caps to see what external semaphore types are supported.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_metal_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "Metal queue pool backend not implemented");
}

static iree_status_t iree_hal_metal_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_pool_t* pool,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_alloca_flags_t flags, iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  if (IREE_UNLIKELY(pool != NULL)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Metal custom queue alloca pools not implemented");
  }

  iree_status_t status = iree_hal_semaphore_list_wait(wait_semaphore_list, iree_infinite_timeout(),
                                                      IREE_ASYNC_WAIT_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device), params,
                                                allocation_size, out_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list,
                                            /*frontier=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_metal_device_advance_frontier(device);
  } else {
    iree_hal_semaphore_list_fail(signal_semaphore_list, iree_status_clone(status));
  }
  return status;
}

static iree_status_t iree_hal_metal_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* buffer,
    iree_hal_dealloca_flags_t flags) {
  // TODO(benvanik): queue-ordered allocations.
  return iree_hal_device_queue_barrier(base_device, queue_affinity, wait_semaphore_list,
                                       signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_metal_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_file_t* source_file,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_file_transfer_options_t options = {
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  return iree_hal_device_queue_read_streaming(base_device, queue_affinity, wait_semaphore_list,
                                              signal_semaphore_list, source_file, source_offset,
                                              target_buffer, target_offset, length, flags, options);
}

static iree_status_t iree_hal_metal_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_file_transfer_options_t options = {
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  return iree_hal_device_queue_write_streaming(base_device, queue_affinity, wait_semaphore_list,
                                               signal_semaphore_list, source_buffer, source_offset,
                                               target_file, target_offset, length, flags, options);
}

static iree_status_t iree_hal_metal_replay_command_buffer(
    iree_hal_metal_device_t* device, iree_hal_command_buffer_t* deferred_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_command_buffer_t** out_direct_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create the transient command buffer. Note that it is one-shot and has no indirect bindings as
  // we will be replaying it once with all the bindings resolved.
  iree_hal_command_buffer_t* direct_command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_direct_command_buffer_create(
              (iree_hal_device_t*)device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
              iree_hal_command_buffer_allowed_categories(deferred_command_buffer),
              /*binding_capacity=*/0, device->command_buffer_resource_reference_mode, device->queue,
              &device->block_pool, &device->staging_buffer, device->builtin_executable,
              device->host_allocator, &direct_command_buffer));

  // Attempt to replay all commands against the transient command buffer. Note that this will fail
  // if any binding does not meet the requirements - having succeeded when recording initially is
  // not a guarantee that this will succeed.
  iree_status_t status = iree_hal_deferred_command_buffer_apply(
      deferred_command_buffer, direct_command_buffer, binding_table);

  if (iree_status_is_ok(status)) {
    *out_direct_command_buffer = direct_command_buffer;
  } else {
    iree_hal_command_buffer_release(direct_command_buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(&device->block_pool, &resource_set));

  // Put the full semaphore list into a resource set, which retains them--we will need to access
  // them until the command buffer completes.
  iree_status_t status = iree_hal_resource_set_insert(resource_set, wait_semaphore_list.count,
                                                      wait_semaphore_list.semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(resource_set, signal_semaphore_list.count,
                                          signal_semaphore_list.semaphores);
  }

  // Translate deferred command buffers into real Metal command buffers.
  // We do this prior to beginning execution so that if we fail we don't leave the system in an
  // inconsistent state.
  iree_hal_command_buffer_t* direct_command_buffer = NULL;
  if (iree_status_is_ok(status) && command_buffer) {
    if (iree_hal_deferred_command_buffer_isa(command_buffer)) {
      // Create a temporary command buffer and replay the deferred command buffer with the
      // binding table provided. Note that any resources used will be retained by the command
      // buffer so we only need to retain the command buffer itself instead of the binding
      // tables provided.
      @autoreleasepool {
        status = iree_hal_metal_replay_command_buffer(device, command_buffer, binding_table,
                                                      &direct_command_buffer);
      }
    } else {
      // Retain the command buffer until the submission has completed.
      iree_hal_command_buffer_retain(command_buffer);
      direct_command_buffer = command_buffer;
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_resource_set_insert(resource_set, 1, &direct_command_buffer);
    }
    iree_hal_command_buffer_release(direct_command_buffer);  // retained in resource set
  }

  // Clone the signal semaphore list onto the heap for the completion handler.
  // The caller's arrays may be stack-allocated and will not survive until GPU
  // completion.
  iree_hal_semaphore_list_t signal_list_clone = iree_hal_semaphore_list_empty();
  iree_allocator_t host_allocator = device->host_allocator;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_semaphore_list_clone(&signal_semaphore_list, host_allocator, &signal_list_clone);
  }

  if (iree_status_is_ok(status)) {
    @autoreleasepool {
      // First create a new command buffer and encode wait commands for all wait semaphores.
      if (wait_semaphore_list.count > 0) {
        id<MTLCommandBuffer> wait_command_buffer = [device->queue
            commandBufferWithDescriptor:device->command_buffer_descriptor];  // autoreleased
        for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
          id<MTLSharedEvent> handle =
              iree_hal_metal_shared_event_handle(wait_semaphore_list.semaphores[i]);
          [wait_command_buffer encodeWaitForEvent:handle
                                            value:wait_semaphore_list.payload_values[i]];
        }
        [wait_command_buffer commit];
      }

      // Then commit all recorded compute command buffers, except the last one, which we will patch
      // up with semaphore signaling.
      id<MTLCommandBuffer> signal_command_buffer = nil;
      if (direct_command_buffer) {
        // NOTE: translation happens above such that we always know these are direct command
        // buffers.
        //
        // TODO(indirect-cmd): support indirect command buffers and switch here, or only use
        // indirect command buffers and assume that instead.
        id<MTLCommandBuffer> handle =
            iree_hal_metal_direct_command_buffer_handle(direct_command_buffer);
        signal_command_buffer = handle;
      }
      if (signal_command_buffer == nil) {
        signal_command_buffer = [device->queue
            commandBufferWithDescriptor:device->command_buffer_descriptor];  // autoreleased
      }

      // Encode signal commands on the GPU for all signal semaphores. These set
      // the MTLSharedEvent values on the device side.
      for (iree_host_size_t i = 0; i < signal_list_clone.count; ++i) {
        id<MTLSharedEvent> handle =
            iree_hal_metal_shared_event_handle(signal_list_clone.semaphores[i]);
        [signal_command_buffer encodeSignalEvent:handle value:signal_list_clone.payload_values[i]];
      }

      // Retain the device to keep the block pool alive past the resource set.
      iree_hal_device_retain(base_device);
      [signal_command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        // Advance host-side async semaphore timelines and dispatch waiting
        // timepoints. The GPU-side encodeSignalEvent already set the
        // MTLSharedEvent values; this synchronizes the async layer.
        if (cb.status == MTLCommandBufferStatusCompleted) {
          iree_status_t signal_status = iree_hal_semaphore_list_signal(signal_list_clone,
                                                                       /*frontier=*/NULL);
          if (IREE_UNLIKELY(!iree_status_is_ok(signal_status))) {
            // Each timeline value must be signaled exactly once. Signal failure
            // indicates a structural error — fail all semaphores so waiters get
            // a proper diagnostic.
            iree_hal_semaphore_list_fail(signal_list_clone, signal_status);
          }
        } else {
          // GPU command buffer failed — fail all signal semaphores so waiters
          // get a proper error instead of timing out.
          iree_hal_semaphore_list_fail(
              signal_list_clone,
              iree_make_status(IREE_STATUS_INTERNAL, "Metal command buffer failed (status %d)",
                               (int)cb.status));
        }
        iree_hal_semaphore_list_free(signal_list_clone, host_allocator);
        // Release all retained resources, then the device handle separately
        // to avoid destroying the block pool before the resource set is done.
        iree_hal_resource_set_free(resource_set);
        iree_hal_device_release(base_device);
      }];
      [signal_command_buffer commit];
      iree_hal_metal_device_advance_frontier(device);
    }
  } else {
    // Fail all signal semaphores so downstream waiters see the error instead
    // of hanging indefinitely.
    iree_hal_semaphore_list_fail(signal_semaphore_list, iree_status_clone(status));
    iree_hal_semaphore_list_free(signal_list_clone, host_allocator);
    iree_hal_resource_set_free(resource_set);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_device_queue_flush(iree_hal_device_t* base_device,
                                                       iree_hal_queue_affinity_t queue_affinity) {
  // Nothing to do for now given we immediately release workload to the GPU on queue execute.
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_profiling_begin(
    iree_hal_device_t* base_device, const iree_hal_device_profiling_options_t* options) {
  (void)base_device;
  (void)options;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Metal HAL-native profiling is not implemented");
}

static iree_status_t iree_hal_metal_device_profiling_flush(iree_hal_device_t* base_device) {
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_profiling_end(iree_hal_device_t* base_device) {
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_external_capture_begin(
    iree_hal_device_t* base_device, const iree_hal_device_external_capture_options_t* options) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  if (!iree_string_view_equal(options->provider, IREE_SV("metal"))) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Metal external capture provider '%.*s' is not implemented",
                            (int)options->provider.size, options->provider.data);
  }

  if (device->capture_manager) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "cannot nest Metal external capture");
  }

  MTLCaptureManager* capture_manager = [[MTLCaptureManager sharedCaptureManager] retain];  // +1
  iree_status_t status = iree_ok_status();

  @autoreleasepool {
    NSURL* capture_url = NULL;
    iree_string_view_t file_path = options->file_path;
    if (!iree_string_view_is_empty(file_path)) {
      if (!iree_string_view_ends_with(file_path, IREE_SV(".gputrace"))) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "capture filename must end with .gputrace");
      } else if (![capture_manager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported capture to file (if invoking as command-line "
                                  "binary, make sure there is companion Info.plist under the same "
                                  "directory with 'MetalCaptureEnabled' key being true)");
      } else {
        NSString* ns_string = [[[NSString alloc] initWithBytes:file_path.data
                                                        length:file_path.size
                                                      encoding:NSUTF8StringEncoding] autorelease];
        if (!ns_string) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "capture filename is not UTF-8");
        } else {
          NSString* capture_path = ns_string.stringByStandardizingPath;
          capture_url = [NSURL fileURLWithPath:capture_path isDirectory:false];
        }
      }
    }

    if (iree_status_is_ok(status)) {
      MTLCaptureDescriptor* capture_descriptor = [[[MTLCaptureDescriptor alloc] init] autorelease];
      capture_descriptor.captureObject = device->device;
      if (capture_url) {
        capture_descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        capture_descriptor.outputURL = capture_url;
      } else {
        capture_descriptor.destination = MTLCaptureDestinationDeveloperTools;
      }

      NSError* error = NULL;
      if (![capture_manager startCaptureWithDescriptor:capture_descriptor error:&error]) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "failed to start external capture");
        const char* ns_c_error = error ? [error.localizedDescription
                                             cStringUsingEncoding:[NSString defaultCStringEncoding]]
                                       : "unknown error";  // autoreleased
        status = iree_status_annotate_f(status, "with NSError: %s",
                                        ns_c_error ? ns_c_error : "unknown error");
      }
    }
  }

  if (!iree_status_is_ok(status)) {
    [capture_manager release];  // -1
    return status;
  }
  device->capture_manager = capture_manager;
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_external_capture_end(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  if (!device->capture_manager) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "no Metal external capture is active");
  }
  [device->capture_manager stopCapture];
  [device->capture_manager release];  // -1
  device->capture_manager = NULL;
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_metal_device_vtable = {
    .destroy = iree_hal_metal_device_destroy,
    .id = iree_hal_metal_device_id,
    .host_allocator = iree_hal_metal_device_host_allocator,
    .device_allocator = iree_hal_metal_device_allocator,
    .replace_device_allocator = iree_hal_metal_replace_device_allocator,
    .replace_channel_provider = iree_hal_metal_replace_channel_provider,
    .trim = iree_hal_metal_device_trim,
    .query_i64 = iree_hal_metal_device_query_i64,
    .query_capabilities = iree_hal_metal_device_query_capabilities,
    .topology_info = iree_hal_metal_device_topology_info,
    .refine_topology_edge = iree_hal_metal_device_refine_topology_edge,
    .assign_topology_info = iree_hal_metal_device_assign_topology_info,
    .create_channel = iree_hal_metal_device_create_channel,
    .create_command_buffer = iree_hal_metal_device_create_command_buffer,
    .create_event = iree_hal_metal_device_create_event,
    .create_executable_cache = iree_hal_metal_device_create_executable_cache,
    .import_file = iree_hal_metal_device_import_file,
    .create_semaphore = iree_hal_metal_device_create_semaphore,
    .query_semaphore_compatibility = iree_hal_metal_device_query_semaphore_compatibility,
    .query_queue_pool_backend = iree_hal_metal_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_metal_device_queue_alloca,
    .queue_dealloca = iree_hal_metal_device_queue_dealloca,
    .queue_fill = iree_hal_device_queue_emulated_fill,
    .queue_update = iree_hal_device_queue_emulated_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_metal_device_queue_read,
    .queue_write = iree_hal_metal_device_queue_write,
    .queue_host_call = iree_hal_device_queue_emulated_host_call,
    .queue_dispatch = iree_hal_device_queue_emulated_dispatch,
    .queue_execute = iree_hal_metal_device_queue_execute,
    .queue_flush = iree_hal_metal_device_queue_flush,
    .profiling_begin = iree_hal_metal_device_profiling_begin,
    .profiling_flush = iree_hal_metal_device_profiling_flush,
    .profiling_end = iree_hal_metal_device_profiling_end,
    .external_capture_begin = iree_hal_metal_device_external_capture_begin,
    .external_capture_end = iree_hal_metal_device_external_capture_end,
};
