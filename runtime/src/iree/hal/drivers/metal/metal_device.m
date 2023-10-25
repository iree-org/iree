// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/metal_device.h"

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/api.h"
#include "iree/hal/drivers/metal/builtin_executables.h"
#include "iree/hal/drivers/metal/direct_allocator.h"
#include "iree/hal/drivers/metal/direct_command_buffer.h"
#include "iree/hal/drivers/metal/nop_executable_cache.h"
#include "iree/hal/drivers/metal/pipeline_layout.h"
#include "iree/hal/drivers/metal/shared_event.h"
#include "iree/hal/drivers/metal/staging_buffer.h"
#include "iree/hal/utils/buffer_transfer.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"
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

  MTLCaptureManager* capture_manager;
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
    id<MTLDevice> metal_device, iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_metal_device_t* device = NULL;

  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_metal_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + iree_sizeof_struct(*device));
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator, &device->block_pool);
  device->params = *params;
  device->host_allocator = host_allocator;

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

  iree_status_t status = iree_hal_metal_allocator_create(metal_device,
#if defined(IREE_PLATFORM_MACOS)
                                                         metal_queue,
#endif  // IREE_PLATFORM_MACOS
                                                         params->resource_hazard_tracking_mode,
                                                         host_allocator, &device->device_allocator);

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
                                           id<MTLDevice> device, iree_allocator_t host_allocator,
                                           iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_hal_metal_device_create_internal(identifier, params, device, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  [device->event_listener release];  // -1
  dispatch_release(device->semaphore_notification_queue);

  iree_hal_metal_builtin_executable_destroy(device->builtin_executable);

  iree_hal_allocator_release(device->device_allocator);
  [device->command_buffer_descriptor release];  // -1
  [device->queue release];                      // -1
  [device->device release];                     // -1

  iree_hal_metal_staging_buffer_deinitialize(&device->staging_buffer);
  iree_arena_block_pool_deinitialize(&device->block_pool);

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

static iree_status_t iree_hal_metal_device_trim(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_metal_device_query_i64(iree_hal_device_t* base_device,
                                                     iree_string_view_t category,
                                                     iree_string_view_t key, int64_t* out_value) {
  *out_value = 0;

  if (iree_string_view_equal(category, iree_make_cstring_view("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, iree_make_cstring_view("metal-msl-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "unknown device configuration key value '%.*s :: %.*s'",
                          (int)category.size, category.data, (int)key.size, key.data);
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

  if (iree_any_bit_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_NESTED))
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "nested command buffer not yet supported");
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT))
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "multi-shot command buffer not yet supported");

  return iree_hal_metal_direct_command_buffer_create(
      base_device, mode, command_categories, binding_capacity,
      device->command_buffer_resource_reference_mode, device->queue, &device->block_pool,
      &device->staging_buffer, device->builtin_executable, device->host_allocator,
      out_command_buffer);
}

static iree_status_t iree_hal_metal_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device, iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count, const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_descriptor_set_layout_create(
      flags, binding_count, bindings, device->host_allocator, out_descriptor_set_layout);
}

static iree_status_t iree_hal_metal_device_create_event(iree_hal_device_t* base_device,
                                                        iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_metal_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier, iree_loop_t loop,
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
  if (iree_io_file_handle_type(handle) != IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "implementation does not support the external file type");
  }
  return iree_hal_memory_file_wrap(queue_affinity, access, handle,
                                   iree_hal_device_allocator(base_device),
                                   iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_metal_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count, iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_pipeline_layout_create(set_layout_count, set_layouts, push_constants,
                                               device->host_allocator, out_pipeline_layout);
}

static iree_status_t iree_hal_metal_device_create_semaphore(iree_hal_device_t* base_device,
                                                            uint64_t initial_value,
                                                            iree_hal_semaphore_t** out_semaphore) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  return iree_hal_metal_shared_event_create(device->device, initial_value, device->event_listener,
                                            device->host_allocator, out_semaphore);
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

static iree_status_t iree_hal_metal_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_allocator_pool_t pool,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // TODO(benvanik): queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list, iree_infinite_timeout()));
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                                          params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* buffer) {
  // TODO(benvanik): queue-ordered allocations.
  return iree_hal_device_queue_barrier(base_device, queue_affinity, wait_semaphore_list,
                                       signal_semaphore_list);
}

static iree_status_t iree_hal_metal_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_file_t* source_file,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list, source_file,
      source_offset, target_buffer, target_offset, length, flags, options));
  return loop_status;
}

static iree_status_t iree_hal_metal_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list, source_buffer,
      source_offset, target_file, target_offset, length, flags, options));
  return loop_status;
}

static iree_status_t iree_hal_metal_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(&device->block_pool, &resource_set));

  iree_status_t status =
      iree_hal_resource_set_insert(resource_set, command_buffer_count, command_buffers);

  // Put the full semaphore list into a resource set, which retains them--we will need to access
  // them until the command buffer completes.
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(resource_set, wait_semaphore_list.count,
                                          wait_semaphore_list.semaphores);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(resource_set, signal_semaphore_list.count,
                                          signal_semaphore_list.semaphores);
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
      for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
        iree_hal_command_buffer_t* command_buffer = command_buffers[i];
        id<MTLCommandBuffer> handle = iree_hal_metal_direct_command_buffer_handle(command_buffer);
        if (i + 1 != command_buffer_count) [handle commit];
        signal_command_buffer = handle;
      }
      if (signal_command_buffer == nil) {
        signal_command_buffer = [device->queue
            commandBufferWithDescriptor:device->command_buffer_descriptor];  // autoreleased
      }

      // Finally encode signal commands for all signal semaphores.
      for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
        id<MTLSharedEvent> handle =
            iree_hal_metal_shared_event_handle(signal_semaphore_list.semaphores[i]);
        [signal_command_buffer encodeSignalEvent:handle
                                           value:signal_semaphore_list.payload_values[i]];
      }

      // We use a resource set to keep track of resources in the above. So here we need to retain
      // the device to make sure the block pool behind outlives the resource set.
      iree_hal_device_retain(base_device);
      [signal_command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        // Now we can release all retained resources.
        iree_hal_resource_set_free(resource_set);
        // And then release the device handle. Note that this must happen separately--if we put the
        // device itself in the resource set, we can destroy the block pool data structure inside
        // the device prematurely, before the resource set free procedure done scanning it.
        iree_hal_device_release(base_device);
      }];
      [signal_command_buffer commit];
    }
  } else {
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

static iree_status_t iree_hal_metal_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  return iree_hal_metal_shared_event_multi_wait(wait_mode, &semaphore_list, timeout);
}

static iree_status_t iree_hal_metal_device_profiling_begin(
    iree_hal_device_t* base_device, const iree_hal_device_profiling_options_t* options) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);

  if (device->capture_manager) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "cannot nest profile capture");
  }

  if (iree_all_bits_set(options->mode, IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS)) {
    device->capture_manager = [[MTLCaptureManager sharedCaptureManager] retain];  // +1

    @autoreleasepool {
      NSURL* capture_url = NULL;
      if (strlen(options->file_path) != 0) {
        if (!iree_string_view_ends_with(IREE_SV(options->file_path), IREE_SV(".gputrace"))) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "capture filename must end with .gputrace");
        }
        if (![device->capture_manager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported capture to file (if invoking as command-line "
                                  "binary, make sure there is companion Info.plist under the same "
                                  "directory with 'MetalCaptureEnabled' key being true)");
        }

        NSString* ns_string = [NSString stringWithCString:options->file_path
                                                 encoding:[NSString defaultCStringEncoding]];
        NSString* capture_path = ns_string.stringByStandardizingPath;
        capture_url = [NSURL fileURLWithPath:capture_path isDirectory:false];
      }

      MTLCaptureDescriptor* capture_descriptor = [[[MTLCaptureDescriptor alloc] init] autorelease];
      capture_descriptor.captureObject = device->device;
      if (capture_url) {
        capture_descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        capture_descriptor.outputURL = capture_url;
      } else {
        capture_descriptor.destination = MTLCaptureDestinationDeveloperTools;
      }

      NSError* error = NULL;
      if (![device->capture_manager startCaptureWithDescriptor:capture_descriptor error:&error]) {
        iree_status_t status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "failed to start profile capture");
        const char* ns_c_error = [error.localizedDescription
            cStringUsingEncoding:[NSString defaultCStringEncoding]];  // autoreleased
        return iree_status_annotate_f(status, "with NSError: %s", ns_c_error);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_profiling_flush(iree_hal_device_t* base_device) {
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_profiling_end(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  if (device->capture_manager) {
    [device->capture_manager stopCapture];
    [device->capture_manager release];  // -1
    device->capture_manager = NULL;
  }
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_metal_device_vtable = {
    .destroy = iree_hal_metal_device_destroy,
    .id = iree_hal_metal_device_id,
    .host_allocator = iree_hal_metal_device_host_allocator,
    .device_allocator = iree_hal_metal_device_allocator,
    .replace_device_allocator = iree_hal_metal_replace_device_allocator,
    .trim = iree_hal_metal_device_trim,
    .query_i64 = iree_hal_metal_device_query_i64,
    .create_channel = iree_hal_metal_device_create_channel,
    .create_command_buffer = iree_hal_metal_device_create_command_buffer,
    .create_descriptor_set_layout = iree_hal_metal_device_create_descriptor_set_layout,
    .create_event = iree_hal_metal_device_create_event,
    .create_executable_cache = iree_hal_metal_device_create_executable_cache,
    .import_file = iree_hal_metal_device_import_file,
    .create_pipeline_layout = iree_hal_metal_device_create_pipeline_layout,
    .create_semaphore = iree_hal_metal_device_create_semaphore,
    .query_semaphore_compatibility = iree_hal_metal_device_query_semaphore_compatibility,
    .transfer_range = iree_hal_device_submit_transfer_range_and_wait,
    .queue_alloca = iree_hal_metal_device_queue_alloca,
    .queue_dealloca = iree_hal_metal_device_queue_dealloca,
    .queue_read = iree_hal_metal_device_queue_read,
    .queue_write = iree_hal_metal_device_queue_write,
    .queue_execute = iree_hal_metal_device_queue_execute,
    .queue_flush = iree_hal_metal_device_queue_flush,
    .wait_semaphores = iree_hal_metal_device_wait_semaphores,
    .profiling_begin = iree_hal_metal_device_profiling_begin,
    .profiling_flush = iree_hal_metal_device_profiling_flush,
    .profiling_end = iree_hal_metal_device_profiling_end,
};
