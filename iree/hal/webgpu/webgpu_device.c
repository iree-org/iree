// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/webgpu/webgpu_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/tracing.h"
#include "iree/hal/webgpu/bind_group_cache.h"
#include "iree/hal/webgpu/builtins.h"
#include "iree/hal/webgpu/command_buffer.h"
#include "iree/hal/webgpu/descriptor_set_layout.h"
#include "iree/hal/webgpu/executable_layout.h"
#include "iree/hal/webgpu/nop_event.h"
#include "iree/hal/webgpu/nop_executable_cache.h"
#include "iree/hal/webgpu/nop_semaphore.h"
#include "iree/hal/webgpu/simple_allocator.h"
#include "iree/hal/webgpu/staging_buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_device_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_WEBGPU_SMALL_POOL_BLOCK_SIZE (4 * 1024)
#define IREE_HAL_WEBGPU_LARGE_POOL_BLOCK_SIZE (32 * 1024)

IREE_API_EXPORT void iree_hal_webgpu_device_options_initialize(
    iree_hal_webgpu_device_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  out_options->flags = IREE_HAL_WEBGPU_DEVICE_RESERVED;
  out_options->queue_uniform_buffer_size =
      IREE_HAL_WEBGPU_STAGING_BUFFER_DEFAULT_CAPACITY;
}

typedef struct iree_hal_webgpu_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;

  bool owns_device_handle;
  WGPUInstance instance;
  WGPUAdapter adapter;
  WGPUDevice handle;
  WGPUQueue queue;

  // Block pool used for small allocations like submissions and callbacks.
  iree_arena_block_pool_t small_block_pool;
  // Block pool used for command buffers with a large block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t large_block_pool;

  iree_hal_allocator_t* device_allocator;

  // Builtin shaders emulating functionality not present in WebGPU.
  iree_hal_webgpu_builtins_t builtins;

  // Cached bind groups used during command buffer recording.
  iree_hal_webgpu_bind_group_cache_t bind_group_cache;

  // Staging buffer for parameter uploads.
  // Host storage is allocated as part of the device structure.
  iree_hal_webgpu_staging_buffer_t staging_buffer;
  uint8_t staging_buffer_host_data[];
} iree_hal_webgpu_device_t;

extern const iree_hal_device_vtable_t iree_hal_webgpu_device_vtable;

static iree_hal_webgpu_device_t* iree_hal_webgpu_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_device_vtable);
  return (iree_hal_webgpu_device_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_webgpu_wrap_device(
    iree_string_view_t identifier,
    const iree_hal_webgpu_device_options_t* options, WGPUInstance instance,
    WGPUAdapter adapter, WGPUDevice handle, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(adapter);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  WGPUSupportedLimits supported_limits;
  memset(&supported_limits, 0, sizeof(supported_limits));
  if (!wgpuDeviceGetLimits(handle, &supported_limits)) {
    // Failed to query limits - conservatively set what we need.
    // TODO(benvanik): see if it's realistic for this to fail - it cannot in
    // the browser implementation so I'm not sure why it returns bool here.
    supported_limits.limits.minStorageBufferOffsetAlignment = 256;
    supported_limits.limits.minUniformBufferOffsetAlignment = 256;
  }

  iree_hal_webgpu_device_t* device = NULL;
  iree_host_size_t total_size =
      sizeof(*device) + options->queue_uniform_buffer_size + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  iree_hal_resource_initialize(&iree_hal_webgpu_device_vtable,
                               &device->resource);
  device->host_allocator = host_allocator;
  uint8_t* buffer_ptr = (uint8_t*)device + sizeof(*device);
  buffer_ptr += options->queue_uniform_buffer_size;
  buffer_ptr += iree_string_view_append_to_buffer(
      identifier, &device->identifier, (char*)buffer_ptr);

  device->owns_device_handle = false;
  device->instance = instance;
  device->adapter = adapter;
  device->handle = handle;
  device->queue = wgpuDeviceGetQueue(handle);

  iree_arena_block_pool_initialize(IREE_HAL_WEBGPU_SMALL_POOL_BLOCK_SIZE,
                                   host_allocator, &device->small_block_pool);
  iree_arena_block_pool_initialize(IREE_HAL_WEBGPU_LARGE_POOL_BLOCK_SIZE,
                                   host_allocator, &device->large_block_pool);

  iree_hal_webgpu_bind_group_cache_initialize(device->handle,
                                              &device->bind_group_cache);

  iree_status_t status = iree_hal_webgpu_simple_allocator_create(
      device->handle, device->identifier, device->host_allocator,
      &device->device_allocator);

  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_staging_buffer_initialize(
        device->handle, &supported_limits.limits, device->device_allocator,
        device->staging_buffer_host_data, options->queue_uniform_buffer_size,
        &device->staging_buffer);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_builtins_initialize(
        device->handle, &device->staging_buffer, &device->builtins);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Builtins may be retaining resources; must be dropped first.
  iree_hal_webgpu_builtins_deinitialize(&device->builtins);

  // Bind groups can retain buffers.
  iree_hal_webgpu_bind_group_cache_deinitialize(&device->bind_group_cache);

  // There must be no more buffers live that use the allocator.
  iree_hal_webgpu_staging_buffer_deinitialize(&device->staging_buffer);
  iree_hal_allocator_release(device->device_allocator);

  // All outstanding blocks must have been returned to the pool (all command
  // buffers/submissions/etc disposed).
  iree_arena_block_pool_deinitialize(&device->small_block_pool);
  iree_arena_block_pool_deinitialize(&device->large_block_pool);

  // If we wrapped an existing device we don't want to destroy it on shutdown as
  // it may still be in use by the hosting application.
  if (device->owns_device_handle) {
    // NOTE: this destroys the device immediately (vs dropping it); that's fine
    // as we have the same requirement in the HAL.
    wgpuDeviceDestroy(device->handle);
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

static iree_status_t iree_hal_webgpu_device_query_i32(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int32_t* out_value) {
  // iree_hal_webgpu_device_t* device =
  // iree_hal_webgpu_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category,
                             iree_make_cstring_view("hal.executable.format"))) {
    *out_value =
        iree_string_view_equal(key, iree_make_cstring_view("webgpu-wgsl-fb"))
            ? 1
            : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_webgpu_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_command_buffer_create(
      device->handle, mode, command_categories, queue_affinity,
      &device->large_block_pool, &device->staging_buffer,
      &device->bind_group_cache, &device->builtins, device->host_allocator,
      out_command_buffer);
}

static iree_status_t iree_hal_webgpu_device_create_descriptor_set(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "iree_hal_webgpu_device_create_descriptor_set not yet implemented");
}

static iree_status_t iree_hal_webgpu_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_descriptor_set_layout_create(
      device->handle, usage_type, binding_count, bindings,
      device->host_allocator, out_descriptor_set_layout);
}

static iree_status_t iree_hal_webgpu_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_nop_event_create(device->host_allocator, out_event);
}

static iree_status_t iree_hal_webgpu_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_nop_executable_cache_create(
      device->handle, identifier, device->host_allocator, out_executable_cache);
}

static iree_status_t iree_hal_webgpu_device_create_executable_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_hal_executable_layout_t** out_executable_layout) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_executable_layout_create(
      device->handle, set_layout_count, set_layouts, push_constants,
      device->host_allocator, out_executable_layout);
}

static iree_status_t iree_hal_webgpu_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_hal_webgpu_nop_semaphore_create(
      initial_value, device->host_allocator, out_semaphore);
}

static iree_status_t iree_hal_webgpu_device_queue_submit(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);

  // TODO(benvanik): this currently assumes we are synchronizing on semaphores
  // and that any passed in to wait on will already be signaled. This would need
  // to change a bit to properly support waiting on host-signaled semaphores.
  // All work is ordered against the WebGPU queues and there's only one queue so
  // there's really not much to do.

  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    for (iree_host_size_t j = 0; j < batches[i].command_buffer_count; ++j) {
      iree_hal_command_buffer_t* command_buffer = batches[i].command_buffers[j];
      IREE_RETURN_IF_ERROR(
          iree_hal_webgpu_command_buffer_issue(command_buffer, device->queue));
    }
  }

  // HACK: fully synchronize with the device... maybe.
  // TODO(benvanik): make semaphores work so that we don't have to synchronize.
  return iree_hal_device_wait_idle(base_device, iree_infinite_timeout());
}

static iree_status_t iree_hal_webgpu_device_submit_and_wait(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_timeout_t timeout) {
  // Submit...
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_device_queue_submit(
      base_device, command_categories, queue_affinity, batch_count, batches));

  // ...and wait.
  return iree_hal_semaphore_wait(wait_semaphore, wait_value, timeout);
}

static iree_status_t iree_hal_webgpu_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "iree_hal_webgpu_device_wait_semaphores not yet implemented");
}

static iree_status_t iree_hal_webgpu_device_wait_idle(
    iree_hal_device_t* base_device, iree_timeout_t timeout) {
  iree_hal_webgpu_device_t* device = iree_hal_webgpu_device_cast(base_device);
  return iree_webgpu_queue_wait_idle(device->instance, device->handle,
                                     device->queue, timeout);
}

const iree_hal_device_vtable_t iree_hal_webgpu_device_vtable = {
    .destroy = iree_hal_webgpu_device_destroy,
    .id = iree_hal_webgpu_device_id,
    .host_allocator = iree_hal_webgpu_device_host_allocator,
    .device_allocator = iree_hal_webgpu_device_allocator,
    .query_i32 = iree_hal_webgpu_device_query_i32,
    .create_command_buffer = iree_hal_webgpu_device_create_command_buffer,
    .create_descriptor_set = iree_hal_webgpu_device_create_descriptor_set,
    .create_descriptor_set_layout =
        iree_hal_webgpu_device_create_descriptor_set_layout,
    .create_event = iree_hal_webgpu_device_create_event,
    .create_executable_cache = iree_hal_webgpu_device_create_executable_cache,
    .create_executable_layout = iree_hal_webgpu_device_create_executable_layout,
    .create_semaphore = iree_hal_webgpu_device_create_semaphore,
    .queue_submit = iree_hal_webgpu_device_queue_submit,
    .submit_and_wait = iree_hal_webgpu_device_submit_and_wait,
    .wait_semaphores = iree_hal_webgpu_device_wait_semaphores,
    .wait_idle = iree_hal_webgpu_device_wait_idle,
};
