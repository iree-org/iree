// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/sync_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/tracing.h"
#include "iree/hal/local/inline_command_buffer.h"
#include "iree/hal/local/local_descriptor_set.h"
#include "iree/hal/local/local_descriptor_set_layout.h"
#include "iree/hal/local/local_executable_cache.h"
#include "iree/hal/local/local_executable_layout.h"
#include "iree/hal/local/sync_event.h"
#include "iree/hal/local/sync_semaphore.h"
#include "iree/hal/utils/buffer_transfer.h"

typedef struct iree_hal_sync_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

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
}

static iree_status_t iree_hal_sync_device_check_params(
    const iree_hal_sync_device_params_t* params) {
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

static iree_status_t iree_hal_sync_device_trim(iree_hal_device_t* base_device) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_sync_device_query_i32(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int32_t* out_value) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category,
                             iree_make_cstring_view("hal.executable.format"))) {
    *out_value =
        iree_hal_query_any_executable_loader_support(
            device->loader_count, device->loaders, /*caching_mode=*/0, key)
            ? 1
            : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_sync_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer) {
  // TODO(#4680): implement a non-inline command buffer that stores its commands
  // and can be submitted later on/multiple-times.
  return iree_hal_inline_command_buffer_create(
      base_device, mode, command_categories, queue_affinity,
      iree_hal_device_host_allocator(base_device), out_command_buffer);
}

static iree_status_t iree_hal_sync_device_create_descriptor_set(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  return iree_hal_local_descriptor_set_create(set_layout, binding_count,
                                              bindings, out_descriptor_set);
}

static iree_status_t iree_hal_sync_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  return iree_hal_local_descriptor_set_layout_create(
      usage_type, binding_count, bindings,
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
      identifier, device->loader_count, device->loaders,
      iree_hal_device_host_allocator(base_device), out_executable_cache);
}

static iree_status_t iree_hal_sync_device_create_executable_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_hal_executable_layout_t** out_executable_layout) {
  return iree_hal_local_executable_layout_create(
      push_constants, set_layout_count, set_layouts,
      iree_hal_device_host_allocator(base_device), out_executable_layout);
}

static iree_status_t iree_hal_sync_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return iree_hal_sync_semaphore_create(&device->semaphore_state, initial_value,
                                        device->host_allocator, out_semaphore);
}

static iree_status_t iree_hal_sync_device_queue_submit(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);

  // TODO(#4680): there is some better error handling here needed; we should
  // propagate failures to all signal semaphores. Today we aren't as there
  // shouldn't be any failures or if there are there's not much we'd be able to
  // do - we already executed everything inline!

  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    const iree_hal_submission_batch_t* batch = &batches[i];

    // Wait for semaphores to be signaled before performing any work.
    IREE_RETURN_IF_ERROR(iree_hal_sync_semaphore_multi_wait(
        &device->semaphore_state, IREE_HAL_WAIT_MODE_ALL,
        &batch->wait_semaphores, iree_infinite_timeout()));

    // TODO(#4680): if we were doing deferred submissions we would issue them
    // here. With only inline command buffers we have nothing to do here.

    // Signal all semaphores now that batch work has completed.
    IREE_RETURN_IF_ERROR(iree_hal_sync_semaphore_multi_signal(
        &device->semaphore_state, &batch->signal_semaphores));
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_sync_device_submit_and_wait(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_timeout_t timeout) {
  // Submit...
  IREE_RETURN_IF_ERROR(iree_hal_sync_device_queue_submit(
      base_device, command_categories, queue_affinity, batch_count, batches));

  // ...and wait.
  return iree_hal_semaphore_wait(wait_semaphore, wait_value, timeout);
}

static iree_status_t iree_hal_sync_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout) {
  iree_hal_sync_device_t* device = iree_hal_sync_device_cast(base_device);
  return iree_hal_sync_semaphore_multi_wait(&device->semaphore_state, wait_mode,
                                            semaphore_list, timeout);
}

static iree_status_t iree_hal_sync_device_wait_idle(
    iree_hal_device_t* base_device, iree_timeout_t timeout) {
  // No-op (in intended usages). If we allowed multiple threads to call into
  // the same device then we may want to change this to an atomic flag as to
  // whether any thread is actively performing work.
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_sync_device_vtable = {
    .destroy = iree_hal_sync_device_destroy,
    .id = iree_hal_sync_device_id,
    .host_allocator = iree_hal_sync_device_host_allocator,
    .device_allocator = iree_hal_sync_device_allocator,
    .trim = iree_hal_sync_device_trim,
    .query_i32 = iree_hal_sync_device_query_i32,
    .create_command_buffer = iree_hal_sync_device_create_command_buffer,
    .create_descriptor_set = iree_hal_sync_device_create_descriptor_set,
    .create_descriptor_set_layout =
        iree_hal_sync_device_create_descriptor_set_layout,
    .create_event = iree_hal_sync_device_create_event,
    .create_executable_cache = iree_hal_sync_device_create_executable_cache,
    .create_executable_layout = iree_hal_sync_device_create_executable_layout,
    .create_semaphore = iree_hal_sync_device_create_semaphore,
    .transfer_range = iree_hal_device_transfer_mappable_range,
    .queue_submit = iree_hal_sync_device_queue_submit,
    .submit_and_wait = iree_hal_sync_device_submit_and_wait,
    .wait_semaphores = iree_hal_sync_device_wait_semaphores,
    .wait_idle = iree_hal_sync_device_wait_idle,
};
