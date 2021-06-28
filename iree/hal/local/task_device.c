// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/task_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/tracing.h"
#include "iree/hal/local/event_pool.h"
#include "iree/hal/local/local_descriptor_set.h"
#include "iree/hal/local/local_descriptor_set_layout.h"
#include "iree/hal/local/local_executable_cache.h"
#include "iree/hal/local/local_executable_layout.h"
#include "iree/hal/local/task_command_buffer.h"
#include "iree/hal/local/task_event.h"
#include "iree/hal/local/task_queue.h"
#include "iree/hal/local/task_semaphore.h"

#define IREE_HAL_LOCAL_TASK_EVENT_POOL_CAPACITY 32

typedef struct iree_hal_task_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for small allocations like tasks and submissions.
  iree_arena_block_pool_t small_block_pool;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t large_block_pool;

  // iree_event_t pool for semaphore wait operations.
  iree_hal_local_event_pool_t* event_pool;

  iree_task_executor_t* executor;

  iree_host_size_t loader_count;
  iree_hal_executable_loader_t** loaders;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

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
  out_params->queue_count = 8;
}

static iree_status_t iree_hal_task_device_check_params(
    const iree_hal_task_device_params_t* params) {
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

iree_status_t iree_hal_task_device_create(
    iree_string_view_t identifier, const iree_hal_task_device_params_t* params,
    iree_task_executor_t* executor, iree_host_size_t loader_count,
    iree_hal_executable_loader_t** loaders, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(!loader_count || loaders);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_task_device_check_params(params));

  iree_hal_task_device_t* device = NULL;
  iree_host_size_t total_size =
      sizeof(*device) + params->queue_count * sizeof(*device->queues) +
      identifier.size + loader_count * sizeof(*device->loaders);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device);
  if (iree_status_is_ok(status)) {
    memset(device, 0, total_size);
    iree_hal_resource_initialize(&iree_hal_task_device_vtable,
                                 &device->resource);
    iree_string_view_append_to_buffer(
        identifier, &device->identifier,
        (char*)device + sizeof(*device) +
            params->queue_count * sizeof(*device->queues));
    device->host_allocator = host_allocator;
    iree_arena_block_pool_initialize(4096, host_allocator,
                                     &device->small_block_pool);
    iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                     &device->large_block_pool);
    device->event_pool = NULL;

    device->executor = executor;
    iree_task_executor_retain(device->executor);

    device->loader_count = loader_count;
    device->loaders =
        (iree_hal_executable_loader_t**)((uint8_t*)device->identifier.data +
                                         identifier.size);
    for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
      device->loaders[i] = loaders[i];
      iree_hal_executable_loader_retain(device->loaders[i]);
    }

    device->queue_count = params->queue_count;
    for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
      // TODO(benvanik): add a number to each queue ID.
      iree_hal_task_queue_initialize(device->identifier, device->executor,
                                     &device->small_block_pool,
                                     &device->queues[i]);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_local_event_pool_allocate(
        IREE_HAL_LOCAL_TASK_EVENT_POOL_CAPACITY, host_allocator,
        &device->event_pool);
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
  iree_task_executor_release(device->executor);
  iree_hal_local_event_pool_free(device->event_pool);
  iree_arena_block_pool_deinitialize(&device->large_block_pool);
  iree_arena_block_pool_deinitialize(&device->small_block_pool);
  iree_hal_allocator_release(device->device_allocator);
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

static iree_status_t iree_hal_task_device_query_i32(
    iree_hal_device_t* base_device, iree_string_view_t key,
    int32_t* out_value) {
  // iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  *out_value = 0;
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "unknown device configuration key value '%.*s'",
                          (int)key.size, key.data);
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

static iree_status_t iree_hal_task_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_host_size_t queue_index = iree_hal_task_device_select_queue(
      device, command_categories, queue_affinity);
  return iree_hal_task_command_buffer_create(
      &device->queues[queue_index].scope, mode, command_categories,
      queue_affinity, &device->large_block_pool, device->host_allocator,
      out_command_buffer);
}

static iree_status_t iree_hal_task_device_create_descriptor_set(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  return iree_hal_local_descriptor_set_create(set_layout, binding_count,
                                              bindings, out_descriptor_set);
}

static iree_status_t iree_hal_task_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  return iree_hal_local_descriptor_set_layout_create(
      usage_type, binding_count, bindings,
      iree_hal_device_host_allocator(base_device), out_descriptor_set_layout);
}

static iree_status_t iree_hal_task_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  return iree_hal_task_event_create(iree_hal_device_host_allocator(base_device),
                                    out_event);
}

static iree_status_t iree_hal_task_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return iree_hal_local_executable_cache_create(
      identifier, device->loader_count, device->loaders,
      iree_hal_device_host_allocator(base_device), out_executable_cache);
}

static iree_status_t iree_hal_task_device_create_executable_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_hal_executable_layout_t** out_executable_layout) {
  return iree_hal_local_executable_layout_create(
      push_constants, set_layout_count, set_layouts,
      iree_hal_device_host_allocator(base_device), out_executable_layout);
}

static iree_status_t iree_hal_task_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return iree_hal_task_semaphore_create(device->event_pool, initial_value,
                                        device->host_allocator, out_semaphore);
}

static iree_status_t iree_hal_task_device_queue_submit(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  iree_host_size_t queue_index = iree_hal_task_device_select_queue(
      device, command_categories, queue_affinity);
  return iree_hal_task_queue_submit(&device->queues[queue_index], batch_count,
                                    batches);
}

static iree_status_t iree_hal_task_device_submit_and_wait(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_timeout_t timeout) {
  // Submit...
  IREE_RETURN_IF_ERROR(iree_hal_task_device_queue_submit(
      base_device, command_categories, queue_affinity, batch_count, batches));

  // ...and wait.
  return iree_hal_semaphore_wait(wait_semaphore, wait_value, timeout);
}

static iree_status_t iree_hal_task_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  return iree_hal_task_semaphore_multi_wait(wait_mode, semaphore_list, timeout,
                                            device->event_pool,
                                            &device->large_block_pool);
}

static iree_status_t iree_hal_task_device_wait_idle(
    iree_hal_device_t* base_device, iree_timeout_t timeout) {
  iree_hal_task_device_t* device = iree_hal_task_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    status = iree_hal_task_queue_wait_idle(&device->queues[i], timeout);
    if (!iree_status_is_ok(status)) break;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_device_vtable_t iree_hal_task_device_vtable = {
    .destroy = iree_hal_task_device_destroy,
    .id = iree_hal_task_device_id,
    .host_allocator = iree_hal_task_device_host_allocator,
    .device_allocator = iree_hal_task_device_allocator,
    .query_i32 = iree_hal_task_device_query_i32,
    .create_command_buffer = iree_hal_task_device_create_command_buffer,
    .create_descriptor_set = iree_hal_task_device_create_descriptor_set,
    .create_descriptor_set_layout =
        iree_hal_task_device_create_descriptor_set_layout,
    .create_event = iree_hal_task_device_create_event,
    .create_executable_cache = iree_hal_task_device_create_executable_cache,
    .create_executable_layout = iree_hal_task_device_create_executable_layout,
    .create_semaphore = iree_hal_task_device_create_semaphore,
    .queue_submit = iree_hal_task_device_queue_submit,
    .submit_and_wait = iree_hal_task_device_submit_and_wait,
    .wait_semaphores = iree_hal_task_device_wait_semaphores,
    .wait_idle = iree_hal_task_device_wait_idle,
};
