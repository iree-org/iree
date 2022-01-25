// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/rocm_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/rocm/context_wrapper.h"
#include "experimental/rocm/descriptor_set_layout.h"
#include "experimental/rocm/direct_command_buffer.h"
#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/event_semaphore.h"
#include "experimental/rocm/executable_layout.h"
#include "experimental/rocm/nop_executable_cache.h"
#include "experimental/rocm/rocm_allocator.h"
#include "experimental/rocm/rocm_event.h"
#include "experimental/rocm/status_util.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tracing.h"
#include "iree/hal/utils/buffer_transfer.h"

//===----------------------------------------------------------------------===//
// iree_hal_rocm_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_rocm_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the ROCM symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  hipDevice_t device;

  // TODO: support multiple streams.
  hipStream_t stream;
  iree_hal_rocm_context_wrapper_t context_wrapper;
  iree_hal_allocator_t* device_allocator;

} iree_hal_rocm_device_t;

static const iree_hal_device_vtable_t iree_hal_rocm_device_vtable;

static iree_hal_rocm_device_t* iree_hal_rocm_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_device_vtable);
  return (iree_hal_rocm_device_t*)base_value;
}

static void iree_hal_rocm_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);
  ROCM_IGNORE_ERROR(device->context_wrapper.syms,
                    hipStreamDestroy(device->stream));

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_rocm_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    hipDevice_t rocm_device, hipStream_t stream, hipCtx_t context,
    iree_hal_rocm_dynamic_symbols_t* syms, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_rocm_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_rocm_device_vtable, &device->resource);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  uint8_t* buffer_ptr = (uint8_t*)device + sizeof(*device);
  buffer_ptr += iree_string_view_append_to_buffer(
      identifier, &device->identifier, (char*)buffer_ptr);
  device->device = rocm_device;
  device->stream = stream;
  device->context_wrapper.rocm_context = context;
  device->context_wrapper.host_allocator = host_allocator;
  device->context_wrapper.syms = syms;
  iree_status_t status = iree_hal_rocm_allocator_create(
      (iree_hal_device_t*)device, &device->context_wrapper,
      &device->device_allocator);
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_rocm_device_create(iree_hal_driver_t* driver,
                                          iree_string_view_t identifier,
                                          iree_hal_rocm_dynamic_symbols_t* syms,
                                          hipDevice_t device,
                                          iree_allocator_t host_allocator,
                                          iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hipCtx_t context;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, ROCM_RESULT_TO_STATUS(syms, hipCtxCreate(&context, 0, device)));
  hipStream_t stream;
  iree_status_t status = ROCM_RESULT_TO_STATUS(
      syms, hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_device_create_internal(driver, identifier, device,
                                                  stream, context, syms,
                                                  host_allocator, out_device);
  }
  if (!iree_status_is_ok(status)) {
    if (stream) {
      syms->hipStreamDestroy(stream);
    }
    syms->hipCtxDestroy(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_string_view_t iree_hal_rocm_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_rocm_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return device->context_wrapper.host_allocator;
}

static iree_hal_allocator_t* iree_hal_rocm_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return device->device_allocator;
}

static iree_status_t iree_hal_rocm_device_query_i32(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int32_t* out_value) {
  // iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category,
                             iree_make_cstring_view("hal.executable.format"))) {
    *out_value =
        iree_string_view_equal(key, iree_make_cstring_view("rocm-hsaco-fb"))
            ? 1
            : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_rocm_device_trim(iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_rocm_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_direct_command_buffer_create(
      base_device, &device->context_wrapper, mode, command_categories,
      queue_affinity, &device->block_pool, out_command_buffer);
}

static iree_status_t iree_hal_rocm_device_create_descriptor_set(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "non-push descriptor sets still need work");
}

static iree_status_t iree_hal_rocm_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_descriptor_set_layout_create(
      &device->context_wrapper, usage_type, binding_count, bindings,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_rocm_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_event_create(&device->context_wrapper, out_event);
}

static iree_status_t iree_hal_rocm_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_nop_executable_cache_create(
      &device->context_wrapper, identifier, out_executable_cache);
}

static iree_status_t iree_hal_rocm_device_create_executable_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_hal_executable_layout_t** out_executable_layout) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_executable_layout_create(
      &device->context_wrapper, set_layout_count, set_layouts, push_constants,
      out_executable_layout);
}

static iree_status_t iree_hal_rocm_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_semaphore_create(&device->context_wrapper, initial_value,
                                        out_semaphore);
}

static iree_status_t iree_hal_rocm_device_queue_submit(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  // TODO(raikonenfnu): Once semaphore is implemented wait for semaphores
  // TODO(thomasraoux): implement semaphores - for now this conservatively
  // synchronizes after every submit.
  // TODO(raikonenfnu): currently run on default/null stream, when cmd buffer
  // stream work with device->stream, we'll change
  ROCM_RETURN_IF_ERROR(device->context_wrapper.syms, hipStreamSynchronize(0),
                       "hipStreamSynchronize");
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_submit_and_wait(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_timeout_t timeout) {
  // Submit...
  IREE_RETURN_IF_ERROR(iree_hal_rocm_device_queue_submit(
      base_device, command_categories, queue_affinity, batch_count, batches));

  // ...and wait.
  return iree_hal_semaphore_wait(wait_semaphore, wait_value, timeout);
}

static iree_status_t iree_hal_rocm_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "semaphore not implemented");
}

static iree_status_t iree_hal_rocm_device_wait_idle(
    iree_hal_device_t* base_device, iree_timeout_t timeout) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  // Wait until the stream is done.
  // TODO(thomasraoux): HIP doesn't support a deadline for wait, figure out how
  // to handle it better.
  ROCM_RETURN_IF_ERROR(device->context_wrapper.syms,
                       hipStreamSynchronize(device->stream),
                       "hipStreamSynchronize");
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_rocm_device_vtable = {
    .destroy = iree_hal_rocm_device_destroy,
    .id = iree_hal_rocm_device_id,
    .host_allocator = iree_hal_rocm_device_host_allocator,
    .device_allocator = iree_hal_rocm_device_allocator,
    .trim = iree_hal_rocm_device_trim,
    .query_i32 = iree_hal_rocm_device_query_i32,
    .create_command_buffer = iree_hal_rocm_device_create_command_buffer,
    .create_descriptor_set = iree_hal_rocm_device_create_descriptor_set,
    .create_descriptor_set_layout =
        iree_hal_rocm_device_create_descriptor_set_layout,
    .create_event = iree_hal_rocm_device_create_event,
    .create_executable_cache = iree_hal_rocm_device_create_executable_cache,
    .create_executable_layout = iree_hal_rocm_device_create_executable_layout,
    .create_semaphore = iree_hal_rocm_device_create_semaphore,
    .transfer_range = iree_hal_device_submit_transfer_range_and_wait,
    .queue_submit = iree_hal_rocm_device_queue_submit,
    .submit_and_wait = iree_hal_rocm_device_submit_and_wait,
    .wait_semaphores = iree_hal_rocm_device_wait_semaphores,
    .wait_idle = iree_hal_rocm_device_wait_idle,
};
