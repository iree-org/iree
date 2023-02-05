// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/metal/metal_device.h"

#include "experimental/metal/direct_allocator.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/buffer_transfer.h"

typedef struct iree_hal_metal_device_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  iree_string_view_t identifier;

  // Original driver that owns this device.
  iree_hal_driver_t* driver;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  id<MTLDevice> device;
} iree_hal_metal_device_t;

static const iree_hal_device_vtable_t iree_hal_metal_device_vtable;

static iree_hal_metal_device_t* iree_hal_metal_device_cast(iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_device_vtable);
  return (iree_hal_metal_device_t*)base_value;
}

static iree_status_t iree_hal_metal_device_create_internal(iree_hal_driver_t* driver,
                                                           iree_string_view_t identifier,
                                                           id<MTLDevice> metal_device,
                                                           iree_allocator_t host_allocator,
                                                           iree_hal_device_t** out_device) {
  iree_hal_metal_device_t* device = NULL;

  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_status_t status = iree_hal_metal_allocator_create((iree_hal_device_t*)device, metal_device,
                                                         host_allocator, &device->device_allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_metal_device_vtable, &device->resource);
    iree_string_view_append_to_buffer(identifier, &device->identifier,
                                      (char*)device + iree_sizeof_struct(*device));
    device->driver = driver;
    iree_hal_driver_retain(device->driver);
    device->host_allocator = host_allocator;
    device->device = [metal_device retain];  // +1

    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_metal_device_create(iree_hal_driver_t* driver, iree_string_view_t identifier,
                                           id<MTLDevice> device, iree_allocator_t host_allocator,
                                           iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_hal_metal_device_create_internal(driver, identifier, device, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_metal_device_t* device = iree_hal_metal_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(device->device_allocator);
  [device->device release];  // -1
  iree_hal_driver_release(device->driver);

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
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_metal_device_query_i64(iree_hal_device_t* base_device,
                                                     iree_string_view_t category,
                                                     iree_string_view_t key, int64_t* out_value) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented device i64 query");
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
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented command buffer create");
}

static iree_status_t iree_hal_metal_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device, iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count, const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented descriptor set create");
}

static iree_status_t iree_hal_metal_device_create_event(iree_hal_device_t* base_device,
                                                        iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented event create");
}

static iree_status_t iree_hal_metal_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier, iree_loop_t loop,
    iree_hal_executable_cache_t** out_executable_cache) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented executable cache create");
}

static iree_status_t iree_hal_metal_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count, iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented pipeline layout create");
}

static iree_status_t iree_hal_metal_device_create_semaphore(iree_hal_device_t* base_device,
                                                            uint64_t initial_value,
                                                            iree_hal_semaphore_t** out_semaphore) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented semaphore create");
}

static iree_hal_semaphore_compatibility_t iree_hal_metal_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
}

static iree_status_t iree_hal_metal_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_allocator_pool_t pool,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented queue alloca");
}

static iree_status_t iree_hal_metal_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented queue dealloca");
}

static iree_status_t iree_hal_metal_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented queue execute");
}

static iree_status_t iree_hal_metal_device_queue_flush(iree_hal_device_t* base_device,
                                                       iree_hal_queue_affinity_t queue_affinity) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented queue flush");
}

static iree_status_t iree_hal_metal_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplmented semaphore wait");
}

static iree_status_t iree_hal_metal_device_profiling_begin(
    iree_hal_device_t* device, const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_device_profiling_end(iree_hal_device_t* device) {
  // Unimplemented (and that's ok).
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
    .create_pipeline_layout = iree_hal_metal_device_create_pipeline_layout,
    .create_semaphore = iree_hal_metal_device_create_semaphore,
    .query_semaphore_compatibility = iree_hal_metal_device_query_semaphore_compatibility,
    .transfer_range = iree_hal_device_submit_transfer_range_and_wait,
    .queue_alloca = iree_hal_metal_device_queue_alloca,
    .queue_dealloca = iree_hal_metal_device_queue_dealloca,
    .queue_execute = iree_hal_metal_device_queue_execute,
    .queue_flush = iree_hal_metal_device_queue_flush,
    .wait_semaphores = iree_hal_metal_device_wait_semaphores,
    .profiling_begin = iree_hal_metal_device_profiling_begin,
    .profiling_end = iree_hal_metal_device_profiling_end,
};
