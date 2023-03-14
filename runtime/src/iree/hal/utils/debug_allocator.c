// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/debug_allocator.h"

#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_debug_allocator_t
//===----------------------------------------------------------------------===//

// We could make this configurable in order to rotate it during trials. For now
// it's fixed so that it's possible to pick this up in tooling.
//
// Expected values for each interpretation (signed/unsigned):
//  i8: -51 / 205
// i16: -12851 / 52685
// i32: -842150451 / 3452816845
// i64: -3617008641903833651 / 14829735431805717965
// f16: -23.20313
// f32: -4.316021e+08
// f64: -6.27743856220419e+66
#define IREE_HAL_DEBUG_ALLOCATOR_FILL_PATTERN 0xCD

struct iree_hal_debug_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_device_t* device;
  iree_hal_allocator_t* device_allocator;
};

static const iree_hal_allocator_vtable_t iree_hal_debug_allocator_vtable;

iree_hal_debug_allocator_t* iree_hal_debug_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_debug_allocator_vtable);
  return (iree_hal_debug_allocator_t*)base_value;
}

iree_status_t iree_hal_debug_allocator_create(
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_debug_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));

  iree_hal_resource_initialize(&iree_hal_debug_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->device = device;
  iree_hal_device_retain(allocator->device);
  allocator->device_allocator = device_allocator;
  iree_hal_allocator_retain(allocator->device_allocator);

  *out_allocator = (iree_hal_allocator_t*)allocator;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_debug_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_debug_allocator_t* allocator =
      iree_hal_debug_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(allocator->device_allocator);
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_debug_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_debug_allocator_t* allocator =
      (iree_hal_debug_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_debug_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_debug_allocator_t* allocator =
      iree_hal_debug_allocator_cast(base_allocator);
  return iree_hal_allocator_trim(allocator->device_allocator);
}

static void iree_hal_debug_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  iree_hal_debug_allocator_t* allocator =
      iree_hal_debug_allocator_cast(base_allocator);
  iree_hal_allocator_query_statistics(allocator->device_allocator,
                                      out_statistics);
}

static iree_status_t iree_hal_debug_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_debug_allocator_t* allocator =
      iree_hal_debug_allocator_cast(base_allocator);
  return iree_hal_allocator_query_memory_heaps(allocator->device_allocator,
                                               capacity, heaps, out_count);
}

static iree_hal_buffer_compatibility_t
iree_hal_debug_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_debug_allocator_t* allocator =
      iree_hal_debug_allocator_cast(base_allocator);
  return iree_hal_allocator_query_buffer_compatibility(
      allocator->device_allocator, *params, *allocation_size, params,
      allocation_size);
}

static iree_status_t iree_hal_debug_allocator_fill_on_host(
    iree_hal_buffer_t* buffer, uint8_t fill_pattern) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_buffer_map_fill(
      buffer, 0, IREE_WHOLE_BUFFER, &fill_pattern, sizeof(fill_pattern));
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_debug_allocator_fill_on_device(
    iree_hal_device_t* device, iree_hal_buffer_t* buffer,
    uint8_t fill_pattern) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_transfer_command_t command = {
      .type = IREE_HAL_TRANSFER_COMMAND_TYPE_FILL,
      .fill =
          {
              .target_buffer = buffer,
              .target_offset = 0,
              .length = iree_hal_buffer_allocation_size(buffer),
              .pattern = &fill_pattern,
              .pattern_length = sizeof(fill_pattern),
          },
  };

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_create_transfer_command_buffer(
              device,
              IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
                  IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
              IREE_HAL_QUEUE_AFFINITY_ANY, 1, &command, &command_buffer));

  iree_hal_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_hal_semaphore_create(device, 0ull, &semaphore);

  uint64_t signal_value = 1ull;
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &semaphore,
        .payload_values = &signal_value,
    };
    status = iree_hal_device_queue_execute(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           iree_hal_semaphore_list_empty(),
                                           signal_list, 1, &command_buffer);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout());
  }

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_debug_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_debug_allocator_t* allocator =
      iree_hal_debug_allocator_cast(base_allocator);

  // Allocate the buffer from the underlying allocator. It may come back with
  // undefined contents (including those from prior allocations which may appear
  // correct).
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator->device_allocator, *params, allocation_size, initial_data,
      out_buffer));

  // If the buffer is read-only we can't fill it even if we wanted to. This
  // usually happens with initial data.
  iree_hal_buffer_t* base_buffer = *out_buffer;
  if (initial_data.data_length > 0 ||
      !iree_all_bits_set(iree_hal_buffer_allowed_access(base_buffer),
                         IREE_HAL_MEMORY_ACCESS_WRITE)) {
    return iree_ok_status();
  }

  // We could rotate this here if we wanted to have it vary over time (per
  // allocation, per trim, etc).
  uint8_t fill_pattern = IREE_HAL_DEBUG_ALLOCATOR_FILL_PATTERN;

  if (iree_all_bits_set(iree_hal_buffer_allowed_usage(base_buffer),
                        IREE_HAL_BUFFER_USAGE_MAPPING) &&
      iree_all_bits_set(iree_hal_buffer_memory_type(base_buffer),
                        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return iree_hal_debug_allocator_fill_on_host(base_buffer, fill_pattern);
  } else {
    return iree_hal_debug_allocator_fill_on_device(allocator->device,
                                                   base_buffer, fill_pattern);
  }
}

static void iree_hal_debug_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer) {
  // No-op; we never point a buffer back at us for deallocation.
}

static iree_status_t iree_hal_debug_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_debug_allocator_t* allocator =
      iree_hal_debug_allocator_cast(base_allocator);
  return iree_hal_allocator_import_buffer(allocator->device_allocator, *params,
                                          external_buffer, release_callback,
                                          out_buffer);
}

static iree_status_t iree_hal_debug_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  iree_hal_debug_allocator_t* allocator =
      iree_hal_debug_allocator_cast(base_allocator);
  return iree_hal_allocator_export_buffer(allocator->device_allocator, buffer,
                                          requested_type, requested_flags,
                                          out_external_buffer);
}

static const iree_hal_allocator_vtable_t iree_hal_debug_allocator_vtable = {
    .destroy = iree_hal_debug_allocator_destroy,
    .host_allocator = iree_hal_debug_allocator_host_allocator,
    .trim = iree_hal_debug_allocator_trim,
    .query_statistics = iree_hal_debug_allocator_query_statistics,
    .query_memory_heaps = iree_hal_debug_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_debug_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_debug_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_debug_allocator_deallocate_buffer,
    .import_buffer = iree_hal_debug_allocator_import_buffer,
    .export_buffer = iree_hal_debug_allocator_export_buffer,
};
