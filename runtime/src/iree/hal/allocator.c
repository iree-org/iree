// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/allocator.h"

#include <stddef.h>
#include <stdio.h>

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_format(
    const iree_hal_allocator_statistics_t* statistics,
    iree_string_builder_t* builder) {
#if IREE_STATISTICS_ENABLE

  // This could be prettier/have nice number formatting/etc.

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  HOST_LOCAL: %12" PRIdsz "B peak / %12" PRIdsz
      "B allocated / %12" PRIdsz "B freed / %12" PRIdsz "B live\n",
      statistics->host_bytes_peak, statistics->host_bytes_allocated,
      statistics->host_bytes_freed,
      (statistics->host_bytes_allocated - statistics->host_bytes_freed)));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "DEVICE_LOCAL: %12" PRIdsz "B peak / %12" PRIdsz
      "B allocated / %12" PRIdsz "B freed / %12" PRIdsz "B live\n",
      statistics->device_bytes_peak, statistics->device_bytes_allocated,
      statistics->device_bytes_freed,
      (statistics->device_bytes_allocated - statistics->device_bytes_freed)));

#else
  // No-op when disabled.
#endif  // IREE_STATISTICS_ENABLE
  return iree_ok_status();
}

#define _VTABLE_DISPATCH(allocator, method_name) \
  IREE_HAL_VTABLE_DISPATCH(allocator, iree_hal_allocator, method_name)

IREE_HAL_API_RETAIN_RELEASE(allocator);

IREE_API_EXPORT iree_allocator_t iree_hal_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT allocator) {
  IREE_ASSERT_ARGUMENT(allocator);
  return _VTABLE_DISPATCH(allocator, host_allocator)(allocator);
}

IREE_API_EXPORT
iree_status_t iree_hal_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT allocator) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, trim)(allocator);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_ASSERT_ARGUMENT(allocator);
  memset(out_statistics, 0, sizeof(*out_statistics));
  IREE_STATISTICS({
    _VTABLE_DISPATCH(allocator, query_statistics)(allocator, out_statistics);
  });
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_fprint(
    FILE* file, iree_hal_allocator_t* IREE_RESTRICT allocator) {
#if IREE_STATISTICS_ENABLE
  iree_hal_allocator_statistics_t statistics;
  iree_hal_allocator_query_statistics(allocator, &statistics);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_hal_allocator_host_allocator(allocator),
                                 &builder);

  // TODO(benvanik): query identifier for the allocator so we can denote which
  // device is being reported.
  iree_status_t status = iree_string_builder_append_cstring(
      &builder, "[[ iree_hal_allocator_t memory statistics ]]\n");

  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_statistics_format(&statistics, &builder);
  }

  if (iree_status_is_ok(status)) {
    fprintf(file, "%.*s", (int)iree_string_builder_size(&builder),
            iree_string_builder_buffer(&builder));
  }

  iree_string_builder_deinitialize(&builder);
  return status;
#else
  // No-op.
  return iree_ok_status();
#endif  // IREE_STATISTICS_ENABLE
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT allocator, iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  IREE_ASSERT_ARGUMENT(allocator);
  if (out_count) *out_count = 0;
  return _VTABLE_DISPATCH(allocator, query_memory_heaps)(allocator, capacity,
                                                         heaps, out_count);
}

IREE_API_EXPORT iree_hal_buffer_compatibility_t
iree_hal_allocator_query_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size) {
  IREE_ASSERT_ARGUMENT(allocator);
  iree_hal_buffer_params_canonicalize(&params);
  return _VTABLE_DISPATCH(allocator, query_compatibility)(allocator, &params,
                                                          allocation_size);
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (int64_t)allocation_size);
  iree_hal_buffer_params_canonicalize(&params);
  iree_status_t status = _VTABLE_DISPATCH(allocator, allocate_buffer)(
      allocator, &params, allocation_size, initial_data, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator, iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(
      z0, (int64_t)iree_hal_buffer_allocation_size(buffer));
  _VTABLE_DISPATCH(allocator, deallocate_buffer)(allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(external_buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_buffer_params_canonicalize(&params);
  iree_status_t status = _VTABLE_DISPATCH(allocator, import_buffer)(
      allocator, &params, external_buffer, release_callback, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_external_buffer);
  memset(out_external_buffer, 0, sizeof(*out_external_buffer));
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, export_buffer)(
      allocator, buffer, requested_type, requested_flags, out_external_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
