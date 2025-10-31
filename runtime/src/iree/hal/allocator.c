// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/allocator.h"

#include <stddef.h>
#include <stdio.h>

#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

//===----------------------------------------------------------------------===//
// String utils
//===----------------------------------------------------------------------===//

static const iree_bitfield_string_mapping_t
    iree_hal_buffer_compatibility_mappings[] = {
        {IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE, IREE_SVL("ALLOCATABLE")},
        {IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE, IREE_SVL("IMPORTABLE")},
        {IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE, IREE_SVL("EXPORTABLE")},
        {IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
         IREE_SVL("QUEUE_TRANSFER")},
        {IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
         IREE_SVL("QUEUE_DISPATCH")},
        {IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE,
         IREE_SVL("LOW_PERFORMANCE")},
};

IREE_API_EXPORT iree_status_t iree_hal_buffer_compatibility_parse(
    iree_string_view_t value, iree_hal_buffer_compatibility_t* out_value) {
  return iree_bitfield_parse(
      value, IREE_ARRAYSIZE(iree_hal_buffer_compatibility_mappings),
      iree_hal_buffer_compatibility_mappings, out_value);
}

IREE_API_EXPORT iree_string_view_t
iree_hal_buffer_compatibility_format(iree_hal_buffer_compatibility_t value,
                                     iree_bitfield_string_temp_t* out_temp) {
  return iree_bitfield_format_inline(
      value, IREE_ARRAYSIZE(iree_hal_buffer_compatibility_mappings),
      iree_hal_buffer_compatibility_mappings, out_temp);
}

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

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t
//===----------------------------------------------------------------------===//

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
  if (capacity && !heaps) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "heap storage must be provided when capacity is defined");
  }
  return _VTABLE_DISPATCH(allocator, query_memory_heaps)(allocator, capacity,
                                                         heaps, out_count);
}

IREE_API_EXPORT iree_hal_buffer_compatibility_t
iree_hal_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t* out_params,
    iree_device_size_t* out_allocation_size) {
  IREE_ASSERT_ARGUMENT(allocator);
  iree_hal_buffer_params_canonicalize(&params);
  iree_hal_buffer_compatibility_t result =
      _VTABLE_DISPATCH(allocator, query_buffer_compatibility)(
          allocator, &params, &allocation_size);
  if (result != IREE_HAL_BUFFER_COMPATIBILITY_NONE) {
    if (out_params) *out_params = params;
    if (out_allocation_size) *out_allocation_size = allocation_size;
  }
  return result;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);
  iree_hal_buffer_params_canonicalize(&params);
  iree_status_t status = _VTABLE_DISPATCH(allocator, allocate_buffer)(
      allocator, &params, allocation_size, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT allocator, iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
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

//===----------------------------------------------------------------------===//
// Virtual Memory Management
//===----------------------------------------------------------------------===//

IREE_API_EXPORT bool iree_hal_allocator_supports_virtual_memory(
    iree_hal_allocator_t* IREE_RESTRICT allocator) {
  IREE_ASSERT_ARGUMENT(allocator);
  if (!_VTABLE_DISPATCH(allocator, supports_virtual_memory)) {
    return false;
  }
  return _VTABLE_DISPATCH(allocator, supports_virtual_memory)(allocator);
}

IREE_API_EXPORT iree_status_t
iree_hal_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params,
    iree_device_size_t* IREE_RESTRICT out_minimum_page_size,
    iree_device_size_t* IREE_RESTRICT out_recommended_page_size) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_minimum_page_size);
  IREE_ASSERT_ARGUMENT(out_recommended_page_size);
  *out_minimum_page_size = 0;
  *out_recommended_page_size = 0;
  if (!_VTABLE_DISPATCH(allocator, virtual_memory_query_granularity)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "virtual memory not supported by allocator");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(allocator, virtual_memory_query_granularity)(
          allocator, params, out_minimum_page_size, out_recommended_page_size);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t** IREE_RESTRICT out_virtual_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_virtual_buffer);
  *out_virtual_buffer = NULL;
  if (!_VTABLE_DISPATCH(allocator, virtual_memory_reserve)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "virtual memory not supported by allocator");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)size);
  iree_status_t status = _VTABLE_DISPATCH(allocator, virtual_memory_reserve)(
      allocator, queue_affinity, size, out_virtual_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_virtual_memory_release(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  if (!_VTABLE_DISPATCH(allocator, virtual_memory_release)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "virtual memory not supported by allocator");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, virtual_memory_release)(
      allocator, virtual_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_physical_memory_allocate(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_params_t params, iree_device_size_t size,
    iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** IREE_RESTRICT out_physical_memory) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_physical_memory);
  *out_physical_memory = NULL;
  if (!_VTABLE_DISPATCH(allocator, physical_memory_allocate)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "virtual memory not supported by allocator");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)size);
  iree_status_t status = _VTABLE_DISPATCH(allocator, physical_memory_allocate)(
      allocator, params, size, host_allocator, out_physical_memory);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_physical_memory_free(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(physical_memory);
  if (!_VTABLE_DISPATCH(allocator, physical_memory_free)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "virtual memory not supported by allocator");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, physical_memory_free)(
      allocator, physical_memory);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_virtual_memory_map(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(physical_memory);
  if (!_VTABLE_DISPATCH(allocator, virtual_memory_map)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "virtual memory not supported by allocator");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)size);
  iree_status_t status = _VTABLE_DISPATCH(allocator, virtual_memory_map)(
      allocator, virtual_buffer, virtual_offset, physical_memory,
      physical_offset, size);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  if (!_VTABLE_DISPATCH(allocator, virtual_memory_unmap)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "virtual memory not supported by allocator");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, virtual_memory_unmap)(
      allocator, virtual_buffer, virtual_offset, size);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_virtual_memory_protect(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  if (!_VTABLE_DISPATCH(allocator, virtual_memory_protect)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "virtual memory not supported by allocator");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, virtual_memory_protect)(
      allocator, virtual_buffer, virtual_offset, size, queue_affinity,
      protection);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_virtual_memory_advise(
    iree_hal_allocator_t* IREE_RESTRICT allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  if (!_VTABLE_DISPATCH(allocator, virtual_memory_advise)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "virtual memory not supported by allocator");
  }
  // No tracing for advise - it's a hint that may be called frequently.
  return _VTABLE_DISPATCH(allocator, virtual_memory_advise)(
      allocator, virtual_buffer, virtual_offset, size, queue_affinity, advice);
}
