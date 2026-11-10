// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hexagon/allocator.h"

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hexagon_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_hexagon_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_hexagon_allocator_vtable;

static iree_hal_hexagon_allocator_t* iree_hal_hexagon_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_allocator_vtable);
  return (iree_hal_hexagon_allocator_t*)base_value;
}

iree_status_t iree_hal_hexagon_allocator_create(
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;

  iree_hal_hexagon_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_hexagon_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;

  *out_allocator = (iree_hal_allocator_t*)allocator;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hexagon_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  iree_hal_hexagon_allocator_t* allocator =
      iree_hal_hexagon_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_hexagon_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hexagon_allocator_t* allocator =
      (iree_hal_hexagon_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_hexagon_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_hexagon_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
}

static iree_status_t iree_hal_hexagon_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  // TODO(hexagon): report the heaps once buffer allocation is implemented.
  *out_count = 0;
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_hexagon_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  // TODO(hexagon): set compatibility rules once buffer allocation is
  // implemented.
  return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
}

static iree_status_t iree_hal_hexagon_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon buffer allocation not implemented");
}

static void iree_hal_hexagon_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  // No buffers are ever allocated from this allocator.
  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_hexagon_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon buffer import not implemented");
}

static iree_status_t iree_hal_hexagon_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon buffer export not implemented");
}

static bool iree_hal_hexagon_allocator_supports_virtual_memory(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return false;
}

static iree_status_t
iree_hal_hexagon_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params,
    iree_device_size_t* IREE_RESTRICT out_minimum_page_size,
    iree_device_size_t* IREE_RESTRICT out_recommended_page_size) {
  *out_minimum_page_size = 0;
  *out_recommended_page_size = 0;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "hexagon allocator does not support virtual memory");
}

static iree_status_t iree_hal_hexagon_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t** IREE_RESTRICT out_virtual_buffer) {
  *out_virtual_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "hexagon allocator does not support virtual memory");
}

static iree_status_t iree_hal_hexagon_allocator_virtual_memory_release(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "hexagon allocator does not support virtual memory");
}

static iree_status_t iree_hal_hexagon_allocator_physical_memory_allocate(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params, iree_device_size_t size,
    iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** IREE_RESTRICT out_physical_memory) {
  *out_physical_memory = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "hexagon allocator does not support virtual memory");
}

static iree_status_t iree_hal_hexagon_allocator_physical_memory_free(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "hexagon allocator does not support virtual memory");
}

static iree_status_t iree_hal_hexagon_allocator_virtual_memory_map(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "hexagon allocator does not support virtual memory");
}

static iree_status_t iree_hal_hexagon_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "hexagon allocator does not support virtual memory");
}

static iree_status_t iree_hal_hexagon_allocator_virtual_memory_protect(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "hexagon allocator does not support virtual memory");
}

static iree_status_t iree_hal_hexagon_allocator_virtual_memory_advise(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "hexagon allocator does not support virtual memory");
}

static const iree_hal_allocator_vtable_t iree_hal_hexagon_allocator_vtable = {
    .destroy = iree_hal_hexagon_allocator_destroy,
    .host_allocator = iree_hal_hexagon_allocator_host_allocator,
    .trim = iree_hal_hexagon_allocator_trim,
    .query_statistics = iree_hal_hexagon_allocator_query_statistics,
    .query_memory_heaps = iree_hal_hexagon_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_hexagon_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_hexagon_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_hexagon_allocator_deallocate_buffer,
    .import_buffer = iree_hal_hexagon_allocator_import_buffer,
    .export_buffer = iree_hal_hexagon_allocator_export_buffer,
    .supports_virtual_memory =
        iree_hal_hexagon_allocator_supports_virtual_memory,
    .virtual_memory_query_granularity =
        iree_hal_hexagon_allocator_virtual_memory_query_granularity,
    .virtual_memory_reserve = iree_hal_hexagon_allocator_virtual_memory_reserve,
    .virtual_memory_release = iree_hal_hexagon_allocator_virtual_memory_release,
    .physical_memory_allocate =
        iree_hal_hexagon_allocator_physical_memory_allocate,
    .physical_memory_free = iree_hal_hexagon_allocator_physical_memory_free,
    .virtual_memory_map = iree_hal_hexagon_allocator_virtual_memory_map,
    .virtual_memory_unmap = iree_hal_hexagon_allocator_virtual_memory_unmap,
    .virtual_memory_protect = iree_hal_hexagon_allocator_virtual_memory_protect,
    .virtual_memory_advise = iree_hal_hexagon_allocator_virtual_memory_advise,
};
