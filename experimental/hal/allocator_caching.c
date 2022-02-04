// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hal/allocator_caching.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"
#include <pthread.h>

pthread_mutex_t lock;

typedef struct iree_hal_buffer_node_t {
  iree_hal_buffer_t* cache_data;
  struct iree_hal_buffer_node_t* next;
} iree_hal_buffer_node_t;

typedef struct iree_hal_caching_allocator_t {
  iree_hal_resource_t resource;
  iree_hal_allocator_t* delegate_allocator;
  iree_hal_buffer_node_t* cache_list;
} iree_hal_caching_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_caching_allocator_vtable;

iree_hal_caching_allocator_t* iree_hal_caching_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_caching_allocator_vtable);
  return (iree_hal_caching_allocator_t*)base_value;
}

IREE_API_EXPORT iree_hal_allocator_t* iree_hal_caching_allocator_get_delegate(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_caching_allocator_vtable);
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_value);
  return (iree_hal_allocator_t*)allocator->delegate_allocator;
}

static iree_status_t iree_hal_caching_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_create_caching(
    iree_hal_allocator_t* delegate_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_caching_allocator_t* allocator = NULL;
  iree_host_size_t total_size =
      iree_sizeof_struct(*allocator);
  iree_status_t status = iree_allocator_malloc(
      iree_hal_allocator_host_allocator(delegate_allocator), total_size,
      (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_caching_allocator_vtable,
                                 &allocator->resource);
    allocator->delegate_allocator = delegate_allocator;
    allocator->cache_list = NULL;
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_caching_allocator_remove_buffer_from_cache(
    iree_hal_caching_allocator_t* allocator, int buffer_index_to_remove) {
  iree_hal_buffer_node_t** cache_list = &(allocator->cache_list);
  iree_hal_buffer_node_t* cache_list_ptr = *cache_list;
  if (buffer_index_to_remove == 0) {
    *cache_list = cache_list_ptr->next;
    iree_allocator_free(
        iree_hal_allocator_host_allocator(allocator->delegate_allocator),
        cache_list_ptr);
    return;
  }
  for (int cur_idx = 0; cur_idx < buffer_index_to_remove - 1; cur_idx++) {
    cache_list_ptr = cache_list_ptr->next;
  }
  iree_hal_buffer_node_t* next = NULL;
  if (cache_list_ptr->next) next = cache_list_ptr->next->next;
  iree_allocator_free(
      iree_hal_allocator_host_allocator(allocator->delegate_allocator),
      cache_list_ptr->next);
  cache_list_ptr->next = next;
}

static bool iree_hal_caching_allocator_allocate_from_cache(
    iree_hal_caching_allocator_t* allocator, iree_host_size_t requested_size,
    iree_hal_buffer_t** out_buffer) {
  size_t buffer_index_in_cache = 0;
  iree_hal_buffer_node_t* cache_list_ptr = allocator->cache_list;
  while (cache_list_ptr) {
    if (cache_list_ptr->cache_data->allocation_size >= requested_size) {
      break;
    }
    cache_list_ptr = cache_list_ptr->next;
    buffer_index_in_cache++;
  }
  if (!cache_list_ptr) return false;
  // TODO(SWINATA): Implement Blocks splitting by generating subspan from
  // remaining buffer as new cache using byte_offset
  *out_buffer = cache_list_ptr->cache_data;
  iree_hal_caching_allocator_remove_buffer_from_cache(allocator,
                                                      buffer_index_in_cache);
  return true;
}

IREE_API_EXPORT iree_status_t
iree_hal_allocator_add_buffer_to_cache(iree_hal_buffer_t* buffer) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(buffer->device_allocator);
  iree_hal_buffer_node_t* cache_buffer_node = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_hal_allocator_host_allocator(allocator->delegate_allocator),
      iree_sizeof_struct(*cache_buffer_node), (void**)&cache_buffer_node);
  if (!iree_status_is_ok(status)) return status;
  cache_buffer_node->cache_data = buffer;
  cache_buffer_node->next = NULL;
  if (!allocator->cache_list) {
    allocator->cache_list = cache_buffer_node;
  } else {
    iree_hal_buffer_node_t* cache_list_ptr = allocator->cache_list;
    // Add buffer to the end of list
    if(cache_list_ptr->cache_data->allocation_size > cache_buffer_node->cache_data->allocation_size) {
      cache_buffer_node->next = cache_list_ptr;
      allocator->cache_list = cache_buffer_node;
      return iree_ok_status();
    }
    while (cache_list_ptr) {
      if (cache_list_ptr->next == NULL) break;
      cache_list_ptr = cache_list_ptr->next;
    }
    cache_list_ptr->next = cache_buffer_node;
  }
  return iree_ok_status();
}

static void iree_hal_caching_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  iree_hal_allocator_t* delegate_allocator = allocator->delegate_allocator;
  iree_hal_buffer_node_t* cache_list_ptr = allocator->cache_list;
  while (cache_list_ptr) {
    cache_list_ptr->cache_data->device_allocator = delegate_allocator;
    iree_hal_buffer_destroy(cache_list_ptr->cache_data);
    cache_list_ptr = cache_list_ptr->next;
  }
  iree_hal_allocator_destroy(allocator->delegate_allocator);
}

static iree_allocator_t iree_hal_caching_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  iree_hal_caching_allocator_t* allocator =
      (iree_hal_caching_allocator_t*)base_allocator;
  return iree_hal_allocator_host_allocator(allocator->delegate_allocator);
}

static void iree_hal_caching_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  iree_hal_allocator_query_statistics(allocator->delegate_allocator,
                                      out_statistics);
}

static iree_hal_buffer_compatibility_t
iree_hal_caching_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  return iree_hal_allocator_query_compatibility(allocator->delegate_allocator,
                                                *params, allocation_size);
}

static iree_status_t iree_hal_caching_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_host_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    if (iree_hal_caching_allocator_allocate_from_cache(
            allocator, allocation_size, out_buffer)) {
      return iree_ok_status();
    }
  }
  iree_status_t status;
  status = iree_hal_allocator_allocate_buffer(allocator->delegate_allocator,
                                              *params, allocation_size,
                                              initial_data, out_buffer);
  (*out_buffer)->device_allocator = (iree_hal_allocator_t*)allocator;
  return status;
}

static iree_status_t iree_hal_caching_allocator_wrap_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params, iree_byte_span_t data,
    iree_allocator_t data_allocator,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "wrapping of external buffers not supported");
}

static void iree_hal_caching_allocator_deallocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* base_buffer) {
  iree_hal_memory_type_t memory_type = iree_hal_buffer_memory_type(base_buffer);
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    iree_status_t status = iree_hal_allocator_add_buffer_to_cache(base_buffer);
    if (iree_status_is_ok(status)) {
      return;
    }
  }
  iree_hal_allocator_deallocate_buffer(allocator->delegate_allocator,
                                       base_buffer);
}

static iree_status_t iree_hal_caching_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "importing from external buffers not supported");
}

static iree_status_t iree_hal_caching_allocator_export_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "exporting to external buffers not supported");
}

static const iree_hal_allocator_vtable_t iree_hal_caching_allocator_vtable = {
    .destroy = iree_hal_caching_allocator_destroy,
    .host_allocator = iree_hal_caching_allocator_host_allocator,
    .trim = iree_hal_caching_allocator_trim,
    .query_statistics = iree_hal_caching_allocator_query_statistics,
    .query_compatibility =
        iree_hal_caching_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_caching_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_caching_allocator_deallocate_buffer,
    .import_buffer = iree_hal_caching_allocator_import_buffer,
    .export_buffer = iree_hal_caching_allocator_export_buffer,
};
