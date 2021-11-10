// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/allocator_caching.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

typedef struct iree_hal_caching_allocator_t {
  iree_hal_resource_t resource;
  iree_hal_allocator_t* delegate_allocator;
  iree_string_view_t identifier;

} iree_hal_caching_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_caching_allocator_vtable;

iree_hal_caching_allocator_t* iree_hal_caching_allocator_cast(
    iree_hal_allocator_t* base_value) {
  return (iree_hal_caching_allocator_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_create_caching(
    iree_string_view_t identifier, iree_hal_allocator_t* delegate_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_cache_buffer = true;

  iree_hal_caching_allocator_t* allocator = NULL;
  iree_host_size_t total_size =
      iree_sizeof_struct(*allocator) + identifier.size;
  iree_status_t status = iree_allocator_malloc(
      iree_hal_allocator_host_allocator(delegate_allocator), total_size, (void**)&allocator);
  if (iree_status_is_ok(status)) {
    
    iree_hal_resource_initialize(&iree_hal_caching_allocator_vtable,
                                  &allocator->resource);

    allocator->delegate_allocator = delegate_allocator;

    iree_string_view_append_to_buffer(
        identifier, &allocator->identifier,
        (char*)allocator + iree_sizeof_struct(*allocator));
    
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

typedef struct iree_hal_caching_allocator_cache {
  iree_hal_buffer_t *buffer;
  struct iree_hal_caching_allocator_cache *next;
} iree_hal_caching_allocator_cache;

bool iree_hal_allocator_cache_buffer;

iree_hal_caching_allocator_cache *iree_hal_caching_allocator_cache_list = NULL;

static bool iree_hal_caching_allocator_buffer_available(iree_host_size_t requested_size) {
  iree_hal_caching_allocator_cache
    *iree_hal_caching_allocator_cache_ptr = iree_hal_caching_allocator_cache_list;
  while(iree_hal_caching_allocator_cache_ptr != NULL) {
    if(iree_hal_caching_allocator_cache_ptr->buffer->allocation_size >= requested_size) {
      return true;
    }
    iree_hal_caching_allocator_cache_ptr = iree_hal_caching_allocator_cache_ptr->next;
  }
  return false;
}

static void iree_hal_caching_allocator_remove_buffer_from_cache(int buffer_index_to_remove) {
  iree_hal_caching_allocator_cache *buffer_to_be_removed_from_cache,
    *iree_hal_caching_allocator_cache_ptr = iree_hal_caching_allocator_cache_list;
  if(buffer_index_to_remove == 0) {
    buffer_to_be_removed_from_cache = iree_hal_caching_allocator_cache_ptr;
    if(buffer_to_be_removed_from_cache->next == NULL) {
      iree_hal_caching_allocator_cache_list = NULL;
    } else {
      iree_hal_caching_allocator_cache_list = iree_hal_caching_allocator_cache_list->next;
    }
  } else {
    int current_buffer_index = 0;
    while(current_buffer_index < buffer_index_to_remove-1) {
      iree_hal_caching_allocator_cache_ptr = iree_hal_caching_allocator_cache_ptr->next;
      current_buffer_index++;
    }
    buffer_to_be_removed_from_cache = iree_hal_caching_allocator_cache_ptr->next;

    if(buffer_to_be_removed_from_cache->next == NULL) {
      iree_hal_caching_allocator_cache_ptr->next = NULL;
    } else {
      iree_hal_caching_allocator_cache_ptr->next = buffer_to_be_removed_from_cache->next;
    }
  }
  free(buffer_to_be_removed_from_cache);
  return;
}

static iree_status_t iree_hal_caching_allocator_allocate_from_cache(
    iree_host_size_t requested_size, iree_hal_buffer_t** out_buffer) {
  size_t buffer_index_in_cache = 0;
  iree_hal_caching_allocator_cache* 
    iree_hal_caching_allocator_cache_ptr = iree_hal_caching_allocator_cache_list;
  while(iree_hal_caching_allocator_cache_ptr != NULL) {
    if(iree_hal_caching_allocator_cache_ptr->buffer->allocation_size >= requested_size) {
      break;
    }
    iree_hal_caching_allocator_cache_ptr = iree_hal_caching_allocator_cache_ptr->next;
    buffer_index_in_cache++;
  }
  iree_hal_buffer_t *buffer_to_allocate = iree_hal_caching_allocator_cache_ptr->buffer;
  
  *out_buffer = buffer_to_allocate;
  iree_hal_caching_allocator_remove_buffer_from_cache(buffer_index_in_cache);

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_add_buffer_to_cache(iree_hal_buffer_t* buffer) {
  struct iree_hal_caching_allocator_cache* cache_buffer_node = 
    (iree_hal_caching_allocator_cache*) malloc(sizeof(iree_hal_caching_allocator_cache));
  cache_buffer_node->buffer = buffer;
  cache_buffer_node->next = NULL;

  if(iree_hal_caching_allocator_cache_list == NULL) {
    iree_hal_caching_allocator_cache_list = cache_buffer_node;
  } else {
    iree_hal_caching_allocator_cache
      *iree_hal_caching_allocator_cache_ptr = iree_hal_caching_allocator_cache_list;
    // Add buffer to the end of list
    while(iree_hal_caching_allocator_cache_ptr != NULL) {
      if(iree_hal_caching_allocator_cache_ptr->next == NULL)
        break;
      else
        iree_hal_caching_allocator_cache_ptr = iree_hal_caching_allocator_cache_ptr->next;
    }
    iree_hal_caching_allocator_cache_ptr->next = cache_buffer_node;
  }
  return iree_ok_status();
}

static void iree_hal_caching_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_allocator_cache_buffer = false;
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);
  iree_hal_allocator_destroy(allocator->delegate_allocator);
}

static iree_allocator_t iree_hal_caching_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  
  return iree_hal_allocator_host_allocator(base_allocator);
}

static void iree_hal_caching_allocator_query_statistics(
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);
  iree_hal_allocator_query_statistics(allocator->delegate_allocator, out_statistics);
}

static iree_hal_buffer_compatibility_t
iree_hal_caching_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage,
    iree_device_size_t allocation_size) {

  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);
  return iree_hal_allocator_query_buffer_compatibility(allocator->delegate_allocator,
    memory_type, allowed_usage, intended_usage, allocation_size);
}

static iree_status_t iree_hal_caching_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {

  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);

  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      if(iree_hal_caching_allocator_buffer_available(allocation_size)) {
          return iree_hal_caching_allocator_allocate_from_cache(allocation_size, out_buffer);
      }
    }
  }

  return iree_hal_allocator_allocate_buffer(allocator->delegate_allocator, memory_type,
      allowed_usage, allocation_size, out_buffer);
}

static iree_status_t iree_hal_caching_allocator_wrap_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);
  return iree_hal_allocator_wrap_buffer(allocator->delegate_allocator, memory_type,
      allowed_access, allowed_usage, data, data_allocator, out_buffer);
}

static const iree_hal_allocator_vtable_t iree_hal_caching_allocator_vtable = {
    .destroy = iree_hal_caching_allocator_destroy,
    .host_allocator = iree_hal_caching_allocator_host_allocator,
    .query_statistics = iree_hal_caching_allocator_query_statistics,
    .query_buffer_compatibility =
        iree_hal_caching_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_caching_allocator_allocate_buffer,
    .wrap_buffer = iree_hal_caching_allocator_wrap_buffer,
};
