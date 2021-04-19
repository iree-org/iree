// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/detail.h"

typedef struct iree_hal_heap_allocator_s {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;
} iree_hal_heap_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_heap_allocator_vtable;

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_create_heap(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_heap_allocator_t* allocator = NULL;
  iree_host_size_t total_size = sizeof(*allocator) + identifier.size;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_heap_allocator_vtable,
                                 &allocator->resource);
    allocator->host_allocator = host_allocator;
    iree_string_view_append_to_buffer(
        identifier, &allocator->identifier,
        (char*)allocator + total_size - identifier.size);
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_heap_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_heap_allocator_t* allocator =
      (iree_hal_heap_allocator_t*)base_allocator;
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_heap_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  iree_hal_heap_allocator_t* allocator =
      (iree_hal_heap_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_hal_buffer_compatibility_t
iree_hal_heap_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage,
    iree_device_size_t allocation_size) {
  // Disallow usage not permitted by the buffer itself. Since we then use this
  // to determine compatibility below we'll naturally set the right compat flags
  // based on what's both allowed and intended.
  intended_usage &= allowed_usage;

  // All buffers can be allocated on the heap and all heap-accessible buffers
  // can be imported.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
      IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE;

  // Buffers can only be used on the queue if they are device visible.
  // This is not a strict requirement of heap buffers but matches devices that
  // have discrete memory spaces (remoting/sandboxed, GPUs, etc) and makes it
  // much easier to find issues of buffer definition with local devices that
  // will cause issues when used with real devices.
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  return compatibility;
}

static iree_status_t iree_hal_heap_allocator_make_compatible(
    iree_hal_memory_type_t* memory_type,
    iree_hal_memory_access_t* allowed_access,
    iree_hal_buffer_usage_t* allowed_usage) {
  // Always ensure we are host-visible.
  *memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;

  // Host currently uses mapping to copy buffers, which is done a lot.
  // We could probably remove this mutation by preventing copies in those cases.
  *allowed_usage |= IREE_HAL_BUFFER_USAGE_MAPPING;

  // TODO(benvanik): check if transfer is still required for DMA copy source.
  *allowed_usage |= IREE_HAL_BUFFER_USAGE_TRANSFER;

  return iree_ok_status();
}

static iree_status_t iree_hal_heap_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_heap_allocator_t* allocator =
      (iree_hal_heap_allocator_t*)base_allocator;

  // Coerce options into those required for use by heap-based devices.
  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_ALL;
  IREE_RETURN_IF_ERROR(iree_hal_heap_allocator_make_compatible(
      &memory_type, &allowed_access, &allowed_usage));

  iree_byte_span_t data = iree_make_byte_span(NULL, allocation_size);
  if (allocation_size > 0) {
    // Zero-length buffers are valid but we don't want to try to malloc them.
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        allocator->host_allocator, allocation_size, (void**)&data.data));
  }
  iree_status_t status = iree_hal_heap_buffer_wrap(
      base_allocator, memory_type, allowed_access, allowed_usage,
      allocation_size, data, allocator->host_allocator, out_buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator->host_allocator, data.data);
  }
  return status;
}

static iree_status_t iree_hal_heap_allocator_wrap_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  // Coerce options into those required for use by heap-based devices.
  IREE_RETURN_IF_ERROR(iree_hal_heap_allocator_make_compatible(
      &memory_type, &allowed_access, &allowed_usage));

  return iree_hal_heap_buffer_wrap(base_allocator, memory_type, allowed_access,
                                   allowed_usage, data.data_length, data,
                                   data_allocator, out_buffer);
}

static const iree_hal_allocator_vtable_t iree_hal_heap_allocator_vtable = {
    .destroy = iree_hal_heap_allocator_destroy,
    .host_allocator = iree_hal_heap_allocator_host_allocator,
    .query_buffer_compatibility =
        iree_hal_heap_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_heap_allocator_allocate_buffer,
    .wrap_buffer = iree_hal_heap_allocator_wrap_buffer,
};
