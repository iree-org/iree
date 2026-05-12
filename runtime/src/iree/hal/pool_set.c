// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/pool_set.h"

#include <string.h>

typedef struct iree_hal_pool_set_entry_t {
  // Retained pool reference selected when |capabilities| match.
  iree_hal_pool_t* pool;

  // Selection priority. Higher values win when multiple pools match.
  int32_t priority;

  // Cached pool capabilities captured at registration time.
  iree_hal_pool_capabilities_t capabilities;
} iree_hal_pool_set_entry_t;

static iree_hal_memory_type_t iree_hal_pool_set_required_memory_type(
    iree_hal_memory_type_t memory_type) {
  return memory_type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
}

static void iree_hal_pool_set_apply_optimal_memory_type(
    const iree_hal_pool_capabilities_t* capabilities,
    iree_hal_buffer_params_t* params) {
  if (iree_any_bit_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
    params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
    params->type |= capabilities->memory_type;
  }
}

iree_status_t iree_hal_pool_set_initialize(iree_host_size_t initial_capacity,
                                           iree_allocator_t host_allocator,
                                           iree_hal_pool_set_t* out_pool_set) {
  IREE_ASSERT_ARGUMENT(out_pool_set);
  memset(out_pool_set, 0, sizeof(*out_pool_set));
  IREE_TRACE_ZONE_BEGIN(z0);

  out_pool_set->host_allocator = host_allocator;
  if (initial_capacity == 0) {
    initial_capacity = 8;
  }
  iree_hal_pool_set_entry_t* entries = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, initial_capacity * sizeof(*entries), (void**)&entries);
  if (iree_status_is_ok(status)) {
    out_pool_set->entries = entries;
    out_pool_set->entry_capacity = initial_capacity;
    out_pool_set->entry_count = 0;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_pool_set_deinitialize(iree_hal_pool_set_t* pool_set) {
  if (!pool_set) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < pool_set->entry_count; ++i) {
    iree_hal_pool_release(pool_set->entries[i].pool);
  }
  iree_allocator_free(pool_set->host_allocator, pool_set->entries);
  memset(pool_set, 0, sizeof(*pool_set));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_pool_set_register(iree_hal_pool_set_t* pool_set,
                                         int32_t priority,
                                         iree_hal_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool_set);
  IREE_ASSERT_ARGUMENT(pool);

  // Grow the entry array if needed.
  if (pool_set->entry_count >= pool_set->entry_capacity) {
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "grow");
    iree_status_t status = iree_allocator_grow_array(
        pool_set->host_allocator, pool_set->entry_count + 1,
        sizeof(iree_hal_pool_set_entry_t), &pool_set->entry_capacity,
        (void**)&pool_set->entries);
    IREE_TRACE_ZONE_END(z0);
    IREE_RETURN_IF_ERROR(status);
  }

  iree_host_size_t insert_index = 0;
  while (insert_index < pool_set->entry_count &&
         pool_set->entries[insert_index].priority >= priority) {
    ++insert_index;
  }
  if (insert_index < pool_set->entry_count) {
    memmove(&pool_set->entries[insert_index + 1],
            &pool_set->entries[insert_index],
            (pool_set->entry_count - insert_index) *
                sizeof(iree_hal_pool_set_entry_t));
  }

  iree_hal_pool_set_entry_t* entry = &pool_set->entries[insert_index];
  iree_hal_pool_retain(pool);
  entry->pool = pool;
  entry->priority = priority;
  iree_hal_pool_query_capabilities(pool, &entry->capabilities);
  ++pool_set->entry_count;
  return iree_ok_status();
}

iree_hal_pool_t* iree_hal_pool_set_select(const iree_hal_pool_set_t* pool_set,
                                          iree_hal_buffer_params_t params,
                                          iree_device_size_t allocation_size) {
  IREE_ASSERT_ARGUMENT(pool_set);
  iree_hal_buffer_params_canonicalize(&params);
  const iree_hal_memory_type_t required_type =
      iree_hal_pool_set_required_memory_type(params.type);

  for (iree_host_size_t i = 0; i < pool_set->entry_count; ++i) {
    const iree_hal_pool_set_entry_t* entry = &pool_set->entries[i];
    const iree_hal_pool_capabilities_t* capabilities = &entry->capabilities;

    // Memory type: pool must provide at least the required type bits.
    if ((capabilities->memory_type & required_type) != required_type) {
      continue;
    }

    // Usage: pool must support at least the required usage bits.
    if ((capabilities->supported_usage & params.usage) != params.usage) {
      continue;
    }

    // Size: allocation must fit within pool's constraints.
    if (capabilities->min_allocation_size > 0 &&
        allocation_size < capabilities->min_allocation_size) {
      continue;
    }
    if (capabilities->max_allocation_size > 0 &&
        allocation_size > capabilities->max_allocation_size) {
      continue;
    }

    return entry->pool;
  }

  return NULL;
}

iree_status_t iree_hal_pool_set_allocate_buffer(
    iree_hal_pool_set_t* pool_set, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    const iree_async_frontier_t* requester_frontier, iree_timeout_t timeout,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(pool_set);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  iree_hal_pool_t* pool =
      iree_hal_pool_set_select(pool_set, params, allocation_size);
  if (!pool) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no registered pool can satisfy allocation of %" PRIdsz
        " bytes with the requested buffer parameters",
        allocation_size);
  }
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  iree_hal_pool_set_apply_optimal_memory_type(&capabilities, &params);
  return iree_hal_pool_allocate_buffer(pool, params, allocation_size,
                                       requester_frontier, timeout, out_buffer);
}
