// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/handle_table.h"

#include <string.h>

iree_status_t iree_hal_webgpu_handle_table_initialize(
    uint32_t initial_capacity, iree_allocator_t allocator,
    iree_hal_webgpu_handle_table_t* out_table) {
  IREE_ASSERT_ARGUMENT(out_table);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_table, 0, sizeof(*out_table));

  // Need at least 2 slots: index 0 (reserved null) + one usable slot.
  if (initial_capacity < 2) initial_capacity = 2;

  out_table->allocator = allocator;

  // Allocate the entries array (zero-initialized → all NULL).
  iree_status_t status = iree_allocator_malloc_array(
      allocator, initial_capacity, sizeof(void*), (void**)&out_table->entries);

  // Allocate the free stack array. Maximum possible free count is
  // initial_capacity - 1 (everything except index 0).
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, initial_capacity - 1,
                                         sizeof(uint32_t),
                                         (void**)&out_table->free_stack);
  }

  if (iree_status_is_ok(status)) {
    out_table->capacity = initial_capacity;
    out_table->high_water = 1;  // Index 0 is reserved.
  } else {
    // Clean up on partial failure.
    iree_allocator_free(allocator, out_table->entries);
    memset(out_table, 0, sizeof(*out_table));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_webgpu_handle_table_deinitialize(
    iree_hal_webgpu_handle_table_t* table) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(table->count == 0,
              "handle table still has %" PRIu32 " occupied entries",
              table->count);
  iree_allocator_free(table->allocator, table->free_stack);
  iree_allocator_free(table->allocator, table->entries);
  memset(table, 0, sizeof(*table));
  IREE_TRACE_ZONE_END(z0);
}

// Doubles the capacity of the entries and free_stack arrays.
static iree_status_t iree_hal_webgpu_handle_table_grow(
    iree_hal_webgpu_handle_table_t* table) {
  uint32_t new_capacity = table->capacity * 2;
  if (new_capacity < table->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "handle table capacity overflow");
  }

  // Grow entries array. iree_allocator_realloc takes the old pointer via
  // inout_ptr and returns the new pointer in the same location.
  void** entries = table->entries;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      table->allocator, new_capacity, sizeof(void*), (void**)&entries));
  table->entries = entries;

  // Zero-initialize the new portion.
  memset(&table->entries[table->capacity], 0,
         (iree_host_size_t)(new_capacity - table->capacity) * sizeof(void*));

  // Grow free stack array. New max free count is new_capacity - 1.
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      table->allocator, new_capacity - 1, sizeof(uint32_t),
      (void**)&table->free_stack));
  table->capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_handle_table_insert(
    iree_hal_webgpu_handle_table_t* table, void* object,
    iree_hal_webgpu_handle_t* out_handle) {
  IREE_ASSERT_ARGUMENT(object);
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = IREE_HAL_WEBGPU_HANDLE_NULL;

  uint32_t index;
  if (table->free_count > 0) {
    // Reuse a previously freed slot.
    index = table->free_stack[--table->free_count];
  } else if (table->high_water < table->capacity) {
    // Allocate from the high-water region.
    index = table->high_water++;
  } else {
    // Table is full — grow and allocate from high-water.
    IREE_RETURN_IF_ERROR(iree_hal_webgpu_handle_table_grow(table));
    index = table->high_water++;
  }

  IREE_ASSERT(index > 0 && index < table->capacity);
  IREE_ASSERT(table->entries[index] == NULL);
  table->entries[index] = object;
  table->count++;
  *out_handle = (iree_hal_webgpu_handle_t)index;
  return iree_ok_status();
}

void* iree_hal_webgpu_handle_table_get(
    const iree_hal_webgpu_handle_table_t* table,
    iree_hal_webgpu_handle_t handle) {
  if (handle == IREE_HAL_WEBGPU_HANDLE_NULL || handle >= table->capacity) {
    return NULL;
  }
  return table->entries[handle];
}

void* iree_hal_webgpu_handle_table_remove(iree_hal_webgpu_handle_table_t* table,
                                          iree_hal_webgpu_handle_t handle) {
  IREE_ASSERT(handle != IREE_HAL_WEBGPU_HANDLE_NULL && handle < table->capacity,
              "removing invalid handle %" PRIu32, handle);
  void* object = table->entries[handle];
  IREE_ASSERT(object != NULL, "removing already-free handle %" PRIu32, handle);
  table->entries[handle] = NULL;
  table->free_stack[table->free_count++] = handle;
  table->count--;
  return object;
}
