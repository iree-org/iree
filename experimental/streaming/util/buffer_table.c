// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/util/buffer_table.h"

#include "experimental/streaming/internal.h"

// Initial capacity for the buffer table.
#define IREE_HAL_STREAMING_BUFFER_TABLE_INITIAL_CAPACITY 256

struct iree_hal_streaming_buffer_table_t {
  // Array of buffer pointers.
  iree_hal_streaming_buffer_t** buffers;

  // Current number of buffers in the table.
  iree_host_size_t count;

  // Capacity of the buffers array.
  iree_host_size_t capacity;

  // Host allocator for table memory.
  iree_allocator_t host_allocator;
};

//===----------------------------------------------------------------------===//
// Internal abstracted operations
//===----------------------------------------------------------------------===//

// Calculates storage size needed for the given capacity.
static iree_host_size_t iree_hal_streaming_buffer_table_calculate_storage(
    iree_host_size_t capacity) {
  return capacity * sizeof(iree_hal_streaming_buffer_t*);
}

// Finds the index of a buffer containing the given pointer.
// The pointer may be a host or device pointer, and may be anywhere within
// the buffer's range. Returns count if not found.
static iree_host_size_t iree_hal_streaming_buffer_table_find_index(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_any_ptr_t any_ptr) {
  // Linear search through the table.
  // This is acceptable for typical buffer counts.
  // Can be replaced with binary search or hash lookup if needed.
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    iree_hal_streaming_buffer_t* buffer = table->buffers[i];
    if (!buffer) continue;

    // Check device pointer range first (98% case).
    if (any_ptr >= buffer->device_ptr &&
        any_ptr < buffer->device_ptr + buffer->size) {
      return i;
    }

    // Check host pointer range if available.
    if (buffer->host_ptr) {
      uint64_t host_addr = (uint64_t)(uintptr_t)buffer->host_ptr;
      if (any_ptr >= host_addr && any_ptr < host_addr + buffer->size) {
        return i;
      }
    }
  }
  return table->count;  // Not found.
}

// Grows the table capacity by 2x.
static iree_status_t iree_hal_streaming_buffer_table_grow(
    iree_hal_streaming_buffer_table_t* table) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t new_capacity =
      table->capacity ? table->capacity * 2
                      : IREE_HAL_STREAMING_BUFFER_TABLE_INITIAL_CAPACITY;
  const iree_host_size_t new_size =
      iree_hal_streaming_buffer_table_calculate_storage(new_capacity);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, new_capacity);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_realloc(table->host_allocator, new_size,
                             (void**)&table->buffers),
      "growing buffer table");

  table->capacity = new_capacity;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Inserts a buffer at the given index.
// Assumes the index is valid and there is capacity.
static void iree_hal_streaming_buffer_table_insert_at(
    iree_hal_streaming_buffer_table_t* table, iree_host_size_t index,
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT(index <= table->count);
  IREE_ASSERT(table->count < table->capacity);

  // Shift elements if inserting in the middle.
  if (index < table->count) {
    memmove(&table->buffers[index + 1], &table->buffers[index],
            (table->count - index) * sizeof(iree_hal_streaming_buffer_t*));
  }

  table->buffers[index] = buffer;
  table->count++;
}

// Erases a buffer at the given index.
// Assumes the index is valid.
static void iree_hal_streaming_buffer_table_erase_at(
    iree_hal_streaming_buffer_table_t* table, iree_host_size_t index) {
  IREE_ASSERT(index < table->count);

  // Shift elements if erasing from the middle.
  if (index < table->count - 1) {
    memmove(&table->buffers[index], &table->buffers[index + 1],
            (table->count - index - 1) * sizeof(iree_hal_streaming_buffer_t*));
  }

  table->count--;
}

//===----------------------------------------------------------------------===//
// iree_hal_streaming_buffer_table_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_buffer_table_allocate(
    iree_allocator_t host_allocator,
    iree_hal_streaming_buffer_table_t** out_table) {
  IREE_ASSERT_ARGUMENT(out_table);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_table = NULL;

  iree_hal_streaming_buffer_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*table), (void**)&table));

  memset(table, 0, sizeof(*table));
  table->host_allocator = host_allocator;

  *out_table = table;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_streaming_buffer_table_free(
    iree_hal_streaming_buffer_table_t* table) {
  if (!table) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Free the buffers array.
  if (table->buffers) {
    iree_allocator_free(table->host_allocator, table->buffers);
  }

  // Free the table itself.
  iree_allocator_t host_allocator = table->host_allocator;
  iree_allocator_free(host_allocator, table);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_streaming_buffer_table_insert(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if device pointer already exists.
  iree_host_size_t index =
      iree_hal_streaming_buffer_table_find_index(table, buffer->device_ptr);
  if (index < table->count) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                             "device pointer %p already registered",
                             (void*)buffer->device_ptr));
  }

  // Check if host pointer already exists (if assigned).
  if (buffer->host_ptr) {
    uint64_t host_addr = (uint64_t)(uintptr_t)buffer->host_ptr;
    index = iree_hal_streaming_buffer_table_find_index(table, host_addr);
    if (index < table->count) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                               "host pointer %p already registered",
                               buffer->host_ptr));
    }
  }

  // Grow if needed.
  if (table->count >= table->capacity) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_buffer_table_grow(table));
  }

  // TODO(benvanik): update find to return insertion point, insert there.
  // This would let us binary search. We may want a different data structure
  // entirely, though, so for now we insert at the end.
  iree_hal_streaming_buffer_table_insert_at(table, table->count, buffer);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_buffer_table_remove(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_any_ptr_t any_ptr) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t index =
      iree_hal_streaming_buffer_table_find_index(table, any_ptr);
  if (index >= table->count) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_NOT_FOUND,
                             "pointer %p not found in table", (void*)any_ptr));
  }

  // Remove from table.
  iree_hal_streaming_buffer_table_erase_at(table, index);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_buffer_table_lookup(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_any_ptr_t any_ptr,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  iree_host_size_t index =
      iree_hal_streaming_buffer_table_find_index(table, any_ptr);
  if (index >= table->count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "pointer %p not found in table", (void*)any_ptr);
  }

  *out_buffer = table->buffers[index];
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_buffer_table_lookup_range(
    iree_hal_streaming_buffer_table_t* table,
    iree_hal_streaming_any_ptr_t any_ptr, iree_device_size_t size,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range size must be greater than 0");
  }

  // Check for overflow.
  iree_hal_streaming_any_ptr_t range_end = any_ptr + size;
  if (range_end < any_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range [%p, %p) would overflow", (void*)any_ptr,
                            (void*)range_end);
  }

  // Linear search through all buffers to find one that contains the range.
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    iree_hal_streaming_buffer_t* buffer = table->buffers[i];
    if (!buffer) continue;

    // Check device pointer range first (98% case).
    iree_hal_streaming_deviceptr_t buffer_start = buffer->device_ptr;
    iree_hal_streaming_deviceptr_t buffer_end = buffer_start + buffer->size;
    if (any_ptr >= buffer_start && range_end <= buffer_end) {
      *out_buffer = buffer;
      return iree_ok_status();
    }

    // Check host pointer range if available.
    if (buffer->host_ptr) {
      uint64_t host_start = (uint64_t)(uintptr_t)buffer->host_ptr;
      uint64_t host_end = host_start + buffer->size;
      if (any_ptr >= host_start && range_end <= host_end) {
        *out_buffer = buffer;
        return iree_ok_status();
      }
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no buffer contains range [%p, %p)", (void*)any_ptr,
                          (void*)range_end);
}
