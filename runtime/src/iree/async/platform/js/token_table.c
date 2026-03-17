// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/js/token_table.h"

#include <stddef.h>
#include <string.h>

#include "iree/async/operation.h"

// Verify the completion entry layout matches the JS TypedArray ABI.
static_assert(sizeof(iree_async_js_completion_entry_t) == 8,
              "completion entry must be 8 bytes for JS interop");
static_assert(offsetof(iree_async_js_completion_entry_t, token) == 0,
              "token must be at offset 0");
static_assert(offsetof(iree_async_js_completion_entry_t, status_code) == 4,
              "status_code must be at offset 4");

iree_status_t iree_async_js_token_table_initialize(
    iree_host_size_t capacity, iree_allocator_t allocator,
    iree_async_js_token_table_t* out_table) {
  IREE_ASSERT_ARGUMENT(out_table);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_table, 0, sizeof(*out_table));

  if (capacity == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "token table capacity must be > 0");
  }

  out_table->allocator = allocator;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, capacity, sizeof(iree_async_operation_t*),
      (void**)&out_table->entries);
  if (iree_status_is_ok(status)) {
    out_table->capacity = capacity;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_js_token_table_deinitialize(
    iree_async_js_token_table_t* table) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(table->count == 0,
              "token table still has %" PRIhsz " occupied slots", table->count);
  iree_allocator_free(table->allocator, table->entries);
  memset(table, 0, sizeof(*table));
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_async_js_token_table_acquire(
    iree_async_js_token_table_t* table, iree_async_operation_t* operation,
    uint32_t* out_token) {
  IREE_ASSERT_ARGUMENT(operation);
  IREE_ASSERT_ARGUMENT(out_token);

  if (table->count >= table->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "token table full (capacity=%" PRIhsz ")",
                            table->capacity);
  }

  iree_host_size_t index = table->next_token % table->capacity;
  if (table->entries[index] != NULL) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "token table slot %" PRIhsz
                            " occupied (next_token=%" PRIu32
                            ", capacity=%" PRIhsz ")",
                            index, table->next_token, table->capacity);
  }

  table->entries[index] = operation;
  *out_token = table->next_token;
  table->next_token++;
  table->count++;
  return iree_ok_status();
}

iree_async_operation_t* iree_async_js_token_table_lookup(
    const iree_async_js_token_table_t* table, uint32_t token) {
  iree_host_size_t index = token % table->capacity;
  return table->entries[index];
}

void iree_async_js_token_table_release(iree_async_js_token_table_t* table,
                                       uint32_t token) {
  iree_host_size_t index = token % table->capacity;
  IREE_ASSERT(table->entries[index] != NULL,
              "releasing already-free token table slot %" PRIhsz, index);
  table->entries[index] = NULL;
  table->count--;
}
