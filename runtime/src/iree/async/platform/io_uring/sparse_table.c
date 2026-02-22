// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/sparse_table.h"

#include <string.h>

iree_status_t iree_io_uring_sparse_table_allocate(
    uint16_t capacity, iree_allocator_t allocator,
    iree_io_uring_sparse_table_t** out_table) {
  IREE_ASSERT_ARGUMENT(out_table);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_table = NULL;

  if (capacity == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sparse table capacity must be > 0");
  }

  iree_host_size_t word_count = iree_bitmap_calculate_words(capacity);
  iree_host_size_t total_size =
      sizeof(iree_io_uring_sparse_table_t) + word_count * sizeof(uint64_t);

  iree_io_uring_sparse_table_t* table = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&table));
  memset(table, 0, total_size);

  iree_slim_mutex_initialize(&table->mutex);
  table->bitmap.bit_count = capacity;
  table->bitmap.words = (uint64_t*)(table + 1);

  *out_table = table;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_io_uring_sparse_table_free(iree_io_uring_sparse_table_t* table,
                                     iree_allocator_t allocator) {
  if (!table) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_deinitialize(&table->mutex);
  iree_allocator_free(allocator, table);
  IREE_TRACE_ZONE_END(z0);
}

void iree_io_uring_sparse_table_lock(iree_io_uring_sparse_table_t* table) {
  iree_slim_mutex_lock(&table->mutex);
}

void iree_io_uring_sparse_table_unlock(iree_io_uring_sparse_table_t* table) {
  iree_slim_mutex_unlock(&table->mutex);
}

int32_t iree_io_uring_sparse_table_acquire(iree_io_uring_sparse_table_t* table,
                                           uint16_t count) {
  if (count == 0) return -1;
  iree_host_size_t start =
      iree_bitmap_find_first_unset_span(table->bitmap, 0, count);
  if (start >= table->bitmap.bit_count) return -1;
  iree_bitmap_set_span(table->bitmap, start, count);
  return (int32_t)start;
}

void iree_io_uring_sparse_table_release(iree_io_uring_sparse_table_t* table,
                                        uint16_t start, uint16_t count) {
  IREE_ASSERT(start + count <= table->bitmap.bit_count);
  iree_bitmap_reset_span(table->bitmap, start, count);
}
