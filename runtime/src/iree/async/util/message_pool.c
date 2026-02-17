// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/message_pool.h"

#include <string.h>

void iree_async_message_pool_initialize(
    iree_host_size_t capacity, iree_async_message_pool_entry_t* entries,
    iree_async_message_pool_t* out_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->capacity = capacity;

  iree_atomic_slist_initialize(&out_pool->free_list);
  iree_atomic_slist_initialize(&out_pool->pending_list);
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    memset(&entries[i], 0, sizeof(entries[i]));
    iree_atomic_slist_push(&out_pool->free_list, &entries[i].slist_entry);
  }
  IREE_TRACE_ZONE_END(z0);
}

void iree_async_message_pool_deinitialize(iree_async_message_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_atomic_slist_deinitialize(&pool->free_list);
  iree_atomic_slist_deinitialize(&pool->pending_list);
  memset(pool, 0, sizeof(*pool));
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_async_message_pool_send(iree_async_message_pool_t* pool,
                                           uint64_t message_data) {
  iree_atomic_slist_entry_t* slist_entry =
      iree_atomic_slist_pop(&pool->free_list);
  if (!slist_entry) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "message pool exhausted (capacity=%" PRIhsz ")",
                            pool->capacity);
  }

  iree_async_message_pool_entry_t* entry =
      (iree_async_message_pool_entry_t*)slist_entry;
  entry->message_data = message_data;

  iree_atomic_slist_push(&pool->pending_list, &entry->slist_entry);
  return iree_ok_status();
}

iree_async_message_pool_entry_t* iree_async_message_pool_flush(
    iree_async_message_pool_t* pool) {
  iree_atomic_slist_entry_t* head = NULL;
  iree_atomic_slist_entry_t* tail = NULL;
  if (!iree_atomic_slist_flush(&pool->pending_list,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                               &head, &tail)) {
    return NULL;
  }
  return (iree_async_message_pool_entry_t*)head;
}

void iree_async_message_pool_release(iree_async_message_pool_t* pool,
                                     iree_async_message_pool_entry_t* entry) {
  entry->message_data = 0;
  iree_atomic_slist_push(&pool->free_list, &entry->slist_entry);
}
