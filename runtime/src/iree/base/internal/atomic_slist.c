// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/atomic_slist.h"

#include <string.h>

#include "iree/base/attributes.h"

// Loads the head pointer with the given memory ordering.
static inline iree_atomic_slist_entry_t* iree_atomic_slist_load_head(
    iree_atomic_slist_t* list, iree_memory_order_t order) {
  return (iree_atomic_slist_entry_t*)iree_atomic_load(&list->head, order);
}

// Stores the head pointer with release ordering (publishes prior writes).
static inline void iree_atomic_slist_store_head(
    iree_atomic_slist_t* list, iree_atomic_slist_entry_t* value) {
  iree_atomic_store(&list->head, (intptr_t)value, iree_memory_order_release);
}

void iree_atomic_slist_initialize(iree_atomic_slist_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  iree_slim_mutex_initialize(&out_list->mutex);
}

void iree_atomic_slist_deinitialize(iree_atomic_slist_t* list) {
  iree_slim_mutex_deinitialize(&list->mutex);
  memset(list, 0, sizeof(*list));
}

void iree_atomic_slist_concat(iree_atomic_slist_t* list,
                              iree_atomic_slist_entry_t* head,
                              iree_atomic_slist_entry_t* tail) {
  if (IREE_UNLIKELY(!head)) return;
  iree_slim_mutex_lock(&list->mutex);
  tail->next = iree_atomic_slist_load_head(list, iree_memory_order_relaxed);
  iree_atomic_slist_store_head(list, head);
  iree_slim_mutex_unlock(&list->mutex);
}

void iree_atomic_slist_push(iree_atomic_slist_t* list,
                            iree_atomic_slist_entry_t* entry) {
  iree_slim_mutex_lock(&list->mutex);
  entry->next = iree_atomic_slist_load_head(list, iree_memory_order_relaxed);
  iree_atomic_slist_store_head(list, entry);
  iree_slim_mutex_unlock(&list->mutex);
}

void iree_atomic_slist_push_unsafe(iree_atomic_slist_t* list,
                                   iree_atomic_slist_entry_t* entry) {
  entry->next = iree_atomic_slist_load_head(list, iree_memory_order_relaxed);
  iree_atomic_slist_store_head(list, entry);
}

void iree_atomic_slist_discard(iree_atomic_slist_t* list) {
  iree_slim_mutex_lock(&list->mutex);
  iree_atomic_slist_store_head(list, NULL);
  iree_slim_mutex_unlock(&list->mutex);
}

iree_atomic_slist_entry_t* iree_atomic_slist_pop(iree_atomic_slist_t* list) {
  // Fast path: check if the list is empty without taking the mutex.
  // The relaxed load is safe because we re-check under the mutex if non-NULL.
  // False negatives (read NULL when an entry was just pushed) are benign:
  // the entry will be seen on the next pop call.
  if (!iree_atomic_slist_load_head(list, iree_memory_order_relaxed)) {
    return NULL;
  }
  iree_slim_mutex_lock(&list->mutex);
  iree_atomic_slist_entry_t* entry =
      iree_atomic_slist_load_head(list, iree_memory_order_relaxed);
  if (entry != NULL) {
    iree_atomic_slist_store_head(list, entry->next);
    entry->next = NULL;
  }
  iree_slim_mutex_unlock(&list->mutex);
  return entry;
}

bool iree_atomic_slist_flush(iree_atomic_slist_t* list,
                             iree_atomic_slist_flush_order_t flush_order,
                             iree_atomic_slist_entry_t** out_head,
                             iree_atomic_slist_entry_t** out_tail) {
  // Fast path: check if the list is empty without taking the mutex.
  if (!iree_atomic_slist_load_head(list, iree_memory_order_relaxed)) {
    return false;
  }
  iree_slim_mutex_lock(&list->mutex);
  iree_atomic_slist_entry_t* head =
      iree_atomic_slist_load_head(list, iree_memory_order_relaxed);
  iree_atomic_slist_store_head(list, NULL);
  iree_slim_mutex_unlock(&list->mutex);
  if (!head) return false;

  switch (flush_order) {
    case IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO: {
      *out_head = head;
      if (out_tail) {
        iree_atomic_slist_entry_t* p = head;
        while (p->next) p = p->next;
        *out_tail = p;
      }
      break;
    }
    case IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO: {
      iree_atomic_slist_entry_t* tail = head;
      if (out_tail) *out_tail = tail;
      iree_atomic_slist_entry_t* p = head;
      do {
        iree_atomic_slist_entry_t* next = p->next;
        p->next = head;
        head = p;
        p = next;
      } while (p != NULL);
      tail->next = NULL;
      *out_head = head;
      break;
    }
    default:
      return false;
  }

  return true;
}
