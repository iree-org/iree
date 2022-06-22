// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/atomic_slist.h"

#include <string.h>

#include "iree/base/attributes.h"

// TODO(benvanik): add TSAN annotations when switched to atomics:
// https://github.com/gcc-mirror/gcc/blob/master/libsanitizer/include/sanitizer/tsan_interface_atomic.h
// https://reviews.llvm.org/D18500

void iree_atomic_slist_initialize(iree_atomic_slist_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
}

void iree_atomic_slist_deinitialize(iree_atomic_slist_t* list) {
  // TODO(benvanik): assert empty.
  memset(list, 0, sizeof(*list));
}

void iree_atomic_slist_concat(iree_atomic_slist_t* list,
                              iree_atomic_slist_entry_t* head,
                              iree_atomic_slist_entry_t* tail) {
  if (IREE_UNLIKELY(!head)) return;
  while (true) {
    intptr_t list_head =
        iree_atomic_load_intptr(&list->head, iree_memory_order_relaxed);
    iree_atomic_store_intptr(&tail->next, list_head, iree_memory_order_release);
    if (iree_atomic_compare_exchange_weak_intptr(
            &list->head, &list_head, (intptr_t)head, iree_memory_order_release,
            iree_memory_order_relaxed)) {
      break;
    }
  }
}

void iree_atomic_slist_push(iree_atomic_slist_t* list,
                            iree_atomic_slist_entry_t* entry) {
  iree_atomic_slist_concat(list, entry, entry);
}

void iree_atomic_slist_push_unsafe(iree_atomic_slist_t* list,
                                   iree_atomic_slist_entry_t* entry) {
  intptr_t list_head =
      iree_atomic_load_intptr(&list->head, iree_memory_order_relaxed);
  iree_atomic_store_intptr(&entry->next, list_head, iree_memory_order_relaxed);
  iree_atomic_store_intptr(&list->head, (intptr_t)entry,
                           iree_memory_order_relaxed);
}

iree_atomic_slist_entry_t* iree_atomic_slist_pop(iree_atomic_slist_t* list) {
  // We are about to delete the old list head. Using memory_order_acquire here
  // ensures that read accesses to it can't be reordered past this.
  intptr_t list_head;
  while (true) {
    list_head = iree_atomic_load_intptr(&list->head, iree_memory_order_acquire);
    if (!list_head) return NULL;
    intptr_t list_head_next =
        iree_atomic_load_intptr(&((iree_atomic_slist_entry_t*)list_head)->next,
                                iree_memory_order_relaxed);
    if (iree_atomic_compare_exchange_weak_intptr(
            &list->head, &list_head, list_head_next, iree_memory_order_release,
            iree_memory_order_relaxed)) {
      break;
    }
  }
  iree_atomic_slist_entry_t* popped = (iree_atomic_slist_entry_t*)list_head;
  popped->next = 0;
  return popped;
}

bool iree_atomic_slist_flush(iree_atomic_slist_t* list,
                             iree_atomic_slist_flush_order_t flush_order,
                             iree_atomic_slist_entry_t** out_head,
                             iree_atomic_slist_entry_t** out_tail) {
  // Exchange list head with NULL to steal the entire list. The list will be in
  // the native LIFO order of the slist.
  intptr_t list_head;
  while (true) {
    list_head = iree_atomic_load_intptr(&list->head, iree_memory_order_acquire);
    if (!list_head) return false;
    if (iree_atomic_compare_exchange_weak_intptr(&list->head, &list_head, 0,
                                                 iree_memory_order_release,
                                                 iree_memory_order_relaxed)) {
      break;
    }
  }
  iree_atomic_slist_entry_t* head = (iree_atomic_slist_entry_t*)list_head;

  switch (flush_order) {
    case IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO: {
      // List is already in native LIFO order. If the user wants a tail we have
      // to scan for it, though, which we really only want to do when required
      // as it's a linked list pointer walk.
      *out_head = head;
      if (out_tail) {
        iree_atomic_slist_entry_t* p = head;
        while (p->next)
          p = (iree_atomic_slist_entry_t*)iree_atomic_load_intptr(
              &p->next, iree_memory_order_relaxed);
        *out_tail = p;
      }
      break;
    }
    case IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO: {
      // Reverse the list in a single scan. list_head is our tail, so scan
      // forward to find our head. Since we have to walk the whole list anyway
      // we can cheaply give both the head and tail to the caller.
      iree_atomic_slist_entry_t* tail = head;
      if (out_tail) *out_tail = tail;
      iree_atomic_slist_entry_t* p = head;
      do {
        iree_atomic_slist_entry_t* next =
            (iree_atomic_slist_entry_t*)iree_atomic_load_intptr(
                &p->next, iree_memory_order_relaxed);
        iree_atomic_store_intptr(&p->next, (intptr_t)head,
                                 iree_memory_order_relaxed);
        head = p;
        p = next;
      } while (p != NULL);
      iree_atomic_store_intptr(&tail->next, 0, iree_memory_order_relaxed);
      *out_head = head;
      break;
    }
    default:
      return false;
  }

  return true;
}
