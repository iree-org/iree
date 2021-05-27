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

#include "iree/base/internal/atomic_slist.h"

#include <assert.h>

// TODO(benvanik): add TSAN annotations when switched to atomics:
// https://github.com/gcc-mirror/gcc/blob/master/libsanitizer/include/sanitizer/tsan_interface_atomic.h
// https://reviews.llvm.org/D18500

void iree_atomic_slist_initialize(iree_atomic_slist_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  iree_slim_mutex_initialize(&out_list->mutex);
}

void iree_atomic_slist_deinitialize(iree_atomic_slist_t* list) {
  // TODO(benvanik): assert empty.
  iree_slim_mutex_deinitialize(&list->mutex);
  memset(list, 0, sizeof(*list));
}

void iree_atomic_slist_concat(iree_atomic_slist_t* list,
                              iree_atomic_slist_entry_t* head,
                              iree_atomic_slist_entry_t* tail) {
  if (IREE_UNLIKELY(!head)) return;
  iree_slim_mutex_lock(&list->mutex);
  tail->next = list->head;
  list->head = head;
  iree_slim_mutex_unlock(&list->mutex);
}

void iree_atomic_slist_push(iree_atomic_slist_t* list,
                            iree_atomic_slist_entry_t* entry) {
  iree_slim_mutex_lock(&list->mutex);
  iree_atomic_slist_push_unsafe(list, entry);
  iree_slim_mutex_unlock(&list->mutex);
}

void iree_atomic_slist_push_unsafe(iree_atomic_slist_t* list,
                                   iree_atomic_slist_entry_t* entry) {
  // NOTE: no lock is held here and no atomic operation will be used when this
  // is actually made atomic.
  entry->next = list->head;
  list->head = entry;
}

iree_atomic_slist_entry_t* iree_atomic_slist_pop(iree_atomic_slist_t* list) {
  iree_slim_mutex_lock(&list->mutex);
  iree_atomic_slist_entry_t* entry = list->head;
  if (entry != NULL) {
    list->head = entry->next;
    entry->next = NULL;
  }
  iree_slim_mutex_unlock(&list->mutex);
  return entry;
}

bool iree_atomic_slist_flush(iree_atomic_slist_t* list,
                             iree_atomic_slist_flush_order_t flush_order,
                             iree_atomic_slist_entry_t** out_head,
                             iree_atomic_slist_entry_t** out_tail) {
  // Exchange list head with NULL to steal the entire list. The list will be in
  // the native LIFO order of the slist.
  iree_slim_mutex_lock(&list->mutex);
  iree_atomic_slist_entry_t* head = list->head;
  list->head = NULL;
  iree_slim_mutex_unlock(&list->mutex);
  if (!head) return false;

  switch (flush_order) {
    case IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO: {
      // List is already in native LIFO order. If the user wants a tail we have
      // to scan for it, though, which we really only want to do when required
      // as it's a linked list pointer walk.
      *out_head = head;
      if (out_tail) {
        iree_atomic_slist_entry_t* p = head;
        while (p->next) p = p->next;
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
