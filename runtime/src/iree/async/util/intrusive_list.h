// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Intrusive doubly-linked list for O(1) insertion and removal.
//
// Usage:
//   1. Embed iree_intrusive_list_entry_t in your struct
//   2. Use iree_intrusive_list_push_front() to add entries
//   3. Use iree_intrusive_list_remove() to remove entries
//
// Example:
//   typedef struct my_resource_t {
//     iree_intrusive_list_entry_t list_entry;  // Must be first for easy cast.
//     int data;
//   } my_resource_t;
//
//   iree_intrusive_list_t list = iree_intrusive_list_empty();
//   my_resource_t* r = allocate_resource();
//   iree_intrusive_list_entry_initialize(&r->list_entry);
//   iree_intrusive_list_push_front(&list, &r->list_entry);
//   // Later:
//   iree_intrusive_list_remove(&list, &r->list_entry);
//
// Thread-safety: None. Caller must synchronize access.

#ifndef IREE_ASYNC_UTIL_INTRUSIVE_LIST_H_
#define IREE_ASYNC_UTIL_INTRUSIVE_LIST_H_

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// List entry (embed in your struct)
//===----------------------------------------------------------------------===//

// Intrusive list entry. Embed in your struct, typically as the first member
// for easy casting.
typedef struct iree_intrusive_list_entry_t {
  struct iree_intrusive_list_entry_t* next;
  struct iree_intrusive_list_entry_t* prev;
} iree_intrusive_list_entry_t;

// Initializes a list entry as unlinked.
static inline void iree_intrusive_list_entry_initialize(
    iree_intrusive_list_entry_t* entry) {
  entry->next = NULL;
  entry->prev = NULL;
}

// Returns true if the entry is currently linked into a list.
static inline bool iree_intrusive_list_entry_is_linked(
    const iree_intrusive_list_entry_t* entry) {
  // An entry is linked if either pointer is non-NULL.
  // This works because remove() clears both pointers.
  return entry->next != NULL || entry->prev != NULL;
}

//===----------------------------------------------------------------------===//
// List head
//===----------------------------------------------------------------------===//

// Head of an intrusive doubly-linked list.
typedef struct iree_intrusive_list_t {
  iree_intrusive_list_entry_t* head;
} iree_intrusive_list_t;

// Returns an empty list.
static inline iree_intrusive_list_t iree_intrusive_list_empty(void) {
  iree_intrusive_list_t list = {NULL};
  return list;
}

// Returns true if the list is empty.
static inline bool iree_intrusive_list_is_empty(
    const iree_intrusive_list_t* list) {
  return list->head == NULL;
}

//===----------------------------------------------------------------------===//
// List operations
//===----------------------------------------------------------------------===//

// Pushes an entry at the front of the list.
// The entry must not already be linked.
static inline void iree_intrusive_list_push_front(
    iree_intrusive_list_t* list, iree_intrusive_list_entry_t* entry) {
  entry->prev = NULL;
  entry->next = list->head;
  if (list->head) {
    list->head->prev = entry;
  }
  list->head = entry;
}

// Removes an entry from the list.
// Safe to call on an unlinked entry (no-op).
// The entry is unlinked after this call (both pointers cleared).
static inline void iree_intrusive_list_remove(
    iree_intrusive_list_t* list, iree_intrusive_list_entry_t* entry) {
  if (entry->prev) {
    entry->prev->next = entry->next;
  } else if (list->head == entry) {
    // Entry is at head.
    list->head = entry->next;
  }
  if (entry->next) {
    entry->next->prev = entry->prev;
  }
  // Clear pointers to mark as unlinked.
  entry->next = NULL;
  entry->prev = NULL;
}

//===----------------------------------------------------------------------===//
// Iteration macros
//===----------------------------------------------------------------------===//

// Iterates over list entries.
// |list| is a pointer to iree_intrusive_list_t.
// |entry| is the loop variable (iree_intrusive_list_entry_t*).
#define IREE_INTRUSIVE_LIST_FOR_EACH(list, entry) \
  for ((entry) = (list)->head; (entry) != NULL; (entry) = (entry)->next)

// Iterates over list entries with safe removal.
// |list| is a pointer to iree_intrusive_list_t.
// |entry| is the loop variable (iree_intrusive_list_entry_t*).
// |temp| is a temporary variable (iree_intrusive_list_entry_t*).
//
// The current entry may be removed during iteration.
#define IREE_INTRUSIVE_LIST_FOR_EACH_SAFE(list, entry, temp)            \
  for ((entry) = (list)->head, (temp) = (entry) ? (entry)->next : NULL; \
       (entry) != NULL;                                                 \
       (entry) = (temp), (temp) = (entry) ? (entry)->next : NULL)

// Helper to get the containing struct from a list entry.
// |entry| is a pointer to iree_intrusive_list_entry_t.
// |type| is the containing struct type.
// |member| is the name of the list_entry field in the struct.
#define IREE_INTRUSIVE_LIST_CONTAINER(entry, type, member) \
  ((type*)((char*)(entry) - offsetof(type, member)))

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_INTRUSIVE_LIST_H_
