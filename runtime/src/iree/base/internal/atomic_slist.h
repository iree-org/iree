// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: the best kind of synchronization is no synchronization; always try to
// design your algorithm so that you don't need anything from this file :)
// See https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html

#ifndef IREE_BASE_INTERNAL_ATOMIC_SLIST_H_
#define IREE_BASE_INTERNAL_ATOMIC_SLIST_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"

#ifdef __cplusplus
extern "C" {
#endif

// The embedded pointer to the next entry in the slist. This points to the
// internal iree_atomic_slist_entry_t, *not* the user-provided pointer.
typedef void* iree_atomic_slist_intrusive_ptr_t;

// DO NOT USE: implementation detail.
typedef struct iree_atomic_slist_entry_t {
  struct iree_atomic_slist_entry_t* next;
} iree_atomic_slist_entry_t;

// Lightweight contention-avoiding singly linked list.
// This models optimistically-ordered LIFO behavior (stack push/pop) using
// atomic primitives.
//
//           ***************************************************
//           ******** ONLY APPROXIMATE ORDER GUARANTEES ********
//           ***************************************************
//
// This makes it extremely efficient for when only eventual consistency across
// producers and consumers is required. The most common example is free lists
// where all that matters is that entries make it into the list and not that
// they have any particular order between them. Work queues where all tasks
// within the queue are able to execute in any order like with wavefront-style
// scheduling can also benefit from this relaxed behavior.
//
// If a strict ordering is required this can be used as a primitive to construct
// a flat-combining data structure where data structure change requests are
// published to this list and a combiner is chosen to land the published data in
// an appropriate order:
// http://people.csail.mit.edu/shanir/publications/Flat%20Combining%20SPAA%2010.pdf
//
// There's often still a benefit in unordered scenarios of having LIFO behavior
// as it promotes cache-friendly small linked lists when there is a small number
// of producers and consumers (1:1 is the best case), though as the producer and
// consumer count increases the LIFO behavior can pessimize performance as there
// is more contention for the list head pointer. Prefer to shard across multiple
// per-core/thread lists and use techniques like flat-combining for the
// cross-core/thread aggregation/sequencing.
//
// This API modeled roughly on the Windows SList type:
// https://docs.microsoft.com/en-us/windows/win32/sync/interlocked-singly-linked-lists
// which is roughly compatible with the Apple OSAtomic queue:
// https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/OSAtomicEnqueue.3.html
// https://opensource.apple.com/source/libplatform/libplatform-125/include/libkern/OSAtomicQueue.h.auto.html
//
// Usage:
// https://docs.microsoft.com/en-us/windows/win32/sync/using-singly-linked-lists
//
// WARNING: this is an extremely sharp pufferfish-esque API. Don't use it. 🐡
//
// TODO(benvanik): verify behavior (and worthwhileness) of supporting platform
// primitives. The benefit of something like OSAtomicEnqueue/Dequeue is that it
// may have better tooling (TSAN), special intrinsic handling in the compiler,
// etc. That said, the Windows Interlocked* variants don't seem to. Having a
// single heavily tested implementation seems more worthwhile than several.
typedef iree_alignas(iree_max_align_t) struct {
  // TODO(benvanik): spend some time golfing this. Unblocking myself for now :)
  iree_slim_mutex_t mutex;
  iree_atomic_slist_entry_t* head;
} iree_atomic_slist_t;

// Initializes an slist handle to an empty list.
// Lists must be flushed to empty and deinitialized when no longer needed with
// iree_atomic_slist_deinitialize.
//
// NOTE: not thread-safe; existing |out_list| contents are discarded.
void iree_atomic_slist_initialize(iree_atomic_slist_t* out_list);

// Deinitializes an slist.
// The list must be empty; callers are expected to flush the list from the same
// thread making this call when it is guaranteed no other thread may be trying
// to use the list.
//
// NOTE: not thread-safe; |list| must not be used by any other thread.
void iree_atomic_slist_deinitialize(iree_atomic_slist_t* list);

// Concatenates a span of entries into the list in the order they are provided.
//
// Example:
//   existing slist: C B A
//    provided span: 1 2 3
//  resulting slist: 1 2 3 C B A
void iree_atomic_slist_concat(iree_atomic_slist_t* list,
                              iree_atomic_slist_entry_t* head,
                              iree_atomic_slist_entry_t* tail);

// Pushes an entry into the list.
//
//   existing slist: C B A
//   provided entry: 1
//  resulting slist: 1 C B A
void iree_atomic_slist_push(iree_atomic_slist_t* list,
                            iree_atomic_slist_entry_t* entry);

// Pushes an entry into the list without using an atomic update.
// This is useful for when |list| is known to be inaccessible to any other
// thread, such as when populating a stack-local list prior to sharing it.
void iree_atomic_slist_push_unsafe(iree_atomic_slist_t* list,
                                   iree_atomic_slist_entry_t* entry);

// Pops the most recently pushed entry from the list and returns it.
// Returns NULL if the list was empty at the time it was queried.
//
//   existing slist: C B A
//  resulting slist: B A
//   returned entry: C
iree_atomic_slist_entry_t* iree_atomic_slist_pop(iree_atomic_slist_t* list);

// Defines the approximate order in which a span of flushed entries is returned.
typedef enum iree_atomic_slist_flush_order_e {
  // |out_head| and |out_tail| will be set to a span of the entries roughly in
  // the order they were pushed to the list in LIFO (stack) order.
  //
  // Example:
  //    slist: C B A
  //   result: C B A (or when contended possibly C A B)
  IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO = 0,
  // |out_head| and |out_tail| will be set to the first and last entries
  // pushed respectively, turning this LIFO slist into a FIFO queue.
  //
  // Example:
  //    slist: C B A
  //   result: A B C (or when contended possibly B A C)
  IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
} iree_atomic_slist_flush_order_t;

// Removes all items from the list and returns them in **APPROXIMATELY** the
// |flush_order| requested. As there are no order guarantees there may be slight
// transpositions of entries that were pushed from multiple processors or even
// interleaved entries within spans of entries pushed with
// iree_atomic_slist_concat.
//
// If |out_tail| is not required it can be omitted and this may avoid the
// need for the flush to walk the list and touch each entry.
//
// Returns true if any items were present and false if the output list is empty.
// Note that because atomic data structures can race it's possible for there to
// both be something in the list prior to this call and something in the list
// after the call and yet the return can still be false.
bool iree_atomic_slist_flush(iree_atomic_slist_t* list,
                             iree_atomic_slist_flush_order_t flush_order,
                             iree_atomic_slist_entry_t** out_head,
                             iree_atomic_slist_entry_t** out_tail);

//==============================================================================
// Typed wrapper generator for iree_atomic_slist_t
//==============================================================================

// Typed and named wrappers for making atomic slists easier to work with.
//
// Usage:
//  typedef struct {
//    int some_fields;
//    iree_atomic_slist_intrusive_ptr_t slist_next;
//    int more_fields;
//  } my_type_t;
//  IREE_TYPED_ATOMIC_SLIST_WRAPPER(my_type, my_type_t,
//                                  offsetof(my_type_t, slist_next));
//
//  my_type_slist_t list;
//  my_type_slist_initialize(&list);
//  my_type_t* entry = allocate_my_type(123);
//  my_type_slist_push(&list, entry);
//  entry = my_type_slist_pop(&list);
#define IREE_TYPED_ATOMIC_SLIST_WRAPPER(name, type, next_offset)               \
  static inline iree_atomic_slist_entry_t* name##_slist_entry_from_ptr(        \
      type* entry) {                                                           \
    return entry                                                               \
               ? ((iree_atomic_slist_entry_t*)((uint8_t*)entry + next_offset)) \
               : NULL;                                                         \
  }                                                                            \
  static inline type* name##_slist_entry_to_ptr(                               \
      iree_atomic_slist_entry_t* entry) {                                      \
    return entry ? (type*)(((uint8_t*)entry) - next_offset) : NULL;            \
  }                                                                            \
                                                                               \
  static inline type* name##_slist_get_next(type* entry) {                     \
    if (!entry) return NULL;                                                   \
    return name##_slist_entry_to_ptr(                                          \
        ((iree_atomic_slist_entry_t*)((uint8_t*)entry + next_offset))->next);  \
  }                                                                            \
  static inline void name##_slist_set_next(type* entry, type* next) {          \
    name##_slist_entry_from_ptr(entry)->next =                                 \
        name##_slist_entry_from_ptr(next);                                     \
  }                                                                            \
                                                                               \
  typedef iree_alignas(iree_max_align_t) struct {                              \
    iree_atomic_slist_t impl;                                                  \
  } name##_slist_t;                                                            \
                                                                               \
  static inline void name##_slist_initialize(name##_slist_t* out_list) {       \
    iree_atomic_slist_initialize(&out_list->impl);                             \
  }                                                                            \
  static inline void name##_slist_deinitialize(name##_slist_t* list) {         \
    iree_atomic_slist_deinitialize(&list->impl);                               \
  }                                                                            \
                                                                               \
  static inline void name##_slist_push(name##_slist_t* list, type* entry) {    \
    iree_atomic_slist_push(&list->impl, name##_slist_entry_from_ptr(entry));   \
  }                                                                            \
  static inline void name##_slist_push_unsafe(name##_slist_t* list,            \
                                              type* entry) {                   \
    iree_atomic_slist_push_unsafe(&list->impl,                                 \
                                  name##_slist_entry_from_ptr(entry));         \
  }                                                                            \
  static inline void name##_slist_concat(name##_slist_t* list, type* head,     \
                                         type* tail) {                         \
    iree_atomic_slist_concat(&list->impl, name##_slist_entry_from_ptr(head),   \
                             name##_slist_entry_from_ptr(tail));               \
  }                                                                            \
  static inline type* name##_slist_pop(name##_slist_t* list) {                 \
    return name##_slist_entry_to_ptr(iree_atomic_slist_pop(&list->impl));      \
  }                                                                            \
                                                                               \
  static inline bool name##_slist_flush(                                       \
      name##_slist_t* list, iree_atomic_slist_flush_order_t flush_order,       \
      type** out_head, type** out_tail) {                                      \
    iree_atomic_slist_entry_t* head = NULL;                                    \
    iree_atomic_slist_entry_t* tail = NULL;                                    \
    if (!iree_atomic_slist_flush(&list->impl, flush_order, &head,              \
                                 out_tail ? &tail : NULL)) {                   \
      return false; /* empty list */                                           \
    }                                                                          \
    *out_head = name##_slist_entry_to_ptr(head);                               \
    if (out_tail) *out_tail = name##_slist_entry_to_ptr(tail);                 \
    return true;                                                               \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_ATOMIC_SLIST_H_
