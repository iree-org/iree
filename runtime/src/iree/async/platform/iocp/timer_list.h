// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sorted intrusive timer list for userspace timer management.
//
// Uses a doubly-linked list sorted by deadline (earliest first). O(n) insert,
// O(1) remove - acceptable for typical timer counts (~100 concurrent timers).
// Can be upgraded to a min-heap if profiling shows insert is a bottleneck.

#ifndef IREE_ASYNC_PLATFORM_IOCP_TIMER_LIST_H_
#define IREE_ASYNC_PLATFORM_IOCP_TIMER_LIST_H_

#include "iree/async/operations/scheduling.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A sorted list of pending timers, ordered by deadline (earliest first).
// The list uses intrusive pointers stored in timer->platform.iocp.{next,prev}.
typedef struct iree_async_iocp_timer_list_t {
  iree_async_timer_operation_t* head;  // Earliest deadline.
  iree_async_timer_operation_t* tail;  // Most distant deadline.
} iree_async_iocp_timer_list_t;

// Initializes an empty timer list.
static inline void iree_async_iocp_timer_list_initialize(
    iree_async_iocp_timer_list_t* list) {
  list->head = NULL;
  list->tail = NULL;
}

// Inserts a timer into the list, maintaining deadline order.
// O(n) where n is the number of timers in the list.
void iree_async_iocp_timer_list_insert(iree_async_iocp_timer_list_t* list,
                                       iree_async_timer_operation_t* timer);

// Removes a timer from the list.
// O(1) since we have prev/next pointers.
void iree_async_iocp_timer_list_remove(iree_async_iocp_timer_list_t* list,
                                       iree_async_timer_operation_t* timer);

// Returns true if the timer is in the list.
bool iree_async_iocp_timer_list_contains(
    const iree_async_iocp_timer_list_t* list,
    const iree_async_timer_operation_t* timer);

// Removes and returns the earliest expired timer (deadline <= now), or NULL.
// Call repeatedly to drain all expired timers.
iree_async_timer_operation_t* iree_async_iocp_timer_list_pop_expired(
    iree_async_iocp_timer_list_t* list, iree_time_t now);

// Returns the deadline of the earliest timer, or IREE_TIME_INFINITE_FUTURE
// if the list is empty.
iree_time_t iree_async_iocp_timer_list_next_deadline_ns(
    const iree_async_iocp_timer_list_t* list);

// Returns true if the list is empty.
static inline bool iree_async_iocp_timer_list_is_empty(
    const iree_async_iocp_timer_list_t* list) {
  return list->head == NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IOCP_TIMER_LIST_H_
