// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/timer_list.h"

void iree_async_posix_timer_list_insert(iree_async_posix_timer_list_t* list,
                                        iree_async_timer_operation_t* timer) {
  timer->platform.posix.next = NULL;
  timer->platform.posix.prev = NULL;

  // Empty list case.
  if (!list->head) {
    list->head = timer;
    list->tail = timer;
    return;
  }

  // Walk list to find insertion point (sorted by deadline ascending).
  iree_async_timer_operation_t* current = list->head;
  while (current && current->deadline_ns <= timer->deadline_ns) {
    current = current->platform.posix.next;
  }

  if (!current) {
    // Insert at tail (deadline is furthest out).
    timer->platform.posix.prev = list->tail;
    list->tail->platform.posix.next = timer;
    list->tail = timer;
  } else if (!current->platform.posix.prev) {
    // Insert at head (deadline is earliest).
    timer->platform.posix.next = list->head;
    list->head->platform.posix.prev = timer;
    list->head = timer;
  } else {
    // Insert in middle.
    timer->platform.posix.prev = current->platform.posix.prev;
    timer->platform.posix.next = current;
    current->platform.posix.prev->platform.posix.next = timer;
    current->platform.posix.prev = timer;
  }
}

void iree_async_posix_timer_list_remove(iree_async_posix_timer_list_t* list,
                                        iree_async_timer_operation_t* timer) {
  if (timer->platform.posix.prev) {
    timer->platform.posix.prev->platform.posix.next =
        timer->platform.posix.next;
  } else {
    list->head = timer->platform.posix.next;
  }

  if (timer->platform.posix.next) {
    timer->platform.posix.next->platform.posix.prev =
        timer->platform.posix.prev;
  } else {
    list->tail = timer->platform.posix.prev;
  }

  timer->platform.posix.next = NULL;
  timer->platform.posix.prev = NULL;
}

bool iree_async_posix_timer_list_contains(
    const iree_async_posix_timer_list_t* list,
    const iree_async_timer_operation_t* timer) {
  // A timer is in the list if it's the head/tail OR has a prev/next link.
  return timer == list->head || timer == list->tail ||
         timer->platform.posix.prev != NULL ||
         timer->platform.posix.next != NULL;
}

iree_async_timer_operation_t* iree_async_posix_timer_list_pop_expired(
    iree_async_posix_timer_list_t* list, iree_time_t now) {
  if (!list->head || list->head->deadline_ns > now) {
    return NULL;
  }
  iree_async_timer_operation_t* timer = list->head;
  iree_async_posix_timer_list_remove(list, timer);
  return timer;
}

iree_time_t iree_async_posix_timer_list_next_deadline_ns(
    const iree_async_posix_timer_list_t* list) {
  if (!list->head) {
    return IREE_TIME_INFINITE_FUTURE;
  }
  return list->head->deadline_ns;
}
