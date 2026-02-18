// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/iocp/timer_list.h"

void iree_async_iocp_timer_list_insert(iree_async_iocp_timer_list_t* list,
                                       iree_async_timer_operation_t* timer) {
  timer->platform.iocp.next = NULL;
  timer->platform.iocp.prev = NULL;

  // Empty list case.
  if (!list->head) {
    list->head = timer;
    list->tail = timer;
    return;
  }

  // Walk list to find insertion point (sorted by deadline ascending).
  iree_async_timer_operation_t* current = list->head;
  while (current && current->deadline_ns <= timer->deadline_ns) {
    current = current->platform.iocp.next;
  }

  if (!current) {
    // Insert at tail (deadline is furthest out).
    timer->platform.iocp.prev = list->tail;
    list->tail->platform.iocp.next = timer;
    list->tail = timer;
  } else if (!current->platform.iocp.prev) {
    // Insert at head (deadline is earliest).
    timer->platform.iocp.next = list->head;
    list->head->platform.iocp.prev = timer;
    list->head = timer;
  } else {
    // Insert in middle.
    timer->platform.iocp.prev = current->platform.iocp.prev;
    timer->platform.iocp.next = current;
    current->platform.iocp.prev->platform.iocp.next = timer;
    current->platform.iocp.prev = timer;
  }
}

void iree_async_iocp_timer_list_remove(iree_async_iocp_timer_list_t* list,
                                       iree_async_timer_operation_t* timer) {
  if (timer->platform.iocp.prev) {
    timer->platform.iocp.prev->platform.iocp.next = timer->platform.iocp.next;
  } else {
    list->head = timer->platform.iocp.next;
  }

  if (timer->platform.iocp.next) {
    timer->platform.iocp.next->platform.iocp.prev = timer->platform.iocp.prev;
  } else {
    list->tail = timer->platform.iocp.prev;
  }

  timer->platform.iocp.next = NULL;
  timer->platform.iocp.prev = NULL;
}

bool iree_async_iocp_timer_list_contains(
    const iree_async_iocp_timer_list_t* list,
    const iree_async_timer_operation_t* timer) {
  // A timer is in the list if it's the head/tail OR has a prev/next link.
  return timer == list->head || timer == list->tail ||
         timer->platform.iocp.prev != NULL || timer->platform.iocp.next != NULL;
}

iree_async_timer_operation_t* iree_async_iocp_timer_list_pop_expired(
    iree_async_iocp_timer_list_t* list, iree_time_t now) {
  if (!list->head || list->head->deadline_ns > now) {
    return NULL;
  }
  iree_async_timer_operation_t* timer = list->head;
  iree_async_iocp_timer_list_remove(list, timer);
  return timer;
}

iree_time_t iree_async_iocp_timer_list_next_deadline_ns(
    const iree_async_iocp_timer_list_t* list) {
  if (!list->head) {
    return IREE_TIME_INFINITE_FUTURE;
  }
  return list->head->deadline_ns;
}
