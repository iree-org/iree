// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/queue.h"

#include <stddef.h>
#include <string.h>

void iree_task_queue_initialize(iree_task_queue_t* out_queue) {
  memset(out_queue, 0, sizeof(*out_queue));
  iree_slim_mutex_initialize(&out_queue->mutex);
  iree_task_list_initialize(&out_queue->list);
}

void iree_task_queue_deinitialize(iree_task_queue_t* queue) {
  iree_task_list_discard(&queue->list);
  iree_slim_mutex_deinitialize(&queue->mutex);
}

bool iree_task_queue_is_empty(iree_task_queue_t* queue) {
  iree_slim_mutex_lock(&queue->mutex);
  bool is_empty = iree_task_list_is_empty(&queue->list);
  iree_slim_mutex_unlock(&queue->mutex);
  return is_empty;
}

void iree_task_queue_push_front(iree_task_queue_t* queue, iree_task_t* task) {
  iree_slim_mutex_lock(&queue->mutex);
  iree_task_list_push_front(&queue->list, task);
  iree_slim_mutex_unlock(&queue->mutex);
}

void iree_task_queue_append_from_lifo_list_unsafe(iree_task_queue_t* queue,
                                                  iree_task_list_t* list) {
  // NOTE: reversing the list outside of the lock.
  iree_task_list_reverse(list);
  iree_slim_mutex_lock(&queue->mutex);
  iree_task_list_append(&queue->list, list);
  iree_slim_mutex_unlock(&queue->mutex);
}

iree_task_t* iree_task_queue_flush_from_lifo_slist(
    iree_task_queue_t* queue, iree_atomic_task_slist_t* source_slist) {
  // Perform the flush and swap outside of the lock; acquiring the list is
  // atomic and then we own it exclusively.
  iree_task_list_t suffix;
  iree_task_list_initialize(&suffix);
  const bool did_flush = iree_atomic_task_slist_flush(
      source_slist, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
      &suffix.head, &suffix.tail);

  // Append the tasks and pop off the front for return.
  iree_slim_mutex_lock(&queue->mutex);
  if (did_flush) iree_task_list_append(&queue->list, &suffix);
  iree_task_t* next_task = iree_task_list_pop_front(&queue->list);
  iree_slim_mutex_unlock(&queue->mutex);

  return next_task;
}

iree_task_t* iree_task_queue_pop_front(iree_task_queue_t* queue) {
  iree_slim_mutex_lock(&queue->mutex);
  iree_task_t* next_task = iree_task_list_pop_front(&queue->list);
  iree_slim_mutex_unlock(&queue->mutex);
  return next_task;
}

iree_task_t* iree_task_queue_try_steal(iree_task_queue_t* source_queue,
                                       iree_task_queue_t* target_queue,
                                       iree_host_size_t max_tasks) {
  // First attempt to steal up to max_tasks from the source queue.
  iree_task_list_t stolen_tasks;
  iree_task_list_initialize(&stolen_tasks);
  if (iree_slim_mutex_try_lock(&source_queue->mutex)) {
    iree_task_list_split(&source_queue->list, max_tasks, &stolen_tasks);
    iree_slim_mutex_unlock(&source_queue->mutex);
  }

  // Add any stolen tasks to the target queue and pop off the head for return.
  iree_task_t* next_task = NULL;
  if (!iree_task_list_is_empty(&stolen_tasks)) {
    iree_slim_mutex_lock(&target_queue->mutex);
    iree_task_list_append(&target_queue->list, &stolen_tasks);
    next_task = iree_task_list_pop_front(&target_queue->list);
    iree_slim_mutex_unlock(&target_queue->mutex);
  }
  return next_task;
}
