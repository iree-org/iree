// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/submission.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"

void iree_task_submission_initialize(iree_task_submission_t* out_submission) {
  iree_task_list_initialize(&out_submission->ready_list);
  iree_task_list_initialize(&out_submission->waiting_list);
}

void iree_task_submission_initialize_from_lifo_slist(
    iree_atomic_task_slist_t* ready_slist,
    iree_task_submission_t* out_submission) {
  // Flush from the LIFO ready list to the LIFO submission queue.
  // We have to walk everything here to get the tail pointer, which could be
  // improved by sourcing from something other than an slist.
  iree_task_submission_initialize(out_submission);
  iree_atomic_task_slist_flush(
      ready_slist, IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
      &out_submission->ready_list.head, &out_submission->ready_list.tail);
}

void iree_task_submission_reset(iree_task_submission_t* submission) {
  memset(&submission->ready_list, 0, sizeof(submission->ready_list));
  memset(&submission->waiting_list, 0, sizeof(submission->waiting_list));
}

void iree_task_submission_discard(iree_task_submission_t* submission) {
  iree_task_list_discard(&submission->ready_list);
  iree_task_list_discard(&submission->waiting_list);
}

bool iree_task_submission_is_empty(iree_task_submission_t* submission) {
  return iree_task_list_is_empty(&submission->ready_list) &&
         iree_task_list_is_empty(&submission->waiting_list);
}

void iree_task_submission_enqueue(iree_task_submission_t* submission,
                                  iree_task_t* task) {
  IREE_ASSERT_TRUE(iree_task_is_ready(task),
                   "must be a root task to be enqueued on a submission");
  if (task->type == IREE_TASK_TYPE_WAIT &&
      (task->flags & IREE_TASK_FLAG_WAIT_COMPLETED) == 0) {
    // A wait that we know is unresolved and can immediately route to the
    // waiting list. This avoids the need to try to schedule the wait when it's
    // almost certain that the wait would not be satisfied.
    iree_task_list_push_front(&submission->waiting_list, task);
  } else {
    // Task is ready to execute immediately.
    iree_task_list_push_front(&submission->ready_list, task);
  }
}

void iree_task_submission_enqueue_list(iree_task_submission_t* submission,
                                       iree_task_list_t* list) {
  iree_task_t* task = list->head;
  list->head = list->tail = NULL;
  while (task) {
    iree_task_t* next = task->next_task;
    iree_task_submission_enqueue(submission, task);
    task = next;
  }
}
