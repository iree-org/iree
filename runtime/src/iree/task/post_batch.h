// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_POST_BATCH_H_
#define IREE_TASK_POST_BATCH_H_

#include <stdbool.h>

#include "iree/base/api.h"
#include "iree/task/affinity_set.h"
#include "iree/task/executor.h"
#include "iree/task/list.h"
#include "iree/task/task.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_task_worker_t iree_task_worker_t;

// Transient/stack-allocated structure for batching up tasks for posting to
// worker mailboxes in single operations. This avoids the need to repeatedly
// thrash caches during coordination as only during submission are the worker
// mailboxes touched and only once per worker.
typedef struct iree_task_post_batch_t {
  iree_task_executor_t* executor;

  // Local worker constructing the post batch.
  // This is used to know when lighter-weight queuing can occur (no need to
  // post across a mailbox channel to yourself!).
  // May be NULL if not being posted from a worker (such as a submission).
  iree_task_worker_t* current_worker;

  // A bitmask of workers indicating which have pending tasks in their lists.
  // Used to quickly scan the lists and perform the posts only when required.
  iree_task_affinity_set_t worker_pending_mask;

  // A per-worker LIFO task list waiting to be posted.
  iree_task_list_t worker_pending_lifos[0];
} iree_task_post_batch_t;

void iree_task_post_batch_initialize(iree_task_executor_t* executor,
                                     iree_task_worker_t* current_worker,
                                     iree_task_post_batch_t* out_post_batch);

// Returns the total number of workers that the post batch is targeting.
iree_host_size_t iree_task_post_batch_worker_count(
    const iree_task_post_batch_t* post_batch);

// Selects a random worker from the given affinity set.
iree_host_size_t iree_task_post_batch_select_worker(
    iree_task_post_batch_t* post_batch, iree_task_affinity_set_t affinity_set);

// Enqueues a task to the given worker. Note that the pending work lists for
// each work is kept in LIFO order so that we can easily concatenate it with the
// worker mailbox slist that's in LIFO order.
void iree_task_post_batch_enqueue(iree_task_post_batch_t* post_batch,
                                  iree_host_size_t worker_index,
                                  iree_task_t* task);

// Submits all pending tasks to their worker mailboxes and resets state.
// Returns true if any tasks were posted to workers.
bool iree_task_post_batch_submit(iree_task_post_batch_t* post_batch);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_POST_BATCH_H_
