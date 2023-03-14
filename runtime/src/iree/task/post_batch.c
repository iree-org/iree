// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/post_batch.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/base/tracing.h"
#include "iree/task/executor_impl.h"
#include "iree/task/queue.h"
#include "iree/task/worker.h"

void iree_task_post_batch_initialize(iree_task_executor_t* executor,
                                     iree_task_worker_t* current_worker,
                                     iree_task_post_batch_t* out_post_batch) {
  out_post_batch->executor = executor;
  out_post_batch->current_worker = current_worker;
  out_post_batch->worker_pending_mask = 0;
  memset(&out_post_batch->worker_pending_lifos, 0,
         executor->worker_count * sizeof(iree_task_list_t));
}

iree_host_size_t iree_task_post_batch_worker_count(
    const iree_task_post_batch_t* post_batch) {
  return post_batch->executor->worker_count;
}

static iree_host_size_t iree_task_post_batch_select_random_worker(
    iree_task_post_batch_t* post_batch, iree_task_affinity_set_t affinity_set) {
  // The masks are accessed with 'relaxed' order because they are just hints.
  iree_task_affinity_set_t worker_live_mask =
      iree_atomic_task_affinity_set_load(
          &post_batch->executor->worker_live_mask, iree_memory_order_relaxed);
  iree_task_affinity_set_t valid_worker_mask = affinity_set & worker_live_mask;
  if (!valid_worker_mask) {
    // No valid workers as desired; for now just bail to worker 0.
    return 0;
  }

  // TODO(benvanik): rotate through workers here. Instead, if the affinity set
  // has the current_worker allowed we just use that to avoid needing a
  // cross-thread hop.
  return iree_task_affinity_set_count_trailing_zeros(valid_worker_mask);
}

iree_host_size_t iree_task_post_batch_select_worker(
    iree_task_post_batch_t* post_batch, iree_task_affinity_set_t affinity_set) {
  if (post_batch->current_worker) {
    // Posting from a worker - prefer sending right back to this worker if we
    // haven't already scheduled for it.
    if ((affinity_set & post_batch->current_worker->worker_bit) &&
        !(post_batch->worker_pending_mask &
          post_batch->current_worker->worker_bit)) {
      return iree_task_affinity_set_count_trailing_zeros(
          post_batch->current_worker->worker_bit);
    }
  }

  // Prefer workers that are idle as though they'll need to wake up it is
  // guaranteed that they aren't working on something else and the latency of
  // waking should (hopefully) be less than the latency of waiting for a
  // worker's queue to finish. Note that we only consider workers idle if we
  // ourselves in this batch haven't already queued work for them (as then they
  // aren't going to be idle).
  // The masks are accessed with 'relaxed' order because they are just hints.
  iree_task_affinity_set_t worker_idle_mask =
      iree_atomic_task_affinity_set_load(
          &post_batch->executor->worker_idle_mask, iree_memory_order_relaxed);
  worker_idle_mask &= ~post_batch->worker_pending_mask;
  iree_task_affinity_set_t idle_affinity_set = affinity_set & worker_idle_mask;
  if (idle_affinity_set) {
    return iree_task_post_batch_select_random_worker(post_batch,
                                                     idle_affinity_set);
  }

  // No more workers are idle; farm out at random. In the worst case work
  // stealing will help balance things out on the backend.
  return iree_task_post_batch_select_random_worker(post_batch, affinity_set);
}

void iree_task_post_batch_enqueue(iree_task_post_batch_t* post_batch,
                                  iree_host_size_t worker_index,
                                  iree_task_t* task) {
  iree_task_list_push_front(&post_batch->worker_pending_lifos[worker_index],
                            task);
  post_batch->worker_pending_mask |=
      iree_task_affinity_for_worker(worker_index);
}

// Wakes each worker indicated in the |wake_mask|, if needed.
static void iree_task_post_batch_wake_workers(
    iree_task_post_batch_t* post_batch, iree_task_affinity_set_t wake_mask) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, iree_math_count_ones_u64(wake_mask));

  // TODO(#4016): use a FUTEX_WAKE_BITSET here to wake all of the workers that
  // have pending work in a single syscall (vs. popcnt(worker_pending_mask)
  // syscalls). This will reduce wake latency for workers later in the set;
  // for example today worker[31] will wait until workers[0-30] have had their
  // syscalls performed before it's even requested to wake. This also loses
  // information the kernel could use to avoid core migration as it knows when N
  // threads will be needed simultaneously and can hopefully perform any needed
  // migrations prior to beginning execution.
  iree_task_executor_t* executor = post_batch->executor;
  int wake_count = iree_task_affinity_set_count_ones(wake_mask);
  int worker_index = 0;
  for (int i = 0; i < wake_count; ++i) {
    int offset = iree_task_affinity_set_count_trailing_zeros(wake_mask);
    int wake_index = worker_index + offset;
    worker_index += offset + 1;
    wake_mask = iree_shr(wake_mask, offset + 1);

    // Wake workers if they are waiting - workers are the only thing that can
    // wait on this notification so this should almost always be either free (an
    // atomic load) if a particular worker isn't waiting or it's required to
    // actually wake it and we can't avoid it.
    iree_task_worker_t* worker = &executor->workers[wake_index];
    iree_notification_post(&worker->wake_notification, 1);
  }

  IREE_TRACE_ZONE_END(z0);
}

bool iree_task_post_batch_submit(iree_task_post_batch_t* post_batch) {
  if (!post_batch->worker_pending_mask) return false;

  IREE_TRACE_ZONE_BEGIN(z0);

  // Run through each worker that has a bit set in the pending mask and post
  // the pending tasks.
  iree_task_affinity_set_t worker_mask = post_batch->worker_pending_mask;
  post_batch->worker_pending_mask = 0;
  int worker_index = 0;
  int post_count = iree_task_affinity_set_count_ones(worker_mask);
  iree_task_affinity_set_t worker_wake_mask = 0;
  for (int i = 0; i < post_count; ++i) {
    int offset = iree_task_affinity_set_count_trailing_zeros(worker_mask);
    int target_index = worker_index + offset;
    worker_index += offset + 1;
    worker_mask = iree_shr(worker_mask, offset + 1);

    iree_task_worker_t* worker = &post_batch->executor->workers[target_index];
    iree_task_list_t* target_pending_lifo =
        &post_batch->worker_pending_lifos[target_index];
    if (worker == post_batch->current_worker) {
      // Fast-path for posting to self; this happens when a worker plays the
      // role of coordinator and we want to ensure we aren't doing a fully
      // block-and-flush loop when we could just be popping the next new task
      // off the list.
      iree_task_queue_append_from_lifo_list_unsafe(&worker->local_task_queue,
                                                   target_pending_lifo);
    } else {
      iree_task_worker_post_tasks(worker, target_pending_lifo);
      worker_wake_mask |= iree_task_affinity_for_worker(target_index);
    }
  }

  // Wake all workers that now have pending work. If a worker is not already
  // waiting this will be cheap (no syscall).
  if (worker_wake_mask != 0) {
    iree_task_post_batch_wake_workers(post_batch, worker_wake_mask);
  }

  IREE_TRACE_ZONE_END(z0);
  return post_count != 0;
}
