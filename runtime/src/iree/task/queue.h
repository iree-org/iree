// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_QUEUE_H_
#define IREE_TASK_QUEUE_H_

#include <stdbool.h>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/task/list.h"
#include "iree/task/task.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A simple work-stealing LIFO queue modeled on a Chase-Lev concurrent deque.
// This is used by workers to maintain their thread-local working lists. The
// workers keep the tasks they will process in FIFO order. They allow it to
// empty and then refresh it with more tasks from the incoming worker mailbox.
// The performance bias here is to the workers as they are >90% of the
// accesses and the only other accesses are thieves that hopefully we can just
// improve our distribution to vs. introducing a slowdown here.
//
// A futex is used to synchronize access; because the common case is that of
// only the worker that owns the queue touching it for pushing and popping items
// this puts us into the sweet-spot of uncontended lightweight exclusive locks.
// Since futices are effectively just single machine words managed with atomic
// ops we can avoid a lot of the traditional atomic tomfoolery one finds in
// systems like these that originated prior to the introduction of futices while
// also keeping the tiny overhead of the pure atomic solutions.
//
// We can also take advantage of the futex providing an actual exclusive region
// such that our data structure can be whatever we want as opposed to needing to
// be something that someone had figured out how to make atomic. For example,
// common implementations of work-stealing queues are all bounded as unbounded
// atomic deques are an unsolved problem in CS.
//
// Very rarely when another worker runs out of work it'll try to steal tasks
// from nearby workers and use this queue type to do it: the assumption is that
// it's better to take the last task the victim worker will get to so that in a
// long list of tasks it remains chugging through the head of the list with good
// cache locality. If we end up with a lot of theft, though, it's possible for
// the cache benefits of the pop_back approach to the worker to outweigh the
// cache pessimism for all thieves. Let's hope we can schedule deterministic-
// enough tiles such that theft is rare!
//
// Our queue variant here is tuned for the use case we have: we exclusively
// push in multiple tasks at a time (flushed from the mailbox) and exclusively
// pop a single task a time (what to work on next). The stealing part is batched
// so that when a remote worker has to perform a theft it takes a good chunk of
// tasks in one go (hopefully roughly half) to reduce the total overhead when
// there is high imbalance in workloads.
//
// Flushing from the mailbox slist (LIFO) to our list (FIFO) requires a full
// walk of the incoming task linked list. This is generally fine as the number
// of tasks in any given flush is low(ish) and by walking in reverse order to
// then process forward the cache should be hot as the worker starts making its
// way back through the tasks. As we walk forward we'll be using the task fields
// for execution and retiring of tasks (notifing dependencies/etc) and the
// intrusive next pointer sitting next to those should be in-cache when we need
// to access it. This, combined with slab allocation of tasks in command buffers
// to begin with gives us the (probabilistically) same characteristics of a flat
// array walked with an index as is common in other work queues but with the
// flexibility to reorder tasks as we see fit (theft, redistribution/rotation,
// reprioritization, etc).
//
// Similar concepts, though implemented with atomics:
//   "Dynamic Circular Work-Stealing Deque":
//   http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.170.1097&rep=rep1&type=pdf
//   "Correct and Efficient Work-Stealing for Weak Memory Models":
//   https://fzn.fr/readings/ppopp13.pdf
//   Motivating article:
//   https://blog.molecular-matters.com/2015/08/24/job-system-2-0-lock-free-work-stealing-part-1-basics/
//
// Useful diagram from https://github.com/injinj/WSQ
// Much of this implementation is inspired from that; though significant
// reworking was required for our FIFO->LIFO->FIFO sandwich.
//  +--------+ <- tasks[0]
//  |  top   | <- stealers consume here: task = tasks[top++]
//  |        |
//  |   ||   |
//  |        |
//  |   vv   |
//  | bottom | <- owner pushes here:    tasks[bottom++] = task
//  |        |    owner consumes here:  task = tasks[--bottom]
//  |        |
//  +--------+ <- tasks[IREE_TASK_QUEUE_CAPACITY-1]
//
// Unlike that implementation, though, our task list is unbounded because we use
// a linked list. To keep our options open, though, I've left the API of this
// implementation compatible with classic atomic work-stealing queues. I'm
// hopeful this will not need to be revisted for awhile, though!
//
// Future improvement idea: have the owner of the queue maintain a theft point
// skip list that makes it possible for thieves to quickly come in and slice
// off batches of tasks at the tail of the queue. Since we are a singly-linked
// list we can't easily just walk backward and we don't want to be introducing
// cache line contention as thieves start touching the same tasks as the worker
// is while processing.
typedef struct iree_task_queue_t {
  // Must be held when manipulating the queue. >90% accesses are by the owner.
  iree_slim_mutex_t mutex;

  // FIFO task list.
  iree_task_list_t list IREE_GUARDED_BY(mutex);
} iree_task_queue_t;

// Initializes a work-stealing task queue in-place.
void iree_task_queue_initialize(iree_task_queue_t* out_queue);

// Deinitializes a task queue and clears all references.
// Must not be called while any other worker may be attempting to steal tasks.
void iree_task_queue_deinitialize(iree_task_queue_t* queue);

// Returns true if the queue is empty.
// Note that due to races this may return both false-positives and -negatives.
bool iree_task_queue_is_empty(iree_task_queue_t* queue);

// Pushes a task to the front of the queue.
// Always prefer the multi-push variants (prepend/append) when adding more than
// one task to the queue. This is mostly useful for exceptional cases such as
// when a task may yield and need to be reprocessed after the worker resumes.
//
// Must only be called from the owning worker's thread.
void iree_task_queue_push_front(iree_task_queue_t* queue, iree_task_t* task);

// Appends a LIFO |list| of tasks to the queue.
//
// Must only be called from the owning worker's thread.
void iree_task_queue_append_from_lifo_list_unsafe(iree_task_queue_t* queue,
                                                  iree_task_list_t* list);

// Flushes the |source_slist| LIFO mailbox into the task queue in FIFO order.
// Returns the first task in the queue upon success; the task may be
// pre-existing or from the newly flushed tasks.
//
// Must only be called from the owning worker's thread.
iree_task_t* iree_task_queue_flush_from_lifo_slist(
    iree_task_queue_t* queue, iree_atomic_task_slist_t* source_slist);

// Pops a task from the front of the queue if any are available.
//
// Must only be called from the owning worker's thread.
iree_task_t* iree_task_queue_pop_front(iree_task_queue_t* queue);

// Tries to steal up to |max_tasks| from the back of the queue.
//
// On success, up to |max_tasks| tasks that were at the tail of the
// |source_queue| will be moved to the |target_queue| and the first of the
// stolen tasks is returned.
//
// On failure, NULL is returned.
//
// This function is allowed to fail spuriously, i.e. even if there are
// tasks to steal.
//
// It's expected this is not called from the queue's owning worker, though it's
// valid to do so.
iree_task_t* iree_task_queue_try_steal(iree_task_queue_t* source_queue,
                                       iree_task_queue_t* target_queue,
                                       iree_host_size_t max_tasks);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_QUEUE_H_
