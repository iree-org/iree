// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_POOL_H_
#define IREE_TASK_POOL_H_

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/task/list.h"
#include "iree/task/task.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// An allocation of tasks in a task pool containing multiple tasks.
// This struct is at the head of all task allocations made from the allocator.
// It is used to form a linked list of all allocations made so that they can be
// easily freed during pool teardown.
typedef struct iree_task_allocation_header_t {
  // Next allocation in the linked list of allocations.
  iree_atomic_slist_intrusive_ptr_t* next;
} iree_task_allocation_header_t;

// An atomic approximately LIFO singly-linked list.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_atomic_task_allocation,
                                iree_task_allocation_header_t,
                                offsetof(iree_task_allocation_header_t, next));

// Shared thread-safe pool of iree_task_t structures of a particular size.
// This can be used to quickly allocate blocks of tasks to be initialized by
// task producers, enqueued, and then eventually recycled back to the pool.
//
// The lifetime of all tasks must be less than the pool they were acquired
// from. Tasks acquired from one pool must not be released to another pool or
// via any other mechanism.
//
// Pools can either be fixed-size with a maximum number of available tasks that
// can be outstanding at any time or growable to allow the pool to be grown
// unbounded after initialization.
typedef struct iree_task_pool_t {
  // Allocator used for allocating/freeing each allocation block.
  iree_allocator_t allocator;

  // Task size, in bytes.
  iree_host_size_t task_size;

  // NOTE: we don't track current usage count as that would introduce additional
  // contention as tasks are acquired/released. If we end up finding a lot of
  // memory idling here we can add a threshold over which we reclaim it, but the
  // easiest (and most efficient) solution is to force the user to synchronize
  // with the executor on a low memory event and use iree_task_pool_trim.

  // Head of a linked list of all allocations made by the pool.
  iree_atomic_task_allocation_slist_t allocations_slist;

  // Linked list of free tasks used as a stack (LIFO).
  // This is not a great structure for this as over time the tasks will get out
  // of order and walking the linked list will incur cache misses. We offset
  // that cost a bit by knowing that the time between walking the list to
  // acquire tasks and when we initialize the tasks is short and that we would
  // have triggered a cache miss anyway. In the future we can explore other
  // approaches (such as small chunked linear lists) that better exploit spatial
  // locality, if needed.
  iree_atomic_task_slist_t available_slist;
} iree_task_pool_t;

// Initializes a task pool and optionally performs an initial task allocation.
iree_status_t iree_task_pool_initialize(iree_allocator_t allocator,
                                        iree_host_size_t task_size,
                                        iree_host_size_t initial_capacity,
                                        iree_task_pool_t* out_pool);

// Deinitializes a task pool and releases all task allocations back to the
// allocator specified during initialization. All tasks must have already been
// released back to the pool.
void iree_task_pool_deinitialize(iree_task_pool_t* pool);

// Attempts to trim unused allocations from the task pool.
// Must not be called while any tasks that were acquired from this pool are
// still live; callers must synchronize with the executor and ensure they aren't
// pushing any more work during the trim operation.
void iree_task_pool_trim(iree_task_pool_t* pool);

// Acquires a task from the task pool. The returned task will have undefined
// contents and must be initialized by the caller.
iree_status_t iree_task_pool_acquire(iree_task_pool_t* pool,
                                     iree_task_t** out_task);

// Acquires a set of tasks from the task pool. The returned tasks will have
// undefined contents besides their intrusive next pointers and must be
// intialized by the caller.
//
// WARNING: this may cause growth during races if multiple threads are trying to
// acquire at the same time. Our usage patterns here are such that this is never
// the case, though, as all acquisition from the internal executor pools happens
// with the coordination lock held.
iree_status_t iree_task_pool_acquire_many(iree_task_pool_t* pool,
                                          iree_host_size_t count,
                                          iree_task_list_t* out_list);

// Releases a task to the task pool.
// Callers must ensure the task is no longer in use.
void iree_task_pool_release(iree_task_pool_t* pool, iree_task_t* task);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TASK_POOL_H_
