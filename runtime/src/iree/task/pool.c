// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/pool.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"

// Minimum byte size of a block in bytes, including the tasks as well as the
// allocation header. This is here to allow us to reduce the number of times
// we go to the allocator and amortize the overhead of our block header.
#define IREE_TASK_POOL_MIN_BLOCK_SIZE (4 * 1024)

// Alignment for block allocations; roughly a (likely) page size.
// Since many allocators after the small byte range (~thousands of bytes) will
// round up this just prevents us from being 1 over the allocator block size and
// wasting space in a larger bucket.
#define IREE_TASK_POOL_BLOCK_ALIGNMENT (4 * 1024)

// The minimum number of tasks that will be allocated when growth is needed.
// The total number may be larger once rounded to meet block size and alignment
// requirements. Note that we leave a bit of room here for the block header
// such that we don't always allocate a nice round number + N bytes that then
// bumps us into the next power of two bucket.
#define IREE_TASK_POOL_MIN_GROWTH_CAPACITY (255)

// Grows the task pool by at least |minimum_capacity| on top of its current
// capacity. The actual number of tasks available may be rounded up to make the
// allocated blocks more allocator-friendly sizes.
//
// As an optimization for on-demand growth cases an |out_task| can be specified
// to receive a task without the need for acquiring one from the pool
// immediately after the growth completes. This avoids a race condition where
// another thread could snipe the tasks we just allocated for the caller prior
// to the caller getting a chance to acquire one.
static iree_status_t iree_task_pool_grow(iree_task_pool_t* pool,
                                         iree_host_size_t minimum_capacity,
                                         iree_task_t** out_task) {
  if (IREE_UNLIKELY(!minimum_capacity)) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate a new block of tasks. To try to prevent the allocator from
  // fragmenting we try to always allocate blocks that are page-aligned and
  // powers of two.
  //
  // Note that we pad out our header to iree_max_align_t bytes so that all tasks
  // are aligned on the same boundaries as required by atomic operations.
  iree_host_size_t header_size =
      iree_host_align(sizeof(iree_task_allocation_header_t), iree_max_align_t);
  iree_host_size_t pow2_block_size = iree_math_round_up_to_pow2_u64(
      header_size + minimum_capacity * pool->task_size);
  iree_host_size_t aligned_block_size =
      iree_host_align(pow2_block_size, IREE_TASK_POOL_BLOCK_ALIGNMENT);
  if (aligned_block_size < IREE_TASK_POOL_MIN_BLOCK_SIZE) {
    aligned_block_size = IREE_TASK_POOL_MIN_BLOCK_SIZE;
  }
  iree_task_allocation_header_t* allocation = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(pool->allocator, aligned_block_size,
                                (void**)&allocation));

  // Insert the allocation into the tracking list. Nothing reads the list until
  // the pool is trimmed/deinitialized so it's safe to do now prior to
  // populating anything. It's all just empty data anyway.
  iree_atomic_task_allocation_slist_push(&pool->allocations_slist, allocation);

  // Since we may have rounded up the allocation we may have gotten more space
  // for tasks than we were asked for. Ensure we actually make use of them.
  iree_host_size_t actual_capacity =
      (aligned_block_size - header_size) / pool->task_size;

  // Stitch together the tasks by setting all next pointers.
  // Since we are going to be touching all the pages the order here is important
  // as once we insert these new tasks into the available_slist they'll be
  // popped out head->tail. To ensure the head that gets popped first is still
  // warm in cache we construct the list backwards, with the tail tasks being
  // fine to be evicted.
  //
  // The nice thing about this walk is that it ensures that if there were any
  // zero-fill-on-demand trickery going on the pages are all wired here vs.
  // when the tasks are first acquired from the list where it'd be harder to
  // track.
  uintptr_t p = ((uintptr_t)allocation + aligned_block_size) - pool->task_size;
  iree_task_t* head = (iree_task_t*)p;
  iree_task_t* tail = head;
  head->next_task = NULL;
  head->pool = pool;

  // Work around a loop vectorizer bug that causes memory corruption in this
  // loop. Only Android NDK r25 is known to be affected. See
  // https://github.com/openxla/iree/issues/9953 for details.
#if defined(__NDK_MAJOR__) && __NDK_MAJOR__ == 25
#pragma clang loop unroll(disable) vectorize(disable)
#endif
  for (iree_host_size_t i = 0; i < actual_capacity; ++i, p -= pool->task_size) {
    iree_task_t* task = (iree_task_t*)p;
    task->next_task = head;
    task->pool = pool;
    head = task;
  }

  // If the caller needs a task we can slice off the head to return prior to
  // adding it to the slist where it may get stolen.
  if (out_task) {
    *out_task = head;
    head = head->next_task;
  }

  // Concatenate the list of new free tasks into the pool.
  iree_atomic_task_slist_concat(&pool->available_slist, head, tail);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_task_pool_initialize(iree_allocator_t allocator,
                                        iree_host_size_t task_size,
                                        iree_host_size_t initial_capacity,
                                        iree_task_pool_t* out_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, task_size);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, initial_capacity);

  out_pool->allocator = allocator;
  out_pool->task_size = task_size;
  iree_atomic_task_allocation_slist_initialize(&out_pool->allocations_slist);
  iree_atomic_task_slist_initialize(&out_pool->available_slist);
  iree_status_t status =
      iree_task_pool_grow(out_pool, initial_capacity, /*out_task=*/NULL);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_task_pool_deinitialize(iree_task_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_task_allocation_header_t* allocation = NULL;
  if (iree_atomic_task_allocation_slist_flush(
          &pool->allocations_slist,
          IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &allocation, NULL)) {
    while (allocation) {
      iree_task_allocation_header_t* next =
          iree_atomic_task_allocation_slist_get_next(allocation);
      iree_allocator_free(pool->allocator, allocation);
      allocation = next;
    }
  }
  iree_atomic_task_allocation_slist_deinitialize(&pool->allocations_slist);
  iree_atomic_task_slist_deinitialize(&pool->available_slist);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_pool_trim(iree_task_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // NOTE: this is only safe if there are no outstanding tasks.
  // Hopefully the caller read the docstring!

  // We only need to flush the list to empty it - these are just references into
  // the allocations and don't need to be released.
  iree_task_t* task_head = NULL;
  iree_atomic_task_slist_flush(&pool->available_slist,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                               &task_head, /*tail=*/NULL);

  iree_task_allocation_header_t* allocation_head = NULL;
  if (iree_atomic_task_allocation_slist_flush(
          &pool->allocations_slist,
          IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &allocation_head,
          /*tail=*/NULL)) {
    do {
      iree_task_allocation_header_t* next =
          iree_atomic_task_allocation_slist_get_next(allocation_head);
      iree_allocator_free(pool->allocator, allocation_head);
      allocation_head = next;
    } while (allocation_head != NULL);
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_task_pool_acquire(iree_task_pool_t* pool,
                                     iree_task_t** out_task) {
  if (!pool) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);

  // Attempt to acquire a task from the available list.
  iree_task_t* task = iree_atomic_task_slist_pop(&pool->available_slist);
  if (task) {
    *out_task = task;
    return iree_ok_status();
  }

  // No tasks were available when we tried; force growth now.
  // Note that due to races it's possible that there are now tasks that have
  // been released back into the pool, but the fact that we failed once means
  // we are sitting right at the current limit of the pool and growing will
  // help ensure we go down the fast path more frequently in the future.
  return iree_task_pool_grow(pool, IREE_TASK_POOL_MIN_GROWTH_CAPACITY,
                             out_task);
}

iree_status_t iree_task_pool_acquire_many(iree_task_pool_t* pool,
                                          iree_host_size_t count,
                                          iree_task_list_t* out_list) {
  if (!pool) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);

  // If we acquire more than the requested count we need to give those leftovers
  // back to the pool before we leave.
  iree_task_list_t leftover_tasks;
  iree_task_list_initialize(&leftover_tasks);
  iree_task_list_initialize(out_list);

  iree_status_t status = iree_ok_status();
  while (count) {
    // Flush the entire available list so we can start operating on it.
    // This is where the potential race comes in: if another thread goes to
    // acquire a task while we have the list local here it'll grow the list so
    // it can meet its demand. That's still correct behavior but will result in
    // potentially more wasted memory than if the other thread would have
    // waited. Thankfully we save memory in so many other places that in the
    // rare case there are multiple concurrent schedulers acquiring tasks it's
    // not the end of the world.
    iree_task_list_t acquired_tasks;
    iree_task_list_initialize(&acquired_tasks);
    if (iree_atomic_task_slist_flush(
            &pool->available_slist,
            IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
            &acquired_tasks.head,
            /*tail=*/NULL)) {
      // Had some items in the pool; eat up to the requested count.
      // Note that we may run out and need to allocate more or have gotten
      // too many during the flush and need to track those leftovers.
      //
      // Instead of having the slist flush walk the list and give us a tail we
      // do that here: we need to walk the list anyway to partition it.
      iree_task_t* p = acquired_tasks.head;
      while (count > 0) {
        p = iree_atomic_task_slist_get_next(p);
        if (!p) break;
        acquired_tasks.tail = p;
        --count;
      }

      // If we got everything we need then we have to put all of the flushed
      // tasks we didn't use into the leftover list.
      if (count == 0) {
        iree_task_list_t acquire_leftovers;
        iree_task_list_initialize(&acquire_leftovers);
        acquire_leftovers.head =
            iree_atomic_task_slist_get_next(acquired_tasks.tail);
        iree_atomic_task_slist_set_next(acquired_tasks.tail, NULL);
        p = acquire_leftovers.head;
        iree_task_t* next;
        while ((next = iree_atomic_task_slist_get_next(p))) p = next;
        acquire_leftovers.tail = p;
        iree_task_list_append(&leftover_tasks, &acquire_leftovers);
      }

      // Add the tasks we did acquire to our result list.
      // NOTE: this is unmeasured but the intuition is that we want to put the
      // tasks we just acquired at the head of the list so that they are warm
      // upon return to the caller who will then be touching the head of the
      // list immediately.
      iree_task_list_prepend(out_list, &acquired_tasks);
    }

    // If we still need more tasks but ran out of ones in the flush list then we
    // need to grow some more.
    if (count > 0) {
      status = iree_task_pool_grow(pool, count, /*out_task=*/NULL);
      if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;
    }
  }

  // Return leftovers that we acquired but didn't need to the pool.
  iree_atomic_task_slist_concat(&pool->available_slist, leftover_tasks.head,
                                leftover_tasks.tail);

  // Upon failure return any tasks we may have already acquired from the pool.
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_atomic_task_slist_concat(&pool->available_slist, out_list->head,
                                  out_list->tail);
  }

  return status;
}

void iree_task_pool_release(iree_task_pool_t* pool, iree_task_t* task) {
  if (!pool) return;
  IREE_ASSERT_EQ(task->pool, pool);
  iree_atomic_task_slist_push(&pool->available_slist, task);
}
