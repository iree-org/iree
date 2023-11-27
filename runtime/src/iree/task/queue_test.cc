// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/queue.h"

#include "iree/base/internal/threading.h"
#include "iree/testing/gtest.h"

// Like iree_task_queue_try_steal but retries until success.
// This is used in this test as iree_task_queue_try_steal may (rarely) fail even
// in simple single-threaded tests, see #15488.
static iree_task_t* iree_task_queue_try_steal_until_success(
    iree_task_queue_t* source_queue, iree_task_queue_t* target_queue,
    iree_host_size_t max_tasks) {
  while (true) {
    iree_task_t* task =
        iree_task_queue_try_steal(source_queue, target_queue, max_tasks);
    if (task) {
      return task;
    }
    iree_thread_yield();
  }
}

namespace {

TEST(QueueTest, Lifetime) {
  iree_task_queue_t queue;
  iree_task_queue_initialize(&queue);
  iree_task_queue_deinitialize(&queue);
}

TEST(QueueTest, Empty) {
  iree_task_queue_t queue;
  iree_task_queue_initialize(&queue);
  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  EXPECT_FALSE(iree_task_queue_pop_front(&queue));
  iree_task_queue_deinitialize(&queue);
}

TEST(QueueTest, PushPop) {
  iree_task_queue_t queue;
  iree_task_queue_initialize(&queue);

  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  EXPECT_FALSE(iree_task_queue_pop_front(&queue));

  iree_task_t task_a = {0};
  iree_task_queue_push_front(&queue, &task_a);

  EXPECT_FALSE(iree_task_queue_is_empty(&queue));

  iree_task_t task_b = {0};
  iree_task_queue_push_front(&queue, &task_b);

  EXPECT_FALSE(iree_task_queue_is_empty(&queue));
  EXPECT_EQ(&task_b, iree_task_queue_pop_front(&queue));

  EXPECT_FALSE(iree_task_queue_is_empty(&queue));
  EXPECT_EQ(&task_a, iree_task_queue_pop_front(&queue));

  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  EXPECT_FALSE(iree_task_queue_pop_front(&queue));

  iree_task_queue_deinitialize(&queue);
}

TEST(QueueTest, AppendListEmpty) {
  iree_task_queue_t queue;
  iree_task_queue_initialize(&queue);

  iree_task_list_t list = {0};

  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  iree_task_queue_append_from_lifo_list_unsafe(&queue, &list);
  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  EXPECT_TRUE(iree_task_list_is_empty(&list));

  iree_task_queue_deinitialize(&queue);
}

TEST(QueueTest, AppendList1) {
  iree_task_queue_t queue;
  iree_task_queue_initialize(&queue);

  iree_task_list_t list = {0};
  iree_task_t task_a = {0};
  iree_task_list_push_front(&list, &task_a);

  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  iree_task_queue_append_from_lifo_list_unsafe(&queue, &list);
  EXPECT_FALSE(iree_task_queue_is_empty(&queue));
  EXPECT_TRUE(iree_task_list_is_empty(&list));

  EXPECT_EQ(&task_a, iree_task_queue_pop_front(&queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&queue));

  iree_task_queue_deinitialize(&queue);
}

TEST(QueueTest, AppendListOrdered) {
  iree_task_queue_t queue;
  iree_task_queue_initialize(&queue);

  // Make a lifo list: b<-a.
  iree_task_list_t list = {0};
  iree_task_t task_a = {0};
  iree_task_list_push_front(&list, &task_a);
  iree_task_t task_b = {0};
  iree_task_list_push_front(&list, &task_b);

  // Append the list to the queue; it should swap LIFO->FIFO.
  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  iree_task_queue_append_from_lifo_list_unsafe(&queue, &list);
  EXPECT_FALSE(iree_task_queue_is_empty(&queue));
  EXPECT_TRUE(iree_task_list_is_empty(&list));

  // Pop list and ensure order: a->b.
  EXPECT_EQ(&task_a, iree_task_queue_pop_front(&queue));
  EXPECT_EQ(&task_b, iree_task_queue_pop_front(&queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&queue));

  iree_task_queue_deinitialize(&queue);
}

TEST(QueueTest, FlushSlistEmpty) {
  iree_task_queue_t queue;
  iree_task_queue_initialize(&queue);

  iree_atomic_task_slist_t slist;
  iree_atomic_task_slist_initialize(&slist);

  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  EXPECT_FALSE(iree_task_queue_flush_from_lifo_slist(&queue, &slist));
  EXPECT_TRUE(iree_task_queue_is_empty(&queue));

  iree_atomic_task_slist_deinitialize(&slist);

  iree_task_queue_deinitialize(&queue);
}

TEST(QueueTest, FlushSlist1) {
  iree_task_queue_t queue;
  iree_task_queue_initialize(&queue);

  iree_atomic_task_slist_t slist;
  iree_atomic_task_slist_initialize(&slist);
  iree_task_t task_a = {0};
  iree_atomic_task_slist_push(&slist, &task_a);

  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  EXPECT_EQ(&task_a, iree_task_queue_flush_from_lifo_slist(&queue, &slist));
  EXPECT_TRUE(iree_task_queue_is_empty(&queue));

  iree_atomic_task_slist_deinitialize(&slist);

  iree_task_queue_deinitialize(&queue);
}

TEST(QueueTest, FlushSlistOrdered) {
  iree_task_queue_t queue;
  iree_task_queue_initialize(&queue);

  // Make a lifo list: c<-b<-a.
  iree_atomic_task_slist_t slist;
  iree_atomic_task_slist_initialize(&slist);
  iree_task_t task_a = {0};
  iree_atomic_task_slist_push(&slist, &task_a);
  iree_task_t task_b = {0};
  iree_atomic_task_slist_push(&slist, &task_b);
  iree_task_t task_c = {0};
  iree_atomic_task_slist_push(&slist, &task_c);

  // Flush the list to the queue; it should swap LIFO->FIFO and return the
  // first task in the queue.
  EXPECT_TRUE(iree_task_queue_is_empty(&queue));
  EXPECT_EQ(&task_a, iree_task_queue_flush_from_lifo_slist(&queue, &slist));
  EXPECT_FALSE(iree_task_queue_is_empty(&queue));

  // Pop list and ensure order: [a->]b->c.
  EXPECT_EQ(&task_b, iree_task_queue_pop_front(&queue));
  EXPECT_EQ(&task_c, iree_task_queue_pop_front(&queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&queue));

  iree_atomic_task_slist_deinitialize(&slist);

  iree_task_queue_deinitialize(&queue);
}

TEST(QueueTest, TryStealEmpty) {
  iree_task_queue_t source_queue;
  iree_task_queue_initialize(&source_queue);
  iree_task_queue_t target_queue;
  iree_task_queue_initialize(&target_queue);

  iree_task_t task_a = {0};
  iree_task_queue_push_front(&source_queue, &task_a);
  iree_task_t task_b = {0};
  iree_task_queue_push_front(&source_queue, &task_b);
  iree_task_t task_c = {0};
  iree_task_queue_push_front(&source_queue, &task_c);

  EXPECT_EQ(&task_a, iree_task_queue_try_steal_until_success(&source_queue,
                                                             &target_queue, 1));

  iree_task_queue_deinitialize(&source_queue);
  iree_task_queue_deinitialize(&target_queue);
}

TEST(QueueTest, TryStealLast) {
  iree_task_queue_t source_queue;
  iree_task_queue_initialize(&source_queue);
  iree_task_queue_t target_queue;
  iree_task_queue_initialize(&target_queue);

  iree_task_t task_a = {0};
  iree_task_queue_push_front(&source_queue, &task_a);

  EXPECT_EQ(&task_a, iree_task_queue_try_steal_until_success(
                         &source_queue, &target_queue, 100));
  EXPECT_TRUE(iree_task_queue_is_empty(&target_queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&source_queue));

  iree_task_queue_deinitialize(&source_queue);
  iree_task_queue_deinitialize(&target_queue);
}

TEST(QueueTest, TrySteal1) {
  iree_task_queue_t source_queue;
  iree_task_queue_initialize(&source_queue);
  iree_task_queue_t target_queue;
  iree_task_queue_initialize(&target_queue);

  iree_task_t task_a = {0};
  iree_task_t task_b = {0};
  iree_task_t task_c = {0};
  iree_task_queue_push_front(&source_queue, &task_c);
  iree_task_queue_push_front(&source_queue, &task_b);
  iree_task_queue_push_front(&source_queue, &task_a);

  EXPECT_EQ(&task_c, iree_task_queue_try_steal_until_success(&source_queue,
                                                             &target_queue, 1));
  EXPECT_TRUE(iree_task_queue_is_empty(&target_queue));

  EXPECT_EQ(&task_a, iree_task_queue_pop_front(&source_queue));
  EXPECT_EQ(&task_b, iree_task_queue_pop_front(&source_queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&source_queue));

  iree_task_queue_deinitialize(&source_queue);
  iree_task_queue_deinitialize(&target_queue);
}

TEST(QueueTest, TryStealIntoExisting) {
  iree_task_queue_t source_queue;
  iree_task_queue_initialize(&source_queue);
  iree_task_queue_t target_queue;
  iree_task_queue_initialize(&target_queue);

  iree_task_t task_a = {0};
  iree_task_t task_b = {0};
  iree_task_queue_push_front(&source_queue, &task_b);
  iree_task_queue_push_front(&source_queue, &task_a);

  iree_task_t task_existing = {0};
  iree_task_queue_push_front(&target_queue, &task_existing);

  EXPECT_EQ(&task_existing, iree_task_queue_try_steal_until_success(
                                &source_queue, &target_queue, 1));

  EXPECT_EQ(&task_a, iree_task_queue_pop_front(&source_queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&source_queue));

  EXPECT_EQ(&task_b, iree_task_queue_pop_front(&target_queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&target_queue));

  iree_task_queue_deinitialize(&source_queue);
  iree_task_queue_deinitialize(&target_queue);
}

TEST(QueueTest, TryStealMany) {
  iree_task_queue_t source_queue;
  iree_task_queue_initialize(&source_queue);
  iree_task_queue_t target_queue;
  iree_task_queue_initialize(&target_queue);

  iree_task_t task_a = {0};
  iree_task_t task_b = {0};
  iree_task_t task_c = {0};
  iree_task_t task_d = {0};
  iree_task_queue_push_front(&source_queue, &task_d);
  iree_task_queue_push_front(&source_queue, &task_c);
  iree_task_queue_push_front(&source_queue, &task_b);
  iree_task_queue_push_front(&source_queue, &task_a);

  EXPECT_EQ(&task_c, iree_task_queue_try_steal_until_success(&source_queue,
                                                             &target_queue, 2));
  EXPECT_EQ(&task_d, iree_task_queue_pop_front(&target_queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&target_queue));

  EXPECT_EQ(&task_a, iree_task_queue_pop_front(&source_queue));
  EXPECT_EQ(&task_b, iree_task_queue_pop_front(&source_queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&source_queue));

  iree_task_queue_deinitialize(&source_queue);
  iree_task_queue_deinitialize(&target_queue);
}

TEST(QueueTest, TryStealAll) {
  iree_task_queue_t source_queue;
  iree_task_queue_initialize(&source_queue);
  iree_task_queue_t target_queue;
  iree_task_queue_initialize(&target_queue);

  iree_task_t task_a = {0};
  iree_task_t task_b = {0};
  iree_task_t task_c = {0};
  iree_task_t task_d = {0};
  iree_task_queue_push_front(&source_queue, &task_d);
  iree_task_queue_push_front(&source_queue, &task_c);
  iree_task_queue_push_front(&source_queue, &task_b);
  iree_task_queue_push_front(&source_queue, &task_a);

  EXPECT_EQ(&task_c, iree_task_queue_try_steal_until_success(
                         &source_queue, &target_queue, 1000));
  EXPECT_EQ(&task_d, iree_task_queue_pop_front(&target_queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&target_queue));

  EXPECT_EQ(&task_a, iree_task_queue_pop_front(&source_queue));
  EXPECT_EQ(&task_b, iree_task_queue_pop_front(&source_queue));
  EXPECT_TRUE(iree_task_queue_is_empty(&source_queue));

  iree_task_queue_deinitialize(&source_queue);
  iree_task_queue_deinitialize(&target_queue);
}

}  // namespace
