// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/list.h"

#include "iree/task/testing/test_util.h"
#include "iree/testing/gtest.h"

namespace {

TEST(TaskListTest, Empty) {
  iree_task_list_t list;
  iree_task_list_initialize(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));
  EXPECT_EQ(0, iree_task_list_calculate_size(&list));
  iree_task_list_discard(&list);
}

TEST(TaskListTest, CalculateSize) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);

  EXPECT_TRUE(iree_task_list_is_empty(&list));
  EXPECT_EQ(0, iree_task_list_calculate_size(&list));

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list, task0);
  EXPECT_FALSE(iree_task_list_is_empty(&list));
  EXPECT_EQ(1, iree_task_list_calculate_size(&list));

  iree_task_list_push_back(&list, task1);
  EXPECT_EQ(2, iree_task_list_calculate_size(&list));
  iree_task_list_push_back(&list, task2);
  EXPECT_EQ(3, iree_task_list_calculate_size(&list));
  iree_task_list_push_back(&list, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list));
}

TEST(TaskListTest, Move) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list_a, list_b;
  iree_task_list_initialize(&list_a);
  iree_task_list_initialize(&list_b);

  EXPECT_TRUE(iree_task_list_is_empty(&list_a));
  EXPECT_TRUE(iree_task_list_is_empty(&list_b));

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);
  iree_task_list_push_back(&list_a, task0);
  iree_task_list_push_back(&list_a, task1);
  iree_task_list_push_back(&list_a, task2);
  iree_task_list_push_back(&list_a, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list_a));
  EXPECT_TRUE(CheckListOrderFIFO(&list_a));

  iree_task_list_move(&list_a, &list_b);
  EXPECT_TRUE(iree_task_list_is_empty(&list_a));
  EXPECT_EQ(4, iree_task_list_calculate_size(&list_b));
  EXPECT_TRUE(CheckListOrderFIFO(&list_b));
}

TEST(TaskListTest, DiscardEmpty) {
  iree_task_list_t list;
  iree_task_list_initialize(&list);

  EXPECT_TRUE(iree_task_list_is_empty(&list));
  iree_task_list_discard(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));
}

TEST(TaskListTest, Discard) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);
  iree_task_list_push_back(&list, task0);
  iree_task_list_push_back(&list, task1);
  iree_task_list_push_back(&list, task2);
  iree_task_list_push_back(&list, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));

  iree_task_list_discard(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));

  // IMPLICIT: if the tasks were not released back to the pool we'll leak.
}

TEST(TaskListTest, DiscardSequence) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);
  iree_task_set_completion_task(task0, task1);
  iree_task_set_completion_task(task1, task2);
  iree_task_set_completion_task(task2, task3);
  iree_task_list_push_back(&list, task0);
  iree_task_list_push_back(&list, task1);
  iree_task_list_push_back(&list, task2);
  iree_task_list_push_back(&list, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));

  iree_task_list_discard(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));

  // IMPLICIT: if the tasks were not released back to the pool we'll leak.
}

TEST(TaskListTest, DiscardJoin) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);
  iree_task_set_completion_task(task0, task3);
  iree_task_set_completion_task(task1, task3);
  iree_task_set_completion_task(task2, task3);
  iree_task_list_push_back(&list, task0);
  iree_task_list_push_back(&list, task1);
  iree_task_list_push_back(&list, task2);
  iree_task_list_push_back(&list, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));

  iree_task_list_discard(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));

  // IMPLICIT: if the tasks were not released back to the pool we'll leak.
}

TEST(TaskListTest, PushFront) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_front(&list, task0);
  iree_task_list_push_front(&list, task1);
  iree_task_list_push_front(&list, task2);
  iree_task_list_push_front(&list, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderLIFO(&list));

  EXPECT_EQ(3, iree_task_list_pop_front(&list)->flags);
  EXPECT_EQ(2, iree_task_list_pop_front(&list)->flags);
  EXPECT_EQ(1, iree_task_list_pop_front(&list)->flags);
  EXPECT_EQ(0, iree_task_list_pop_front(&list)->flags);
  EXPECT_TRUE(iree_task_list_is_empty(&list));
}

TEST(TaskListTest, PopFront) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list, task0);
  iree_task_list_push_back(&list, task1);
  iree_task_list_push_back(&list, task2);
  iree_task_list_push_back(&list, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));

  EXPECT_EQ(0, iree_task_list_pop_front(&list)->flags);
  EXPECT_EQ(1, iree_task_list_pop_front(&list)->flags);
  EXPECT_EQ(2, iree_task_list_pop_front(&list)->flags);
  EXPECT_EQ(3, iree_task_list_pop_front(&list)->flags);
  EXPECT_TRUE(iree_task_list_is_empty(&list));
}

TEST(TaskListTest, Erase) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list, task0);
  iree_task_list_push_back(&list, task1);
  iree_task_list_push_back(&list, task2);
  iree_task_list_push_back(&list, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));

  // Remove head.
  iree_task_list_erase(&list, NULL, task0);
  EXPECT_EQ(3, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));
  EXPECT_EQ(task1, iree_task_list_front(&list));

  // Remove tail.
  iree_task_list_erase(&list, task2, task3);
  EXPECT_EQ(2, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));
  EXPECT_EQ(task2, iree_task_list_back(&list));

  // Remove the rest.
  iree_task_list_erase(&list, task1, task2);
  EXPECT_EQ(1, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));
  EXPECT_EQ(task1, iree_task_list_front(&list));
  EXPECT_EQ(task1, iree_task_list_back(&list));

  iree_task_list_erase(&list, NULL, task1);
  EXPECT_TRUE(iree_task_list_is_empty(&list));
  EXPECT_EQ(NULL, iree_task_list_front(&list));
  EXPECT_EQ(NULL, iree_task_list_back(&list));
}

TEST(TaskListTest, PrependEmpty) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list_a, list_b;
  iree_task_list_initialize(&list_a);
  iree_task_list_initialize(&list_b);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);

  iree_task_list_push_back(&list_a, task0);
  iree_task_list_push_back(&list_a, task1);

  EXPECT_TRUE(iree_task_list_is_empty(&list_b));
  iree_task_list_prepend(&list_a, &list_b);
  EXPECT_EQ(2, iree_task_list_calculate_size(&list_a));
  EXPECT_TRUE(CheckListOrderFIFO(&list_a));
}

TEST(TaskListTest, PrependIntoEmpty) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list_a, list_b;
  iree_task_list_initialize(&list_a);
  iree_task_list_initialize(&list_b);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list_b, task0);
  iree_task_list_push_back(&list_b, task1);
  iree_task_list_push_back(&list_b, task2);
  iree_task_list_push_back(&list_b, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list_b));
  EXPECT_TRUE(CheckListOrderFIFO(&list_b));

  EXPECT_TRUE(iree_task_list_is_empty(&list_a));
  iree_task_list_prepend(&list_a, &list_b);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list_a));
  EXPECT_TRUE(CheckListOrderFIFO(&list_a));
  EXPECT_TRUE(iree_task_list_is_empty(&list_b));
}

TEST(TaskListTest, PrependInto1) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list_a, list_b;
  iree_task_list_initialize(&list_a);
  iree_task_list_initialize(&list_b);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list_b, task0);
  iree_task_list_push_back(&list_b, task1);
  iree_task_list_push_back(&list_b, task2);

  iree_task_list_push_back(&list_a, task3);
  iree_task_list_prepend(&list_a, &list_b);

  EXPECT_EQ(4, iree_task_list_calculate_size(&list_a));
  EXPECT_TRUE(CheckListOrderFIFO(&list_a));
  EXPECT_TRUE(iree_task_list_is_empty(&list_b));
}

TEST(TaskListTest, PrependInto2) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list_a, list_b;
  iree_task_list_initialize(&list_a);
  iree_task_list_initialize(&list_b);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list_b, task0);
  iree_task_list_push_back(&list_b, task1);
  iree_task_list_push_back(&list_a, task2);
  iree_task_list_push_back(&list_a, task3);
  iree_task_list_prepend(&list_a, &list_b);

  EXPECT_EQ(4, iree_task_list_calculate_size(&list_a));
  EXPECT_TRUE(CheckListOrderFIFO(&list_a));
  EXPECT_TRUE(iree_task_list_is_empty(&list_b));
}

TEST(TaskListTest, AppendIntoEmpty) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list_a, list_b;
  iree_task_list_initialize(&list_a);
  iree_task_list_initialize(&list_b);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list_b, task0);
  iree_task_list_push_back(&list_b, task1);
  iree_task_list_push_back(&list_b, task2);
  iree_task_list_push_back(&list_b, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list_b));
  EXPECT_TRUE(CheckListOrderFIFO(&list_b));

  EXPECT_TRUE(iree_task_list_is_empty(&list_a));
  iree_task_list_append(&list_a, &list_b);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list_a));
  EXPECT_TRUE(CheckListOrderFIFO(&list_a));
  EXPECT_TRUE(iree_task_list_is_empty(&list_b));
}

TEST(TaskListTest, AppendInto1) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list_a, list_b;
  iree_task_list_initialize(&list_a);
  iree_task_list_initialize(&list_b);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list_b, task1);
  iree_task_list_push_back(&list_b, task2);

  iree_task_list_push_back(&list_b, task3);
  iree_task_list_push_back(&list_a, task0);

  iree_task_list_append(&list_a, &list_b);

  EXPECT_EQ(4, iree_task_list_calculate_size(&list_a));
  EXPECT_TRUE(CheckListOrderFIFO(&list_a));
  EXPECT_TRUE(iree_task_list_is_empty(&list_b));
}

TEST(TaskListTest, AppendInto2) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list_a, list_b;
  iree_task_list_initialize(&list_a);
  iree_task_list_initialize(&list_b);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list_b, task2);
  iree_task_list_push_back(&list_b, task3);

  iree_task_list_push_back(&list_a, task0);
  iree_task_list_push_back(&list_a, task1);

  iree_task_list_append(&list_a, &list_b);

  EXPECT_EQ(4, iree_task_list_calculate_size(&list_a));
  EXPECT_TRUE(CheckListOrderFIFO(&list_a));
  EXPECT_TRUE(iree_task_list_is_empty(&list_b));
}

TEST(TaskListTest, Reverse0) {
  iree_task_list_t list;
  iree_task_list_initialize(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));
  iree_task_list_reverse(&list);
  EXPECT_TRUE(iree_task_list_is_empty(&list));
}

TEST(TaskListTest, Reverse1) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);

  auto task0 = AcquireNopTask(pool, scope, 0);

  iree_task_list_push_back(&list, task0);
  EXPECT_EQ(1, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));
  iree_task_list_reverse(&list);
  EXPECT_TRUE(CheckListOrderLIFO(&list));
}

TEST(TaskListTest, Reverse2) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);

  iree_task_list_push_back(&list, task0);
  iree_task_list_push_back(&list, task1);
  EXPECT_EQ(2, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));
  iree_task_list_reverse(&list);
  EXPECT_TRUE(CheckListOrderLIFO(&list));
}

TEST(TaskListTest, Reverse4) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t list;
  iree_task_list_initialize(&list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&list, task0);
  iree_task_list_push_back(&list, task1);
  iree_task_list_push_back(&list, task2);
  iree_task_list_push_back(&list, task3);
  EXPECT_EQ(4, iree_task_list_calculate_size(&list));
  EXPECT_TRUE(CheckListOrderFIFO(&list));
  iree_task_list_reverse(&list);
  EXPECT_TRUE(CheckListOrderLIFO(&list));
}

TEST(TaskListTest, SplitEmpty) {
  iree_task_list_t head_list;
  iree_task_list_initialize(&head_list);

  iree_task_list_t tail_list;
  iree_task_list_split(&head_list, /*max_tasks=*/64, &tail_list);

  EXPECT_TRUE(iree_task_list_is_empty(&head_list));
  EXPECT_TRUE(iree_task_list_is_empty(&tail_list));
}

TEST(TaskListTest, Split1) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t head_list;
  iree_task_list_initialize(&head_list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  iree_task_list_push_back(&head_list, task0);
  EXPECT_EQ(1, iree_task_list_calculate_size(&head_list));

  iree_task_list_t tail_list;
  iree_task_list_split(&head_list, /*max_tasks=*/64, &tail_list);

  EXPECT_TRUE(iree_task_list_is_empty(&head_list));
  EXPECT_EQ(1, iree_task_list_calculate_size(&tail_list));
}

TEST(TaskListTest, Split2) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t head_list;
  iree_task_list_initialize(&head_list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);

  iree_task_list_push_back(&head_list, task0);
  iree_task_list_push_back(&head_list, task1);

  iree_task_list_t tail_list;
  iree_task_list_split(&head_list, /*max_tasks=*/64, &tail_list);

  EXPECT_EQ(1, iree_task_list_calculate_size(&head_list));
  EXPECT_TRUE(CheckListOrderFIFO(&head_list));
  EXPECT_EQ(1, iree_task_list_calculate_size(&tail_list));
  EXPECT_TRUE(CheckListOrderFIFO(&tail_list));
}

TEST(TaskListTest, Split3) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t head_list;
  iree_task_list_initialize(&head_list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);

  iree_task_list_push_back(&head_list, task0);
  iree_task_list_push_back(&head_list, task1);
  iree_task_list_push_back(&head_list, task2);

  iree_task_list_t tail_list;
  iree_task_list_split(&head_list, /*max_tasks=*/64, &tail_list);

  EXPECT_EQ(1, iree_task_list_calculate_size(&head_list));
  EXPECT_TRUE(CheckListOrderFIFO(&head_list));
  EXPECT_EQ(2, iree_task_list_calculate_size(&tail_list));
  EXPECT_TRUE(CheckListOrderFIFO(&tail_list));
}

TEST(TaskListTest, Split4) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t head_list;
  iree_task_list_initialize(&head_list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&head_list, task0);
  iree_task_list_push_back(&head_list, task1);
  iree_task_list_push_back(&head_list, task2);
  iree_task_list_push_back(&head_list, task3);

  iree_task_list_t tail_list;
  iree_task_list_split(&head_list, /*max_tasks=*/64, &tail_list);

  EXPECT_EQ(2, iree_task_list_calculate_size(&head_list));
  EXPECT_TRUE(CheckListOrderFIFO(&head_list));
  EXPECT_EQ(2, iree_task_list_calculate_size(&tail_list));
  EXPECT_TRUE(CheckListOrderFIFO(&tail_list));
}

TEST(TaskListTest, SplitMaxTasks1) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t head_list;
  iree_task_list_initialize(&head_list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&head_list, task0);
  iree_task_list_push_back(&head_list, task1);
  iree_task_list_push_back(&head_list, task2);
  iree_task_list_push_back(&head_list, task3);

  iree_task_list_t tail_list;
  iree_task_list_split(&head_list, /*max_tasks=*/1, &tail_list);

  EXPECT_EQ(3, iree_task_list_calculate_size(&head_list));
  EXPECT_TRUE(CheckListOrderFIFO(&head_list));
  EXPECT_EQ(1, iree_task_list_calculate_size(&tail_list));
  EXPECT_TRUE(CheckListOrderFIFO(&tail_list));
}

TEST(TaskListTest, SplitMaxTasks2) {
  auto pool = AllocateNopPool();
  auto scope = AllocateScope("a");

  iree_task_list_t head_list;
  iree_task_list_initialize(&head_list);

  auto task0 = AcquireNopTask(pool, scope, 0);
  auto task1 = AcquireNopTask(pool, scope, 1);
  auto task2 = AcquireNopTask(pool, scope, 2);
  auto task3 = AcquireNopTask(pool, scope, 3);

  iree_task_list_push_back(&head_list, task0);
  iree_task_list_push_back(&head_list, task1);
  iree_task_list_push_back(&head_list, task2);
  iree_task_list_push_back(&head_list, task3);

  iree_task_list_t tail_list;
  iree_task_list_split(&head_list, /*max_tasks=*/2, &tail_list);

  EXPECT_EQ(2, iree_task_list_calculate_size(&head_list));
  EXPECT_TRUE(CheckListOrderFIFO(&head_list));
  EXPECT_EQ(2, iree_task_list_calculate_size(&tail_list));
  EXPECT_TRUE(CheckListOrderFIFO(&tail_list));
}

}  // namespace
