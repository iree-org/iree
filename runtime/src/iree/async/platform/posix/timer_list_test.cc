// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/timer_list.h"

#include "iree/testing/gtest.h"

namespace {

class TimerListTest : public ::testing::Test {
 protected:
  void SetUp() override { iree_async_posix_timer_list_initialize(&list_); }

  // Helper to create a timer with a given deadline.
  void InitTimer(iree_async_timer_operation_t* timer, iree_time_t deadline) {
    memset(timer, 0, sizeof(*timer));
    timer->base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
    timer->deadline_ns = deadline;
  }

  iree_async_posix_timer_list_t list_;
};

TEST_F(TimerListTest, EmptyList) {
  EXPECT_TRUE(iree_async_posix_timer_list_is_empty(&list_));
  EXPECT_EQ(iree_async_posix_timer_list_next_deadline_ns(&list_),
            IREE_TIME_INFINITE_FUTURE);
  EXPECT_EQ(iree_async_posix_timer_list_pop_expired(&list_, 1000), nullptr);
}

TEST_F(TimerListTest, SingleInsertRemove) {
  iree_async_timer_operation_t timer;
  InitTimer(&timer, 100);

  iree_async_posix_timer_list_insert(&list_, &timer);
  EXPECT_FALSE(iree_async_posix_timer_list_is_empty(&list_));
  EXPECT_TRUE(iree_async_posix_timer_list_contains(&list_, &timer));
  EXPECT_EQ(iree_async_posix_timer_list_next_deadline_ns(&list_), 100);

  iree_async_posix_timer_list_remove(&list_, &timer);
  EXPECT_TRUE(iree_async_posix_timer_list_is_empty(&list_));
  EXPECT_FALSE(iree_async_posix_timer_list_contains(&list_, &timer));
}

TEST_F(TimerListTest, InsertMaintainsOrder) {
  iree_async_timer_operation_t t1, t2, t3;
  InitTimer(&t1, 100);
  InitTimer(&t2, 50);
  InitTimer(&t3, 150);

  // Insert out of order.
  iree_async_posix_timer_list_insert(&list_, &t1);  // 100
  iree_async_posix_timer_list_insert(&list_, &t2);  // 50 (becomes head)
  iree_async_posix_timer_list_insert(&list_, &t3);  // 150 (becomes tail)

  // Should be ordered: t2(50) -> t1(100) -> t3(150)
  EXPECT_EQ(list_.head, &t2);
  EXPECT_EQ(list_.tail, &t3);
  EXPECT_EQ(iree_async_posix_timer_list_next_deadline_ns(&list_), 50);
}

TEST_F(TimerListTest, InsertEqualDeadlinesPreservesInsertOrder) {
  iree_async_timer_operation_t t1, t2, t3;
  InitTimer(&t1, 100);
  InitTimer(&t2, 100);
  InitTimer(&t3, 100);

  iree_async_posix_timer_list_insert(&list_, &t1);
  iree_async_posix_timer_list_insert(&list_, &t2);
  iree_async_posix_timer_list_insert(&list_, &t3);

  // All equal deadlines - should be in insert order: t1 -> t2 -> t3
  EXPECT_EQ(list_.head, &t1);
  EXPECT_EQ(t1.platform.posix.next, &t2);
  EXPECT_EQ(t2.platform.posix.next, &t3);
  EXPECT_EQ(list_.tail, &t3);
}

TEST_F(TimerListTest, RemoveHead) {
  iree_async_timer_operation_t t1, t2;
  InitTimer(&t1, 50);
  InitTimer(&t2, 100);

  iree_async_posix_timer_list_insert(&list_, &t1);
  iree_async_posix_timer_list_insert(&list_, &t2);

  iree_async_posix_timer_list_remove(&list_, &t1);
  EXPECT_EQ(list_.head, &t2);
  EXPECT_EQ(list_.tail, &t2);
  EXPECT_FALSE(iree_async_posix_timer_list_contains(&list_, &t1));
}

TEST_F(TimerListTest, RemoveTail) {
  iree_async_timer_operation_t t1, t2;
  InitTimer(&t1, 50);
  InitTimer(&t2, 100);

  iree_async_posix_timer_list_insert(&list_, &t1);
  iree_async_posix_timer_list_insert(&list_, &t2);

  iree_async_posix_timer_list_remove(&list_, &t2);
  EXPECT_EQ(list_.head, &t1);
  EXPECT_EQ(list_.tail, &t1);
  EXPECT_FALSE(iree_async_posix_timer_list_contains(&list_, &t2));
}

TEST_F(TimerListTest, RemoveMiddle) {
  iree_async_timer_operation_t t1, t2, t3;
  InitTimer(&t1, 50);
  InitTimer(&t2, 100);
  InitTimer(&t3, 150);

  iree_async_posix_timer_list_insert(&list_, &t1);
  iree_async_posix_timer_list_insert(&list_, &t2);
  iree_async_posix_timer_list_insert(&list_, &t3);

  iree_async_posix_timer_list_remove(&list_, &t2);
  EXPECT_EQ(list_.head, &t1);
  EXPECT_EQ(list_.tail, &t3);
  EXPECT_EQ(t1.platform.posix.next, &t3);
  EXPECT_EQ(t3.platform.posix.prev, &t1);
  EXPECT_FALSE(iree_async_posix_timer_list_contains(&list_, &t2));
}

TEST_F(TimerListTest, PopExpiredNoneExpired) {
  iree_async_timer_operation_t timer;
  InitTimer(&timer, 100);

  iree_async_posix_timer_list_insert(&list_, &timer);
  EXPECT_EQ(iree_async_posix_timer_list_pop_expired(&list_, 50), nullptr);
  EXPECT_TRUE(iree_async_posix_timer_list_contains(&list_, &timer));
}

TEST_F(TimerListTest, PopExpiredExactDeadline) {
  iree_async_timer_operation_t timer;
  InitTimer(&timer, 100);

  iree_async_posix_timer_list_insert(&list_, &timer);
  EXPECT_EQ(iree_async_posix_timer_list_pop_expired(&list_, 100), &timer);
  EXPECT_TRUE(iree_async_posix_timer_list_is_empty(&list_));
}

TEST_F(TimerListTest, PopExpiredPastDeadline) {
  iree_async_timer_operation_t timer;
  InitTimer(&timer, 100);

  iree_async_posix_timer_list_insert(&list_, &timer);
  EXPECT_EQ(iree_async_posix_timer_list_pop_expired(&list_, 200), &timer);
  EXPECT_TRUE(iree_async_posix_timer_list_is_empty(&list_));
}

TEST_F(TimerListTest, PopExpiredMultiple) {
  iree_async_timer_operation_t t1, t2, t3;
  InitTimer(&t1, 50);
  InitTimer(&t2, 100);
  InitTimer(&t3, 150);

  iree_async_posix_timer_list_insert(&list_, &t1);
  iree_async_posix_timer_list_insert(&list_, &t2);
  iree_async_posix_timer_list_insert(&list_, &t3);

  // Pop at time 120 - should get t1 and t2.
  EXPECT_EQ(iree_async_posix_timer_list_pop_expired(&list_, 120), &t1);
  EXPECT_EQ(iree_async_posix_timer_list_pop_expired(&list_, 120), &t2);
  EXPECT_EQ(iree_async_posix_timer_list_pop_expired(&list_, 120), nullptr);

  // t3 still in list.
  EXPECT_TRUE(iree_async_posix_timer_list_contains(&list_, &t3));
  EXPECT_EQ(list_.head, &t3);
}

TEST_F(TimerListTest, ContainsNotInList) {
  iree_async_timer_operation_t in_list, not_in_list;
  InitTimer(&in_list, 100);
  InitTimer(&not_in_list, 200);

  iree_async_posix_timer_list_insert(&list_, &in_list);

  EXPECT_TRUE(iree_async_posix_timer_list_contains(&list_, &in_list));
  EXPECT_FALSE(iree_async_posix_timer_list_contains(&list_, &not_in_list));
}

TEST_F(TimerListTest, MonotonicDeadlinesFastPath) {
  iree_async_timer_operation_t t1, t2, t3, t4, t5;
  InitTimer(&t1, 100);
  InitTimer(&t2, 200);
  InitTimer(&t3, 300);
  InitTimer(&t4, 400);
  InitTimer(&t5, 500);

  // Insert in monotonically increasing order (exercises the tail fast path).
  iree_async_posix_timer_list_insert(&list_, &t1);
  iree_async_posix_timer_list_insert(&list_, &t2);
  iree_async_posix_timer_list_insert(&list_, &t3);
  iree_async_posix_timer_list_insert(&list_, &t4);
  iree_async_posix_timer_list_insert(&list_, &t5);

  // Verify ordering: t1 -> t2 -> t3 -> t4 -> t5.
  EXPECT_EQ(list_.head, &t1);
  EXPECT_EQ(t1.platform.posix.next, &t2);
  EXPECT_EQ(t2.platform.posix.next, &t3);
  EXPECT_EQ(t3.platform.posix.next, &t4);
  EXPECT_EQ(t4.platform.posix.next, &t5);
  EXPECT_EQ(list_.tail, &t5);

  // Verify backward links.
  EXPECT_EQ(t5.platform.posix.prev, &t4);
  EXPECT_EQ(t4.platform.posix.prev, &t3);
  EXPECT_EQ(t3.platform.posix.prev, &t2);
  EXPECT_EQ(t2.platform.posix.prev, &t1);
  EXPECT_EQ(t1.platform.posix.prev, nullptr);
}

TEST_F(TimerListTest, MixedInsertionOrder) {
  iree_async_timer_operation_t t1, t2, t3, t4, t5;
  InitTimer(&t1, 300);
  InitTimer(&t2, 100);
  InitTimer(&t3, 500);
  InitTimer(&t4, 200);
  InitTimer(&t5, 400);

  // Insert in mixed order: some hit the fast path, some walk.
  iree_async_posix_timer_list_insert(&list_,
                                     &t1);  // 300 (first, becomes head+tail)
  iree_async_posix_timer_list_insert(&list_,
                                     &t2);  // 100 (walk, insert at head)
  iree_async_posix_timer_list_insert(&list_,
                                     &t3);  // 500 (fast path, append at tail)
  iree_async_posix_timer_list_insert(&list_,
                                     &t4);  // 200 (walk, insert in middle)
  iree_async_posix_timer_list_insert(&list_,
                                     &t5);  // 400 (walk, insert in middle)

  // Expected order: t2(100) -> t4(200) -> t1(300) -> t5(400) -> t3(500).
  EXPECT_EQ(list_.head, &t2);
  EXPECT_EQ(t2.platform.posix.next, &t4);
  EXPECT_EQ(t4.platform.posix.next, &t1);
  EXPECT_EQ(t1.platform.posix.next, &t5);
  EXPECT_EQ(t5.platform.posix.next, &t3);
  EXPECT_EQ(list_.tail, &t3);
}

TEST_F(TimerListTest, ReinsertAfterRemove) {
  iree_async_timer_operation_t timer;
  InitTimer(&timer, 100);

  iree_async_posix_timer_list_insert(&list_, &timer);
  iree_async_posix_timer_list_remove(&list_, &timer);
  EXPECT_TRUE(iree_async_posix_timer_list_is_empty(&list_));

  // Change deadline and reinsert.
  timer.deadline_ns = 200;
  iree_async_posix_timer_list_insert(&list_, &timer);
  EXPECT_TRUE(iree_async_posix_timer_list_contains(&list_, &timer));
  EXPECT_EQ(iree_async_posix_timer_list_next_deadline_ns(&list_), 200);
}

}  // namespace
