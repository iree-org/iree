// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for timer operations.

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"

namespace iree::async::cts {

class TimerTest : public CtsTestBase<> {};

// Basic timer: fires after a short delay.
TEST_P(TimerTest, BasicTimer) {
  iree_async_timer_operation_t timer;
  memset(&timer, 0, sizeof(timer));
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

  // Set deadline 50ms in the future.
  iree_time_t start_time = iree_time_now();
  timer.deadline_ns = start_time + iree_make_duration_ms(50);  // 50ms

  CompletionTracker tracker;
  timer.base.completion_fn = CompletionTracker::Callback;
  timer.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));

  // Poll until the timer fires. 200ms budget should be plenty.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(200));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Verify the timer fired at or after the deadline.
  iree_time_t end_time = iree_time_now();
  EXPECT_GE(end_time, timer.deadline_ns);
}

// Timer with immediate deadline (already passed).
TEST_P(TimerTest, ImmediateTimer) {
  iree_async_timer_operation_t timer;
  memset(&timer, 0, sizeof(timer));
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

  // Deadline in the past - should fire immediately.
  timer.deadline_ns = iree_time_now() - 1000000LL;  // 1ms ago

  CompletionTracker tracker;
  timer.base.completion_fn = CompletionTracker::Callback;
  timer.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));

  // Should complete quickly since deadline is already passed.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(100));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Multiple timers with different deadlines complete in deadline order.
TEST_P(TimerTest, MultipleTimersOrder) {
  iree_async_timer_operation_t timer1, timer2, timer3;
  memset(&timer1, 0, sizeof(timer1));
  memset(&timer2, 0, sizeof(timer2));
  memset(&timer3, 0, sizeof(timer3));

  timer1.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer2.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer3.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

  iree_time_t now = iree_time_now();
  // Submit in reverse order of deadlines.
  timer1.deadline_ns = now + iree_make_duration_ms(100);  // 100ms - fires last
  timer2.deadline_ns = now + iree_make_duration_ms(50);   // 50ms - fires second
  timer3.deadline_ns = now + iree_make_duration_ms(25);   // 25ms - fires first

  // Track completion order via a shared vector.
  struct OrderTracker {
    std::vector<int> order;
    static void Callback1(void* user_data, iree_async_operation_t* op,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
      static_cast<OrderTracker*>(user_data)->order.push_back(1);
      iree_status_ignore(status);
    }
    static void Callback2(void* user_data, iree_async_operation_t* op,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
      static_cast<OrderTracker*>(user_data)->order.push_back(2);
      iree_status_ignore(status);
    }
    static void Callback3(void* user_data, iree_async_operation_t* op,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
      static_cast<OrderTracker*>(user_data)->order.push_back(3);
      iree_status_ignore(status);
    }
  };

  OrderTracker order_tracker;
  timer1.base.completion_fn = OrderTracker::Callback1;
  timer1.base.user_data = &order_tracker;
  timer2.base.completion_fn = OrderTracker::Callback2;
  timer2.base.user_data = &order_tracker;
  timer3.base.completion_fn = OrderTracker::Callback3;
  timer3.base.user_data = &order_tracker;

  // Submit all three.
  iree_async_operation_t* ops[] = {&timer1.base, &timer2.base, &timer3.base};
  iree_async_operation_list_t list = {ops, 3};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Wait for all three to complete.
  PollUntil(/*min_completions=*/3, /*total_budget=*/iree_make_duration_ms(300));

  // Verify order: timer3 (25ms), timer2 (50ms), timer1 (100ms).
  ASSERT_EQ(order_tracker.order.size(), 3u);
  EXPECT_EQ(order_tracker.order[0], 3);
  EXPECT_EQ(order_tracker.order[1], 2);
  EXPECT_EQ(order_tracker.order[2], 1);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(TimerTest);

}  // namespace iree::async::cts
