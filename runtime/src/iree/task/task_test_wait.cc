// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <chrono>
#include <thread>

#include "iree/task/task.h"
#include "iree/task/testing/task_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

// NOTE: we intentionally perform most signaling to/from C++ std::threads.
// This models a real application that may be passing in handles tied to custom
// or system primitives unrelated to the task system.

class TaskWaitTest : public TaskTest {};

// Issues a wait task on a handle that has already been signaled.
// The poller will query the status of the handle and immediately retire the
// task.
TEST_F(TaskWaitTest, IssueSignaled) {
  IREE_TRACE_SCOPE();

  iree_event_t event;
  iree_event_initialize(/*initial_state=*/true, &event);

  iree_task_wait_t task;
  iree_task_wait_initialize(&scope_, iree_event_await(&event),
                            IREE_TIME_INFINITE_FUTURE, &task);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
  IREE_EXPECT_OK(iree_task_scope_consume_status(&scope_));

  iree_event_deinitialize(&event);
}

// Issues a wait task on an unsignaled handle such that the poller must wait.
// We'll spin up a thread that sets it a short time in the future and ensure
// that the poller woke and retired the task.
TEST_F(TaskWaitTest, IssueUnsignaled) {
  IREE_TRACE_SCOPE();

  iree_event_t event;
  iree_event_initialize(/*initial_state=*/false, &event);

  iree_task_wait_t task;
  iree_task_wait_initialize(&scope_, iree_event_await(&event),
                            IREE_TIME_INFINITE_FUTURE, &task);

  // Spin up a thread that will signal the event after we start waiting on it.
  std::atomic<bool> has_signaled = {false};
  std::thread signal_thread([&]() {
    IREE_TRACE_SCOPE();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_FALSE(has_signaled);
    has_signaled = true;
    iree_event_set(&event);
  });

  EXPECT_FALSE(has_signaled);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
  EXPECT_TRUE(has_signaled);
  IREE_EXPECT_OK(iree_task_scope_consume_status(&scope_));

  signal_thread.join();
  iree_event_deinitialize(&event);
}

// Issues a wait task on a handle that will never be signaled.
// We set the deadline in the near future and ensure that the poller correctly
// fails the wait with a DEADLINE_EXCEEDED.
TEST_F(TaskWaitTest, IssueTimeout) {
  IREE_TRACE_SCOPE();

  iree_event_t event;
  iree_event_initialize(/*initial_state=*/false, &event);

  iree_task_wait_t task;
  iree_task_wait_initialize(&scope_, iree_event_await(&event),
                            iree_time_now() + (150 * 1000000), &task);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
  EXPECT_THAT(Status(iree_task_scope_consume_status(&scope_)),
              StatusIs(StatusCode::kDeadlineExceeded));

  iree_event_deinitialize(&event);
}

// Issues a delay task that should wait until the requested time.
// NOTE: this kind of test can be flaky - if we have issues we can bump the
// sleep time up.
TEST_F(TaskWaitTest, IssueDelay) {
  IREE_TRACE_SCOPE();

  iree_time_t start_time_ns = iree_time_now();

  iree_task_wait_t task;
  iree_task_wait_initialize_delay(&scope_, start_time_ns + (50 * 1000000),
                                  &task);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
  IREE_EXPECT_OK(iree_task_scope_consume_status(&scope_));

  iree_time_t end_time_ns = iree_time_now();
  EXPECT_GE(end_time_ns - start_time_ns, 25 * 1000000);
}

// Issues multiple waits that join on a single task. This models a wait-all.
TEST_F(TaskWaitTest, WaitAll) {
  IREE_TRACE_SCOPE();

  iree_event_t event_a;
  iree_event_initialize(/*initial_state=*/false, &event_a);
  iree_task_wait_t task_a;
  iree_task_wait_initialize(&scope_, iree_event_await(&event_a),
                            IREE_TIME_INFINITE_FUTURE, &task_a);

  iree_event_t event_b;
  iree_event_initialize(/*initial_state=*/false, &event_b);
  iree_task_wait_t task_b;
  iree_task_wait_initialize(&scope_, iree_event_await(&event_b),
                            IREE_TIME_INFINITE_FUTURE, &task_b);

  iree_task_t* wait_tasks[] = {&task_a.header, &task_b.header};
  iree_task_barrier_t barrier;
  iree_task_barrier_initialize(&scope_, IREE_ARRAYSIZE(wait_tasks), wait_tasks,
                               &barrier);

  iree_task_fence_t fence;
  iree_task_fence_initialize(&scope_, iree_wait_primitive_immediate(), &fence);
  iree_task_set_completion_task(&task_a.header, &fence.header);
  iree_task_set_completion_task(&task_b.header, &fence.header);

  // Spin up a thread that will signal the event after we start waiting on it.
  std::atomic<bool> has_signaled = {false};
  std::thread signal_thread([&]() {
    IREE_TRACE_SCOPE();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(has_signaled);
    iree_event_set(&event_a);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    has_signaled = true;
    iree_event_set(&event_b);
  });

  EXPECT_FALSE(has_signaled);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&barrier.header, &fence.header));
  EXPECT_TRUE(has_signaled);
  IREE_EXPECT_OK(iree_task_scope_consume_status(&scope_));

  signal_thread.join();
  iree_event_deinitialize(&event_a);
  iree_event_deinitialize(&event_b);
}

// Issues multiple waits that join on a single task but where one times out.
TEST_F(TaskWaitTest, WaitAllTimeout) {
  IREE_TRACE_SCOPE();

  iree_event_t event_a;
  iree_event_initialize(/*initial_state=*/true, &event_a);
  iree_task_wait_t task_a;
  iree_task_wait_initialize(&scope_, iree_event_await(&event_a),
                            IREE_TIME_INFINITE_FUTURE, &task_a);

  iree_event_t event_b;
  iree_event_initialize(/*initial_state=*/false, &event_b);
  iree_task_wait_t task_b;
  iree_task_wait_initialize(&scope_, iree_event_await(&event_b),
                            iree_time_now() + (50 * 1000000), &task_b);

  iree_task_t* wait_tasks[] = {&task_a.header, &task_b.header};
  iree_task_barrier_t barrier;
  iree_task_barrier_initialize(&scope_, IREE_ARRAYSIZE(wait_tasks), wait_tasks,
                               &barrier);

  iree_task_fence_t fence;
  iree_task_fence_initialize(&scope_, iree_wait_primitive_immediate(), &fence);
  iree_task_set_completion_task(&task_a.header, &fence.header);
  iree_task_set_completion_task(&task_b.header, &fence.header);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&barrier.header, &fence.header));
  EXPECT_THAT(Status(iree_task_scope_consume_status(&scope_)),
              StatusIs(StatusCode::kDeadlineExceeded));

  iree_event_deinitialize(&event_a);
  iree_event_deinitialize(&event_b);
}

// Issues multiple waits that join on a single task in wait-any mode.
// This means that if one wait finishes all other waits will be cancelled and
// the completion task will continue.
//
// Here event_a is signaled but event_b is not.
TEST_F(TaskWaitTest, WaitAny) {
  IREE_TRACE_SCOPE();

  // Flag shared between all waits in a group.
  iree_atomic_int32_t cancellation_flag = IREE_ATOMIC_VAR_INIT(0);

  iree_event_t event_a;
  iree_event_initialize(/*initial_state=*/false, &event_a);
  iree_task_wait_t task_a;
  iree_task_wait_initialize(&scope_, iree_event_await(&event_a),
                            IREE_TIME_INFINITE_FUTURE, &task_a);
  iree_task_wait_set_wait_any(&task_a, &cancellation_flag);

  iree_event_t event_b;
  iree_event_initialize(/*initial_state=*/false, &event_b);
  iree_task_wait_t task_b;
  iree_task_wait_initialize(&scope_, iree_event_await(&event_b),
                            IREE_TIME_INFINITE_FUTURE, &task_b);
  iree_task_wait_set_wait_any(&task_b, &cancellation_flag);

  iree_task_t* wait_tasks[] = {&task_a.header, &task_b.header};
  iree_task_barrier_t barrier;
  iree_task_barrier_initialize(&scope_, IREE_ARRAYSIZE(wait_tasks), wait_tasks,
                               &barrier);

  iree_task_fence_t fence;
  iree_task_fence_initialize(&scope_, iree_wait_primitive_immediate(), &fence);
  iree_task_set_completion_task(&task_a.header, &fence.header);
  iree_task_set_completion_task(&task_b.header, &fence.header);

  // Spin up a thread that will signal the event after we start waiting on it.
  std::atomic<bool> has_signaled = {false};
  std::thread signal_thread([&]() {
    IREE_TRACE_SCOPE();
    // NOTE: we only signal event_a - event_b remains unsignaled.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(has_signaled);
    has_signaled = true;
    iree_event_set(&event_a);
  });

  EXPECT_FALSE(has_signaled);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&barrier.header, &fence.header));
  EXPECT_TRUE(has_signaled);
  IREE_EXPECT_OK(iree_task_scope_consume_status(&scope_));

  signal_thread.join();
  iree_event_deinitialize(&event_a);
  iree_event_deinitialize(&event_b);
}

// Issues multiple waits that join on a single task in wait-any mode.
// Here instead of signaling anything we cause event_a to timeout so that the
// entire wait is cancelled.
TEST_F(TaskWaitTest, WaitAnyTimeout) {
  IREE_TRACE_SCOPE();

  // Flag shared between all waits in a group.
  iree_atomic_int32_t cancellation_flag = IREE_ATOMIC_VAR_INIT(0);

  iree_event_t event_a;
  iree_event_initialize(/*initial_state=*/false, &event_a);
  iree_task_wait_t task_a;
  iree_task_wait_initialize(&scope_, iree_event_await(&event_a),
                            iree_time_now() + (50 * 1000000), &task_a);
  iree_task_wait_set_wait_any(&task_a, &cancellation_flag);

  iree_event_t event_b;
  iree_event_initialize(/*initial_state=*/false, &event_b);
  iree_task_wait_t task_b;
  iree_task_wait_initialize(&scope_, iree_event_await(&event_b),
                            IREE_TIME_INFINITE_FUTURE, &task_b);
  iree_task_wait_set_wait_any(&task_b, &cancellation_flag);

  iree_task_t* wait_tasks[] = {&task_a.header, &task_b.header};
  iree_task_barrier_t barrier;
  iree_task_barrier_initialize(&scope_, IREE_ARRAYSIZE(wait_tasks), wait_tasks,
                               &barrier);

  iree_task_fence_t fence;
  iree_task_fence_initialize(&scope_, iree_wait_primitive_immediate(), &fence);
  iree_task_set_completion_task(&task_a.header, &fence.header);
  iree_task_set_completion_task(&task_b.header, &fence.header);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&barrier.header, &fence.header));
  EXPECT_THAT(Status(iree_task_scope_consume_status(&scope_)),
              StatusIs(StatusCode::kDeadlineExceeded));

  iree_event_deinitialize(&event_a);
  iree_event_deinitialize(&event_b);
}

}  // namespace
