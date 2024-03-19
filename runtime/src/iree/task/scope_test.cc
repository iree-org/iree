// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/scope.h"

#include <chrono>
#include <thread>

#include "iree/task/submission.h"
#include "iree/task/task_impl.h"
#include "iree/testing/gtest.h"

namespace {

TEST(ScopeTest, Lifetime) {
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("scope_a"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  iree_task_scope_deinitialize(&scope);
}

// NOTE: the exact capacity (and whether we store the name at all) is an
// implementation detail.
TEST(ScopeTest, LongNameTruncation) {
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("01234567890123456789"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);
  EXPECT_TRUE(iree_string_view_equal(iree_make_cstring_view("012345678901234"),
                                     iree_task_scope_name(&scope)));
  iree_task_scope_deinitialize(&scope);
}

TEST(ScopeTest, AbortEmpty) {
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("scope_a"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);

  // Current state is OK.
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  EXPECT_TRUE(iree_status_is_ok(iree_task_scope_consume_status(&scope)));

  // Enter aborted state.
  iree_task_scope_abort(&scope);
  iree_status_t consumed_status = iree_task_scope_consume_status(&scope);
  EXPECT_TRUE(iree_status_is_aborted(consumed_status));
  iree_status_ignore(consumed_status);

  // Ensure aborted state is sticky.
  EXPECT_TRUE(iree_status_is_aborted(iree_task_scope_consume_status(&scope)));

  iree_task_scope_deinitialize(&scope);
}

TEST(ScopeTest, FailEmpty) {
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("scope_a"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);

  // Current state is OK.
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  EXPECT_TRUE(iree_status_is_ok(iree_task_scope_consume_status(&scope)));

  // Enter failure state.
  iree_task_t failed_task = {0};
  failed_task.scope = &scope;
  iree_task_scope_fail(failed_task.scope,
                       iree_make_status(IREE_STATUS_DATA_LOSS, "whoops!"));
  iree_status_t consumed_status = iree_task_scope_consume_status(&scope);
  EXPECT_TRUE(iree_status_is_data_loss(consumed_status));
  iree_status_ignore(consumed_status);

  // Ensure failure state is sticky.
  EXPECT_TRUE(iree_status_is_data_loss(iree_task_scope_consume_status(&scope)));

  iree_task_scope_deinitialize(&scope);
}

// NOTE: only the first failure is recorded and made sticky; subsequent failure
// calls are ignored.
TEST(ScopeTest, FailAgain) {
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("scope_a"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);

  // Current state is OK.
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  EXPECT_TRUE(iree_status_is_ok(iree_task_scope_consume_status(&scope)));

  // Enter initial failure state.
  iree_task_t failed_task_a = {0};
  failed_task_a.scope = &scope;
  iree_task_scope_fail(failed_task_a.scope,
                       iree_make_status(IREE_STATUS_DATA_LOSS, "whoops 1"));
  iree_status_t consumed_status_a = iree_task_scope_consume_status(&scope);
  EXPECT_TRUE(iree_status_is_data_loss(consumed_status_a));
  iree_status_ignore(consumed_status_a);

  // Ensure failure s tate is sticky.
  EXPECT_TRUE(iree_status_is_data_loss(iree_task_scope_consume_status(&scope)));

  // Try failing again - it should be ignored and correctly iree_status_free'd.
  iree_task_t failed_task_b = {0};
  failed_task_b.scope = &scope;
  iree_task_scope_fail(
      failed_task_b.scope,
      iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "whoops 2"));
  iree_status_t consumed_status_b = iree_task_scope_consume_status(&scope);
  EXPECT_TRUE(iree_status_is_data_loss(consumed_status_b));
  iree_status_ignore(consumed_status_b);

  // Still the first failure status.
  EXPECT_TRUE(iree_status_is_data_loss(iree_task_scope_consume_status(&scope)));

  iree_task_scope_deinitialize(&scope);
}

TEST(ScopeTest, WaitIdleWhenIdle) {
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("scope_a"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);

  // Current state is OK and idle.
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  EXPECT_TRUE(iree_status_is_ok(iree_task_scope_consume_status(&scope)));

  // Wait until idle... which is now.
  EXPECT_TRUE(iree_status_is_ok(
      iree_task_scope_wait_idle(&scope, IREE_TIME_INFINITE_FUTURE)));
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));

  iree_task_scope_deinitialize(&scope);
}

TEST(ScopeTest, WaitIdleDeadlineExceeded) {
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("scope_a"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);

  // Current state is OK and idle.
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  EXPECT_TRUE(iree_status_is_ok(iree_task_scope_consume_status(&scope)));

  // Enqueue a task to the scope so it is no longer idle.
  iree_task_fence_t fence_task;
  iree_task_fence_initialize(&scope, iree_wait_primitive_immediate(),
                             &fence_task);
  EXPECT_FALSE(iree_task_scope_is_idle(&scope));

  // Poll, which should fail immediately because we have the outstanding task.
  iree_status_t wait_status =
      iree_task_scope_wait_idle(&scope, IREE_TIME_INFINITE_PAST);
  EXPECT_TRUE(iree_status_is_deadline_exceeded(wait_status));
  EXPECT_FALSE(iree_task_scope_is_idle(&scope));

  // Complete the task (required as part of the scope contract).
  iree_task_submission_t pending_submission;
  iree_task_submission_initialize(&pending_submission);
  iree_task_fence_retire(&fence_task, &pending_submission);
  EXPECT_TRUE(iree_task_submission_is_empty(&pending_submission));

  iree_task_scope_deinitialize(&scope);
}

TEST(ScopeTest, WaitIdleSuccess) {
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("scope_a"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);

  // Current state is OK and idle.
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  EXPECT_TRUE(iree_status_is_ok(iree_task_scope_consume_status(&scope)));

  // Enqueue a task to the scope so it is no longer idle.
  iree_task_fence_t fence_task;
  iree_task_fence_initialize(&scope, iree_wait_primitive_immediate(),
                             &fence_task);
  EXPECT_FALSE(iree_task_scope_is_idle(&scope));

  // Spin up a thread to wait on the scope.
  std::thread wait_thread([&]() {
    EXPECT_FALSE(iree_task_scope_is_idle(&scope));
    EXPECT_TRUE(iree_status_is_ok(
        iree_task_scope_wait_idle(&scope, IREE_TIME_INFINITE_FUTURE)));
    EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  });

  // Wait a moment for the thread to spin up.
  // NOTE: this may flake. Need to see if there's a better way to do this.
  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  // Complete the task.
  iree_task_submission_t pending_submission;
  iree_task_submission_initialize(&pending_submission);
  iree_task_fence_retire(&fence_task, &pending_submission);
  EXPECT_TRUE(iree_task_submission_is_empty(&pending_submission));
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));

  // Join with the thread - this will hang if it didn't wake correctly.
  wait_thread.join();

  iree_task_scope_deinitialize(&scope);
}

TEST(ScopeTest, WaitIdleFailure) {
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("scope_a"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);

  // Current state is OK and idle.
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  EXPECT_TRUE(iree_status_is_ok(iree_task_scope_consume_status(&scope)));

  // Enqueue a task to the scope so it is no longer idle.
  iree_task_fence_t fence_task;
  iree_task_fence_initialize(&scope, iree_wait_primitive_immediate(),
                             &fence_task);
  EXPECT_FALSE(iree_task_scope_is_idle(&scope));

  // Spin up a thread to wait on the scope.
  std::thread wait_thread([&]() {
    EXPECT_FALSE(iree_task_scope_is_idle(&scope));
    EXPECT_TRUE(iree_status_is_ok(
        iree_task_scope_wait_idle(&scope, IREE_TIME_INFINITE_FUTURE)));
    EXPECT_TRUE(iree_task_scope_is_idle(&scope));
  });

  // Wait a moment for the thread to spin up.
  // NOTE: this may flake. Need to see if there's a better way to do this.
  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  // Set the failure state.
  iree_task_scope_fail(
      &scope, iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "whoops"));
  EXPECT_FALSE(iree_task_scope_is_idle(&scope));

  // Complete the task.
  // Note that even if a scope fails we still must complete the tasks so it
  // becomes idle. This ensures that if the scope state is used to control
  // deallocation we don't go deallocating the tasks still in flight and waiting
  // to gracefully fail.
  iree_task_submission_t pending_submission;
  iree_task_submission_initialize(&pending_submission);
  iree_task_fence_retire(&fence_task, &pending_submission);
  EXPECT_TRUE(iree_task_submission_is_empty(&pending_submission));
  EXPECT_TRUE(iree_task_scope_is_idle(&scope));

  // Join with the thread - this will hang if it didn't wake correctly.
  wait_thread.join();

  iree_task_scope_deinitialize(&scope);
}

}  // namespace
