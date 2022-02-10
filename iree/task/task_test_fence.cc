// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/task.h"
#include "iree/task/testing/task_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

class TaskFenceTest : public TaskTest {};

// Tests a chain of fences A -> B -> C.
TEST_F(TaskFenceTest, IssueChained) {
  iree_task_fence_t task_a;
  iree_task_fence_initialize(&scope_, iree_wait_primitive_immediate(), &task_a);

  iree_task_fence_t task_b;
  iree_task_fence_initialize(&scope_, iree_wait_primitive_immediate(), &task_b);
  iree_task_set_completion_task(&task_a.header, &task_b.header);

  iree_task_fence_t task_c;
  iree_task_fence_initialize(&scope_, iree_wait_primitive_immediate(), &task_c);
  iree_task_set_completion_task(&task_b.header, &task_c.header);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task_a.header, &task_c.header));
}

// Tests that failures propagate through fences; task B should not be called.
// A fails -> fence -> B
TEST_F(TaskFenceTest, IssueChainedFailure) {
  IREE_TRACE_SCOPE();

  int did_call_a = 0;
  iree_task_call_t task_a;
  iree_task_call_initialize(&scope_,
                            iree_task_make_call_closure(
                                [](void* user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE();
                                  int* did_call_ptr = (int*)user_context;
                                  ++(*did_call_ptr);
                                  return iree_make_status(IREE_STATUS_DATA_LOSS,
                                                          "whoops!");
                                },
                                &did_call_a),
                            &task_a);

  iree_task_fence_t fence_task;
  iree_task_fence_initialize(&scope_, iree_wait_primitive_immediate(),
                             &fence_task);
  iree_task_set_completion_task(&task_a.header, &fence_task.header);

  int did_call_b = 0;
  iree_task_call_t task_b;
  iree_task_call_initialize(&scope_,
                            iree_task_make_call_closure(
                                [](void* user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE();
                                  int* did_call_ptr = (int*)user_context;
                                  ++(*did_call_ptr);
                                  return iree_ok_status();
                                },
                                &did_call_b),
                            &task_b);
  iree_task_set_completion_task(&fence_task.header, &task_b.header);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task_a.header, &task_b.header));
  EXPECT_EQ(1, did_call_a);
  EXPECT_EQ(0, did_call_b);
  EXPECT_THAT(Status(iree_task_scope_consume_status(&scope_)),
              StatusIs(StatusCode::kDataLoss));
}

}  // namespace
