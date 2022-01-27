// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/task/submission.h"
#include "iree/task/task.h"
#include "iree/task/testing/task_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TaskBarrierTest : public TaskTest {};

enum {
  TASK_A = 1 << 0,
  TASK_B = 1 << 1,
  TASK_C = 1 << 2,
  TASK_D = 1 << 3,
};

// We track which tasks were successfully executed
struct TaskCtx {
  std::atomic<uint32_t> tasks_called = {0};
};

#define MAKE_CALL_TASK_CLOSURE(task_ctx, task_id)      \
  iree_task_make_call_closure(                         \
      [](void* user_context, iree_task_t* task,        \
         iree_task_submission_t* pending_submission) { \
        auto* ctx = (TaskCtx*)user_context;            \
        EXPECT_EQ(0, (ctx->tasks_called & (task_id))); \
        ctx->tasks_called |= (task_id);                \
        return iree_ok_status();                       \
      },                                               \
      (void*)task_ctx)

// Issues a standalone empty barrier:
//  { barrier }
TEST_F(TaskBarrierTest, IssueStandalone) {
  iree_task_barrier_t barrier_task;
  iree_task_barrier_initialize_empty(&scope_, &barrier_task);
  IREE_ASSERT_OK(
      SubmitTasksAndWaitIdle(&barrier_task.header, &barrier_task.header));
}

// Issues a serialized sequence:
//  { a | barrier | b }
TEST_F(TaskBarrierTest, IssueSerializedSequence) {
  TaskCtx task_ctx;

  iree_task_call_t task_a;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_A),
                            &task_a);
  iree_task_call_t task_b;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_B),
                            &task_b);

  iree_task_t* dependent_tasks[1] = {&task_b.header};
  iree_task_barrier_t barrier_task;
  iree_task_barrier_initialize(&scope_, IREE_ARRAYSIZE(dependent_tasks),
                               dependent_tasks, &barrier_task);
  iree_task_set_completion_task(&task_a.header, &barrier_task.header);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task_a.header, &task_b.header));
  EXPECT_EQ(TASK_A | TASK_B, task_ctx.tasks_called);
}

// Issues a join:
//  { a, b, c | barrier | d }
TEST_F(TaskBarrierTest, IssueJoin) {
  TaskCtx task_ctx;

  iree_task_call_t task_a;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_A),
                            &task_a);
  iree_task_call_t task_b;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_B),
                            &task_b);
  iree_task_call_t task_c;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_C),
                            &task_c);
  iree_task_call_t task_d;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_D),
                            &task_d);

  iree_task_t* dependent_tasks[1] = {&task_d.header};
  iree_task_barrier_t barrier_task;
  iree_task_barrier_initialize(&scope_, IREE_ARRAYSIZE(dependent_tasks),
                               dependent_tasks, &barrier_task);
  iree_task_set_completion_task(&task_a.header, &barrier_task.header);
  iree_task_set_completion_task(&task_b.header, &barrier_task.header);
  iree_task_set_completion_task(&task_c.header, &barrier_task.header);

  iree_task_submission_t submission;
  iree_task_submission_initialize(&submission);
  iree_task_submission_enqueue(&submission, &task_a.header);
  iree_task_submission_enqueue(&submission, &task_b.header);
  iree_task_submission_enqueue(&submission, &task_c.header);
  IREE_ASSERT_OK(SubmitAndWaitIdle(&submission, &task_d.header));
  EXPECT_EQ(TASK_A | TASK_B | TASK_C | TASK_D, task_ctx.tasks_called);
}

// Issues a fork:
//  { a | barrier | b, c, d | nop }
TEST_F(TaskBarrierTest, IssueFork) {
  TaskCtx task_ctx;

  iree_task_call_t task_a;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_A),
                            &task_a);
  iree_task_call_t task_b;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_B),
                            &task_b);
  iree_task_call_t task_c;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_C),
                            &task_c);
  iree_task_call_t task_d;
  iree_task_call_initialize(&scope_, MAKE_CALL_TASK_CLOSURE(&task_ctx, TASK_D),
                            &task_d);

  iree_task_t* dependent_tasks[3] = {
      &task_b.header,
      &task_c.header,
      &task_d.header,
  };
  iree_task_barrier_t barrier_task;
  iree_task_barrier_initialize(&scope_, IREE_ARRAYSIZE(dependent_tasks),
                               dependent_tasks, &barrier_task);
  iree_task_set_completion_task(&task_a.header, &barrier_task.header);

  // Just to give us a tail task to wait on.
  iree_task_nop_t nop_task;
  iree_task_nop_initialize(&scope_, &nop_task);
  iree_task_set_completion_task(&task_b.header, &nop_task.header);
  iree_task_set_completion_task(&task_c.header, &nop_task.header);
  iree_task_set_completion_task(&task_d.header, &nop_task.header);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task_a.header, &nop_task.header));
  EXPECT_EQ(TASK_A | TASK_B | TASK_C | TASK_D, task_ctx.tasks_called);
}

}  // namespace
