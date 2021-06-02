// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/task/submission.h"
#include "iree/task/task.h"
#include "iree/task/testing/task_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TaskCallTest : public TaskTest {};

TEST_F(TaskCallTest, Issue) {
  struct TestCtx {
    int did_call = 0;
  };
  TestCtx ctx;

  iree_task_call_t task;
  iree_task_call_initialize(&scope_,
                            iree_task_make_call_closure(
                                [](uintptr_t user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  auto* ctx = (TestCtx*)user_context;
                                  EXPECT_TRUE(NULL != ctx);
                                  EXPECT_EQ(0, ctx->did_call);
                                  ++ctx->did_call;
                                  return iree_ok_status();
                                },
                                (uintptr_t)&ctx),
                            &task);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
  EXPECT_EQ(1, ctx.did_call);
}

// Issues task_a which then issues a nested task_b and waits for it to complete
// prior to progressing. This models dynamic parallelism:
// http://developer.download.nvidia.com/GTC/PDF/GTC2012/PresentationPDF/S0338-GTC2012-CUDA-Programming-Model.pdf
TEST_F(TaskCallTest, IssueNested) {
  struct TestCtx {
    std::atomic<int> did_call_a = {0};
    std::atomic<int> did_call_b = {0};
    std::atomic<bool> has_issued = {false};
    iree_task_call_t task_b;
  };
  TestCtx ctx;

  // task_a will get called twice: the first time it will schedule task_b and
  // then it'll get called again when task_b completes. This is not the only way
  // to do this: task_a could set it up so that a task_c ran after task_b
  // completed instead of getting itself called twice. Both approaches have
  // their uses.
  iree_task_call_t task_a;
  iree_task_call_initialize(
      &scope_,
      iree_task_make_call_closure(
          [](uintptr_t user_context, iree_task_t* task,
             iree_task_submission_t* pending_submission) {
            auto* ctx = (TestCtx*)user_context;
            EXPECT_TRUE(NULL != ctx);

            if (!ctx->has_issued) {
              ctx->has_issued = true;
              EXPECT_EQ(0, ctx->did_call_a);
              ++ctx->did_call_a;
              iree_task_call_initialize(
                  task->scope,
                  iree_task_make_call_closure(
                      [](uintptr_t user_context, iree_task_t* task,
                         iree_task_submission_t* pending_submission) {
                        auto* ctx = (TestCtx*)user_context;
                        EXPECT_TRUE(NULL != ctx);
                        EXPECT_EQ(0, ctx->did_call_b);
                        ++ctx->did_call_b;
                        return iree_ok_status();
                      },
                      user_context),
                  &ctx->task_b);
              iree_task_set_completion_task(&ctx->task_b.header, task);
              iree_task_submission_enqueue(pending_submission,
                                           &ctx->task_b.header);
            } else {
              EXPECT_EQ(1, ctx->did_call_a);
              ++ctx->did_call_a;
            }

            return iree_ok_status();
          },
          (uintptr_t)&ctx),
      &task_a);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task_a.header, &task_a.header));
  EXPECT_EQ(2, ctx.did_call_a);
  EXPECT_EQ(1, ctx.did_call_b);
}

}  // namespace
