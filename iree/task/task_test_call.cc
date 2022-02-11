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

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

class TaskCallTest : public TaskTest {};

// Tests issuing a single call and waiting for it to complete.
TEST_F(TaskCallTest, Issue) {
  IREE_TRACE_SCOPE();

  struct TestCtx {
    int did_call = 0;
  };
  TestCtx ctx;

  iree_task_call_t task;
  iree_task_call_initialize(&scope_,
                            iree_task_make_call_closure(
                                [](void* user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE();
                                  auto* ctx = (TestCtx*)user_context;
                                  EXPECT_TRUE(NULL != ctx);
                                  EXPECT_EQ(0, ctx->did_call);
                                  ++ctx->did_call;
                                  return iree_ok_status();
                                },
                                (void*)&ctx),
                            &task);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
  EXPECT_EQ(1, ctx.did_call);
  IREE_EXPECT_OK(iree_task_scope_consume_status(&scope_));
}

// Tests issuing a single call that returns a failure.
// The failure should be propagated back on the task scope.
TEST_F(TaskCallTest, IssueFailure) {
  IREE_TRACE_SCOPE();

  struct TestCtx {
    int did_call = 0;
  };
  TestCtx ctx;

  // Call successfully issues but fails with some user error.
  iree_task_call_t task;
  iree_task_call_initialize(&scope_,
                            iree_task_make_call_closure(
                                [](void* user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE();
                                  auto* ctx = (TestCtx*)user_context;
                                  EXPECT_TRUE(NULL != ctx);
                                  EXPECT_EQ(0, ctx->did_call);
                                  ++ctx->did_call;
                                  return iree_make_status(
                                      IREE_STATUS_UNAUTHENTICATED, "whoops!");
                                },
                                (void*)&ctx),
                            &task);

  // The task should still be cleaned up, even if it fails.
  static int did_cleanup = 0;
  did_cleanup = 0;
  iree_task_set_cleanup_fn(
      &task.header, +[](iree_task_t* task, iree_status_code_t status_code) {
        IREE_TRACE_SCOPE();
        EXPECT_EQ(status_code, IREE_STATUS_ABORTED);
        ++did_cleanup;
      });

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));

  // Expect both the call to have been made and the task cleaned up.
  // The scope has the failure status.
  EXPECT_EQ(1, ctx.did_call);
  EXPECT_EQ(1, did_cleanup);
  EXPECT_THAT(Status(iree_task_scope_consume_status(&scope_)),
              StatusIs(StatusCode::kUnauthenticated));
}

// Tests issuing chained calls where the first fails.
// The failure should be propagated back on the task scope and the chained call
// should be aborted.
TEST_F(TaskCallTest, IssueFailureChained) {
  IREE_TRACE_SCOPE();

  struct TestCtx {
    int did_call_a = 0;
    int did_call_b = 0;
  };
  TestCtx ctx;

  // First call that will fail.
  iree_task_call_t task_a;
  iree_task_call_initialize(&scope_,
                            iree_task_make_call_closure(
                                [](void* user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE();
                                  auto* ctx = (TestCtx*)user_context;
                                  EXPECT_TRUE(NULL != ctx);
                                  EXPECT_EQ(0, ctx->did_call_a);
                                  ++ctx->did_call_a;
                                  // Force a failure.
                                  return iree_make_status(
                                      IREE_STATUS_UNAUTHENTICATED, "whoops!");
                                },
                                (void*)&ctx),
                            &task_a);
  static int did_cleanup_a = 0;
  did_cleanup_a = 0;
  iree_task_set_cleanup_fn(
      &task_a.header, +[](iree_task_t* task, iree_status_code_t status_code) {
        // Expect that the cleanup gets a signal indicating the task failed.
        IREE_TRACE_SCOPE();
        EXPECT_EQ(status_code, IREE_STATUS_ABORTED);
        ++did_cleanup_a;
      });

  // Second call that will be aborted after the first fails.
  iree_task_call_t task_b;
  iree_task_call_initialize(&scope_,
                            iree_task_make_call_closure(
                                [](void* user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  // This should never get called!
                                  IREE_TRACE_SCOPE();
                                  auto* ctx = (TestCtx*)user_context;
                                  EXPECT_TRUE(NULL != ctx);
                                  EXPECT_EQ(0, ctx->did_call_b);
                                  ++ctx->did_call_b;
                                  return iree_ok_status();
                                },
                                (void*)&ctx),
                            &task_b);
  static int did_cleanup_b = 0;
  did_cleanup_b = 0;
  iree_task_set_cleanup_fn(
      &task_b.header, +[](iree_task_t* task, iree_status_code_t status_code) {
        // Expect that the cleanup gets a signal indicating the task failed.
        IREE_TRACE_SCOPE();
        EXPECT_EQ(status_code, IREE_STATUS_ABORTED);
        ++did_cleanup_b;
      });

  // A -> B
  iree_task_set_completion_task(&task_a.header, &task_b.header);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task_a.header, &task_b.header));

  // Expect that A was called but B was not, and both were cleaned up.
  EXPECT_EQ(1, ctx.did_call_a);
  EXPECT_EQ(1, did_cleanup_a);
  EXPECT_EQ(0, ctx.did_call_b);
  EXPECT_EQ(1, did_cleanup_b);
  EXPECT_THAT(Status(iree_task_scope_consume_status(&scope_)),
              StatusIs(StatusCode::kUnauthenticated));
}

// Issues task_a which then issues a nested task_b and waits for it to complete
// prior to progressing. This models dynamic parallelism:
// http://developer.download.nvidia.com/GTC/PDF/GTC2012/PresentationPDF/S0338-GTC2012-CUDA-Programming-Model.pdf
TEST_F(TaskCallTest, IssueNested) {
  IREE_TRACE_SCOPE();

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
          [](void* user_context, iree_task_t* task,
             iree_task_submission_t* pending_submission) {
            IREE_TRACE_SCOPE();
            auto* ctx = (TestCtx*)user_context;
            EXPECT_TRUE(NULL != ctx);

            if (!ctx->has_issued) {
              ctx->has_issued = true;
              EXPECT_EQ(0, ctx->did_call_a);
              ++ctx->did_call_a;
              iree_task_call_initialize(
                  task->scope,
                  iree_task_make_call_closure(
                      [](void* user_context, iree_task_t* task,
                         iree_task_submission_t* pending_submission) {
                        IREE_TRACE_SCOPE();
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
          (void*)&ctx),
      &task_a);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task_a.header, &task_a.header));
  EXPECT_EQ(2, ctx.did_call_a);
  EXPECT_EQ(1, ctx.did_call_b);
  IREE_EXPECT_OK(iree_task_scope_consume_status(&scope_));
}

// Issues task_a which then issues a nested task_b and task_c; task_b fails and
// it's expected that task_c completes before failing task_a.
// Sibling tasks don't abort each other and as such we are guaranteed that C
// will run: A -> [B fail, C ok] -> A fail
TEST_F(TaskCallTest, IssueNestedFailure) {
  IREE_TRACE_SCOPE();

  struct TestCtx {
    std::atomic<int> did_call_a = {0};
    std::atomic<int> did_call_b = {0};
    std::atomic<int> did_call_c = {0};
    std::atomic<bool> has_issued = {false};
    iree_task_call_t task_b;
    iree_task_call_t task_c;
  };
  TestCtx ctx;

  // task_a will get called only once due to the error: the pre-nesting call
  // will schedule task_b/task_c and then the expected call after the tasks
  // complete will not be made as task_b fails.
  iree_task_call_t task_a;
  iree_task_call_initialize(
      &scope_,
      iree_task_make_call_closure(
          [](void* user_context, iree_task_t* task,
             iree_task_submission_t* pending_submission) {
            auto* ctx = (TestCtx*)user_context;
            EXPECT_TRUE(NULL != ctx);

            if (!ctx->has_issued) {
              ctx->has_issued = true;
              EXPECT_EQ(0, ctx->did_call_a);
              ++ctx->did_call_a;

              // task_b: (fails)
              iree_task_call_initialize(
                  task->scope,
                  iree_task_make_call_closure(
                      [](void* user_context, iree_task_t* task,
                         iree_task_submission_t* pending_submission) {
                        IREE_TRACE_SCOPE();
                        auto* ctx = (TestCtx*)user_context;
                        EXPECT_TRUE(NULL != ctx);
                        EXPECT_EQ(0, ctx->did_call_b);
                        ++ctx->did_call_b;
                        return iree_make_status(IREE_STATUS_DATA_LOSS, "uh oh");
                      },
                      user_context),
                  &ctx->task_b);
              iree_task_set_completion_task(&ctx->task_b.header, task);
              iree_task_submission_enqueue(pending_submission,
                                           &ctx->task_b.header);

              // task_c: (ok)
              iree_task_call_initialize(
                  task->scope,
                  iree_task_make_call_closure(
                      [](void* user_context, iree_task_t* task,
                         iree_task_submission_t* pending_submission) {
                        IREE_TRACE_SCOPE();
                        auto* ctx = (TestCtx*)user_context;
                        EXPECT_TRUE(NULL != ctx);
                        EXPECT_EQ(0, ctx->did_call_c);
                        ++ctx->did_call_c;
                        return iree_ok_status();
                      },
                      user_context),
                  &ctx->task_c);
              iree_task_set_completion_task(&ctx->task_c.header, task);
              iree_task_submission_enqueue(pending_submission,
                                           &ctx->task_c.header);
            } else {
              EXPECT_EQ(1, ctx->did_call_a);
              ++ctx->did_call_a;
            }

            return iree_ok_status();
          },
          (void*)&ctx),
      &task_a);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task_a.header, &task_a.header));
  EXPECT_EQ(1, ctx.did_call_a);
  EXPECT_EQ(1, ctx.did_call_b);
  EXPECT_EQ(1, ctx.did_call_c);
  EXPECT_THAT(Status(iree_task_scope_consume_status(&scope_)),
              StatusIs(StatusCode::kDataLoss));
}

}  // namespace
