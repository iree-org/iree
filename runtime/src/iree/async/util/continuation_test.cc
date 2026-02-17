// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/continuation.h"

#include <vector>

#include "iree/async/operation.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Test infrastructure
//===----------------------------------------------------------------------===//

// Tracks completion callback invocations.
struct CompletionRecord {
  iree_async_operation_t* operation;
  iree_status_code_t status_code;
  iree_async_completion_flags_t flags;
};

// Test fixture context passed to mock submit and callbacks.
struct TestContext {
  // Mock submit behavior.
  iree_status_t submit_result = iree_ok_status();
  std::vector<iree_async_operation_t*> submitted_operations;

  // Completion callback tracking.
  std::vector<CompletionRecord> completions;
};

// Completion callback that records invocations.
// Per IREE convention, this callback TAKES OWNERSHIP of status and must
// consume or ignore it.
static void TestCompletionCallback(void* user_data,
                                   iree_async_operation_t* operation,
                                   iree_status_t status,
                                   iree_async_completion_flags_t flags) {
  TestContext* context = static_cast<TestContext*>(user_data);
  context->completions.push_back({operation, iree_status_code(status), flags});
  // Callback takes ownership - must consume the status.
  iree_status_ignore(status);
}

// Mock submit function that records operations and returns configured result.
static iree_status_t MockSubmit(void* ctx,
                                iree_async_operation_list_t operations) {
  TestContext* context = static_cast<TestContext*>(ctx);
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    context->submitted_operations.push_back(operations.values[i]);
  }
  // Return a clone if it's an error (caller will ignore the original).
  if (!iree_status_is_ok(context->submit_result)) {
    return iree_status_clone(context->submit_result);
  }
  return context->submit_result;
}

class ContinuationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize test operations with callbacks pointing to our context.
    for (int i = 0; i < 3; ++i) {
      iree_async_operation_initialize(
          &operations_[i], IREE_ASYNC_OPERATION_TYPE_NOP,
          IREE_ASYNC_OPERATION_FLAG_NONE, TestCompletionCallback, &context_);
    }
  }

  void TearDown() override {
    // Clean up any error status we configured for submit.
    if (!iree_status_is_ok(context_.submit_result)) {
      iree_status_ignore(context_.submit_result);
    }
  }

  TestContext context_;
  iree_async_operation_t operations_[3];
};

//===----------------------------------------------------------------------===//
// Empty continuation list
//===----------------------------------------------------------------------===//

TEST_F(ContinuationTest, EmptyListReturnsZero) {
  iree_async_continuation_list_t continuations =
      iree_async_continuation_list_empty();

  iree_host_size_t invoked = iree_async_continuation_dispatch(
      MockSubmit, &context_, &continuations, iree_ok_status());

  EXPECT_EQ(invoked, 0u);
  EXPECT_TRUE(context_.submitted_operations.empty());
  EXPECT_TRUE(context_.completions.empty());
}

TEST_F(ContinuationTest, EmptyListDoesNotConsumeStatus) {
  iree_async_continuation_list_t continuations =
      iree_async_continuation_list_empty();

  // Create a status with storage (error status).
  // We own this status until we pass it to a callback or ignore it.
  iree_status_t status = iree_make_status(IREE_STATUS_CANCELLED, "test error");

  iree_host_size_t invoked = iree_async_continuation_dispatch(
      MockSubmit, &context_, &continuations, status);

  EXPECT_EQ(invoked, 0u);

  // CRITICAL: dispatch does NOT consume status. Caller still owns it and
  // would pass it to the triggering operation's completion callback.
  // For this test, we just verify it's still valid then clean up.
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_CANCELLED);

  // We still own it, so we must clean up.
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Success path - submit continuations
//===----------------------------------------------------------------------===//

TEST_F(ContinuationTest, SuccessSubmitsContinuations) {
  iree_async_operation_t* operation_ptrs[] = {&operations_[0], &operations_[1],
                                              &operations_[2]};
  iree_async_continuation_list_t continuations =
      iree_async_make_continuation_list(operation_ptrs, 0, 3);

  iree_host_size_t invoked = iree_async_continuation_dispatch(
      MockSubmit, &context_, &continuations, iree_ok_status());

  // All operations should be submitted, none invoked directly.
  EXPECT_EQ(invoked, 0u);
  ASSERT_EQ(context_.submitted_operations.size(), 3u);
  EXPECT_EQ(context_.submitted_operations[0], &operations_[0]);
  EXPECT_EQ(context_.submitted_operations[1], &operations_[1]);
  EXPECT_EQ(context_.submitted_operations[2], &operations_[2]);
  EXPECT_TRUE(context_.completions.empty());

  // Count should be cleared.
  EXPECT_EQ(continuations.count, 0u);
}

TEST_F(ContinuationTest, SuccessWithStartOffset) {
  iree_async_operation_t* operation_ptrs[] = {&operations_[0], &operations_[1],
                                              &operations_[2]};
  iree_async_continuation_list_t continuations =
      iree_async_make_continuation_list(operation_ptrs, /*start=*/1,
                                        /*count=*/2);

  iree_host_size_t invoked = iree_async_continuation_dispatch(
      MockSubmit, &context_, &continuations, iree_ok_status());

  EXPECT_EQ(invoked, 0u);
  ASSERT_EQ(context_.submitted_operations.size(), 2u);
  EXPECT_EQ(context_.submitted_operations[0], &operations_[1]);
  EXPECT_EQ(context_.submitted_operations[1], &operations_[2]);
}

TEST_F(ContinuationTest, SuccessSubmitFailsInvokesCallbacksWithError) {
  // Configure submit to fail. MockSubmit will clone this for each return.
  context_.submit_result =
      iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "SQ full");

  iree_async_operation_t* operation_ptrs[] = {&operations_[0], &operations_[1]};
  iree_async_continuation_list_t continuations =
      iree_async_make_continuation_list(operation_ptrs, 0, 2);

  iree_host_size_t invoked = iree_async_continuation_dispatch(
      MockSubmit, &context_, &continuations, iree_ok_status());

  // Submit was attempted, then callbacks invoked directly with the error.
  // Each callback receives a CLONE of the submit error (ownership transfer).
  EXPECT_EQ(invoked, 2u);
  ASSERT_EQ(context_.completions.size(), 2u);
  EXPECT_EQ(context_.completions[0].operation, &operations_[0]);
  EXPECT_EQ(context_.completions[0].status_code,
            IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(context_.completions[1].operation, &operations_[1]);
  EXPECT_EQ(context_.completions[1].status_code,
            IREE_STATUS_RESOURCE_EXHAUSTED);
}

//===----------------------------------------------------------------------===//
// Failure path - invoke with CANCELLED
//===----------------------------------------------------------------------===//

TEST_F(ContinuationTest, FailureInvokesCancelledCallbacks) {
  iree_async_operation_t* operation_ptrs[] = {&operations_[0], &operations_[1],
                                              &operations_[2]};
  iree_async_continuation_list_t continuations =
      iree_async_make_continuation_list(operation_ptrs, 0, 3);

  // Triggering operation failed with INTERNAL error.
  iree_status_t trigger_status =
      iree_make_status(IREE_STATUS_INTERNAL, "something failed");

  iree_host_size_t invoked = iree_async_continuation_dispatch(
      MockSubmit, &context_, &continuations, trigger_status);

  // All callbacks invoked directly with fresh CANCELLED statuses.
  // The original trigger_status is NOT passed to continuations.
  EXPECT_EQ(invoked, 3u);
  EXPECT_TRUE(context_.submitted_operations.empty());
  ASSERT_EQ(context_.completions.size(), 3u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(context_.completions[i].operation, &operations_[i]);
    EXPECT_EQ(context_.completions[i].status_code, IREE_STATUS_CANCELLED);
    EXPECT_EQ(context_.completions[i].flags, IREE_ASYNC_COMPLETION_FLAG_NONE);
  }

  // Count should be cleared.
  EXPECT_EQ(continuations.count, 0u);

  // CRITICAL: dispatch does NOT consume trigger_status. The caller retains
  // ownership to pass it to the triggering operation's completion callback.
  EXPECT_EQ(iree_status_code(trigger_status), IREE_STATUS_INTERNAL);

  // We still own it, so we must clean up.
  iree_status_ignore(trigger_status);
}

TEST_F(ContinuationTest, FailureStatusNotConsumed) {
  iree_async_operation_t* operation_ptrs[] = {&operations_[0]};
  iree_async_continuation_list_t continuations =
      iree_async_make_continuation_list(operation_ptrs, 0, 1);

  // Create a status with heap-allocated storage (longer message).
  iree_status_t status = iree_make_status(
      IREE_STATUS_DATA_LOSS,
      "this is a longer error message that will allocate storage");

  iree_async_continuation_dispatch(MockSubmit, &context_, &continuations,
                                   status);

  // Verify the continuation got CANCELLED, not the original error.
  ASSERT_EQ(context_.completions.size(), 1u);
  EXPECT_EQ(context_.completions[0].status_code, IREE_STATUS_CANCELLED);

  // Status should still be valid - caller retains ownership.
  // This would be passed to the triggering operation's callback.
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_DATA_LOSS);

  // We still own it, so clean up.
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST_F(ContinuationTest, NullCallbackSkipped) {
  // One operation has no callback.
  operations_[1].completion_fn = nullptr;

  iree_async_operation_t* operation_ptrs[] = {&operations_[0], &operations_[1],
                                              &operations_[2]};
  iree_async_continuation_list_t continuations =
      iree_async_make_continuation_list(operation_ptrs, 0, 3);

  iree_status_t trigger_status =
      iree_make_status(IREE_STATUS_CANCELLED, "cancelled");

  iree_host_size_t invoked = iree_async_continuation_dispatch(
      MockSubmit, &context_, &continuations, trigger_status);

  // Only 2 callbacks invoked (the one with null is skipped).
  EXPECT_EQ(invoked, 2u);
  ASSERT_EQ(context_.completions.size(), 2u);
  EXPECT_EQ(context_.completions[0].operation, &operations_[0]);
  EXPECT_EQ(context_.completions[1].operation, &operations_[2]);

  // Clean up - we still own trigger_status.
  iree_status_ignore(trigger_status);
}

}  // namespace
