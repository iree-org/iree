// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for NOP operations.
//
// NOP is the simplest operation type - it completes immediately on the next
// poll with an OK status. These tests validate basic submit/poll/callback
// mechanics without involving timers, I/O, or complex state.

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"

namespace iree::async::cts {

class NopTest : public CtsTestBase<> {};

// Single NOP: submit, poll, verify callback fires with OK status.
TEST_P(NopTest, SingleNop) {
  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;

  CompletionTracker tracker;
  nop.base.completion_fn = CompletionTracker::Callback;
  nop.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &nop.base));

  // NOP should complete on the first poll.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(100));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(tracker.last_operation, &nop.base);
}

// Multiple NOPs: submit batch, verify all complete.
TEST_P(NopTest, MultipleNops) {
  constexpr int kNopCount = 5;
  iree_async_nop_operation_t nops[kNopCount];
  memset(nops, 0, sizeof(nops));

  int completion_count = 0;
  auto counting_callback = [](void* user_data, iree_async_operation_t* op,
                              iree_status_t status,
                              iree_async_completion_flags_t flags) {
    ++(*static_cast<int*>(user_data));
    iree_status_ignore(status);
  };

  iree_async_operation_t* ops[kNopCount];
  for (int i = 0; i < kNopCount; ++i) {
    nops[i].base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
    nops[i].base.completion_fn = counting_callback;
    nops[i].base.user_data = &completion_count;
    ops[i] = &nops[i].base;
  }

  iree_async_operation_list_t list = {ops, kNopCount};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // All NOPs should complete quickly.
  PollUntil(/*min_completions=*/kNopCount,
            /*total_budget=*/iree_make_duration_ms(100));

  EXPECT_EQ(completion_count, kNopCount);
}

// NOP callback receives correct operation pointer.
TEST_P(NopTest, CallbackReceivesOperationPointer) {
  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;

  iree_async_operation_t* received_op = nullptr;
  auto capture_callback = [](void* user_data, iree_async_operation_t* op,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
    *static_cast<iree_async_operation_t**>(user_data) = op;
    iree_status_ignore(status);
  };

  nop.base.completion_fn = capture_callback;
  nop.base.user_data = &received_op;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &nop.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(100));

  EXPECT_EQ(received_op, &nop.base);
}

// Empty submit list: should succeed with no completions.
TEST_P(NopTest, EmptySubmit) {
  iree_async_operation_list_t empty_list = {nullptr, 0};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, empty_list));

  // Poll should return immediately with no completions (deadline exceeded).
  iree_host_size_t completed = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_async_proactor_poll(
                            proactor_, iree_immediate_timeout(), &completed));
  EXPECT_EQ(completed, 0u);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(NopTest);

}  // namespace iree::async::cts
