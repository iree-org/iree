// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/js/proactor.h"

#include <cstring>

#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class JsProactorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    options.max_concurrent_operations = 16;
    options.debug_name = iree_make_cstring_view("test");
    IREE_ASSERT_OK(iree_async_proactor_create_js(
        options, iree_allocator_system(), &proactor_));
  }

  void TearDown() override {
    if (proactor_) {
      iree_async_proactor_release(proactor_);
      proactor_ = NULL;
    }
  }

  // Initializes a NOP operation with the given completion callback.
  void InitNop(iree_async_nop_operation_t* nop,
               iree_async_completion_fn_t completion_fn, void* user_data) {
    memset(nop, 0, sizeof(*nop));
    nop->base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
    nop->base.completion_fn = completion_fn;
    nop->base.user_data = user_data;
  }

  // Initializes a timer operation with the given deadline and callback.
  void InitTimer(iree_async_timer_operation_t* timer, iree_time_t deadline_ns,
                 iree_async_completion_fn_t completion_fn, void* user_data) {
    memset(timer, 0, sizeof(*timer));
    timer->base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
    timer->base.completion_fn = completion_fn;
    timer->base.user_data = user_data;
    timer->deadline_ns = deadline_ns;
  }

  // Simple completion callback that increments a counter.
  static void CountingCallback(void* user_data,
                               iree_async_operation_t* operation,
                               iree_status_t status,
                               iree_async_completion_flags_t flags) {
    iree_status_ignore(status);
    int* counter = reinterpret_cast<int*>(user_data);
    ++(*counter);
  }

  // Completion callback that records the status code.
  static void StatusRecordingCallback(void* user_data,
                                      iree_async_operation_t* operation,
                                      iree_status_t status,
                                      iree_async_completion_flags_t flags) {
    iree_status_code_t* code = reinterpret_cast<iree_status_code_t*>(user_data);
    *code = iree_status_code(status);
    iree_status_ignore(status);
  }

  iree_async_proactor_t* proactor_ = NULL;
};

TEST_F(JsProactorTest, CreateAndDestroy) {
  // SetUp and TearDown exercise create/destroy.
  EXPECT_NE(proactor_, nullptr);
}

TEST_F(JsProactorTest, QueryCapabilities) {
  iree_async_proactor_capabilities_t capabilities =
      iree_async_proactor_query_capabilities(proactor_);
  EXPECT_TRUE(capabilities & IREE_ASYNC_PROACTOR_CAPABILITY_ABSOLUTE_TIMEOUT);
  EXPECT_TRUE(capabilities & IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS);
  // Sockets, files, etc. are not supported.
  EXPECT_FALSE(capabilities & IREE_ASYNC_PROACTOR_CAPABILITY_DMABUF);
}

TEST_F(JsProactorTest, QueryCapabilitiesWithMask) {
  // Create a proactor with no capabilities allowed.
  iree_async_proactor_t* masked_proactor = NULL;
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.max_concurrent_operations = 4;
  options.allowed_capabilities = IREE_ASYNC_PROACTOR_CAPABILITY_NONE;
  IREE_ASSERT_OK(iree_async_proactor_create_js(options, iree_allocator_system(),
                                               &masked_proactor));

  iree_async_proactor_capabilities_t capabilities =
      iree_async_proactor_query_capabilities(masked_proactor);
  EXPECT_EQ(capabilities, IREE_ASYNC_PROACTOR_CAPABILITY_NONE);

  iree_async_proactor_release(masked_proactor);
}

TEST_F(JsProactorTest, RetainRelease) {
  iree_async_proactor_retain(proactor_);
  // Should not destroy — ref count is now 2.
  iree_async_proactor_release(proactor_);
  // Ref count is back to 1, proactor still alive.
  // TearDown will release the final reference.
}

TEST_F(JsProactorTest, SubmitNop) {
  int completed_count = 0;
  iree_async_nop_operation_t nop;
  InitNop(&nop, CountingCallback, &completed_count);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &nop.base));

  // NOP completes during poll.
  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(poll_completed, 1);
}

TEST_F(JsProactorTest, SubmitMultipleNops) {
  int completed_count = 0;
  iree_async_nop_operation_t nops[4];
  for (int i = 0; i < 4; ++i) {
    InitNop(&nops[i], CountingCallback, &completed_count);
  }

  // Submit as a batch.
  iree_async_operation_t* ops[4];
  for (int i = 0; i < 4; ++i) {
    ops[i] = &nops[i].base;
  }
  iree_async_operation_list_t list = {ops, 4};
  IREE_ASSERT_OK(proactor_->vtable->submit(proactor_, list));

  // All complete on single poll.
  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  EXPECT_EQ(completed_count, 4);
  EXPECT_EQ(poll_completed, 4);
}

TEST_F(JsProactorTest, PollWithNoWork) {
  iree_host_size_t poll_completed = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                               &poll_completed));
  EXPECT_EQ(poll_completed, 0);
}

TEST_F(JsProactorTest, SubmitExpiredTimer) {
  // A timer with a deadline in the past should complete immediately.
  int completed_count = 0;
  iree_async_timer_operation_t timer;
  InitTimer(&timer, 0, CountingCallback, &completed_count);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));

  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(poll_completed, 1);
}

TEST_F(JsProactorTest, SubmitFutureTimer) {
  // A timer with a far-future deadline. In native tests, the timer_start
  // import is a no-op, so the timer will never fire. We can still verify
  // the submit succeeds and the token is allocated.
  int completed_count = 0;
  iree_async_timer_operation_t timer;
  iree_time_t far_future = iree_time_now() + 60ll * 1000000000ll;
  InitTimer(&timer, far_future, CountingCallback, &completed_count);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));

  // Poll should not complete the timer (no ring completions in native stubs).
  iree_host_size_t poll_completed = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                               &poll_completed));
  EXPECT_EQ(completed_count, 0);

  // Cancel to clean up the token table entry.
  IREE_ASSERT_OK(proactor_->vtable->cancel(proactor_, &timer.base));
}

TEST_F(JsProactorTest, CancelNop) {
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_async_nop_operation_t nop;
  InitNop(&nop, StatusRecordingCallback, &status_code);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &nop.base));
  IREE_ASSERT_OK(proactor_->vtable->cancel(proactor_, &nop.base));

  // Poll should dispatch with CANCELLED.
  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  EXPECT_EQ(status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(poll_completed, 1);
}

TEST_F(JsProactorTest, CancelFutureTimer) {
  // Submit a future timer, then cancel it. In native stubs, timer_cancel
  // returns 1 (always cancelled), so the completion fires immediately in
  // the cancel call.
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_async_timer_operation_t timer;
  iree_time_t far_future = iree_time_now() + 60ll * 1000000000ll;
  InitTimer(&timer, far_future, StatusRecordingCallback, &status_code);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &timer.base));
  IREE_ASSERT_OK(proactor_->vtable->cancel(proactor_, &timer.base));

  // The native stub cancels synchronously, so the callback already fired.
  EXPECT_EQ(status_code, IREE_STATUS_CANCELLED);
}

TEST_F(JsProactorTest, UnsupportedOperationReturnsUnimplemented) {
  // Try to submit a socket connect operation (unsupported).
  iree_async_operation_t fake_op;
  memset(&fake_op, 0, sizeof(fake_op));
  fake_op.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT;
  fake_op.completion_fn = CountingCallback;
  int counter = 0;
  fake_op.user_data = &counter;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_async_proactor_submit_one(proactor_, &fake_op));
}

TEST_F(JsProactorTest, UnavailableSocketReturnsUnavailable) {
  iree_async_socket_t* socket = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        proactor_->vtable->create_socket(
                            proactor_, (iree_async_socket_type_t)0,
                            (iree_async_socket_options_t){0}, &socket));
}

TEST_F(JsProactorTest, UnavailableEventReturnsUnavailable) {
  iree_async_event_t* event = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        proactor_->vtable->create_event(proactor_, &event));
}

TEST_F(JsProactorTest, UnavailableNotificationReturnsUnavailable) {
  iree_async_notification_t* notification = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      proactor_->vtable->create_notification(proactor_, 0, &notification));
}

TEST_F(JsProactorTest, UnavailableFenceReturnsUnavailable) {
  iree_async_primitive_t fence;
  memset(&fence, 0, sizeof(fence));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      proactor_->vtable->import_fence(proactor_, fence, NULL, 0));
}

TEST_F(JsProactorTest, UnavailableMessageReturnsUnavailable) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        proactor_->vtable->send_message(proactor_, 42));
}

TEST_F(JsProactorTest, WakeIsNoopInNative) {
  // Just verify it doesn't crash. In native stubs, wake is a no-op.
  proactor_->vtable->wake(proactor_);
}

TEST_F(JsProactorTest, DefaultCapacityCreate) {
  // Create with default options (max_concurrent_operations = 0).
  iree_async_proactor_t* default_proactor = NULL;
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  IREE_ASSERT_OK(iree_async_proactor_create_js(options, iree_allocator_system(),
                                               &default_proactor));

  // Should still work for a NOP.
  int completed_count = 0;
  iree_async_nop_operation_t nop;
  InitNop(&nop, CountingCallback, &completed_count);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(default_proactor, &nop.base));
  IREE_ASSERT_OK(iree_async_proactor_poll(default_proactor,
                                          iree_make_timeout_ms(0), NULL));
  EXPECT_EQ(completed_count, 1);

  iree_async_proactor_release(default_proactor);
}

//===----------------------------------------------------------------------===//
// LINKED operations
//===----------------------------------------------------------------------===//

// Completion callback that records status code and appends the operation
// pointer to an ordered list, allowing verification of completion ordering.
struct OrderedCompletion {
  iree_async_operation_t* operation;
  iree_status_code_t status_code;
};
struct OrderedCompletionState {
  OrderedCompletion completions[16];
  int count;
};

static void OrderedCallback(void* user_data, iree_async_operation_t* operation,
                            iree_status_t status,
                            iree_async_completion_flags_t flags) {
  auto* state = reinterpret_cast<OrderedCompletionState*>(user_data);
  state->completions[state->count].operation = operation;
  state->completions[state->count].status_code = iree_status_code(status);
  state->count++;
  iree_status_ignore(status);
}

TEST_F(JsProactorTest, TwoLinkedNops) {
  OrderedCompletionState state = {};
  iree_async_nop_operation_t nop0, nop1;
  InitNop(&nop0, OrderedCallback, &state);
  InitNop(&nop1, OrderedCallback, &state);
  nop0.base.flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;

  iree_async_operation_t* ops[] = {&nop0.base, &nop1.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(proactor_->vtable->submit(proactor_, list));

  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  // Both should complete: nop0 fires first, then nop1 via linked continuation.
  EXPECT_EQ(state.count, 2);
  EXPECT_EQ(state.completions[0].operation, &nop0.base);
  EXPECT_EQ(state.completions[0].status_code, IREE_STATUS_OK);
  EXPECT_EQ(state.completions[1].operation, &nop1.base);
  EXPECT_EQ(state.completions[1].status_code, IREE_STATUS_OK);
}

TEST_F(JsProactorTest, LinkedNopChain) {
  OrderedCompletionState state = {};
  iree_async_nop_operation_t nops[4];
  for (int i = 0; i < 4; ++i) {
    InitNop(&nops[i], OrderedCallback, &state);
  }
  // Chain: nops[0] -> nops[1] -> nops[2] -> nops[3]
  nops[0].base.flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;
  nops[1].base.flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;
  nops[2].base.flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;

  iree_async_operation_t* ops[4];
  for (int i = 0; i < 4; ++i) ops[i] = &nops[i].base;
  iree_async_operation_list_t list = {ops, 4};
  IREE_ASSERT_OK(proactor_->vtable->submit(proactor_, list));

  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  EXPECT_EQ(state.count, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(state.completions[i].operation, &nops[i].base)
        << "completion " << i << " out of order";
    EXPECT_EQ(state.completions[i].status_code, IREE_STATUS_OK);
  }
}

TEST_F(JsProactorTest, TrailingLinkedRejected) {
  iree_async_nop_operation_t nop;
  InitNop(&nop, CountingCallback, nullptr);
  nop.base.flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;

  iree_async_operation_t* ops[] = {&nop.base};
  iree_async_operation_list_t list = {ops, 1};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        proactor_->vtable->submit(proactor_, list));
}

TEST_F(JsProactorTest, MixedLinkedAndUnlinkedBatch) {
  // Batch: [nop0 -> nop1] [nop2] — two chain heads, three operations total.
  OrderedCompletionState state = {};
  iree_async_nop_operation_t nops[3];
  for (int i = 0; i < 3; ++i) {
    InitNop(&nops[i], OrderedCallback, &state);
  }
  nops[0].base.flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;

  iree_async_operation_t* ops[] = {&nops[0].base, &nops[1].base, &nops[2].base};
  iree_async_operation_list_t list = {ops, 3};
  IREE_ASSERT_OK(proactor_->vtable->submit(proactor_, list));

  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  EXPECT_EQ(state.count, 3);
  // All should complete OK. The linked pair (nop0->nop1) fires in order,
  // and nop2 fires independently.
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(state.completions[i].status_code, IREE_STATUS_OK);
  }
}

//===----------------------------------------------------------------------===//
// SEQUENCE operations
//===----------------------------------------------------------------------===//

TEST_F(JsProactorTest, ZeroStepSequenceCompletesImmediately) {
  iree_status_code_t status_code = IREE_STATUS_INTERNAL;
  iree_async_sequence_operation_t sequence = {};
  sequence.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
  sequence.base.completion_fn = StatusRecordingCallback;
  sequence.base.user_data = &status_code;
  sequence.steps = nullptr;
  sequence.step_count = 0;
  sequence.step_fn = nullptr;

  // Zero-step sequence completes synchronously during submit.
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));
  EXPECT_EQ(status_code, IREE_STATUS_OK);
}

TEST_F(JsProactorTest, SingleStepNopSequenceLinkPath) {
  // Single-step sequence via LINK path (step_fn == NULL).
  // Link trampolines replace step callbacks; only the sequence base callback
  // fires with the test's callback.
  iree_status_code_t seq_status = IREE_STATUS_INTERNAL;
  iree_async_nop_operation_t nop;
  InitNop(&nop, CountingCallback, nullptr);

  iree_async_operation_t* step = &nop.base;
  iree_async_sequence_operation_t sequence = {};
  sequence.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
  sequence.base.completion_fn = StatusRecordingCallback;
  sequence.base.user_data = &seq_status;
  sequence.steps = &step;
  sequence.step_count = 1;
  sequence.step_fn = nullptr;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  EXPECT_EQ(seq_status, IREE_STATUS_OK);
}

TEST_F(JsProactorTest, TwoStepNopSequenceLinkPath) {
  // Two-step sequence via LINK path (step_fn == NULL).
  iree_status_code_t seq_status = IREE_STATUS_INTERNAL;
  iree_async_nop_operation_t nop0, nop1;
  InitNop(&nop0, CountingCallback, nullptr);
  InitNop(&nop1, CountingCallback, nullptr);

  iree_async_operation_t* steps[] = {&nop0.base, &nop1.base};
  iree_async_sequence_operation_t sequence = {};
  sequence.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
  sequence.base.completion_fn = StatusRecordingCallback;
  sequence.base.user_data = &seq_status;
  sequence.steps = steps;
  sequence.step_count = 2;
  sequence.step_fn = nullptr;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  EXPECT_EQ(seq_status, IREE_STATUS_OK);
}

TEST_F(JsProactorTest, FourStepNopSequenceLinkPath) {
  // Four-step sequence exercises the full linked chain: steps[0..2] get LINKED,
  // steps[3] is the chain tail.
  iree_status_code_t seq_status = IREE_STATUS_INTERNAL;
  iree_async_nop_operation_t nops[4];
  for (int i = 0; i < 4; ++i) {
    InitNop(&nops[i], CountingCallback, nullptr);
  }

  iree_async_operation_t* steps[4];
  for (int i = 0; i < 4; ++i) steps[i] = &nops[i].base;
  iree_async_sequence_operation_t sequence = {};
  sequence.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
  sequence.base.completion_fn = StatusRecordingCallback;
  sequence.base.user_data = &seq_status;
  sequence.steps = steps;
  sequence.step_count = 4;
  sequence.step_fn = nullptr;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));
  EXPECT_EQ(seq_status, IREE_STATUS_OK);
}

// Step function that records each inter-step callback and returns OK.
struct EmulationStepState {
  int step_fn_call_count;
  iree_async_operation_t* completed_steps[8];
  iree_async_operation_t* next_steps[8];
  iree_status_code_t sequence_status;
  bool sequence_completed;
};

static iree_status_t RecordingStepFn(void* user_data,
                                     iree_async_operation_t* completed_step,
                                     iree_async_operation_t* next_step) {
  auto* state = reinterpret_cast<EmulationStepState*>(user_data);
  state->completed_steps[state->step_fn_call_count] = completed_step;
  state->next_steps[state->step_fn_call_count] = next_step;
  state->step_fn_call_count++;
  return iree_ok_status();
}

static void EmulationSequenceCallback(void* user_data,
                                      iree_async_operation_t* operation,
                                      iree_status_t status,
                                      iree_async_completion_flags_t flags) {
  auto* state = reinterpret_cast<EmulationStepState*>(user_data);
  state->sequence_status = iree_status_code(status);
  state->sequence_completed = true;
  iree_status_ignore(status);
}

TEST_F(JsProactorTest, TwoStepNopSequenceEmulationPath) {
  // Two-step sequence via emulation path (step_fn != NULL).
  // The emulator replaces step callbacks with internal trampolines, calls
  // step_fn between steps, and fires the base callback when all steps complete.
  EmulationStepState state = {};

  iree_async_nop_operation_t nop0, nop1;
  InitNop(&nop0, CountingCallback, nullptr);
  InitNop(&nop1, CountingCallback, nullptr);

  iree_async_operation_t* steps[] = {&nop0.base, &nop1.base};
  iree_async_sequence_operation_t sequence = {};
  sequence.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
  sequence.base.completion_fn = EmulationSequenceCallback;
  sequence.base.user_data = &state;
  sequence.steps = steps;
  sequence.step_count = 2;
  sequence.step_fn = RecordingStepFn;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  // The emulation path submits one step at a time. The drain_ready loop picks
  // up re-submitted steps during the same pass, so a single poll suffices.
  iree_host_size_t poll_completed = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                                          &poll_completed));

  EXPECT_TRUE(state.sequence_completed);
  EXPECT_EQ(state.sequence_status, IREE_STATUS_OK);

  // step_fn was called twice: after step 0 completes (with next=step1), and
  // after step 1 completes (with next=NULL marking the end).
  EXPECT_EQ(state.step_fn_call_count, 2);
  EXPECT_EQ(state.completed_steps[0], &nop0.base);
  EXPECT_EQ(state.next_steps[0], &nop1.base);
  EXPECT_EQ(state.completed_steps[1], &nop1.base);
  EXPECT_EQ(state.next_steps[1], nullptr);
}

TEST_F(JsProactorTest, SequenceCancellation) {
  // Submit a sequence, then cancel it. The sequence base callback should
  // fire with CANCELLED.
  iree_status_code_t seq_status = IREE_STATUS_INTERNAL;
  iree_async_nop_operation_t nop0;
  int nop_counter = 0;
  InitNop(&nop0, CountingCallback, &nop_counter);

  // Use a future timer as second step so cancellation has time to fire.
  iree_async_timer_operation_t timer;
  iree_time_t far_future = iree_time_now() + 60ll * 1000000000ll;
  InitTimer(&timer, far_future, CountingCallback, &nop_counter);

  iree_async_operation_t* steps[] = {&nop0.base, &timer.base};
  iree_async_sequence_operation_t sequence = {};
  sequence.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
  sequence.base.completion_fn = StatusRecordingCallback;
  sequence.base.user_data = &seq_status;
  sequence.steps = steps;
  sequence.step_count = 2;
  sequence.step_fn = nullptr;  // LINK path.

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  // Cancel the sequence.
  IREE_ASSERT_OK(proactor_->vtable->cancel(proactor_, &sequence.base));

  // Poll to drain pending completions.
  for (int i = 0; i < 4; ++i) {
    iree_host_size_t poll_completed = 0;
    iree_async_proactor_poll(proactor_, iree_make_timeout_ms(0),
                             &poll_completed);
  }

  EXPECT_EQ(seq_status, IREE_STATUS_CANCELLED);
}

}  // namespace
