// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for linked semaphore operation sequences.
//
// Tests integration of SEMAPHORE_WAIT and SEMAPHORE_SIGNAL with linked
// operation chains. Linked sequences enable kernel-enforced ordering where
// a later operation only begins after the prior one completes successfully.
//
// These tests exercise the core remoting use cases:
//   WAIT → NOP:    Wait for semaphore before proceeding
//   SIGNAL → NOP:  Signal semaphore then continue
//   RECV → SIGNAL: Network receive triggers semaphore signal
//   WAIT → SEND:   Wait for semaphore before sending data

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/semaphore.h"
#include "iree/async/span.h"

namespace iree::async::cts {

class SemaphoreLinkedTest : public SocketTestBase<> {
 protected:
  void SetUp() override {
    SocketTestBase::SetUp();
    if (!iree_any_bit_set(capabilities_,
                          IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
      GTEST_SKIP() << "backend lacks linked operations capability";
    }
  }
};

// Helper struct to track completion order.
struct OrderTracker {
  std::vector<int> order;

  void Record(int id) { order.push_back(id); }

  static void MakeCallback(int id) {
    // Returns a callback that records the given ID.
  }
};

//===----------------------------------------------------------------------===//
// WAIT linked tests
//===----------------------------------------------------------------------===//

// WAIT(LINKED) → NOP: signal semaphore, both complete in order.
TEST_P(SemaphoreLinkedTest, LinkedWaitThenNop) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  struct OrderedCallbacks {
    std::vector<int> order;
    static void WaitCallback(void* u, iree_async_operation_t* o,
                             iree_status_t s, iree_async_completion_flags_t f) {
      static_cast<OrderedCallbacks*>(u)->order.push_back(0);
      iree_status_ignore(s);
    }
    static void NopCallback(void* u, iree_async_operation_t* o, iree_status_t s,
                            iree_async_completion_flags_t f) {
      static_cast<OrderedCallbacks*>(u)->order.push_back(1);
      iree_status_ignore(s);
    }
  };

  OrderedCallbacks tracker;

  // Set up WAIT operation (linked to NOP).
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t wait_value = 10;

  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = OrderedCallbacks::WaitCallback;
  wait_op.base.user_data = &tracker;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  wait_op.semaphores = &semaphore_ptr;
  wait_op.values = &wait_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  // Set up NOP operation (end of chain).
  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.completion_fn = OrderedCallbacks::NopCallback;
  nop.base.user_data = &tracker;

  // Submit as linked sequence.
  iree_async_operation_t* ops[] = {&wait_op.base, &nop.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Signal the semaphore after a short delay.
  std::thread signaler([semaphore]() {
    iree_wait_until(iree_time_now() + iree_make_duration_ms(50));
    iree_status_t status = iree_async_semaphore_signal(semaphore, 10, NULL);
    IREE_CHECK_OK(status);
  });

  // Poll until both complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  signaler.join();

  // Verify order: WAIT first, then NOP.
  ASSERT_EQ(tracker.order.size(), 2u);
  EXPECT_EQ(tracker.order[0], 0);  // WAIT
  EXPECT_EQ(tracker.order[1], 1);  // NOP

  iree_async_semaphore_release(semaphore);
}

// SIGNAL(LINKED) → NOP: both complete, semaphore advances before NOP fires.
TEST_P(SemaphoreLinkedTest, LinkedSignalThenNop) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  struct OrderedCallbacks {
    std::vector<int> order;
    iree_async_semaphore_t* semaphore;
    uint64_t semaphore_value_at_nop;

    static void SignalCallback(void* u, iree_async_operation_t* o,
                               iree_status_t s,
                               iree_async_completion_flags_t f) {
      static_cast<OrderedCallbacks*>(u)->order.push_back(0);
      iree_status_ignore(s);
    }
    static void NopCallback(void* u, iree_async_operation_t* o, iree_status_t s,
                            iree_async_completion_flags_t f) {
      auto* ctx = static_cast<OrderedCallbacks*>(u);
      ctx->order.push_back(1);
      // Record semaphore value at NOP callback time.
      ctx->semaphore_value_at_nop = iree_async_semaphore_query(ctx->semaphore);
      iree_status_ignore(s);
    }
  };

  OrderedCallbacks tracker;
  tracker.semaphore = semaphore;
  tracker.semaphore_value_at_nop = 0;

  // Set up SIGNAL operation (linked to NOP).
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t signal_value = 10;

  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = OrderedCallbacks::SignalCallback;
  signal_op.base.user_data = &tracker;
  signal_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  signal_op.semaphores = &semaphore_ptr;
  signal_op.values = &signal_value;
  signal_op.count = 1;
  signal_op.frontier = NULL;

  // Set up NOP operation (end of chain).
  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.completion_fn = OrderedCallbacks::NopCallback;
  nop.base.user_data = &tracker;

  // Submit as linked sequence.
  iree_async_operation_t* ops[] = {&signal_op.base, &nop.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Verify order.
  ASSERT_EQ(tracker.order.size(), 2u);
  EXPECT_EQ(tracker.order[0], 0);  // SIGNAL
  EXPECT_EQ(tracker.order[1], 1);  // NOP

  // Semaphore should have been signaled before NOP callback ran.
  EXPECT_EQ(tracker.semaphore_value_at_nop, 10u);

  iree_async_semaphore_release(semaphore);
}

// RECV(LINKED) → SIGNAL: recv completes, then signal fires.
TEST_P(SemaphoreLinkedTest, LinkedRecvThenSignal) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  struct OrderedCallbacks {
    std::vector<int> order;
    static void RecvCallback(void* u, iree_async_operation_t* o,
                             iree_status_t s, iree_async_completion_flags_t f) {
      static_cast<OrderedCallbacks*>(u)->order.push_back(0);
      iree_status_ignore(s);
    }
    static void SignalCallback(void* u, iree_async_operation_t* o,
                               iree_status_t s,
                               iree_async_completion_flags_t f) {
      static_cast<OrderedCallbacks*>(u)->order.push_back(1);
      iree_status_ignore(s);
    }
  };

  OrderedCallbacks tracker;

  // Set up RECV operation (linked to SIGNAL).
  char recv_buffer[64];
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  memset(&recv_op, 0, sizeof(recv_op));
  recv_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
  recv_op.base.completion_fn = OrderedCallbacks::RecvCallback;
  recv_op.base.user_data = &tracker;
  recv_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  recv_op.socket = server;
  recv_op.buffers.values = &recv_span;
  recv_op.buffers.count = 1;

  // Set up SIGNAL operation.
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t signal_value = 1;

  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = OrderedCallbacks::SignalCallback;
  signal_op.base.user_data = &tracker;
  signal_op.semaphores = &semaphore_ptr;
  signal_op.values = &signal_value;
  signal_op.count = 1;
  signal_op.frontier = NULL;

  // Submit as linked sequence.
  iree_async_operation_t* ops[] = {&recv_op.base, &signal_op.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Send data to trigger recv.
  const char* send_data = "hello";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, strlen(send_data));

  CompletionTracker send_tracker;
  iree_async_socket_send_operation_t send_op;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Poll until all complete (send + recv + signal).
  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Verify order: RECV first, then SIGNAL.
  ASSERT_EQ(tracker.order.size(), 2u);
  EXPECT_EQ(tracker.order[0], 0);  // RECV
  EXPECT_EQ(tracker.order[1], 1);  // SIGNAL

  // Semaphore should be signaled.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  iree_async_semaphore_release(semaphore);
  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// WAIT(LINKED) → SEND: signal triggers send.
TEST_P(SemaphoreLinkedTest, LinkedWaitThenSend) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  struct OrderedCallbacks {
    std::vector<int> order;
    static void WaitCallback(void* u, iree_async_operation_t* o,
                             iree_status_t s, iree_async_completion_flags_t f) {
      static_cast<OrderedCallbacks*>(u)->order.push_back(0);
      iree_status_ignore(s);
    }
    static void SendCallback(void* u, iree_async_operation_t* o,
                             iree_status_t s, iree_async_completion_flags_t f) {
      static_cast<OrderedCallbacks*>(u)->order.push_back(1);
      iree_status_ignore(s);
    }
  };

  OrderedCallbacks tracker;

  // Set up WAIT operation (linked to SEND).
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t wait_value = 1;

  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = OrderedCallbacks::WaitCallback;
  wait_op.base.user_data = &tracker;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  wait_op.semaphores = &semaphore_ptr;
  wait_op.values = &wait_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  // Set up SEND operation.
  const char* send_data = "triggered by semaphore";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, strlen(send_data));

  iree_async_socket_send_operation_t send_op;
  memset(&send_op, 0, sizeof(send_op));
  send_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND;
  send_op.base.completion_fn = OrderedCallbacks::SendCallback;
  send_op.base.user_data = &tracker;
  send_op.socket = client;
  send_op.buffers.values = &send_span;
  send_op.buffers.count = 1;
  send_op.send_flags = IREE_ASYNC_SOCKET_SEND_FLAG_NONE;

  // Submit as linked sequence.
  iree_async_operation_t* ops[] = {&wait_op.base, &send_op.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Set up recv on server side.
  char recv_buffer[64] = {0};
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  CompletionTracker recv_tracker;
  iree_async_socket_recv_operation_t recv_op;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  // Signal the semaphore after delay.
  std::thread signaler([semaphore]() {
    iree_wait_until(iree_time_now() + iree_make_duration_ms(50));
    iree_status_t status = iree_async_semaphore_signal(semaphore, 1, NULL);
    IREE_CHECK_OK(status);
  });

  // Poll until all complete (wait + send + recv).
  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(5000));

  signaler.join();

  // Verify order: WAIT first, then SEND.
  ASSERT_EQ(tracker.order.size(), 2u);
  EXPECT_EQ(tracker.order[0], 0);  // WAIT
  EXPECT_EQ(tracker.order[1], 1);  // SEND

  // Server should have received the data.
  EXPECT_EQ(recv_tracker.call_count, 1);
  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_STREQ(recv_buffer, "triggered by semaphore");

  iree_async_semaphore_release(semaphore);
  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// WAIT(LINKED) → NOP: fail semaphore, NOP gets CANCELLED.
TEST_P(SemaphoreLinkedTest, LinkedWaitFailureCancelsChain) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  CompletionTracker wait_tracker;
  CompletionTracker nop_tracker;

  // Set up WAIT operation (linked to NOP).
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t wait_value = 10;

  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &wait_tracker;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  wait_op.semaphores = &semaphore_ptr;
  wait_op.values = &wait_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  // Set up NOP operation (end of chain).
  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.completion_fn = CompletionTracker::Callback;
  nop.base.user_data = &nop_tracker;

  // Submit as linked sequence.
  iree_async_operation_t* ops[] = {&wait_op.base, &nop.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Fail the semaphore.
  std::thread failer([semaphore]() {
    iree_wait_until(iree_time_now() + iree_make_duration_ms(50));
    iree_async_semaphore_fail(
        semaphore, iree_make_status(IREE_STATUS_ABORTED, "device lost"));
  });

  // Poll until both complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  failer.join();

  // WAIT should complete with ABORTED.
  EXPECT_EQ(wait_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, wait_tracker.ConsumeStatus());

  // NOP should be CANCELLED because the chain was broken.
  EXPECT_EQ(nop_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, nop_tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// SIGNAL(LINKED) → NOP: non-monotonic signal value, NOP gets CANCELLED.
TEST_P(SemaphoreLinkedTest, LinkedSignalFailureCancelsChain) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  CompletionTracker signal_tracker;
  CompletionTracker nop_tracker;

  // Set up SIGNAL operation with bad value (linked to NOP).
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t signal_value = 5;  // Less than current (10), will fail.

  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = CompletionTracker::Callback;
  signal_op.base.user_data = &signal_tracker;
  signal_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  signal_op.semaphores = &semaphore_ptr;
  signal_op.values = &signal_value;
  signal_op.count = 1;
  signal_op.frontier = NULL;

  // Set up NOP operation (end of chain).
  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.completion_fn = CompletionTracker::Callback;
  nop.base.user_data = &nop_tracker;

  // Submit as linked sequence.
  iree_async_operation_t* ops[] = {&signal_op.base, &nop.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Poll until both complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // SIGNAL should complete with INVALID_ARGUMENT.
  EXPECT_EQ(signal_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        signal_tracker.ConsumeStatus());

  // NOP should be CANCELLED because the chain was broken.
  EXPECT_EQ(nop_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, nop_tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// Two independent WAIT chains, signal one, only its chain completes.
TEST_P(SemaphoreLinkedTest, IndependentChains) {
  iree_async_semaphore_t* sem1 = nullptr;
  iree_async_semaphore_t* sem2 = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem1));
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem2));

  CompletionTracker wait1_tracker, nop1_tracker;
  CompletionTracker wait2_tracker, nop2_tracker;

  // First chain: WAIT(sem1) → NOP
  iree_async_semaphore_t* sem1_ptr = sem1;
  uint64_t wait1_value = 1;

  iree_async_semaphore_wait_operation_t wait1_op;
  memset(&wait1_op, 0, sizeof(wait1_op));
  wait1_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait1_op.base.completion_fn = CompletionTracker::Callback;
  wait1_op.base.user_data = &wait1_tracker;
  wait1_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  wait1_op.semaphores = &sem1_ptr;
  wait1_op.values = &wait1_value;
  wait1_op.count = 1;
  wait1_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  iree_async_nop_operation_t nop1;
  memset(&nop1, 0, sizeof(nop1));
  nop1.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop1.base.completion_fn = CompletionTracker::Callback;
  nop1.base.user_data = &nop1_tracker;

  // Second chain: WAIT(sem2) → NOP
  iree_async_semaphore_t* sem2_ptr = sem2;
  uint64_t wait2_value = 1;

  iree_async_semaphore_wait_operation_t wait2_op;
  memset(&wait2_op, 0, sizeof(wait2_op));
  wait2_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait2_op.base.completion_fn = CompletionTracker::Callback;
  wait2_op.base.user_data = &wait2_tracker;
  wait2_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  wait2_op.semaphores = &sem2_ptr;
  wait2_op.values = &wait2_value;
  wait2_op.count = 1;
  wait2_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  iree_async_nop_operation_t nop2;
  memset(&nop2, 0, sizeof(nop2));
  nop2.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop2.base.completion_fn = CompletionTracker::Callback;
  nop2.base.user_data = &nop2_tracker;

  // Submit both chains.
  iree_async_operation_t* chain1[] = {&wait1_op.base, &nop1.base};
  iree_async_operation_list_t list1 = {chain1, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list1));

  iree_async_operation_t* chain2[] = {&wait2_op.base, &nop2.base};
  iree_async_operation_list_t list2 = {chain2, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list2));

  // Signal only sem1.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem1, 1, NULL));

  // Poll until chain1 completes.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Chain1 should be complete.
  EXPECT_EQ(wait1_tracker.call_count, 1);
  EXPECT_EQ(nop1_tracker.call_count, 1);
  IREE_EXPECT_OK(wait1_tracker.ConsumeStatus());
  IREE_EXPECT_OK(nop1_tracker.ConsumeStatus());

  // Chain2 should still be pending.
  EXPECT_EQ(wait2_tracker.call_count, 0);
  EXPECT_EQ(nop2_tracker.call_count, 0);

  // Signal sem2 to complete chain2.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem2, 1, NULL));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(wait2_tracker.call_count, 1);
  EXPECT_EQ(nop2_tracker.call_count, 1);
  IREE_EXPECT_OK(wait2_tracker.ConsumeStatus());
  IREE_EXPECT_OK(nop2_tracker.ConsumeStatus());

  iree_async_semaphore_release(sem1);
  iree_async_semaphore_release(sem2);
}

CTS_REGISTER_TEST_SUITE(SemaphoreLinkedTest);

}  // namespace iree::async::cts
