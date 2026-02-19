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
//
// Mixed kernel/software chain tests verify correct ordering when software
// operations (SIGNAL, WAIT) appear at arbitrary positions in linked chains
// alongside kernel operations (RECV, SEND):
//   RECV → SIGNAL → SEND:  Software in the middle
//   SIGNAL → RECV:          Software head, kernel tail
//   SIGNAL → SIGNAL → NOP:  Pure software chain
//   RECV → SIGNAL (timing): Signal must NOT fire during submit
//   WAIT → RECV → SIGNAL:   Deferred software, kernel, software
//   WAIT → SIGNAL → SEND (failure): Error propagation through mixed chain

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

//===----------------------------------------------------------------------===//
// Mixed kernel/software chain tests
//===----------------------------------------------------------------------===//

// RECV(LINKED) → SIGNAL(LINKED) → SEND: software operation in the middle of
// a chain that starts and ends with kernel operations. Verifies callbacks
// fire in chain order and that the semaphore is signaled before SEND executes.
TEST_P(SemaphoreLinkedTest, LinkedRecvThenSignalThenSend) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Second connection for the SEND at the end of the chain.
  iree_async_socket_t* client2 = nullptr;
  iree_async_socket_t* server2 = nullptr;
  iree_async_socket_t* listener2 = nullptr;
  EstablishConnection(&client2, &server2, &listener2);

  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  struct OrderedCallbacks {
    std::vector<int> order;
    static void RecvCallback(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(0);
      iree_status_ignore(status);
    }
    static void SignalCallback(void* user_data,
                               iree_async_operation_t* operation,
                               iree_status_t status,
                               iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(1);
      iree_status_ignore(status);
    }
    static void SendCallback(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(2);
      iree_status_ignore(status);
    }
  };

  OrderedCallbacks tracker;

  // RECV (head of chain, linked to SIGNAL).
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

  // SIGNAL (middle of chain, linked to SEND).
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t signal_value = 42;

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

  // SEND (tail of chain).
  const char* send_data = "after-signal";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, strlen(send_data));

  iree_async_socket_send_operation_t send_op;
  memset(&send_op, 0, sizeof(send_op));
  send_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND;
  send_op.base.completion_fn = OrderedCallbacks::SendCallback;
  send_op.base.user_data = &tracker;
  send_op.socket = client2;
  send_op.buffers.values = &send_span;
  send_op.buffers.count = 1;
  send_op.send_flags = IREE_ASYNC_SOCKET_SEND_FLAG_NONE;

  // Submit the 3-op chain.
  iree_async_operation_t* ops[] = {&recv_op.base, &signal_op.base,
                                   &send_op.base};
  iree_async_operation_list_t list = {ops, 3};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Set up recv on server2 to receive the SEND output.
  char recv2_buffer[64] = {0};
  iree_async_span_t recv2_span =
      iree_async_span_from_ptr(recv2_buffer, sizeof(recv2_buffer));
  CompletionTracker recv2_tracker;
  iree_async_socket_recv_operation_t recv2_op;
  InitRecvOperation(&recv2_op, server2, &recv2_span, 1,
                    CompletionTracker::Callback, &recv2_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv2_op.base));

  // Trigger the chain by sending data to server (completing RECV).
  const char* trigger_data = "trigger";
  iree_async_span_t trigger_span =
      iree_async_span_from_ptr((void*)trigger_data, strlen(trigger_data));
  CompletionTracker trigger_tracker;
  iree_async_socket_send_operation_t trigger_op;
  InitSendOperation(&trigger_op, client, &trigger_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &trigger_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &trigger_op.base));

  // Poll until chain completes (trigger send + recv + signal + chain send +
  // recv2).
  PollUntil(/*min_completions=*/5,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Verify callback order: RECV → SIGNAL → SEND.
  ASSERT_EQ(tracker.order.size(), 3u);
  EXPECT_EQ(tracker.order[0], 0);  // RECV
  EXPECT_EQ(tracker.order[1], 1);  // SIGNAL
  EXPECT_EQ(tracker.order[2], 2);  // SEND

  // Semaphore was signaled.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 42u);

  // Server2 received the data sent by the chain's SEND.
  EXPECT_EQ(recv2_tracker.call_count, 1);
  IREE_EXPECT_OK(recv2_tracker.ConsumeStatus());
  EXPECT_EQ(memcmp(recv2_buffer, "after-signal", strlen("after-signal")), 0);

  iree_async_semaphore_release(semaphore);
  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
  iree_async_socket_release(client2);
  iree_async_socket_release(server2);
  iree_async_socket_release(listener2);
}

// SIGNAL(LINKED) → RECV: software operation heads the chain, kernel tail.
// Verifies SIGNAL callback fires before RECV callback and that the semaphore
// is already signaled when RECV completes.
TEST_P(SemaphoreLinkedTest, LinkedSignalThenRecv) {
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
    iree_async_semaphore_t* semaphore;
    uint64_t semaphore_value_at_recv;

    static void SignalCallback(void* user_data,
                               iree_async_operation_t* operation,
                               iree_status_t status,
                               iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(0);
      iree_status_ignore(status);
    }
    static void RecvCallback(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
      auto* context = static_cast<OrderedCallbacks*>(user_data);
      context->order.push_back(1);
      context->semaphore_value_at_recv =
          iree_async_semaphore_query(context->semaphore);
      iree_status_ignore(status);
    }
  };

  OrderedCallbacks tracker;
  tracker.semaphore = semaphore;
  tracker.semaphore_value_at_recv = 0;

  // SIGNAL (head of chain, linked to RECV).
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t signal_value = 7;

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

  // RECV (tail of chain).
  char recv_buffer[64];
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  memset(&recv_op, 0, sizeof(recv_op));
  recv_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
  recv_op.base.completion_fn = OrderedCallbacks::RecvCallback;
  recv_op.base.user_data = &tracker;
  recv_op.socket = server;
  recv_op.buffers.values = &recv_span;
  recv_op.buffers.count = 1;

  // Submit the chain.
  iree_async_operation_t* ops[] = {&signal_op.base, &recv_op.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Send data to trigger RECV.
  const char* send_data = "hello";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, strlen(send_data));
  CompletionTracker send_tracker;
  iree_async_socket_send_operation_t send_op;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Poll until chain completes (send + signal + recv).
  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Verify order: SIGNAL first, then RECV.
  ASSERT_EQ(tracker.order.size(), 2u);
  EXPECT_EQ(tracker.order[0], 0);  // SIGNAL
  EXPECT_EQ(tracker.order[1], 1);  // RECV

  // Semaphore was already signaled when RECV callback ran.
  EXPECT_EQ(tracker.semaphore_value_at_recv, 7u);

  iree_async_semaphore_release(semaphore);
  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// SIGNAL1(LINKED) → SIGNAL2(LINKED) → NOP: pure software chain with zero
// kernel SQEs for the signals. Verifies all three callbacks fire in order
// and both semaphores reach their target values.
TEST_P(SemaphoreLinkedTest, LinkedSignalSignalNop) {
  iree_async_semaphore_t* sem1 = nullptr;
  iree_async_semaphore_t* sem2 = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem1));
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem2));

  struct OrderedCallbacks {
    std::vector<int> order;
    static void Signal1Callback(void* user_data,
                                iree_async_operation_t* operation,
                                iree_status_t status,
                                iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(0);
      iree_status_ignore(status);
    }
    static void Signal2Callback(void* user_data,
                                iree_async_operation_t* operation,
                                iree_status_t status,
                                iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(1);
      iree_status_ignore(status);
    }
    static void NopCallback(void* user_data, iree_async_operation_t* operation,
                            iree_status_t status,
                            iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(2);
      iree_status_ignore(status);
    }
  };

  OrderedCallbacks tracker;

  // SIGNAL1 (head, linked to SIGNAL2).
  iree_async_semaphore_t* sem1_ptr = sem1;
  uint64_t signal1_value = 10;

  iree_async_semaphore_signal_operation_t signal1_op;
  memset(&signal1_op, 0, sizeof(signal1_op));
  signal1_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal1_op.base.completion_fn = OrderedCallbacks::Signal1Callback;
  signal1_op.base.user_data = &tracker;
  signal1_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  signal1_op.semaphores = &sem1_ptr;
  signal1_op.values = &signal1_value;
  signal1_op.count = 1;
  signal1_op.frontier = NULL;

  // SIGNAL2 (middle, linked to NOP).
  iree_async_semaphore_t* sem2_ptr = sem2;
  uint64_t signal2_value = 20;

  iree_async_semaphore_signal_operation_t signal2_op;
  memset(&signal2_op, 0, sizeof(signal2_op));
  signal2_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal2_op.base.completion_fn = OrderedCallbacks::Signal2Callback;
  signal2_op.base.user_data = &tracker;
  signal2_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  signal2_op.semaphores = &sem2_ptr;
  signal2_op.values = &signal2_value;
  signal2_op.count = 1;
  signal2_op.frontier = NULL;

  // NOP (tail).
  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.completion_fn = OrderedCallbacks::NopCallback;
  nop.base.user_data = &tracker;

  // Submit the chain.
  iree_async_operation_t* ops[] = {&signal1_op.base, &signal2_op.base,
                                   &nop.base};
  iree_async_operation_list_t list = {ops, 3};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Verify order: SIGNAL1 → SIGNAL2 → NOP.
  ASSERT_EQ(tracker.order.size(), 3u);
  EXPECT_EQ(tracker.order[0], 0);  // SIGNAL1
  EXPECT_EQ(tracker.order[1], 1);  // SIGNAL2
  EXPECT_EQ(tracker.order[2], 2);  // NOP

  // Both semaphores signaled.
  EXPECT_EQ(iree_async_semaphore_query(sem1), 10u);
  EXPECT_EQ(iree_async_semaphore_query(sem2), 20u);

  iree_async_semaphore_release(sem1);
  iree_async_semaphore_release(sem2);
}

// RECV(LINKED) → SIGNAL: verifies that the signal does NOT fire during submit.
// After submit, the semaphore must still be at its initial value (0). The
// signal fires only after RECV completes (triggered by sending data).
// This catches the eager-signal ordering bug where software operations
// execute during submit() before their predecessor kernel operation starts.
TEST_P(SemaphoreLinkedTest, LinkedRecvThenSignalTimingCheck) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  CompletionTracker recv_tracker;
  CompletionTracker signal_tracker;

  // RECV (linked to SIGNAL).
  char recv_buffer[64];
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  memset(&recv_op, 0, sizeof(recv_op));
  recv_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
  recv_op.base.completion_fn = CompletionTracker::Callback;
  recv_op.base.user_data = &recv_tracker;
  recv_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  recv_op.socket = server;
  recv_op.buffers.values = &recv_span;
  recv_op.buffers.count = 1;

  // SIGNAL (tail).
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t signal_value = 100;

  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = CompletionTracker::Callback;
  signal_op.base.user_data = &signal_tracker;
  signal_op.semaphores = &semaphore_ptr;
  signal_op.values = &signal_value;
  signal_op.count = 1;
  signal_op.frontier = NULL;

  // Submit the chain.
  iree_async_operation_t* ops[] = {&recv_op.base, &signal_op.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // CRITICAL CHECK: Immediately after submit, the semaphore must still be 0.
  // If the signal executed eagerly during submit() (the bug), it would
  // already be 100 here.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 0u)
      << "Signal fired during submit before RECV completed — "
         "eager execution ordering bug";

  // Now send data to trigger RECV, which should trigger SIGNAL.
  const char* send_data = "trigger";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, strlen(send_data));
  CompletionTracker send_tracker;
  iree_async_socket_send_operation_t send_op;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Poll until chain completes (send + recv + signal).
  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Both completed successfully.
  EXPECT_EQ(recv_tracker.call_count, 1);
  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(signal_tracker.call_count, 1);
  IREE_EXPECT_OK(signal_tracker.ConsumeStatus());

  // Semaphore now signaled after RECV completed.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 100u);

  iree_async_semaphore_release(semaphore);
  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// WAIT(LINKED) → RECV(LINKED) → SIGNAL: deferred software head, kernel
// middle, software tail. Verifies WAIT holds the chain until signaled, RECV
// starts only after WAIT, and SIGNAL fires only after RECV.
TEST_P(SemaphoreLinkedTest, LinkedWaitRecvSignal) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_async_semaphore_t* wait_semaphore = nullptr;
  iree_async_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &wait_semaphore));
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &signal_semaphore));

  struct OrderedCallbacks {
    std::vector<int> order;
    static void WaitCallback(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(0);
      iree_status_ignore(status);
    }
    static void RecvCallback(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(1);
      iree_status_ignore(status);
    }
    static void SignalCallback(void* user_data,
                               iree_async_operation_t* operation,
                               iree_status_t status,
                               iree_async_completion_flags_t flags) {
      static_cast<OrderedCallbacks*>(user_data)->order.push_back(2);
      iree_status_ignore(status);
    }
  };

  OrderedCallbacks tracker;

  // WAIT (head, linked to RECV).
  iree_async_semaphore_t* wait_sem_ptr = wait_semaphore;
  uint64_t wait_value = 1;

  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = OrderedCallbacks::WaitCallback;
  wait_op.base.user_data = &tracker;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  wait_op.semaphores = &wait_sem_ptr;
  wait_op.values = &wait_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  // RECV (middle, linked to SIGNAL).
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

  // SIGNAL (tail).
  iree_async_semaphore_t* signal_sem_ptr = signal_semaphore;
  uint64_t signal_value = 50;

  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = OrderedCallbacks::SignalCallback;
  signal_op.base.user_data = &tracker;
  signal_op.semaphores = &signal_sem_ptr;
  signal_op.values = &signal_value;
  signal_op.count = 1;
  signal_op.frontier = NULL;

  // Submit the chain.
  iree_async_operation_t* ops[] = {&wait_op.base, &recv_op.base,
                                   &signal_op.base};
  iree_async_operation_list_t list = {ops, 3};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Send data to server (will be consumed by RECV once WAIT completes).
  const char* send_data = "payload";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, strlen(send_data));
  CompletionTracker send_tracker;
  iree_async_socket_send_operation_t send_op;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Wait for the send to complete (data is now in the socket buffer).
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Chain should still be pending (WAIT blocks it).
  EXPECT_EQ(tracker.order.size(), 0u);
  EXPECT_EQ(iree_async_semaphore_query(signal_semaphore), 0u);

  // Signal the wait semaphore to unblock the chain.
  IREE_ASSERT_OK(iree_async_semaphore_signal(wait_semaphore, 1, NULL));

  // Poll until chain completes (wait + recv + signal).
  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Verify order: WAIT → RECV → SIGNAL.
  ASSERT_EQ(tracker.order.size(), 3u);
  EXPECT_EQ(tracker.order[0], 0);  // WAIT
  EXPECT_EQ(tracker.order[1], 1);  // RECV
  EXPECT_EQ(tracker.order[2], 2);  // SIGNAL

  // Signal semaphore advanced after RECV.
  EXPECT_EQ(iree_async_semaphore_query(signal_semaphore), 50u);

  iree_async_semaphore_release(wait_semaphore);
  iree_async_semaphore_release(signal_semaphore);
  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// WAIT(LINKED) → SIGNAL(LINKED) → SEND: WAIT fails (semaphore failed),
// which should cancel SIGNAL and SEND. Verifies the signal semaphore stays
// at 0 (signal side-effect did NOT execute) and all subsequent operations
// in the chain receive CANCELLED.
//
// This extends LinkedWaitFailureCancelsChain (WAIT → NOP) by verifying that
// error propagation correctly skips software ops (SIGNAL) without executing
// their side effects, and also cancels kernel ops (SEND) that follow.
TEST_P(SemaphoreLinkedTest, LinkedFailureCancelsMixedChain) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_async_semaphore_t* wait_semaphore = nullptr;
  iree_async_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &wait_semaphore));
  IREE_ASSERT_OK(iree_async_semaphore_create_software(
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &signal_semaphore));

  CompletionTracker wait_tracker;
  CompletionTracker signal_tracker;
  CompletionTracker send_tracker;

  // WAIT (head, linked to SIGNAL).
  iree_async_semaphore_t* wait_sem_ptr = wait_semaphore;
  uint64_t wait_value = 10;

  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &wait_tracker;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  wait_op.semaphores = &wait_sem_ptr;
  wait_op.values = &wait_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  // SIGNAL (middle, linked to SEND).
  iree_async_semaphore_t* signal_sem_ptr = signal_semaphore;
  uint64_t signal_value = 99;

  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = CompletionTracker::Callback;
  signal_op.base.user_data = &signal_tracker;
  signal_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  signal_op.semaphores = &signal_sem_ptr;
  signal_op.values = &signal_value;
  signal_op.count = 1;
  signal_op.frontier = NULL;

  // SEND (tail).
  const char* send_data = "should-not-arrive";
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, strlen(send_data));

  iree_async_socket_send_operation_t send_op;
  memset(&send_op, 0, sizeof(send_op));
  send_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND;
  send_op.base.completion_fn = CompletionTracker::Callback;
  send_op.base.user_data = &send_tracker;
  send_op.socket = client;
  send_op.buffers.values = &send_span;
  send_op.buffers.count = 1;
  send_op.send_flags = IREE_ASYNC_SOCKET_SEND_FLAG_NONE;

  // Submit the chain.
  iree_async_operation_t* ops[] = {&wait_op.base, &signal_op.base,
                                   &send_op.base};
  iree_async_operation_list_t list = {ops, 3};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Fail the wait semaphore to break the chain.
  std::thread failer([wait_semaphore]() {
    iree_wait_until(iree_time_now() + iree_make_duration_ms(50));
    iree_async_semaphore_fail(
        wait_semaphore, iree_make_status(IREE_STATUS_ABORTED, "device lost"));
  });

  // Poll until all three complete.
  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(5000));

  failer.join();

  // WAIT should complete with ABORTED.
  EXPECT_EQ(wait_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, wait_tracker.ConsumeStatus());

  // SIGNAL should be CANCELLED (chain was broken by WAIT failure).
  // The signal side-effect must NOT have executed.
  EXPECT_EQ(signal_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, signal_tracker.ConsumeStatus());

  // SEND should be CANCELLED.
  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, send_tracker.ConsumeStatus());

  // Signal semaphore must NOT have been signaled (signal was cancelled).
  EXPECT_EQ(iree_async_semaphore_query(signal_semaphore), 0u);

  iree_async_semaphore_release(wait_semaphore);
  iree_async_semaphore_release(signal_semaphore);
  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

CTS_REGISTER_TEST_SUITE(SemaphoreLinkedTest);

}  // namespace iree::async::cts
