// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for linked operation sequences.
//
// Tests verify that operations with IREE_ASYNC_OPERATION_FLAG_LINKED execute
// in kernel-enforced order with proper dependency and error propagation.
// When an operation has LINKED flag set, the kernel will not begin the next
// operation until the linked operation completes successfully.
//
// ## Cross-Proactor Portability Requirements
//
// Any proactor backend claiming
// IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS must implement these
// semantics:
//
// 1. **Ordering**: If operation A has LINKED flag, operation B (next in batch)
//    must not begin until A completes. Callbacks fire in chain order.
//
// 2. **Error propagation**: If A fails (any non-OK status including CANCELLED),
//    B receives IREE_STATUS_CANCELLED without executing. This propagates
//    through the entire chain.
//
// 3. **Independence**: Two chains in the same batch [A(L)->B, C(L)->D] are
//    independent. Failure in A->B does not affect C->D.
//
// 4. **Validation**: LINKED on the last operation in a batch must be rejected
//    with IREE_STATUS_INVALID_ARGUMENT.
//
// Backends may implement this via kernel primitives (io_uring IOSQE_IO_LINK)
// or in user-space (kqueue, IOCP). The CTS tests verify contract compliance.

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/event.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/span.h"

namespace iree::async::cts {

class SequenceTest : public SocketTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (!iree_any_bit_set(capabilities_,
                          IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
      GTEST_SKIP() << "backend lacks linked operations capability";
    }
  }

  // Tracks completion order via operation indices.
  struct OrderTracker {
    std::vector<int> order;
    std::vector<iree_status_code_t> status_codes;

    static void Callback(void* user_data, iree_async_operation_t* operation,
                         iree_status_t status,
                         iree_async_completion_flags_t flags) {
      auto* tracker = static_cast<OrderTracker*>(user_data);
      // The index is stored in the operation's user_data field offset.
      int index = static_cast<int>(
          reinterpret_cast<intptr_t>(static_cast<iree_async_nop_operation_t*>(
                                         reinterpret_cast<void*>(operation))
                                         ->base.pool));
      tracker->order.push_back(index);
      tracker->status_codes.push_back(iree_status_code(status));
      iree_status_ignore(status);
    }
  };

  // Helper to create a NOP operation with tracking index.
  void InitNopWithIndex(iree_async_nop_operation_t* nop, OrderTracker* tracker,
                        int index, bool linked) {
    memset(nop, 0, sizeof(*nop));
    nop->base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
    nop->base.completion_fn = OrderTracker::Callback;
    nop->base.user_data = tracker;
    // Stash index in the pool field (not used in tests).
    nop->base.pool = reinterpret_cast<iree_async_operation_pool_t*>(
        static_cast<intptr_t>(index));
    if (linked) {
      nop->base.flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;
    }
  }
};

//===----------------------------------------------------------------------===//
// Basic linked NOP tests
//===----------------------------------------------------------------------===//

// Two NOPs linked: first has LINKED flag, second does not.
// Both should complete in submission order.
TEST_P(SequenceTest, TwoLinkedNops) {
  iree_async_nop_operation_t nop1, nop2;
  OrderTracker tracker;

  InitNopWithIndex(&nop1, &tracker, 0, /*linked=*/true);
  InitNopWithIndex(&nop2, &tracker, 1, /*linked=*/false);

  iree_async_operation_t* ops[] = {&nop1.base, &nop2.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  PollUntil(/*min_completions=*/2, /*total_budget=*/iree_make_duration_ms(500));

  ASSERT_EQ(tracker.order.size(), 2u);
  // Linked operations complete in submission order.
  EXPECT_EQ(tracker.order[0], 0);
  EXPECT_EQ(tracker.order[1], 1);
  EXPECT_EQ(tracker.status_codes[0], IREE_STATUS_OK);
  EXPECT_EQ(tracker.status_codes[1], IREE_STATUS_OK);
}

// Chain of 5 NOPs with LINKED flags. All should complete in order.
TEST_P(SequenceTest, LinkedNopChain) {
  constexpr int kChainLength = 5;
  iree_async_nop_operation_t nops[kChainLength];
  OrderTracker tracker;

  for (int i = 0; i < kChainLength; ++i) {
    // All but the last have LINKED flag.
    InitNopWithIndex(&nops[i], &tracker, i, /*linked=*/i < kChainLength - 1);
  }

  iree_async_operation_t* ops[kChainLength];
  for (int i = 0; i < kChainLength; ++i) {
    ops[i] = &nops[i].base;
  }
  iree_async_operation_list_t list = {ops, kChainLength};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  PollUntil(/*min_completions=*/kChainLength,
            /*total_budget=*/iree_make_duration_ms(500));

  ASSERT_EQ(tracker.order.size(), static_cast<size_t>(kChainLength));
  for (int i = 0; i < kChainLength; ++i) {
    EXPECT_EQ(tracker.order[i], i) << "Operation " << i << " out of order";
    EXPECT_EQ(tracker.status_codes[i], IREE_STATUS_OK);
  }
}

//===----------------------------------------------------------------------===//
// Timer ordering tests
//===----------------------------------------------------------------------===//

// Timer with LINKED flag forces the next operation to wait.
// Timer A (50ms, LINKED) -> Timer B (immediate deadline).
// Even though B has earlier deadline, it must wait for A.
TEST_P(SequenceTest, LinkedTimerOrder) {
  iree_async_timer_operation_t timer_a, timer_b;
  memset(&timer_a, 0, sizeof(timer_a));
  memset(&timer_b, 0, sizeof(timer_b));

  timer_a.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer_b.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

  iree_time_t now = iree_time_now();
  timer_a.deadline_ns = now + iree_make_duration_ms(50);  // 50ms in future
  timer_b.deadline_ns = now - 1000000LL;  // 1ms in past (immediate)

  // timer_a is linked to timer_b.
  timer_a.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;

  struct TimerTracker {
    std::vector<int> order;
    static void CallbackA(void* user_data, iree_async_operation_t* op,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
      static_cast<TimerTracker*>(user_data)->order.push_back(0);
      iree_status_ignore(status);
    }
    static void CallbackB(void* user_data, iree_async_operation_t* op,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
      static_cast<TimerTracker*>(user_data)->order.push_back(1);
      iree_status_ignore(status);
    }
  };

  TimerTracker tracker;
  timer_a.base.completion_fn = TimerTracker::CallbackA;
  timer_a.base.user_data = &tracker;
  timer_b.base.completion_fn = TimerTracker::CallbackB;
  timer_b.base.user_data = &tracker;

  iree_async_operation_t* ops[] = {&timer_a.base, &timer_b.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Budget enough for timer_a to fire.
  PollUntil(/*min_completions=*/2, /*total_budget=*/iree_make_duration_ms(200));

  ASSERT_EQ(tracker.order.size(), 2u);
  // timer_a must complete before timer_b despite timer_b's earlier deadline.
  EXPECT_EQ(tracker.order[0], 0);
  EXPECT_EQ(tracker.order[1], 1);
}

//===----------------------------------------------------------------------===//
// Chain failure propagation
//===----------------------------------------------------------------------===//

// When the first operation in a chain is cancelled, subsequent linked
// operations receive CANCELLED status (-ECANCELED from kernel).
TEST_P(SequenceTest, ChainWithCancellation) {
  iree_async_timer_operation_t timer_a, timer_b;
  memset(&timer_a, 0, sizeof(timer_a));
  memset(&timer_b, 0, sizeof(timer_b));

  timer_a.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer_b.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

  // Both far in the future so we have time to cancel.
  iree_time_t now = iree_time_now();
  timer_a.deadline_ns = now + iree_make_duration_ms(10000);  // 10s
  timer_b.deadline_ns = now + iree_make_duration_ms(10000);  // 10s

  timer_a.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;

  CompletionTracker tracker_a, tracker_b;
  timer_a.base.completion_fn = CompletionTracker::Callback;
  timer_a.base.user_data = &tracker_a;
  timer_b.base.completion_fn = CompletionTracker::Callback;
  timer_b.base.user_data = &tracker_b;

  iree_async_operation_t* ops[] = {&timer_a.base, &timer_b.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Cancel the first timer.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &timer_a.base));

  // Poll to receive both callbacks.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(1000));

  // timer_a should get CANCELLED.
  EXPECT_EQ(tracker_a.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker_a.ConsumeStatus());

  // timer_b should also get CANCELLED (chain broken by timer_a's failure).
  EXPECT_EQ(tracker_b.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker_b.ConsumeStatus());
}

//===----------------------------------------------------------------------===//
// Mixed linked and unlinked operations
//===----------------------------------------------------------------------===//

// Batch with two independent chains: [A(LINKED)->B] and [C(LINKED)->D].
// The chains execute independently but maintain internal ordering.
TEST_P(SequenceTest, MixedLinkedUnlinked) {
  iree_async_nop_operation_t a, b, c, d;
  memset(&a, 0, sizeof(a));
  memset(&b, 0, sizeof(b));
  memset(&c, 0, sizeof(c));
  memset(&d, 0, sizeof(d));

  a.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  b.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  c.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  d.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;

  // A links to B, C links to D.
  a.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  c.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;

  struct MixedTracker {
    std::vector<char> order;
    static void CallbackA(void* u, iree_async_operation_t* o, iree_status_t s,
                          iree_async_completion_flags_t f) {
      static_cast<MixedTracker*>(u)->order.push_back('A');
      iree_status_ignore(s);
    }
    static void CallbackB(void* u, iree_async_operation_t* o, iree_status_t s,
                          iree_async_completion_flags_t f) {
      static_cast<MixedTracker*>(u)->order.push_back('B');
      iree_status_ignore(s);
    }
    static void CallbackC(void* u, iree_async_operation_t* o, iree_status_t s,
                          iree_async_completion_flags_t f) {
      static_cast<MixedTracker*>(u)->order.push_back('C');
      iree_status_ignore(s);
    }
    static void CallbackD(void* u, iree_async_operation_t* o, iree_status_t s,
                          iree_async_completion_flags_t f) {
      static_cast<MixedTracker*>(u)->order.push_back('D');
      iree_status_ignore(s);
    }
  };

  MixedTracker tracker;
  a.base.completion_fn = MixedTracker::CallbackA;
  a.base.user_data = &tracker;
  b.base.completion_fn = MixedTracker::CallbackB;
  b.base.user_data = &tracker;
  c.base.completion_fn = MixedTracker::CallbackC;
  c.base.user_data = &tracker;
  d.base.completion_fn = MixedTracker::CallbackD;
  d.base.user_data = &tracker;

  iree_async_operation_t* ops[] = {&a.base, &b.base, &c.base, &d.base};
  iree_async_operation_list_t list = {ops, 4};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  PollUntil(/*min_completions=*/4, /*total_budget=*/iree_make_duration_ms(500));

  ASSERT_EQ(tracker.order.size(), 4u);

  // Find positions of each operation.
  int pos_a = -1, pos_b = -1, pos_c = -1, pos_d = -1;
  for (int i = 0; i < 4; ++i) {
    switch (tracker.order[i]) {
      case 'A':
        pos_a = i;
        break;
      case 'B':
        pos_b = i;
        break;
      case 'C':
        pos_c = i;
        break;
      case 'D':
        pos_d = i;
        break;
    }
  }

  // A must complete before B (linked chain).
  EXPECT_LT(pos_a, pos_b) << "A must complete before B";
  // C must complete before D (linked chain).
  EXPECT_LT(pos_c, pos_d) << "C must complete before D";
  // No ordering constraint between the two chains.
}

//===----------------------------------------------------------------------===//
// EVENT_WAIT linked to NOP
//===----------------------------------------------------------------------===//

// EVENT_WAIT has internal POLL_ADD+READ linking. When user adds LINKED flag,
// the internal chain should be extended to include the next user operation.
TEST_P(SequenceTest, LinkedEventWaitNop) {
  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));

  iree_async_event_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_op.event = event;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;

  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;

  struct EventNopTracker {
    std::vector<int> order;
    static void WaitCallback(void* u, iree_async_operation_t* o,
                             iree_status_t s, iree_async_completion_flags_t f) {
      static_cast<EventNopTracker*>(u)->order.push_back(0);
      iree_status_ignore(s);
    }
    static void NopCallback(void* u, iree_async_operation_t* o, iree_status_t s,
                            iree_async_completion_flags_t f) {
      static_cast<EventNopTracker*>(u)->order.push_back(1);
      iree_status_ignore(s);
    }
  };

  EventNopTracker tracker;
  wait_op.base.completion_fn = EventNopTracker::WaitCallback;
  wait_op.base.user_data = &tracker;
  nop.base.completion_fn = EventNopTracker::NopCallback;
  nop.base.user_data = &tracker;

  iree_async_operation_t* ops[] = {&wait_op.base, &nop.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Signal the event.
  IREE_ASSERT_OK(iree_async_event_set(event));

  PollUntil(/*min_completions=*/2, /*total_budget=*/iree_make_duration_ms(500));

  ASSERT_EQ(tracker.order.size(), 2u);
  // Event wait must complete before the NOP.
  EXPECT_EQ(tracker.order[0], 0);
  EXPECT_EQ(tracker.order[1], 1);

  iree_async_event_release(event);
}

//===----------------------------------------------------------------------===//
// Socket connect+send chain
//===----------------------------------------------------------------------===//

// CONNECT (LINKED) -> SEND: send waits for connect to complete.
TEST_P(SequenceTest, LinkedConnectSend) {
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Start an accept on the listener.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Create client socket.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  struct ConnectSendTracker {
    std::vector<int> order;
    static void ConnectCallback(void* u, iree_async_operation_t* o,
                                iree_status_t s,
                                iree_async_completion_flags_t f) {
      static_cast<ConnectSendTracker*>(u)->order.push_back(0);
      iree_status_ignore(s);
    }
    static void SendCallback(void* u, iree_async_operation_t* o,
                             iree_status_t s, iree_async_completion_flags_t f) {
      static_cast<ConnectSendTracker*>(u)->order.push_back(1);
      iree_status_ignore(s);
    }
  };
  ConnectSendTracker tracker;

  // CONNECT linked to SEND.
  iree_async_socket_connect_operation_t connect_op;
  InitConnectOperation(&connect_op, client, listen_address,
                       ConnectSendTracker::ConnectCallback, &tracker);
  connect_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;

  const char* message = "hello";
  iree_async_span_t send_span = iree_async_span_from_ptr((void*)message, 5);
  iree_async_socket_send_operation_t send_op;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    ConnectSendTracker::SendCallback, &tracker);

  // Submit connect and send as a linked chain.
  iree_async_operation_t* ops[] = {&connect_op.base, &send_op.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Wait for accept + connect + send.
  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_EQ(tracker.order.size(), 2u);
  // Connect must complete before send.
  EXPECT_EQ(tracker.order[0], 0);
  EXPECT_EQ(tracker.order[1], 1);

  iree_async_socket_release(client);
  iree_async_socket_release(accept_op.accepted_socket);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Error handling tests
//===----------------------------------------------------------------------===//

// LINKED flag on last operation must be rejected.
// Setting LINKED on the last op is a contract violation: LINKED means
// "link to next" but there is no next operation.
TEST_P(SequenceTest, TrailingLinkedRejected) {
  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;  // Invalid: no next op.

  CompletionTracker tracker;
  nop.base.completion_fn = CompletionTracker::Callback;
  nop.base.user_data = &tracker;

  // Single operation with LINKED flag should be rejected.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_proactor_submit_one(proactor_, &nop.base));

  // Callback should not have been called since submit failed.
  EXPECT_EQ(tracker.call_count, 0);
}

// LINKED on last op in a multi-op batch should also be rejected.
TEST_P(SequenceTest, TrailingLinkedInBatchRejected) {
  iree_async_nop_operation_t nop1, nop2;
  memset(&nop1, 0, sizeof(nop1));
  memset(&nop2, 0, sizeof(nop2));

  nop1.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop2.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;

  // nop1 LINKED to nop2 is valid. But nop2 also has LINKED = invalid.
  nop1.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  nop2.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;  // Invalid: last op.

  CompletionTracker tracker;
  nop1.base.completion_fn = CompletionTracker::Callback;
  nop1.base.user_data = &tracker;
  nop2.base.completion_fn = CompletionTracker::Callback;
  nop2.base.user_data = &tracker;

  iree_async_operation_t* ops[] = {&nop1.base, &nop2.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_proactor_submit(proactor_, list));

  // Neither callback should have been called since submit failed.
  EXPECT_EQ(tracker.call_count, 0);
}

//===----------------------------------------------------------------------===//
// Failure propagation tests (cross-proactor portability)
//===----------------------------------------------------------------------===//

// When a linked operation fails functionally (not just cancelled), the
// subsequent linked operation should receive CANCELLED status.
// This tests that error propagation works with real I/O failures.
TEST_P(SequenceTest, LinkedConnectFailurePropagates) {
  // Create client socket.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  // Connect to a port with no listener — should get ECONNREFUSED.
  iree_async_address_t bad_address = CreateRefusedAddress();

  CompletionTracker connect_tracker, send_tracker;

  iree_async_socket_connect_operation_t connect_op;
  InitConnectOperation(&connect_op, client, bad_address,
                       CompletionTracker::Callback, &connect_tracker);
  connect_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;

  // The send should be cancelled when connect fails.
  const char* message = "hello";
  iree_async_span_t send_span = iree_async_span_from_ptr((void*)message, 5);
  iree_async_socket_send_operation_t send_op;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  iree_async_operation_t* ops[] = {&connect_op.base, &send_op.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Wait for both operations. Connect will fail, send will be cancelled.
  // Use a shorter timeout since we know the connect will fail quickly.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Connect should have failed (not OK).
  EXPECT_EQ(connect_tracker.call_count, 1);
  {
    iree_status_t status = connect_tracker.ConsumeStatus();
    if (iree_status_is_ok(status)) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
    } else {
      iree_status_ignore(status);
    }
  }

  // Send should have been cancelled due to connect failure.
  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, send_tracker.ConsumeStatus());

  iree_async_socket_release(client);
}

// Three-timer chain with mid-chain cancellation.
// Timer A (LINKED) -> Timer B (LINKED) -> Timer C
// Cancel Timer B while A is still pending.
// Expected: A completes OK or CANCELLED (race), B gets CANCELLED, C gets
// CANCELLED.
TEST_P(SequenceTest, LinkedTimerChainMidCancellation) {
  iree_async_timer_operation_t timer_a, timer_b, timer_c;
  memset(&timer_a, 0, sizeof(timer_a));
  memset(&timer_b, 0, sizeof(timer_b));
  memset(&timer_c, 0, sizeof(timer_c));

  timer_a.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer_b.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer_c.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;

  iree_time_t now = iree_time_now();
  timer_a.deadline_ns = now + iree_make_duration_ms(50);  // 50ms
  timer_b.deadline_ns =
      now + iree_make_duration_ms(10000);  // 10s (will be cancelled)
  timer_c.deadline_ns =
      now + iree_make_duration_ms(10000);  // 10s (will be cancelled)

  timer_a.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  timer_b.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;

  CompletionTracker tracker_a, tracker_b, tracker_c;
  timer_a.base.completion_fn = CompletionTracker::Callback;
  timer_a.base.user_data = &tracker_a;
  timer_b.base.completion_fn = CompletionTracker::Callback;
  timer_b.base.user_data = &tracker_b;
  timer_c.base.completion_fn = CompletionTracker::Callback;
  timer_c.base.user_data = &tracker_c;

  iree_async_operation_t* ops[] = {&timer_a.base, &timer_b.base, &timer_c.base};
  iree_async_operation_list_t list = {ops, 3};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Wait for timer_a to fire (50ms), then cancel timer_b.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(200));
  ASSERT_EQ(tracker_a.call_count, 1);
  IREE_EXPECT_OK(tracker_a.ConsumeStatus());

  // Cancel timer_b - this should also cancel timer_c.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &timer_b.base));

  // Poll for remaining completions.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(1000));

  // timer_b should be cancelled.
  EXPECT_EQ(tracker_b.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker_b.ConsumeStatus());

  // timer_c should also be cancelled (chain propagation).
  EXPECT_EQ(tracker_c.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker_c.ConsumeStatus());
}

//===----------------------------------------------------------------------===//
// Batch array lifetime regression test
//===----------------------------------------------------------------------===//

// Submits a linked timer chain from a helper function whose batch array goes
// out of scope immediately. Verifies the proactor uses the intrusive
// linked_next chain (which lives in the operation structs) rather than the
// caller's batch array (which is dead). Without this fix, the continuation
// dispatch reads freed stack memory when the first timer fires.
TEST_P(SequenceTest, LinkedTimerAsyncSubmit) {
  iree_async_timer_operation_t timer_a, timer_b;
  memset(&timer_a, 0, sizeof(timer_a));
  memset(&timer_b, 0, sizeof(timer_b));

  timer_a.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer_b.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer_a.deadline_ns = iree_time_now() + iree_make_duration_ms(50);
  timer_b.deadline_ns = iree_time_now() + iree_make_duration_ms(10);
  timer_a.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;

  CompletionTracker tracker_a, tracker_b;
  timer_a.base.completion_fn = CompletionTracker::Callback;
  timer_a.base.user_data = &tracker_a;
  timer_b.base.completion_fn = CompletionTracker::Callback;
  timer_b.base.user_data = &tracker_b;

  // Submit from a lambda whose stack frame (including the batch array) is
  // destroyed on return. If the proactor holds a pointer to the batch array,
  // ASAN will catch the use-after-return.
  [&]() {
    iree_async_operation_t* ops[] = {&timer_a.base, &timer_b.base};
    iree_async_operation_list_t list = {ops, 2};
    IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));
  }();

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker_a.call_count, 1);
  EXPECT_EQ(tracker_b.call_count, 1);
  IREE_EXPECT_OK(tracker_a.ConsumeStatus());
  IREE_EXPECT_OK(tracker_b.ConsumeStatus());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(SequenceTest);

//===----------------------------------------------------------------------===//
// IREE_ASYNC_OPERATION_TYPE_SEQUENCE tests
//===----------------------------------------------------------------------===//
//
// Tests for the higher-level SEQUENCE operation type which chains multiple
// operations into an ordered pipeline. Two execution paths are tested:
//   - LINK path (step_fn == NULL): expands steps as a linked batch.
//   - Emulation path (step_fn != NULL): submits one step at a time with
//     inter-step callbacks.

class SequenceOperationTest : public CtsTestBase<> {
 protected:
  // Helper: initialize a NOP operation for use as a sequence step.
  void InitStepNop(iree_async_nop_operation_t* nop) {
    memset(nop, 0, sizeof(*nop));
    nop->base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
    // completion_fn and user_data are overwritten by the sequence trampolines.
  }

  // Helper: initialize a timer operation for use as a sequence step.
  void InitStepTimer(iree_async_timer_operation_t* timer,
                     iree_duration_t delay_ns) {
    memset(timer, 0, sizeof(*timer));
    timer->base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
    timer->deadline_ns = iree_time_now() + delay_ns;
  }

  // Helper: set up a sequence operation with pre-filled step pointers.
  // Does NOT slab-allocate; uses the caller's step_array directly.
  void InitSequence(iree_async_sequence_operation_t* sequence,
                    iree_async_operation_t** step_array,
                    iree_host_size_t step_count, iree_async_step_fn_t step_fn,
                    iree_async_completion_fn_t completion_fn, void* user_data) {
    memset(sequence, 0, sizeof(*sequence));
    sequence->base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
    sequence->base.completion_fn = completion_fn;
    sequence->base.user_data = user_data;
    sequence->steps = step_array;
    sequence->step_count = step_count;
    sequence->current_step = 0;
    sequence->step_fn = step_fn;
  }
};

//===----------------------------------------------------------------------===//
// LINK path tests (step_fn == NULL)
//===----------------------------------------------------------------------===//

// Zero-step sequence completes immediately with OK.
TEST_P(SequenceOperationTest, ZeroStepSequence) {
  CompletionTracker tracker;
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, nullptr, 0, nullptr, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  // Zero-step sequence completes synchronously during submit.
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Single NOP step completes with OK.
TEST_P(SequenceOperationTest, SingleStepNop) {
  iree_async_nop_operation_t nop;
  InitStepNop(&nop);

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&nop.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 1, nullptr, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Two NOP steps complete in order. Base callback fires exactly once.
TEST_P(SequenceOperationTest, TwoStepNop) {
  iree_async_nop_operation_t nop0, nop1;
  InitStepNop(&nop0);
  InitStepNop(&nop1);

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&nop0.base, &nop1.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 2, nullptr, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(500));

  // Base callback fires exactly once, not once per step.
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Three NOP steps all complete. Tests the LINK path with longer chains.
TEST_P(SequenceOperationTest, ThreeStepNop) {
  iree_async_nop_operation_t nops[3];
  for (auto& nop : nops) InitStepNop(&nop);

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&nops[0].base, &nops[1].base,
                                     &nops[2].base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 3, nullptr, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Sequence with a timer: [NOP, TIMER(10ms), NOP].
// Exercises io_uring split points (TIMER requires userspace chain emulation).
TEST_P(SequenceOperationTest, SequenceWithTimer) {
  iree_async_nop_operation_t nop0, nop1;
  iree_async_timer_operation_t timer;
  InitStepNop(&nop0);
  InitStepTimer(&timer, iree_make_duration_ms(10));
  InitStepNop(&nop1);

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&nop0.base, &timer.base, &nop1.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 3, nullptr, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Cancel a sequence during execution.
// [TIMER(10s), NOP] — cancel while the timer is pending.
TEST_P(SequenceOperationTest, SequenceCancellation) {
  iree_async_timer_operation_t timer;
  iree_async_nop_operation_t nop;
  InitStepTimer(&timer, iree_make_duration_ms(10000));  // 10s
  InitStepNop(&nop);

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&timer.base, &nop.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 2, nullptr, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  // Cancel the sequence while the timer is pending.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

//===----------------------------------------------------------------------===//
// Emulation path tests (step_fn != NULL)
//===----------------------------------------------------------------------===//

// Verify step_fn fires between steps with correct arguments.
// step_fn receives (completed_step, next_step) with next_step == NULL on last.
TEST_P(SequenceOperationTest, StepFnCalled) {
  iree_async_nop_operation_t nops[3];
  for (auto& nop : nops) InitStepNop(&nop);

  struct StepFnTracker {
    std::vector<std::pair<iree_async_operation_t*, iree_async_operation_t*>>
        calls;
  };
  StepFnTracker step_tracker;

  auto step_fn = [](void* user_data, iree_async_operation_t* completed_step,
                    iree_async_operation_t* next_step) -> iree_status_t {
    auto* tracker = static_cast<StepFnTracker*>(user_data);
    tracker->calls.push_back({completed_step, next_step});
    return iree_ok_status();
  };

  CompletionTracker completion_tracker;
  iree_async_operation_t* steps[] = {&nops[0].base, &nops[1].base,
                                     &nops[2].base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 3, step_fn, CompletionTracker::Callback,
               &completion_tracker);
  // step_fn receives user_data from the sequence's base.user_data, which is
  // set to &completion_tracker by InitSequence. Override for step tracking.
  sequence.base.user_data = &step_tracker;
  sequence.base.completion_fn = [](void* user_data, iree_async_operation_t* op,
                                   iree_status_t status,
                                   iree_async_completion_flags_t flags) {
    // Just consume the status. We check step_tracker below.
    iree_status_ignore(status);
  };

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  // Emulation path: one poll round-trip per step.
  PollUntil(/*min_completions=*/3,
            /*total_budget=*/iree_make_duration_ms(1000));

  // step_fn fires after every step: after step 0, step 1, and step 2 (last).
  // The last call receives next_step == NULL signaling sequence completion.
  ASSERT_EQ(step_tracker.calls.size(), 3u);

  // After step 0: completed = nops[0], next = nops[1].
  EXPECT_EQ(step_tracker.calls[0].first, &nops[0].base);
  EXPECT_EQ(step_tracker.calls[0].second, &nops[1].base);

  // After step 1: completed = nops[1], next = nops[2].
  EXPECT_EQ(step_tracker.calls[1].first, &nops[1].base);
  EXPECT_EQ(step_tracker.calls[1].second, &nops[2].base);

  // After step 2 (last): completed = nops[2], next = NULL.
  EXPECT_EQ(step_tracker.calls[2].first, &nops[2].base);
  EXPECT_EQ(step_tracker.calls[2].second, nullptr);
}

// step_fn modifies the next step's timer deadline dynamically.
TEST_P(SequenceOperationTest, StepFnModifiesNextStep) {
  iree_async_nop_operation_t nop;
  iree_async_timer_operation_t timer;
  InitStepNop(&nop);
  // Timer starts with a far-future deadline (10s).
  InitStepTimer(&timer, iree_make_duration_ms(10000));

  // step_fn overwrites the timer deadline to 10ms, so it actually completes.
  auto step_fn = [](void* user_data, iree_async_operation_t* completed_step,
                    iree_async_operation_t* next_step) -> iree_status_t {
    if (next_step && next_step->type == IREE_ASYNC_OPERATION_TYPE_TIMER) {
      auto* timer_op =
          reinterpret_cast<iree_async_timer_operation_t*>(next_step);
      timer_op->deadline_ns = iree_time_now() + iree_make_duration_ms(10);
    }
    return iree_ok_status();
  };

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&nop.base, &timer.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 2, step_fn, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  // If step_fn didn't modify the timer, this would time out (10s deadline).
  // With the modification, it should complete within 200ms.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// step_fn returns an error, aborting the sequence.
TEST_P(SequenceOperationTest, StepFnAbort) {
  iree_async_nop_operation_t nop0, nop1;
  InitStepNop(&nop0);
  InitStepNop(&nop1);

  // step_fn vetoes continuation after step 0.
  auto step_fn = [](void* user_data, iree_async_operation_t* completed_step,
                    iree_async_operation_t* next_step) -> iree_status_t {
    return iree_make_status(IREE_STATUS_ABORTED, "step_fn vetoed");
  };

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&nop0.base, &nop1.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 2, step_fn, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(500));

  // step_fn error aborts the sequence.
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, tracker.ConsumeStatus());
}

// Cancel a sequence with step_fn during emulation execution.
TEST_P(SequenceOperationTest, EmulationCancellation) {
  iree_async_timer_operation_t timer;
  iree_async_nop_operation_t nop;
  InitStepTimer(&timer, iree_make_duration_ms(10000));  // 10s
  InitStepNop(&nop);

  // Both step_fn and completion_fn share user_data, so use a combined context
  // that tracks step_fn calls while forwarding completions to the tracker.
  struct CancelTestContext {
    CompletionTracker* completion_tracker;
    int step_fn_call_count;
  };
  CompletionTracker tracker;
  CancelTestContext context = {&tracker, 0};

  auto cancel_step_fn = [](void* user_data,
                           iree_async_operation_t* completed_step,
                           iree_async_operation_t* next_step) -> iree_status_t {
    auto* ctx = static_cast<CancelTestContext*>(user_data);
    ctx->step_fn_call_count++;
    return iree_ok_status();
  };
  auto cancel_completion_fn = [](void* user_data, iree_async_operation_t* op,
                                 iree_status_t status,
                                 iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<CancelTestContext*>(user_data);
    CompletionTracker::Callback(ctx->completion_tracker, op, status, flags);
  };

  iree_async_operation_t* steps[] = {&timer.base, &nop.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 2, cancel_step_fn, cancel_completion_fn,
               &context);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  // Cancel while the timer is pending.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());

  // step_fn should NOT have been called (cancel happened before step 0
  // completed, so step_fn between steps was never reached).
  EXPECT_EQ(context.step_fn_call_count, 0);
}

//===----------------------------------------------------------------------===//
// Edge-case tests
//===----------------------------------------------------------------------===//

// Zero-step sequence with step_fn set completes immediately without calling
// step_fn (there are no inter-step gaps in a zero-step sequence).
TEST_P(SequenceOperationTest, ZeroStepWithStepFn) {
  CompletionTracker tracker;
  struct ZeroStepContext {
    CompletionTracker* tracker;
    int step_fn_call_count;
  };
  ZeroStepContext context = {&tracker, 0};

  auto zero_step_fn = [](void* user_data,
                         iree_async_operation_t* completed_step,
                         iree_async_operation_t* next_step) -> iree_status_t {
    auto* ctx = static_cast<ZeroStepContext*>(user_data);
    ctx->step_fn_call_count++;
    return iree_ok_status();
  };
  auto zero_completion_fn = [](void* user_data, iree_async_operation_t* op,
                               iree_status_t status,
                               iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<ZeroStepContext*>(user_data);
    CompletionTracker::Callback(ctx->tracker, op, status, flags);
  };

  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, nullptr, 0, zero_step_fn, zero_completion_fn,
               &context);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  // Zero-step: completes synchronously, step_fn never called.
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(context.step_fn_call_count, 0);
}

// Single NOP step that gets cancelled immediately after submit.
TEST_P(SequenceOperationTest, SingleStepCancellation) {
  iree_async_timer_operation_t timer;
  InitStepTimer(&timer, iree_make_duration_ms(10000));  // 10s

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&timer.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 1, nullptr, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(1000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

// LINKED flag on a SEQUENCE operation must be rejected.
TEST_P(SequenceOperationTest, LINKEDOnSequenceRejected) {
  iree_async_nop_operation_t nop;
  InitStepNop(&nop);

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&nop.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 1, nullptr, CompletionTracker::Callback,
               &tracker);

  // Set LINKED flag on the sequence itself.
  sequence.base.flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_submit_one(proactor_, &sequence.base));

  // Callback should NOT have fired (submit rejected before processing).
  EXPECT_EQ(tracker.call_count, 0);
}

// LINKED flag on a SEQUENCE operation with step_fn must also be rejected.
TEST_P(SequenceOperationTest, LINKEDOnSequenceWithStepFnRejected) {
  iree_async_nop_operation_t nop;
  InitStepNop(&nop);

  auto step_fn = [](void* user_data, iree_async_operation_t* completed_step,
                    iree_async_operation_t* next_step) -> iree_status_t {
    return iree_ok_status();
  };

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&nop.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 1, step_fn, CompletionTracker::Callback,
               &tracker);

  sequence.base.flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_submit_one(proactor_, &sequence.base));
  EXPECT_EQ(tracker.call_count, 0);
}

// step_fn returns error on the last step (when next_step is NULL).
// The error should propagate as the sequence's final status, overriding
// the "all steps complete" happy path.
TEST_P(SequenceOperationTest, StepFnErrorOnLastStep) {
  iree_async_nop_operation_t nop0, nop1;
  InitStepNop(&nop0);
  InitStepNop(&nop1);

  struct LastStepContext {
    CompletionTracker* tracker;
    int step_fn_call_count;
  };
  CompletionTracker tracker;
  LastStepContext context = {&tracker, 0};

  auto step_fn = [](void* user_data, iree_async_operation_t* completed_step,
                    iree_async_operation_t* next_step) -> iree_status_t {
    auto* ctx = static_cast<LastStepContext*>(user_data);
    ctx->step_fn_call_count++;
    // Veto on the last step (next_step == NULL).
    if (!next_step) {
      return iree_make_status(IREE_STATUS_DATA_LOSS, "last step vetoed");
    }
    return iree_ok_status();
  };
  auto completion_fn = [](void* user_data, iree_async_operation_t* op,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<LastStepContext*>(user_data);
    CompletionTracker::Callback(ctx->tracker, op, status, flags);
  };

  iree_async_operation_t* steps[] = {&nop0.base, &nop1.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 2, step_fn, completion_fn, &context);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  // step_fn called twice: once after step 0 (next_step=&nop1, OK), once after
  // step 1 (next_step=NULL, DATA_LOSS).
  EXPECT_EQ(context.step_fn_call_count, 2);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, tracker.ConsumeStatus());
}

// Cancel arrives between the first CANCEL_REQUESTED check and the step_fn call.
// The second CANCEL_REQUESTED check (after step_fn) should catch it.
TEST_P(SequenceOperationTest, CancelDuringStepFn) {
  // Use two timers: a short one that completes quickly, and a long one that
  // would take 10s if not cancelled. The step_fn sleeps briefly to widen the
  // window for cancel arrival.
  iree_async_timer_operation_t timer_short, timer_long;
  InitStepTimer(&timer_short, iree_make_duration_ms(1));
  InitStepTimer(&timer_long, iree_make_duration_ms(10000));

  struct CancelDuringContext {
    CompletionTracker* tracker;
    iree_async_proactor_t* proactor;
    iree_async_sequence_operation_t* sequence;
    int step_fn_call_count;
  };
  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&timer_short.base, &timer_long.base};
  iree_async_sequence_operation_t sequence;

  CancelDuringContext context = {&tracker, proactor_, &sequence, 0};

  auto step_fn = [](void* user_data, iree_async_operation_t* completed_step,
                    iree_async_operation_t* next_step) -> iree_status_t {
    auto* ctx = static_cast<CancelDuringContext*>(user_data);
    ctx->step_fn_call_count++;
    // Cancel the sequence from within step_fn. The second CANCEL_REQUESTED
    // check (after step_fn returns) should prevent timer_long from being
    // submitted.
    iree_status_t cancel_status =
        iree_async_proactor_cancel(ctx->proactor, &ctx->sequence->base);
    iree_status_ignore(cancel_status);
    return iree_ok_status();
  };
  auto completion_fn = [](void* user_data, iree_async_operation_t* op,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<CancelDuringContext*>(user_data);
    CompletionTracker::Callback(ctx->tracker, op, status, flags);
  };

  InitSequence(&sequence, steps, 2, step_fn, completion_fn, &context);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(2000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
  // step_fn should have been called once (after timer_short completes, before
  // timer_long would be submitted).
  EXPECT_EQ(context.step_fn_call_count, 1);
}

// LINK path: cancel after the first step has completed. Tests the link
// trampoline's CANCEL_REQUESTED check that handles the cancel-step race
// (where the cancel-step call is a no-op because the step already completed).
TEST_P(SequenceOperationTest, LinkCancelAfterStepCompletion) {
  // Short timer completes quickly, long timer would take 10s.
  iree_async_timer_operation_t timer_short, timer_long;
  InitStepTimer(&timer_short, iree_make_duration_ms(1));
  InitStepTimer(&timer_long, iree_make_duration_ms(10000));

  CompletionTracker tracker;
  iree_async_operation_t* steps[] = {&timer_short.base, &timer_long.base};
  iree_async_sequence_operation_t sequence;
  InitSequence(&sequence, steps, 2, nullptr, CompletionTracker::Callback,
               &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sequence.base));

  // Wait briefly for timer_short to complete, then cancel.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(200);
  while (iree_time_now() < deadline) {
    iree_host_size_t count = 0;
    iree_status_t poll_status =
        iree_async_proactor_poll(proactor_, iree_make_timeout_ms(10), &count);
    iree_status_ignore(poll_status);
  }

  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &sequence.base));

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(2000));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(SequenceOperationTest);

}  // namespace iree::async::cts
