// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for cross-proactor MESSAGE operations.
//
// MESSAGE enables proactor-to-proactor communication:
//   - Fast path (MSG_RING): Kernel posts CQE directly to target ring (5.18+).
//   - Fallback path: MPSC queue + eventfd wake (all kernels).
//
// Both paths support LINK chains (e.g., RECV -> MESSAGE) and must be tested.

#include "iree/async/operations/message.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/scheduling.h"

namespace iree::async::cts {

// Two-proactor fixture for message tests.
// Creates a second proactor for cross-proactor communication.
class MessageTest : public SocketTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    // Create a second proactor for message targets.
    BackendInfo backend = GetParam();
    auto result = backend.factory();
    if (!result.ok()) {
      GTEST_SKIP() << "Failed to create second proactor: "
                   << result.status().ToString();
    }
    target_proactor_ = *result;
  }

  void TearDown() override {
    if (target_proactor_) {
      DrainTarget();
      iree_async_proactor_release(target_proactor_);
      target_proactor_ = nullptr;
    }
    CtsTestBase::TearDown();
  }

  // Polls the target proactor until the budget expires.
  // Continues polling even on DEADLINE_EXCEEDED to ensure all completions
  // have a chance to arrive (async systems don't guarantee timing).
  void PollTarget(iree_duration_t budget = iree_make_duration_ms(100)) {
    iree_time_t deadline_ns = iree_time_now() + budget;
    while (iree_time_now() < deadline_ns) {
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          target_proactor_, iree_make_timeout_ms(10), &completed);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        continue;  // Keep polling until deadline.
      }
      IREE_ASSERT_OK(status);
    }
  }

  void DrainTarget(iree_duration_t budget = iree_make_duration_ms(100)) {
    iree_time_t deadline_ns = iree_time_now() + budget;
    while (iree_time_now() < deadline_ns) {
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          target_proactor_, iree_immediate_timeout(), &completed);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        break;
      }
      iree_status_ignore(status);
    }
  }

  iree_async_proactor_t* target_proactor_ = nullptr;
};

// Basic message send from source to target.
TEST_P(MessageTest, BasicSend) {
  // Track received messages on target.
  struct MessageReceiver {
    std::atomic<int> count{0};
    uint64_t last_data = 0;
  };
  MessageReceiver receiver;

  auto callback = [](iree_async_proactor_t* proactor, uint64_t message_data,
                     void* user_data) {
    auto* receiver = static_cast<MessageReceiver*>(user_data);
    receiver->last_data = message_data;
    receiver->count.fetch_add(1, std::memory_order_release);
  };

  iree_async_proactor_set_message_callback(
      target_proactor_,
      iree_async_proactor_message_callback_t{callback, &receiver});

  // Submit message from source proactor.
  constexpr uint64_t kTestValue = 0xDEADBEEF12345678ULL;

  iree_async_message_operation_t message;
  memset(&message, 0, sizeof(message));
  message.base.type = IREE_ASYNC_OPERATION_TYPE_MESSAGE;
  message.target = target_proactor_;
  message.message_data = kTestValue;
  message.message_flags = IREE_ASYNC_MESSAGE_FLAG_NONE;

  CompletionTracker tracker;
  message.base.completion_fn = CompletionTracker::Callback;
  message.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &message.base));

  // Poll source to send the message.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(100));
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Poll target to receive the message.
  PollTarget(iree_make_duration_ms(100));
  EXPECT_EQ(receiver.count.load(), 1);
  EXPECT_EQ(receiver.last_data, kTestValue);
}

// Fire-and-forget message (skip source completion).
TEST_P(MessageTest, SkipSourceCompletion) {
  struct MessageReceiver {
    std::atomic<int> count{0};
  };
  MessageReceiver receiver;

  auto callback = [](iree_async_proactor_t* proactor, uint64_t message_data,
                     void* user_data) {
    auto* receiver = static_cast<MessageReceiver*>(user_data);
    receiver->count.fetch_add(1, std::memory_order_release);
  };

  iree_async_proactor_set_message_callback(
      target_proactor_,
      iree_async_proactor_message_callback_t{callback, &receiver});

  iree_async_message_operation_t message;
  memset(&message, 0, sizeof(message));
  message.base.type = IREE_ASYNC_OPERATION_TYPE_MESSAGE;
  message.target = target_proactor_;
  message.message_data = 42;
  message.message_flags = IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION;

  // No callback on source side.
  message.base.completion_fn = nullptr;
  message.base.user_data = nullptr;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &message.base));

  // With SKIP_SOURCE_COMPLETION, there's no source CQE. But we still need to
  // ensure the kernel processes the SQE. Poll the source with a brief timeout
  // to trigger any deferred processing.
  iree_status_t status =
      iree_async_proactor_poll(proactor_, iree_make_timeout_ms(50), nullptr);
  iree_status_ignore(status);  // DEADLINE_EXCEEDED is expected.

  // Poll target until the message callback fires.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(100);
  while (receiver.count.load() == 0 && iree_time_now() < deadline) {
    status = iree_async_proactor_poll(target_proactor_,
                                      iree_make_timeout_ms(10), nullptr);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }
  EXPECT_EQ(receiver.count.load(), 1);
}

// LINK chain: TIMER -> MESSAGE (tests linkability).
TEST_P(MessageTest, LinkChainTimerThenMessage) {
  struct MessageReceiver {
    std::atomic<int> count{0};
    uint64_t last_data = 0;
  };
  MessageReceiver receiver;

  auto callback = [](iree_async_proactor_t* proactor, uint64_t message_data,
                     void* user_data) {
    auto* receiver = static_cast<MessageReceiver*>(user_data);
    receiver->last_data = message_data;
    receiver->count.fetch_add(1, std::memory_order_release);
  };

  iree_async_proactor_set_message_callback(
      target_proactor_,
      iree_async_proactor_message_callback_t{callback, &receiver});

  // Set up linked operations: TIMER -> MESSAGE.
  iree_async_timer_operation_t timer;
  memset(&timer, 0, sizeof(timer));
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;  // Link to next op.
  timer.deadline_ns = iree_time_now() + iree_make_duration_ms(10);

  CompletionTracker timer_tracker;
  timer.base.completion_fn = CompletionTracker::Callback;
  timer.base.user_data = &timer_tracker;

  iree_async_message_operation_t message;
  memset(&message, 0, sizeof(message));
  message.base.type = IREE_ASYNC_OPERATION_TYPE_MESSAGE;
  message.target = target_proactor_;
  message.message_data = 0x1234ABCD;
  message.message_flags = IREE_ASYNC_MESSAGE_FLAG_NONE;

  CompletionTracker message_tracker;
  message.base.completion_fn = CompletionTracker::Callback;
  message.base.user_data = &message_tracker;

  // Submit as a batch (linked operations).
  iree_async_operation_t* ops[] = {&timer.base, &message.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Poll source - timer should fire, then message should be sent.
  PollUntil(/*min_completions=*/2, /*total_budget=*/iree_make_duration_ms(500));
  EXPECT_EQ(timer_tracker.call_count, 1);
  EXPECT_EQ(message_tracker.call_count, 1);
  IREE_EXPECT_OK(timer_tracker.ConsumeStatus());
  IREE_EXPECT_OK(message_tracker.ConsumeStatus());

  // Poll target to receive.
  PollTarget(iree_make_duration_ms(100));
  EXPECT_EQ(receiver.count.load(), 1);
  EXPECT_EQ(receiver.last_data, 0x1234ABCD);
}

// Self-message: proactor sends to itself.
TEST_P(MessageTest, SelfMessage) {
  struct MessageReceiver {
    std::atomic<int> count{0};
    uint64_t last_data = 0;
  };
  MessageReceiver receiver;

  auto callback = [](iree_async_proactor_t* proactor, uint64_t message_data,
                     void* user_data) {
    auto* receiver = static_cast<MessageReceiver*>(user_data);
    receiver->last_data = message_data;
    receiver->count.fetch_add(1, std::memory_order_release);
  };

  iree_async_proactor_set_message_callback(
      proactor_, iree_async_proactor_message_callback_t{callback, &receiver});

  constexpr uint64_t kTestValue = 0xAAAABBBBCCCCDDDDULL;

  iree_async_message_operation_t message;
  memset(&message, 0, sizeof(message));
  message.base.type = IREE_ASYNC_OPERATION_TYPE_MESSAGE;
  message.target = proactor_;  // Self!
  message.message_data = kTestValue;
  message.message_flags = IREE_ASYNC_MESSAGE_FLAG_NONE;

  CompletionTracker tracker;
  message.base.completion_fn = CompletionTracker::Callback;
  message.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &message.base));

  // Poll until the send completion arrives.
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(100));
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Poll until the message callback fires. In an async system, there's no
  // guarantee that all completions arrive in the same poll iteration.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(100);
  while (receiver.count.load() == 0 && iree_time_now() < deadline) {
    iree_status_t status =
        iree_async_proactor_poll(proactor_, iree_make_timeout_ms(10), nullptr);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }
  EXPECT_EQ(receiver.count.load(), 1);
  EXPECT_EQ(receiver.last_data, 0xAAAABBBBCCCCDDDDULL);
}

// Bidirectional: A->B->A round-trip.
TEST_P(MessageTest, Bidirectional) {
  // Track messages on both proactors.
  struct MessageReceiver {
    std::atomic<int> count{0};
    uint64_t last_data = 0;
  };
  MessageReceiver source_receiver;
  MessageReceiver target_receiver;

  auto source_callback = [](iree_async_proactor_t* proactor,
                            uint64_t message_data, void* user_data) {
    auto* receiver = static_cast<MessageReceiver*>(user_data);
    receiver->last_data = message_data;
    receiver->count.fetch_add(1, std::memory_order_release);
  };

  auto target_callback = [](iree_async_proactor_t* proactor,
                            uint64_t message_data, void* user_data) {
    auto* receiver = static_cast<MessageReceiver*>(user_data);
    receiver->last_data = message_data;
    receiver->count.fetch_add(1, std::memory_order_release);
  };

  iree_async_proactor_set_message_callback(
      proactor_, iree_async_proactor_message_callback_t{source_callback,
                                                        &source_receiver});
  iree_async_proactor_set_message_callback(
      target_proactor_, iree_async_proactor_message_callback_t{
                            target_callback, &target_receiver});

  // Send A -> B.
  iree_async_message_operation_t message_to_b;
  memset(&message_to_b, 0, sizeof(message_to_b));
  message_to_b.base.type = IREE_ASYNC_OPERATION_TYPE_MESSAGE;
  message_to_b.target = target_proactor_;
  message_to_b.message_data = 0x1111;
  message_to_b.message_flags = IREE_ASYNC_MESSAGE_FLAG_NONE;

  CompletionTracker tracker_to_b;
  message_to_b.base.completion_fn = CompletionTracker::Callback;
  message_to_b.base.user_data = &tracker_to_b;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &message_to_b.base));
  PollUntil(1, iree_make_duration_ms(100));
  PollTarget(iree_make_duration_ms(100));
  EXPECT_EQ(target_receiver.count.load(), 1);
  EXPECT_EQ(target_receiver.last_data, 0x1111);

  // Send B -> A.
  iree_async_message_operation_t message_to_a;
  memset(&message_to_a, 0, sizeof(message_to_a));
  message_to_a.base.type = IREE_ASYNC_OPERATION_TYPE_MESSAGE;
  message_to_a.target = proactor_;
  message_to_a.message_data = 0x2222;
  message_to_a.message_flags = IREE_ASYNC_MESSAGE_FLAG_NONE;

  CompletionTracker tracker_to_a;
  message_to_a.base.completion_fn = CompletionTracker::Callback;
  message_to_a.base.user_data = &tracker_to_a;

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(target_proactor_, &message_to_a.base));

  // Poll target to get the B→A source completion.
  PollTarget(iree_make_duration_ms(100));

  // Poll source until the message callback fires. Message delivery happens via
  // callback, not via a user completion (the receiver didn't submit anything).
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(100);
  while (source_receiver.count.load() == 0 && iree_time_now() < deadline) {
    iree_status_t status =
        iree_async_proactor_poll(proactor_, iree_make_timeout_ms(10), nullptr);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  EXPECT_EQ(source_receiver.count.load(), 1);
  EXPECT_EQ(source_receiver.last_data, 0x2222);
}

// Multiple in-flight messages (tests unbounded queue).
TEST_P(MessageTest, MultipleInFlight) {
  constexpr int kMessageCount = 100;

  struct MessageReceiver {
    std::atomic<int> count{0};
    std::atomic<uint64_t> sum{0};
  };
  MessageReceiver receiver;

  auto callback = [](iree_async_proactor_t* proactor, uint64_t message_data,
                     void* user_data) {
    auto* receiver = static_cast<MessageReceiver*>(user_data);
    receiver->sum.fetch_add(message_data, std::memory_order_relaxed);
    receiver->count.fetch_add(1, std::memory_order_release);
  };

  iree_async_proactor_set_message_callback(
      target_proactor_,
      iree_async_proactor_message_callback_t{callback, &receiver});

  // Submit all messages.
  std::vector<iree_async_message_operation_t> messages(kMessageCount);
  std::vector<CompletionTracker> trackers(kMessageCount);
  std::vector<iree_async_operation_t*> ops(kMessageCount);

  uint64_t expected_sum = 0;
  for (int i = 0; i < kMessageCount; ++i) {
    memset(&messages[i], 0, sizeof(messages[i]));
    messages[i].base.type = IREE_ASYNC_OPERATION_TYPE_MESSAGE;
    messages[i].target = target_proactor_;
    messages[i].message_data = (uint64_t)i;
    messages[i].message_flags = IREE_ASYNC_MESSAGE_FLAG_NONE;
    messages[i].base.completion_fn = CompletionTracker::Callback;
    messages[i].base.user_data = &trackers[i];
    ops[i] = &messages[i].base;
    expected_sum += i;
  }

  iree_async_operation_list_t list = {ops.data(),
                                      (iree_host_size_t)kMessageCount};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Poll source to send all messages.
  PollUntil(kMessageCount, iree_make_duration_ms(5000));

  // Poll target to receive all messages.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(5000);
  while (receiver.count.load() < kMessageCount && iree_time_now() < deadline) {
    PollTarget(iree_make_duration_ms(100));
  }

  EXPECT_EQ(receiver.count.load(), kMessageCount);
  EXPECT_EQ(receiver.sum.load(), expected_sum);
}

// Tests the simple send_message API (fire-and-forget from any thread).
TEST_P(MessageTest, SendMessage) {
  struct MessageReceiver {
    std::atomic<int> count{0};
    uint64_t last_data = 0;
  };
  MessageReceiver receiver;

  auto callback = [](iree_async_proactor_t* proactor, uint64_t message_data,
                     void* user_data) {
    auto* receiver = static_cast<MessageReceiver*>(user_data);
    receiver->last_data = message_data;
    receiver->count.fetch_add(1, std::memory_order_release);
  };

  iree_async_proactor_set_message_callback(
      target_proactor_,
      iree_async_proactor_message_callback_t{callback, &receiver});

  constexpr uint64_t kTestValue = 0x123456789ABCDEF0ULL;

  // Send using the simple API (no operation struct, no completion callback).
  IREE_ASSERT_OK(
      iree_async_proactor_send_message(target_proactor_, kTestValue));

  // Poll target until message arrives.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(100);
  while (receiver.count.load() == 0 && iree_time_now() < deadline) {
    iree_status_t status = iree_async_proactor_poll(
        target_proactor_, iree_make_timeout_ms(10), nullptr);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  EXPECT_EQ(receiver.count.load(), 1);
  EXPECT_EQ(receiver.last_data, kTestValue);
}

// Thread-safety: multiple threads sending to the same proactor concurrently.
TEST_P(MessageTest, SendMessageFromMultipleThreads) {
  constexpr int kThreadCount = 4;
  constexpr int kMessagesPerThread = 50;

  struct MessageReceiver {
    std::atomic<int> count{0};
    std::atomic<uint64_t> sum{0};
  };
  MessageReceiver receiver;

  auto callback = [](iree_async_proactor_t* proactor, uint64_t message_data,
                     void* user_data) {
    auto* receiver = static_cast<MessageReceiver*>(user_data);
    receiver->sum.fetch_add(message_data, std::memory_order_relaxed);
    receiver->count.fetch_add(1, std::memory_order_release);
  };

  iree_async_proactor_set_message_callback(
      target_proactor_,
      iree_async_proactor_message_callback_t{callback, &receiver});

  // Launch threads that all send to the same target proactor.
  std::vector<std::thread> threads;
  std::atomic<bool> start_flag{false};

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([this, t, &start_flag, kMessagesPerThread]() {
      // Wait for all threads to be ready.
      while (!start_flag.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      // Send messages.
      for (int i = 0; i < kMessagesPerThread; ++i) {
        uint64_t message_data = (uint64_t)(t * kMessagesPerThread + i);
        iree_status_t status =
            iree_async_proactor_send_message(target_proactor_, message_data);
        IREE_ASSERT_OK(status);
      }
    });
  }

  // Start all threads simultaneously.
  start_flag.store(true, std::memory_order_release);

  // Poll target while threads are sending.
  constexpr int kTotalMessages = kThreadCount * kMessagesPerThread;
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(5000);
  while (receiver.count.load() < kTotalMessages && iree_time_now() < deadline) {
    iree_status_t status = iree_async_proactor_poll(
        target_proactor_, iree_make_timeout_ms(10), nullptr);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  // Wait for all threads to complete.
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify all messages were received.
  EXPECT_EQ(receiver.count.load(), kTotalMessages);

  // Verify the sum (0 + 1 + 2 + ... + (kTotalMessages-1)).
  uint64_t expected_sum = (uint64_t)(kTotalMessages - 1) * kTotalMessages / 2;
  EXPECT_EQ(receiver.sum.load(), expected_sum);
}

CTS_REGISTER_TEST_SUITE(MessageTest);

}  // namespace iree::async::cts
