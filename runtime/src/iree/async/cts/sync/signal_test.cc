// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for signal handling via proactor.
//
// Signal handling allows proactor-integrated delivery of process signals
// (SIGINT, SIGTERM, etc.) through the normal poll() callback path. This
// avoids the complexity of signal handlers and ensures signals are processed
// in a thread-safe manner.
//
// NOTE: Signal handling is process-global. Only one proactor per process may
// own signals. Tests must account for this: once a proactor claims ownership,
// no other proactor can subscribe for the rest of the process.

#include <atomic>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE)
#include <signal.h>
#endif  // IREE_PLATFORM_*

namespace iree::async::cts {

// Tracks signal callbacks for test assertions.
struct SignalTracker {
  std::atomic<int> call_count{0};
  std::atomic<iree_async_signal_t> last_signal{IREE_ASYNC_SIGNAL_NONE};

  static void Callback(void* user_data, iree_async_signal_t signal) {
    auto* tracker = static_cast<SignalTracker*>(user_data);
    tracker->call_count.fetch_add(1, std::memory_order_relaxed);
    tracker->last_signal.store(signal, std::memory_order_relaxed);
  }
};

class SignalTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (!proactor_) return;  // Base class skipped (backend unavailable).

    // Skip if signal handling isn't available on this backend.
    if (!iree_async_proactor_supports_signals(proactor_)) {
      GTEST_SKIP() << "Backend does not support signal handling";
    }
  }

  // TearDown uses base class implementation. Signal ownership is released
  // automatically by the proactor destroy path.
};

//===----------------------------------------------------------------------===//
// Cross-platform tests
//===----------------------------------------------------------------------===//

// Basic subscribe/unsubscribe lifecycle. Uses INTERRUPT which is available on
// all platforms (POSIX SIGINT, Windows CTRL_C_EVENT).
TEST_P(SignalTest, SubscribeUnsubscribe) {
  SignalTracker tracker;
  iree_async_signal_subscription_t* sub = nullptr;

  IREE_ASSERT_OK(iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_INTERRUPT,
      {SignalTracker::Callback, &tracker}, &sub));
  ASSERT_NE(sub, nullptr);

  // Unsubscribe should succeed.
  iree_async_proactor_unsubscribe_signal(proactor_, sub);

  // Unsubscribe with NULL should be safe no-op.
  iree_async_proactor_unsubscribe_signal(proactor_, nullptr);
}

// Invalid signal type returns INVALID_ARGUMENT.
TEST_P(SignalTest, InvalidSignal) {
  iree_async_signal_subscription_t* sub = nullptr;
  iree_status_t status = iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_NONE, {SignalTracker::Callback, nullptr},
      &sub);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(sub, nullptr);

  status = iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_COUNT, {SignalTracker::Callback, nullptr},
      &sub);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(sub, nullptr);
}

// Signal name utility returns expected strings.
TEST_P(SignalTest, SignalName) {
  EXPECT_EQ(iree_string_view_compare(
                iree_async_signal_name(IREE_ASYNC_SIGNAL_INTERRUPT),
                iree_make_cstring_view("INTERRUPT")),
            0);
  EXPECT_EQ(iree_string_view_compare(
                iree_async_signal_name(IREE_ASYNC_SIGNAL_TERMINATE),
                iree_make_cstring_view("TERMINATE")),
            0);
  EXPECT_EQ(
      iree_string_view_compare(iree_async_signal_name(IREE_ASYNC_SIGNAL_USER1),
                               iree_make_cstring_view("USER1")),
      0);
}

//===----------------------------------------------------------------------===//
// Windows-specific tests
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)

// Windows only supports INTERRUPT and TERMINATE. Subscribing to POSIX-only
// signals (HANGUP, QUIT, USER1, USER2) returns INVALID_ARGUMENT.
TEST_P(SignalTest, WindowsUnsupportedSignals) {
  iree_async_signal_subscription_t* sub = nullptr;

  iree_status_t status = iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_HANGUP, {SignalTracker::Callback, nullptr},
      &sub);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(sub, nullptr);

  status = iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_QUIT, {SignalTracker::Callback, nullptr},
      &sub);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(sub, nullptr);

  status = iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_USER1, {SignalTracker::Callback, nullptr},
      &sub);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(sub, nullptr);

  status = iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_USER2, {SignalTracker::Callback, nullptr},
      &sub);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(sub, nullptr);
}

#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// POSIX-specific signal delivery tests
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE)

// Signal delivery via raise() and poll().
TEST_P(SignalTest, SignalDelivery) {
  SignalTracker tracker;
  iree_async_signal_subscription_t* sub = nullptr;

  IREE_ASSERT_OK(iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_USER1, {SignalTracker::Callback, &tracker},
      &sub));

  // Send SIGUSR1 to ourselves.
  raise(SIGUSR1);

  // Poll to deliver the signal.
  PollOnce();

  // Callback should have fired.
  EXPECT_GE(tracker.call_count.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(tracker.last_signal.load(std::memory_order_relaxed),
            IREE_ASYNC_SIGNAL_USER1);

  iree_async_proactor_unsubscribe_signal(proactor_, sub);
}

// Multiple subscribers to the same signal.
TEST_P(SignalTest, MultipleSubscribers) {
  SignalTracker tracker1, tracker2;
  iree_async_signal_subscription_t* sub1 = nullptr;
  iree_async_signal_subscription_t* sub2 = nullptr;

  IREE_ASSERT_OK(iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_USER2, {SignalTracker::Callback, &tracker1},
      &sub1));
  IREE_ASSERT_OK(iree_async_proactor_subscribe_signal(
      proactor_, IREE_ASYNC_SIGNAL_USER2, {SignalTracker::Callback, &tracker2},
      &sub2));

  // Send signal.
  raise(SIGUSR2);

  // Poll to deliver the signal.
  PollOnce();

  // Both callbacks should have fired.
  EXPECT_GE(tracker1.call_count.load(std::memory_order_relaxed), 1);
  EXPECT_GE(tracker2.call_count.load(std::memory_order_relaxed), 1);

  iree_async_proactor_unsubscribe_signal(proactor_, sub1);
  iree_async_proactor_unsubscribe_signal(proactor_, sub2);
}

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_APPLE

// Register the test suite. Signal tests only run on backends that support
// signals (which is determined at runtime via the probe in SetUp).
CTS_REGISTER_TEST_SUITE(SignalTest);

}  // namespace iree::async::cts
