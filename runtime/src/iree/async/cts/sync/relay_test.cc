// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async relay operations using notification sources and sinks.
//
// These tests exercise the relay mechanism using only notification primitives,
// making them fully portable across all proactor backends (POSIX, io_uring,
// IOCP). Primitive-source/sink relay tests are in relay_posix_test.cc (Linux
// and macOS) and relay_windows_test.cc (Windows).
//
// Test categories:
//   - One-shot notification-to-notification relay
//   - Persistent relay with multiple fires
//   - Futex-optimized relay path (gated on FUTEX_OPERATIONS capability)
//   - Lifecycle: unregister while pending, multiple concurrent relays
//
// Verification model: relay side effects are verified via epoch queries on the
// sink notification. The relay CQE is processed synchronously during
// DrainPending; after DrainPending returns, the sink epoch reflects whether the
// relay fired. No waiter threads or timing-based synchronization is needed.

#include "iree/async/relay.h"

#include <atomic>
#include <thread>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/notification.h"

namespace iree::async::cts {

class RelayTest : public CtsTestBase<> {};

// One-shot NOTIFICATION source triggers SIGNAL_NOTIFICATION sink.
TEST_P(RelayTest, NotificationToNotification) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  iree_async_notification_t* sink_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink_notification));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  uint32_t epoch_before =
      iree_async_notification_query_epoch(sink_notification);

  iree_async_notification_signal(source_notification, 1);
  DrainPending();

  uint32_t epoch_after = iree_async_notification_query_epoch(sink_notification);
  EXPECT_NE(epoch_before, epoch_after) << "Relay failed to fire sink";

  iree_async_notification_release(source_notification);
  iree_async_notification_release(sink_notification);
}

// Persistent notification relay fires multiple times.
TEST_P(RelayTest, PersistentNotificationRelay) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  iree_async_notification_t* sink_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink_notification));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  // Fire multiple times to exercise the persistent re-arm path.
  // Each iteration: the relay re-arms during CQE processing (submitting a new
  // FUTEX_WAIT/POLL_ADD SQE), so the next signal produces a fresh CQE.
  for (int i = 0; i < 3; ++i) {
    uint32_t epoch_before =
        iree_async_notification_query_epoch(sink_notification);

    iree_async_notification_signal(source_notification, 1);
    DrainPending();

    uint32_t epoch_after =
        iree_async_notification_query_epoch(sink_notification);
    EXPECT_NE(epoch_before, epoch_after)
        << "Relay failed to fire on iteration " << i;
  }

  iree_async_proactor_unregister_relay(proactor_, relay);
  DrainPending();

  iree_async_notification_release(source_notification);
  iree_async_notification_release(sink_notification);
}

// Unregister notification relay before any signal fires.
TEST_P(RelayTest, UnregisterNotificationWhilePending) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  iree_async_notification_t* sink_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink_notification));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  // Unregister cancels the relay's pending SQE and processes the cancel CQE.
  iree_async_proactor_unregister_relay(proactor_, relay);
  DrainPending();

  // Signal after unregister — no relay SQE is watching, so no CQE is produced.
  uint32_t sink_epoch_before =
      iree_async_notification_query_epoch(sink_notification);
  iree_async_notification_signal(source_notification, 1);
  DrainPending();

  uint32_t sink_epoch_after =
      iree_async_notification_query_epoch(sink_notification);
  EXPECT_EQ(sink_epoch_before, sink_epoch_after)
      << "Relay should not fire after unregister";

  iree_async_notification_release(source_notification);
  iree_async_notification_release(sink_notification);
}

// Multiple relays from the same source notification to different sinks.
TEST_P(RelayTest, MultipleNotificationRelays) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  constexpr int kNumRelays = 3;
  iree_async_notification_t* sink_notifications[kNumRelays] = {};
  iree_async_relay_t* relays[kNumRelays] = {};

  for (int i = 0; i < kNumRelays; ++i) {
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink_notifications[i]));
    IREE_ASSERT_OK(iree_async_proactor_register_relay(
        proactor_,
        iree_async_relay_source_from_notification(source_notification),
        iree_async_relay_sink_signal_notification(sink_notifications[i], 1),
        IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
        &relays[i]));
    ASSERT_NE(relays[i], nullptr);
  }

  uint32_t epochs_before[kNumRelays];
  for (int i = 0; i < kNumRelays; ++i) {
    epochs_before[i] =
        iree_async_notification_query_epoch(sink_notifications[i]);
  }

  iree_async_notification_signal(source_notification, 1);
  DrainPending();

  for (int i = 0; i < kNumRelays; ++i) {
    uint32_t epoch_after =
        iree_async_notification_query_epoch(sink_notifications[i]);
    EXPECT_NE(epoch_after, epochs_before[i])
        << "Relay " << i << " failed to fire";
  }

  // One-shot relays auto-clean up after firing.
  for (int i = 0; i < kNumRelays; ++i) {
    iree_async_notification_release(sink_notifications[i]);
  }
  iree_async_notification_release(source_notification);
}

// Notification-to-notification relay using futex mode (when available).
// This exercises the futex re-arm logic in the relay CQE handler.
TEST_P(RelayTest, NotificationToNotificationFutexMode) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS)) {
    GTEST_SKIP() << "backend lacks futex operations capability";
  }

  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  iree_async_notification_t* sink_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink_notification));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  for (int i = 0; i < 3; ++i) {
    uint32_t epoch_before =
        iree_async_notification_query_epoch(sink_notification);

    iree_async_notification_signal(source_notification, 1);
    DrainPending();

    uint32_t epoch_after =
        iree_async_notification_query_epoch(sink_notification);
    EXPECT_NE(epoch_before, epoch_after)
        << "Futex relay failed on iteration " << i;
  }

  iree_async_proactor_unregister_relay(proactor_, relay);
  DrainPending();

  iree_async_notification_release(source_notification);
  iree_async_notification_release(sink_notification);
}

// Relay signal wakes a cross-thread waiter via epoch observation.
// Exercises cross-thread atomic acquire/release on the sink notification's
// epoch, providing TSAN coverage of the relay signal → waiter wake path.
// Uses iree_notification_t as a gate to avoid the three-actor race inherent in
// calling iree_async_notification_wait directly (where the relay can fire
// before the waiter captures its baseline epoch).
TEST_P(RelayTest, NotificationRelayWakesCrossThread) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  iree_async_notification_t* sink_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink_notification));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  // Capture baseline epoch BEFORE starting the waiter thread.
  iree_notification_t gate;
  iree_notification_initialize(&gate);
  EpochWaitContext wait_context = {
      sink_notification,
      iree_async_notification_query_epoch(sink_notification)};

  std::atomic<bool> waiter_woken{false};
  std::thread waiter([&]() {
    bool result = iree_notification_await(&gate, epoch_advanced, &wait_context,
                                          iree_make_timeout_ms(5000));
    waiter_woken.store(result, std::memory_order_release);
  });

  // Signal source and drain — relay fires, bumping sink epoch.
  iree_async_notification_signal(source_notification, 1);
  DrainPending();

  // Wake the waiter. If the waiter already saw the epoch change (fast path),
  // this post is harmless. If the waiter is blocked in commit_wait, this
  // wakes it and the predicate confirms the epoch change.
  iree_notification_post(&gate, IREE_ALL_WAITERS);

  waiter.join();
  EXPECT_TRUE(waiter_woken.load(std::memory_order_acquire))
      << "Waiter should have observed relay-driven epoch change";

  iree_notification_deinitialize(&gate);
  iree_async_notification_release(source_notification);
  iree_async_notification_release(sink_notification);
}

CTS_REGISTER_TEST_SUITE(RelayTest);

}  // namespace iree::async::cts
