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

  // Register the relay first before starting the waiter.
  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_woken{false};

  // Background thread waits on the sink notification.
  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);
    bool result = iree_async_notification_wait(sink_notification,
                                               iree_make_timeout_ms(5000));
    waiter_woken.store(result, std::memory_order_release);
  });

  // Wait for waiter to start.
  while (!waiter_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }
  // Give waiter time to enter wait.
  iree_wait_until(iree_time_now() + iree_make_duration_ms(20));

  // Signal the source notification.
  iree_async_notification_signal(source_notification, 1);

  // Poll until relay fires.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
  while (!waiter_woken.load(std::memory_order_acquire) &&
         iree_time_now() < deadline) {
    PollOnce();
  }

  waiter.join();

  EXPECT_TRUE(waiter_woken.load(std::memory_order_acquire));

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
  for (int i = 0; i < 3; ++i) {
    std::atomic<bool> waiter_started{false};
    std::atomic<bool> waiter_woken{false};

    // Background thread waits on the sink notification.
    std::thread waiter([&]() {
      waiter_started.store(true, std::memory_order_release);
      bool result = iree_async_notification_wait(sink_notification,
                                                 iree_make_timeout_ms(5000));
      waiter_woken.store(result, std::memory_order_release);
    });

    // Wait for waiter to start.
    while (!waiter_started.load(std::memory_order_acquire)) {
      iree_thread_yield();
    }
    iree_wait_until(iree_time_now() + iree_make_duration_ms(20));

    // Signal the source notification.
    iree_async_notification_signal(source_notification, 1);

    // Poll until relay fires.
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
    while (!waiter_woken.load(std::memory_order_acquire) &&
           iree_time_now() < deadline) {
      PollOnce();
    }

    waiter.join();

    EXPECT_TRUE(waiter_woken.load(std::memory_order_acquire))
        << "Relay failed on iteration " << i;
  }

  // Explicitly unregister persistent relay.
  iree_async_proactor_unregister_relay(proactor_, relay);

  // Drain pending operations.
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

  // Unregister before any signal.
  iree_async_proactor_unregister_relay(proactor_, relay);
  DrainPending();

  // Signal after unregister — sink should NOT fire.
  uint32_t sink_epoch_before =
      iree_async_notification_query_epoch(sink_notification);
  iree_async_notification_signal(source_notification, 1);

  // Poll a few cycles to give any stale relay a chance to fire.
  for (int i = 0; i < 5; ++i) {
    PollOnce();
  }

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

  // Capture epochs before signaling.
  uint32_t epochs_before[kNumRelays];
  for (int i = 0; i < kNumRelays; ++i) {
    epochs_before[i] =
        iree_async_notification_query_epoch(sink_notifications[i]);
  }

  // Signal the source once.
  iree_async_notification_signal(source_notification, 1);

  // Poll until all sinks see epoch advancement.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
  bool all_fired = false;
  while (!all_fired && iree_time_now() < deadline) {
    PollOnce();
    all_fired = true;
    for (int i = 0; i < kNumRelays; ++i) {
      if (iree_async_notification_query_epoch(sink_notifications[i]) ==
          epochs_before[i]) {
        all_fired = false;
        break;
      }
    }
  }

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
  // Skip if futex operations are not available.
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

  // Register a persistent relay so we test the futex re-arm path.
  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  // Fire multiple times to exercise the futex re-arm logic.
  for (int i = 0; i < 3; ++i) {
    std::atomic<bool> waiter_started{false};
    std::atomic<bool> waiter_woken{false};

    // Background thread waits on the sink notification.
    std::thread waiter([&]() {
      waiter_started.store(true, std::memory_order_release);
      bool result = iree_async_notification_wait(sink_notification,
                                                 iree_make_timeout_ms(5000));
      waiter_woken.store(result, std::memory_order_release);
    });

    // Wait for waiter to start.
    while (!waiter_started.load(std::memory_order_acquire)) {
      iree_thread_yield();
    }
    // Give waiter time to enter wait.
    iree_wait_until(iree_time_now() + iree_make_duration_ms(20));

    // Signal the source notification.
    iree_async_notification_signal(source_notification, 1);

    // Poll until relay fires.
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
    while (!waiter_woken.load(std::memory_order_acquire) &&
           iree_time_now() < deadline) {
      PollOnce();
    }

    waiter.join();

    EXPECT_TRUE(waiter_woken.load(std::memory_order_acquire))
        << "Futex relay failed on iteration " << i;
  }

  // Explicitly unregister persistent relay.
  iree_async_proactor_unregister_relay(proactor_, relay);

  // Drain pending operations.
  DrainPending();

  iree_async_notification_release(source_notification);
  iree_async_notification_release(sink_notification);
}

CTS_REGISTER_TEST_SUITE(RelayTest);

}  // namespace iree::async::cts
