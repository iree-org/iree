// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async relay operations.
//
// Relays connect event sources (fd becoming ready, notification signaled) to
// event sinks (signal another fd, signal a notification) with minimal userspace
// involvement when kernel LINK chains can be used.
//
// Test categories:
//   - Source/sink type combinations (4 permutations)
//   - Persistence (one-shot vs persistent re-arming)
//   - Flag behavior (OWN_SOURCE_PRIMITIVE, ERROR_SENSITIVE)
//   - Lifecycle (unregister while pending, multiple active relays)

#include "iree/async/relay.h"

#include <sys/eventfd.h>
#include <unistd.h>

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/notification.h"

namespace iree::async::cts {

class RelayTest : public CtsTestBase<> {
 protected:
  // Creates an eventfd for use in tests.
  int CreateEventFd() {
    int fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    EXPECT_GE(fd, 0);
    return fd;
  }

  // Writes to an eventfd to signal it.
  void SignalEventFd(int fd, uint64_t value = 1) {
    ssize_t written = write(fd, &value, sizeof(value));
    EXPECT_EQ(written, static_cast<ssize_t>(sizeof(value)));
  }

  // Reads from an eventfd to drain it. Returns false if nothing to read.
  bool DrainEventFd(int fd, uint64_t* out_value = nullptr) {
    uint64_t value = 0;
    ssize_t bytes_read = read(fd, &value, sizeof(value));
    if (bytes_read == sizeof(value)) {
      if (out_value) *out_value = value;
      return true;
    }
    return false;
  }

  // Waits for an eventfd to become readable with a timeout.
  bool WaitEventFdReadable(int fd, iree_duration_t timeout_ns) {
    iree_time_t deadline = iree_time_now() + timeout_ns;
    while (iree_time_now() < deadline) {
      uint64_t value;
      if (DrainEventFd(fd, &value)) {
        return true;
      }
      // Yield to allow relay processing.
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(10), nullptr);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
      } else if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        return false;  // Error during poll.
      }
    }
    return false;
  }
};

// Basic test: PRIMITIVE source triggers SIGNAL_PRIMITIVE sink.
// eventfd → eventfd relay.
TEST_P(RelayTest, PrimitiveToPrimitive) {
  int source_fd = CreateEventFd();
  int sink_fd = CreateEventFd();
  ASSERT_GE(source_fd, 0);
  ASSERT_GE(sink_fd, 0);

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 1),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  // Signal the source eventfd.
  SignalEventFd(source_fd);

  // Poll until the relay fires.
  bool sink_signaled =
      WaitEventFdReadable(sink_fd, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled);

  // One-shot relay auto-cleans up, so no explicit unregister needed.
  // Clean up file descriptors.
  close(source_fd);
  close(sink_fd);
}

// PRIMITIVE source triggers SIGNAL_NOTIFICATION sink.
// eventfd → notification relay.
TEST_P(RelayTest, PrimitiveToNotification) {
  int source_fd = CreateEventFd();
  ASSERT_GE(source_fd, 0);

  iree_async_notification_t* sink_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink_notification));

  // Register the relay first before starting the waiter.
  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source_fd)),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_woken{false};

  // Background thread waits on the notification.
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

  // Signal the source eventfd.
  SignalEventFd(source_fd);

  // Poll until relay fires (need to process relay CQE).
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
  while (!waiter_woken.load(std::memory_order_acquire) &&
         iree_time_now() < deadline) {
    PollOnce();
  }

  waiter.join();

  EXPECT_TRUE(waiter_woken.load(std::memory_order_acquire));

  close(source_fd);
  iree_async_notification_release(sink_notification);
}

// NOTIFICATION source triggers SIGNAL_PRIMITIVE sink.
// notification → eventfd relay.
TEST_P(RelayTest, NotificationToPrimitive) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  int sink_fd = CreateEventFd();
  ASSERT_GE(sink_fd, 0);

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 42),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  // Signal the source notification.
  iree_async_notification_signal(source_notification, 1);

  // Poll and wait for sink to be signaled.
  bool sink_signaled =
      WaitEventFdReadable(sink_fd, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled);

  close(sink_fd);
  iree_async_notification_release(source_notification);
}

// NOTIFICATION source triggers SIGNAL_NOTIFICATION sink.
// notification → notification relay.
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

// Persistent relay fires multiple times.
TEST_P(RelayTest, PersistentMultipleTransfers) {
  int source_fd = CreateEventFd();
  int sink_fd = CreateEventFd();
  ASSERT_GE(source_fd, 0);
  ASSERT_GE(sink_fd, 0);

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  // Fire multiple times.
  for (int i = 0; i < 3; ++i) {
    // Drain the sink first to reset.
    DrainEventFd(sink_fd);

    // Signal the source.
    SignalEventFd(source_fd);

    // Wait for sink to be signaled.
    bool sink_signaled =
        WaitEventFdReadable(sink_fd, iree_make_duration_ms(1000));
    EXPECT_TRUE(sink_signaled) << "Relay failed to fire on iteration " << i;
  }

  // Explicitly unregister persistent relay.
  iree_async_proactor_unregister_relay(proactor_, relay);

  // Drain any pending operations.
  DrainPending();

  close(source_fd);
  close(sink_fd);
}

// One-shot relay auto-cleans up after single fire.
TEST_P(RelayTest, OneShotAutoCleanup) {
  int source_fd = CreateEventFd();
  int sink_fd = CreateEventFd();
  ASSERT_GE(source_fd, 0);
  ASSERT_GE(sink_fd, 0);

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 1),
      IREE_ASYNC_RELAY_FLAG_NONE,  // No PERSISTENT flag.
      iree_async_relay_error_callback_none(), &relay));
  ASSERT_NE(relay, nullptr);

  // Signal the source.
  SignalEventFd(source_fd);

  // Wait for first fire.
  bool sink_signaled =
      WaitEventFdReadable(sink_fd, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled);

  // Signal source again - relay should not fire again (auto-cleaned up).
  SignalEventFd(source_fd);

  // Drain any in-flight work, then verify sink was NOT signaled again.
  DrainEventFd(sink_fd);  // Clear from first fire if any remnant.
  DrainPending();

  uint64_t value;
  bool got_second_signal = DrainEventFd(sink_fd, &value);
  EXPECT_FALSE(got_second_signal)
      << "One-shot relay should not fire after cleanup";

  close(source_fd);
  close(sink_fd);
}

// OWN_SOURCE_PRIMITIVE flag closes fd on cleanup.
TEST_P(RelayTest, OwnSourcePrimitive) {
  int source_fd = CreateEventFd();
  int sink_fd = CreateEventFd();
  ASSERT_GE(source_fd, 0);
  ASSERT_GE(sink_fd, 0);

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 1),
      IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE,
      iree_async_relay_error_callback_none(), &relay));
  ASSERT_NE(relay, nullptr);

  // Signal to trigger and cleanup the one-shot relay.
  SignalEventFd(source_fd);
  WaitEventFdReadable(sink_fd, iree_make_duration_ms(1000));

  // The source_fd should now be closed by the relay cleanup.
  // Verify by trying to write - should fail with EBADF.
  uint64_t value = 1;
  ssize_t result = write(source_fd, &value, sizeof(value));
  EXPECT_EQ(result, -1);
  EXPECT_EQ(errno, EBADF);

  // sink_fd is still ours to close.
  close(sink_fd);
}

// Multiple active relays work correctly.
TEST_P(RelayTest, MultipleActiveRelays) {
  constexpr int kNumRelays = 3;
  int source_fds[kNumRelays];
  int sink_fds[kNumRelays];
  iree_async_relay_t* relays[kNumRelays];

  // Create multiple relays.
  for (int i = 0; i < kNumRelays; ++i) {
    source_fds[i] = CreateEventFd();
    sink_fds[i] = CreateEventFd();
    ASSERT_GE(source_fds[i], 0);
    ASSERT_GE(sink_fds[i], 0);

    IREE_ASSERT_OK(iree_async_proactor_register_relay(
        proactor_,
        iree_async_relay_source_from_primitive(
            iree_async_primitive_from_fd(source_fds[i])),
        iree_async_relay_sink_signal_primitive(
            iree_async_primitive_from_fd(sink_fds[i]), 1),
        IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
        &relays[i]));
    ASSERT_NE(relays[i], nullptr);
  }

  // Signal all sources.
  for (int i = 0; i < kNumRelays; ++i) {
    SignalEventFd(source_fds[i]);
  }

  // All sinks should be signaled.
  for (int i = 0; i < kNumRelays; ++i) {
    bool sink_signaled =
        WaitEventFdReadable(sink_fds[i], iree_make_duration_ms(1000));
    EXPECT_TRUE(sink_signaled) << "Relay " << i << " failed to fire";
  }

  // Clean up.
  for (int i = 0; i < kNumRelays; ++i) {
    close(source_fds[i]);
    close(sink_fds[i]);
  }
}

// ERROR_SENSITIVE flag suppresses sink firing on source POLLERR/POLLHUP.
// Uses a pipe where closing the write end causes POLLHUP on the read end.
TEST_P(RelayTest, ErrorSensitiveSuppressesSinkOnPollHup) {
  int pipe_fds[2];
  ASSERT_EQ(pipe(pipe_fds), 0);
  int read_fd = pipe_fds[0];
  int write_fd = pipe_fds[1];

  int sink_fd = CreateEventFd();
  ASSERT_GE(sink_fd, 0);

  // Create a relay with ERROR_SENSITIVE flag.
  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(read_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 1),
      IREE_ASYNC_RELAY_FLAG_ERROR_SENSITIVE,
      iree_async_relay_error_callback_none(), &relay));
  ASSERT_NE(relay, nullptr);

  // Close the write end to cause POLLHUP on the read end.
  close(write_fd);

  // Poll to process the POLLHUP event.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(500);
  while (iree_time_now() < deadline) {
    iree_status_t status =
        iree_async_proactor_poll(proactor_, iree_make_timeout_ms(50), nullptr);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }
  }

  // The sink should NOT have been signaled because of ERROR_SENSITIVE.
  uint64_t value;
  bool got_signal = DrainEventFd(sink_fd, &value);
  EXPECT_FALSE(got_signal) << "ERROR_SENSITIVE should suppress sink on POLLHUP";

  close(read_fd);
  close(sink_fd);
}

// ERROR_SENSITIVE flag still fires sink on normal POLLIN.
TEST_P(RelayTest, ErrorSensitiveFiresOnNormalPollin) {
  int source_fd = CreateEventFd();
  int sink_fd = CreateEventFd();
  ASSERT_GE(source_fd, 0);
  ASSERT_GE(sink_fd, 0);

  // Create a relay with ERROR_SENSITIVE flag.
  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 1),
      IREE_ASYNC_RELAY_FLAG_ERROR_SENSITIVE,
      iree_async_relay_error_callback_none(), &relay));
  ASSERT_NE(relay, nullptr);

  // Signal the source normally.
  SignalEventFd(source_fd);

  // Poll and wait for sink to be signaled.
  bool sink_signaled =
      WaitEventFdReadable(sink_fd, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled) << "ERROR_SENSITIVE should fire on normal POLLIN";

  close(source_fd);
  close(sink_fd);
}

// OWN_SOURCE_PRIMITIVE flag closes fd on explicit unregister of persistent
// relay. (The existing OwnSourcePrimitive test only covers one-shot
// auto-cleanup.)
TEST_P(RelayTest, OwnSourcePrimitiveOnUnregister) {
  int source_fd = CreateEventFd();
  int sink_fd = CreateEventFd();
  ASSERT_GE(source_fd, 0);
  ASSERT_GE(sink_fd, 0);

  // Create a persistent relay with OWN_SOURCE_PRIMITIVE.
  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT |
          IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE,
      iree_async_relay_error_callback_none(), &relay));
  ASSERT_NE(relay, nullptr);

  // Fire once to prove it works.
  SignalEventFd(source_fd);
  bool sink_signaled =
      WaitEventFdReadable(sink_fd, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled);

  // Explicitly unregister the persistent relay.
  iree_async_proactor_unregister_relay(proactor_, relay);

  // Drain pending operations to complete cleanup.
  DrainPending();

  // The source_fd should now be closed by the relay cleanup.
  // Verify by trying to write - should fail with EBADF.
  uint64_t value = 1;
  ssize_t result = write(source_fd, &value, sizeof(value));
  EXPECT_EQ(result, -1);
  EXPECT_EQ(errno, EBADF);

  // sink_fd is still ours to close.
  close(sink_fd);
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

// Unregister while source is pending (no signal yet).
TEST_P(RelayTest, UnregisterWhilePending) {
  int source_fd = CreateEventFd();
  int sink_fd = CreateEventFd();
  ASSERT_GE(source_fd, 0);
  ASSERT_GE(sink_fd, 0);

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink_fd), 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  // Unregister before any signal.
  iree_async_proactor_unregister_relay(proactor_, relay);

  // Signal after unregister - sink should NOT fire.
  SignalEventFd(source_fd);

  // Drain any in-flight work, then verify sink was NOT signaled.
  DrainPending();

  uint64_t value;
  bool got_signal = DrainEventFd(sink_fd, &value);
  EXPECT_FALSE(got_signal) << "Relay should not fire after unregister";

  // Drain pending operations.
  DrainPending();

  close(source_fd);
  close(sink_fd);
}

CTS_REGISTER_TEST_SUITE(RelayTest);

}  // namespace iree::async::cts
