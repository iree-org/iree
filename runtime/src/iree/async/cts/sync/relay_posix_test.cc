// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async relay operations using POSIX fd-based primitives.
//
// These tests exercise relay source/sink combinations that require file
// descriptor primitives: PRIMITIVE sources (fd readability monitoring) and
// SIGNAL_PRIMITIVE sinks (fd writes). They work on both Linux (using eventfd)
// and macOS/BSD (using pipe).
//
// The portable notification-only relay tests are in relay_test.cc.
// The Windows primitive relay tests are in relay_windows_test.cc.
//
// Test categories:
//   - Source/sink type combinations (P→P, P→N, N→P)
//   - Persistence (one-shot vs persistent re-arming)
//   - Flag behavior (OWN_SOURCE_PRIMITIVE, ERROR_SENSITIVE)
//   - Lifecycle (unregister while pending, multiple active relays)

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "iree/async/relay.h"

#ifdef __linux__
#include <sys/eventfd.h>
#endif  // __linux__

#include <atomic>
#include <thread>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/notification.h"

namespace iree::async::cts {

// Abstraction for creating a readable/writable primitive pair.
// On Linux, this is an eventfd (single bidirectional fd).
// On macOS/BSD, this is a pipe (read end + write end).
struct TestPrimitive {
  int source_fd;   // Read end: for relay source monitoring and test drain
  int signal_fd;   // Write end: for test triggering and relay sink writes
  bool owns_pair;  // True for pipe (two fds), false for eventfd (one fd)
};

class RelayPosixTest : public CtsTestBase<> {
 protected:
  // Creates a primitive pair suitable for relay testing.
  TestPrimitive CreateTestPrimitive() {
    TestPrimitive primitive = {};
#ifdef __linux__
    int fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    EXPECT_GE(fd, 0);
    primitive.source_fd = fd;
    primitive.signal_fd = fd;
    primitive.owns_pair = false;
#else
    int fds[2];
    EXPECT_EQ(pipe(fds), 0);
    // Set nonblock and cloexec on both ends.
    for (int i = 0; i < 2; ++i) {
      int flags = fcntl(fds[i], F_GETFL, 0);
      fcntl(fds[i], F_SETFL, flags | O_NONBLOCK);
      fcntl(fds[i], F_SETFD, FD_CLOEXEC);
    }
    primitive.source_fd = fds[0];  // read end
    primitive.signal_fd = fds[1];  // write end
    primitive.owns_pair = true;
#endif
    return primitive;
  }

  // Signals a primitive by writing to its write end.
  void SignalPrimitive(const TestPrimitive& primitive, uint64_t value = 1) {
    ssize_t written = write(primitive.signal_fd, &value, sizeof(value));
    EXPECT_EQ(written, static_cast<ssize_t>(sizeof(value)));
  }

  // Drains a primitive by reading from its read end.
  // Returns false if nothing to read (EAGAIN/EWOULDBLOCK).
  bool DrainPrimitive(const TestPrimitive& primitive,
                      uint64_t* out_value = nullptr) {
    uint64_t value = 0;
    ssize_t bytes_read = read(primitive.source_fd, &value, sizeof(value));
    if (bytes_read > 0) {
      if (out_value) *out_value = value;
      return true;
    }
    return false;
  }

  // Waits for a primitive to become readable with a timeout.
  // Polls the proactor between reads to allow relay dispatch.
  bool WaitPrimitiveReadable(const TestPrimitive& primitive,
                             iree_duration_t timeout_ns) {
    iree_time_t deadline = iree_time_now() + timeout_ns;
    while (iree_time_now() < deadline) {
      uint64_t value;
      if (DrainPrimitive(primitive, &value)) return true;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(10), nullptr);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
      } else if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        return false;
      }
    }
    return false;
  }

  // Closes a primitive pair, handling eventfd (one fd) vs pipe (two fds).
  void ClosePrimitive(TestPrimitive& primitive) {
    if (primitive.source_fd >= 0) {
      close(primitive.source_fd);
      primitive.source_fd = -1;
    }
    if (primitive.owns_pair && primitive.signal_fd >= 0) {
      close(primitive.signal_fd);
      primitive.signal_fd = -1;
    }
  }
};

// Basic test: PRIMITIVE source triggers SIGNAL_PRIMITIVE sink.
TEST_P(RelayPosixTest, PrimitiveToPrimitive) {
  TestPrimitive source = CreateTestPrimitive();
  TestPrimitive sink = CreateTestPrimitive();

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source.source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink.signal_fd), 1),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  SignalPrimitive(source);

  bool sink_signaled = WaitPrimitiveReadable(sink, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled);

  ClosePrimitive(source);
  ClosePrimitive(sink);
}

// PRIMITIVE source triggers SIGNAL_NOTIFICATION sink.
TEST_P(RelayPosixTest, PrimitiveToNotification) {
  TestPrimitive source = CreateTestPrimitive();

  iree_async_notification_t* sink_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink_notification));

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source.source_fd)),
      iree_async_relay_sink_signal_notification(sink_notification, 1),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_woken{false};

  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);
    bool result = iree_async_notification_wait(sink_notification,
                                               iree_make_timeout_ms(5000));
    waiter_woken.store(result, std::memory_order_release);
  });

  while (!waiter_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }
  iree_wait_until(iree_time_now() + iree_make_duration_ms(20));

  SignalPrimitive(source);

  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(2000);
  while (!waiter_woken.load(std::memory_order_acquire) &&
         iree_time_now() < deadline) {
    PollOnce();
  }

  waiter.join();
  EXPECT_TRUE(waiter_woken.load(std::memory_order_acquire));

  ClosePrimitive(source);
  iree_async_notification_release(sink_notification);
}

// NOTIFICATION source triggers SIGNAL_PRIMITIVE sink.
TEST_P(RelayPosixTest, NotificationToPrimitive) {
  iree_async_notification_t* source_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &source_notification));

  TestPrimitive sink = CreateTestPrimitive();

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source_notification),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink.signal_fd), 42),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  iree_async_notification_signal(source_notification, 1);

  bool sink_signaled = WaitPrimitiveReadable(sink, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled);

  ClosePrimitive(sink);
  iree_async_notification_release(source_notification);
}

// Persistent relay fires multiple times.
TEST_P(RelayPosixTest, PersistentMultipleTransfers) {
  TestPrimitive source = CreateTestPrimitive();
  TestPrimitive sink = CreateTestPrimitive();

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source.source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink.signal_fd), 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  for (int i = 0; i < 3; ++i) {
    DrainPrimitive(sink);
    SignalPrimitive(source);
    bool sink_signaled =
        WaitPrimitiveReadable(sink, iree_make_duration_ms(1000));
    EXPECT_TRUE(sink_signaled) << "Relay failed to fire on iteration " << i;
  }

  iree_async_proactor_unregister_relay(proactor_, relay);
  DrainPending();

  ClosePrimitive(source);
  ClosePrimitive(sink);
}

// One-shot relay auto-cleans up after single fire.
TEST_P(RelayPosixTest, OneShotAutoCleanup) {
  TestPrimitive source = CreateTestPrimitive();
  TestPrimitive sink = CreateTestPrimitive();

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source.source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink.signal_fd), 1),
      IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  SignalPrimitive(source);
  bool sink_signaled = WaitPrimitiveReadable(sink, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled);

  // Signal source again — relay should not fire (auto-cleaned up).
  SignalPrimitive(source);
  DrainPrimitive(sink);
  DrainPending();

  uint64_t value;
  bool got_second_signal = DrainPrimitive(sink, &value);
  EXPECT_FALSE(got_second_signal)
      << "One-shot relay should not fire after cleanup";

  ClosePrimitive(source);
  ClosePrimitive(sink);
}

// OWN_SOURCE_PRIMITIVE flag closes source fd on one-shot cleanup.
TEST_P(RelayPosixTest, OwnSourcePrimitive) {
  TestPrimitive source = CreateTestPrimitive();
  TestPrimitive sink = CreateTestPrimitive();

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source.source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink.signal_fd), 1),
      IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE,
      iree_async_relay_error_callback_none(), &relay));
  ASSERT_NE(relay, nullptr);

  SignalPrimitive(source);
  WaitPrimitiveReadable(sink, iree_make_duration_ms(1000));

  // The source_fd should now be closed by the relay cleanup.
  // Verify by trying to write — should fail with EBADF.
  uint64_t value = 1;
  ssize_t result = write(source.source_fd, &value, sizeof(value));
  EXPECT_EQ(result, -1);
  EXPECT_EQ(errno, EBADF);

  // Mark as closed so ClosePrimitive doesn't double-close.
  source.source_fd = -1;
  if (!source.owns_pair) source.signal_fd = -1;

  ClosePrimitive(source);
  ClosePrimitive(sink);
}

// Multiple active relays work correctly.
TEST_P(RelayPosixTest, MultipleActiveRelays) {
  constexpr int kNumRelays = 3;
  TestPrimitive sources[kNumRelays];
  TestPrimitive sinks[kNumRelays];
  iree_async_relay_t* relays[kNumRelays];

  for (int i = 0; i < kNumRelays; ++i) {
    sources[i] = CreateTestPrimitive();
    sinks[i] = CreateTestPrimitive();
    IREE_ASSERT_OK(iree_async_proactor_register_relay(
        proactor_,
        iree_async_relay_source_from_primitive(
            iree_async_primitive_from_fd(sources[i].source_fd)),
        iree_async_relay_sink_signal_primitive(
            iree_async_primitive_from_fd(sinks[i].signal_fd), 1),
        IREE_ASYNC_RELAY_FLAG_NONE, iree_async_relay_error_callback_none(),
        &relays[i]));
    ASSERT_NE(relays[i], nullptr);
  }

  for (int i = 0; i < kNumRelays; ++i) {
    SignalPrimitive(sources[i]);
  }

  for (int i = 0; i < kNumRelays; ++i) {
    bool sink_signaled =
        WaitPrimitiveReadable(sinks[i], iree_make_duration_ms(1000));
    EXPECT_TRUE(sink_signaled) << "Relay " << i << " failed to fire";
  }

  for (int i = 0; i < kNumRelays; ++i) {
    ClosePrimitive(sources[i]);
    ClosePrimitive(sinks[i]);
  }
}

// ERROR_SENSITIVE flag suppresses sink firing on source POLLERR/POLLHUP.
// Uses a pipe where closing the write end causes POLLHUP on the read end.
TEST_P(RelayPosixTest, ErrorSensitiveSuppressesSinkOnPollHup) {
  // This test always uses a pipe regardless of platform, because we need
  // to trigger POLLHUP by closing one end.
  int pipe_fds[2];
  ASSERT_EQ(pipe(pipe_fds), 0);
  int read_fd = pipe_fds[0];
  int write_fd = pipe_fds[1];

  TestPrimitive sink = CreateTestPrimitive();

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(read_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink.signal_fd), 1),
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

  uint64_t value;
  bool got_signal = DrainPrimitive(sink, &value);
  EXPECT_FALSE(got_signal) << "ERROR_SENSITIVE should suppress sink on POLLHUP";

  close(read_fd);
  ClosePrimitive(sink);
}

// ERROR_SENSITIVE flag still fires sink on normal POLLIN.
TEST_P(RelayPosixTest, ErrorSensitiveFiresOnNormalPollin) {
  TestPrimitive source = CreateTestPrimitive();
  TestPrimitive sink = CreateTestPrimitive();

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source.source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink.signal_fd), 1),
      IREE_ASYNC_RELAY_FLAG_ERROR_SENSITIVE,
      iree_async_relay_error_callback_none(), &relay));
  ASSERT_NE(relay, nullptr);

  SignalPrimitive(source);

  bool sink_signaled = WaitPrimitiveReadable(sink, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled) << "ERROR_SENSITIVE should fire on normal POLLIN";

  ClosePrimitive(source);
  ClosePrimitive(sink);
}

// OWN_SOURCE_PRIMITIVE closes source fd on explicit unregister of persistent
// relay.
TEST_P(RelayPosixTest, OwnSourcePrimitiveOnUnregister) {
  TestPrimitive source = CreateTestPrimitive();
  TestPrimitive sink = CreateTestPrimitive();

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source.source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink.signal_fd), 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT |
          IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE,
      iree_async_relay_error_callback_none(), &relay));
  ASSERT_NE(relay, nullptr);

  SignalPrimitive(source);
  bool sink_signaled = WaitPrimitiveReadable(sink, iree_make_duration_ms(1000));
  EXPECT_TRUE(sink_signaled);

  iree_async_proactor_unregister_relay(proactor_, relay);
  DrainPending();

  // The source_fd should now be closed by the relay cleanup.
  uint64_t value = 1;
  ssize_t result = write(source.source_fd, &value, sizeof(value));
  EXPECT_EQ(result, -1);
  EXPECT_EQ(errno, EBADF);

  source.source_fd = -1;
  if (!source.owns_pair) source.signal_fd = -1;

  ClosePrimitive(source);
  ClosePrimitive(sink);
}

// Unregister while source is pending (no signal yet).
TEST_P(RelayPosixTest, UnregisterWhilePending) {
  TestPrimitive source = CreateTestPrimitive();
  TestPrimitive sink = CreateTestPrimitive();

  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_,
      iree_async_relay_source_from_primitive(
          iree_async_primitive_from_fd(source.source_fd)),
      iree_async_relay_sink_signal_primitive(
          iree_async_primitive_from_fd(sink.signal_fd), 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  ASSERT_NE(relay, nullptr);

  iree_async_proactor_unregister_relay(proactor_, relay);

  // Signal after unregister — sink should NOT fire.
  SignalPrimitive(source);
  DrainPending();

  uint64_t value;
  bool got_signal = DrainPrimitive(sink, &value);
  EXPECT_FALSE(got_signal) << "Relay should not fire after unregister";

  DrainPending();

  ClosePrimitive(source);
  ClosePrimitive(sink);
}

CTS_REGISTER_TEST_SUITE(RelayPosixTest);

}  // namespace iree::async::cts
