// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for event source operations.
//
// Event sources provide persistent monitoring of external file descriptors
// with callback-based notification. These tests verify the registration,
// callback invocation, and unregistration lifecycle.
//
// Two test fixtures are provided:
// - EventSourceEventfdTest: Uses Linux eventfd, skipped on non-Linux platforms.
// - EventSourcePosixTest: Uses POSIX pipe(), runs on all POSIX platforms.
//   Use this for cross-platform tests (error propagation, hangup detection).

#include <errno.h>
#include <poll.h>
#include <string.h>
#include <unistd.h>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/proactor.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#include <sys/eventfd.h>
#endif

#include <atomic>
#include <thread>

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// EventSourceEventfdTest - Linux-specific tests using eventfd
//===----------------------------------------------------------------------===//

// Test fixture for event source tests using Linux eventfd.
// Skipped on non-Linux platforms.
class EventSourceEventfdTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
#if !defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_ANDROID)
    GTEST_SKIP() << "EventSourceEventfdTest requires Linux eventfd";
#endif
  }

  // Creates a non-blocking eventfd for testing.
  int CreateTestEventFd() {
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
    int fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    EXPECT_GE(fd, 0) << "eventfd creation failed";
    return fd;
#else
    return -1;
#endif
  }

  // Signals an eventfd by writing to it.
  void SignalEventFd(int fd) {
    uint64_t value = 1;
    ssize_t written = write(fd, &value, sizeof(value));
    EXPECT_EQ(written, sizeof(value)) << "eventfd write failed";
  }

  // Drains an eventfd by reading from it.
  void DrainEventFd(int fd) {
    uint64_t value = 0;
    ssize_t bytes_read = read(fd, &value, sizeof(value));
    (void)bytes_read;  // May fail if already drained; that's fine.
  }
};

// Basic lifecycle: register, unregister, no crash.
TEST_P(EventSourceEventfdTest, RegisterUnregister) {
  int fd = CreateTestEventFd();
  if (fd < 0) return;

  // Track callback invocations.
  std::atomic<int> callback_count{0};
  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* count = static_cast<std::atomic<int>*>(user_data);
        (*count)++;
      },
      &callback_count};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t handle = iree_async_primitive_from_fd(fd);

  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, handle,
                                                           callback, &source));
  ASSERT_NE(source, nullptr);

  // Unregister before any signals.
  iree_async_proactor_unregister_event_source(proactor_, source);

  // Cleanup.
  close(fd);
}

// Signal fd after registration, verify callback fires.
TEST_P(EventSourceEventfdTest, CallbackFires) {
  int fd = CreateTestEventFd();
  if (fd < 0) return;

  struct CallbackState {
    std::atomic<int> call_count{0};
    iree_async_event_source_t* source = nullptr;
    int fd = -1;
  };
  CallbackState state;
  state.fd = fd;

  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* state = static_cast<CallbackState*>(user_data);
        state->call_count++;
        // Drain the eventfd to prevent re-triggering (level-triggered).
        uint64_t value = 0;
        ssize_t result = read(state->fd, &value, sizeof(value));
        IREE_ASSERT(result >= 0 || errno == EAGAIN);
      },
      &state};

  iree_async_primitive_t handle = iree_async_primitive_from_fd(fd);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(
      proactor_, handle, callback, &state.source));

  // Signal the eventfd.
  SignalEventFd(fd);

  // Poll until callback fires. Internal callbacks don't count toward
  // "completions", so we poll with short timeouts and check the state directly.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(200);
  while (state.call_count.load() == 0 && iree_time_now() < deadline) {
    PollOnce();
  }

  EXPECT_GE(state.call_count.load(), 1) << "Callback should have fired";

  // Cleanup.
  iree_async_proactor_unregister_event_source(proactor_, state.source);
  close(fd);
}

// Multiple signals should fire multiple callbacks (multishot behavior).
TEST_P(EventSourceEventfdTest, MultipleSignals) {
  int fd = CreateTestEventFd();
  if (fd < 0) return;

  struct CallbackState {
    std::atomic<int> call_count{0};
    int fd = -1;
  };
  CallbackState state;
  state.fd = fd;

  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* state = static_cast<CallbackState*>(user_data);
        state->call_count++;
        // Drain the eventfd.
        uint64_t value = 0;
        ssize_t result = read(state->fd, &value, sizeof(value));
        IREE_ASSERT(result >= 0 || errno == EAGAIN);
      },
      &state};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t handle = iree_async_primitive_from_fd(fd);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, handle,
                                                           callback, &source));

  // Signal multiple times with poll() calls in between.
  constexpr int kSignalCount = 3;
  for (int i = 0; i < kSignalCount; ++i) {
    int count_before = state.call_count.load();
    SignalEventFd(fd);
    // Poll until this signal is processed.
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(100);
    while (state.call_count.load() <= count_before &&
           iree_time_now() < deadline) {
      PollOnce();
    }
  }

  EXPECT_GE(state.call_count.load(), kSignalCount)
      << "Should have received at least " << kSignalCount << " callbacks";

  // Cleanup.
  iree_async_proactor_unregister_event_source(proactor_, source);
  close(fd);
}

// After unregister, signals should not fire callbacks.
TEST_P(EventSourceEventfdTest, UnregisterStopsCallbacks) {
  int fd = CreateTestEventFd();
  if (fd < 0) return;

  struct CallbackState {
    std::atomic<int> call_count{0};
    int fd = -1;
  };
  CallbackState state;
  state.fd = fd;

  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* state = static_cast<CallbackState*>(user_data);
        state->call_count++;
        uint64_t value = 0;
        ssize_t result = read(state->fd, &value, sizeof(value));
        IREE_ASSERT(result >= 0 || errno == EAGAIN);
      },
      &state};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t handle = iree_async_primitive_from_fd(fd);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, handle,
                                                           callback, &source));

  // Verify callback works initially.
  SignalEventFd(fd);
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(100);
  while (state.call_count.load() == 0 && iree_time_now() < deadline) {
    PollOnce();
  }
  int count_before_unregister = state.call_count.load();
  EXPECT_GE(count_before_unregister, 1);

  // Unregister.
  iree_async_proactor_unregister_event_source(proactor_, source);

  // Signal again after unregister.
  SignalEventFd(fd);
  PollOnce();

  // Callback count should not have increased.
  EXPECT_EQ(state.call_count.load(), count_before_unregister)
      << "Callback should not fire after unregister";

  // Cleanup (eventfd only, source already unregistered).
  close(fd);
}

// Multiple event sources can be registered simultaneously.
TEST_P(EventSourceEventfdTest, MultipleEventSources) {
  constexpr int kSourceCount = 3;

  int fds[kSourceCount];
  std::atomic<int> callback_counts[kSourceCount];
  iree_async_event_source_t* sources[kSourceCount];

  for (int i = 0; i < kSourceCount; ++i) {
    fds[i] = CreateTestEventFd();
    if (fds[i] < 0) return;
    callback_counts[i].store(0);
    sources[i] = nullptr;
  }

  // Register all sources.
  for (int i = 0; i < kSourceCount; ++i) {
    struct CallbackData {
      std::atomic<int>* counter;
      int fd;
    };
    // We need stable storage for the callback user_data.
    // Use a simple static array for this test.
    static CallbackData callback_data[kSourceCount];
    callback_data[i].counter = &callback_counts[i];
    callback_data[i].fd = fds[i];

    iree_async_event_source_callback_t callback = {
        [](void* user_data, iree_async_event_source_t* source,
           iree_async_poll_events_t events) {
          auto* data = static_cast<CallbackData*>(user_data);
          (*data->counter)++;
          uint64_t value = 0;
          ssize_t result = read(data->fd, &value, sizeof(value));
          IREE_ASSERT(result >= 0 || errno == EAGAIN);
        },
        &callback_data[i]};

    iree_async_primitive_t handle = iree_async_primitive_from_fd(fds[i]);
    IREE_ASSERT_OK(iree_async_proactor_register_event_source(
        proactor_, handle, callback, &sources[i]));
  }

  // Signal only the middle one (index 1).
  SignalEventFd(fds[1]);
  {
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(100);
    while (callback_counts[1].load() == 0 && iree_time_now() < deadline) {
      PollOnce();
    }
  }

  // Only the signaled source's callback should fire.
  EXPECT_EQ(callback_counts[0].load(), 0);
  EXPECT_GE(callback_counts[1].load(), 1);
  EXPECT_EQ(callback_counts[2].load(), 0);

  // Signal all of them.
  for (int i = 0; i < kSourceCount; ++i) {
    SignalEventFd(fds[i]);
  }
  {
    // Poll until all sources have at least one callback.
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(100);
    while (iree_time_now() < deadline) {
      bool all_fired = true;
      for (int i = 0; i < kSourceCount; ++i) {
        if (callback_counts[i].load() < 1) {
          all_fired = false;
          break;
        }
      }
      if (all_fired) break;
      PollOnce();
    }
  }

  // All should have at least one callback now.
  for (int i = 0; i < kSourceCount; ++i) {
    EXPECT_GE(callback_counts[i].load(), 1) << "Source " << i;
  }

  // Cleanup.
  for (int i = 0; i < kSourceCount; ++i) {
    iree_async_proactor_unregister_event_source(proactor_, sources[i]);
    close(fds[i]);
  }
}

CTS_REGISTER_TEST_SUITE(EventSourceEventfdTest);

//===----------------------------------------------------------------------===//
// EventSourcePosixTest - POSIX-portable tests using pipe()
//===----------------------------------------------------------------------===//

// Test fixture for POSIX-portable event source tests.
// Uses pipe() which works on Linux, macOS, BSD, and other POSIX systems.
// These tests verify cross-platform behavior like error propagation.
class EventSourcePosixTest : public CtsTestBase<> {
 protected:
  // No skip - pipe() is available on all POSIX platforms.
};

// Closing the write end of a pipe triggers POLLHUP on the read end.
// This verifies error event propagation through the callback.
//
// This test is critical for cross-platform correctness: when implementing
// kqueue or other backends, this test verifies they translate native hangup
// events to IREE_ASYNC_POLL_EVENT_HUP correctly.
TEST_P(EventSourcePosixTest, PipeHangupDeliversPollHup) {
  int pipe_fds[2];
  ASSERT_EQ(pipe(pipe_fds), 0) << "pipe() failed: " << strerror(errno);
  int read_fd = pipe_fds[0];
  int write_fd = pipe_fds[1];

  struct CallbackState {
    std::atomic<int> call_count{0};
    std::atomic<iree_async_poll_events_t> last_events{0};
  };
  CallbackState state;

  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* state = static_cast<CallbackState*>(user_data);
        state->call_count++;
        state->last_events.store(events);
      },
      &state};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t handle = iree_async_primitive_from_fd(read_fd);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, handle,
                                                           callback, &source));

  // Close the write end - this triggers POLLHUP on the read end.
  close(write_fd);

  // Poll until callback fires.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(200);
  while (state.call_count.load() == 0 && iree_time_now() < deadline) {
    PollOnce();
  }

  ASSERT_GE(state.call_count.load(), 1) << "Callback should have fired";

  // Verify HUP event was delivered.
  iree_async_poll_events_t events = state.last_events.load();
  EXPECT_TRUE(events & IREE_ASYNC_POLL_EVENT_HUP)
      << "Expected POLLHUP, got events=0x" << std::hex << events;

  // iree_async_poll_has_error() should return true for HUP-only events.
  // Note: Some systems may also set POLLIN alongside POLLHUP (indicating
  // EOF is readable). Only check has_error if IN is not set.
  if (!(events & IREE_ASYNC_POLL_EVENT_IN)) {
    EXPECT_TRUE(iree_async_poll_has_error(events))
        << "iree_async_poll_has_error() should return true for HUP without IN";
  }

  // Cleanup.
  iree_async_proactor_unregister_event_source(proactor_, source);
  close(read_fd);
}

// Verify POLLIN is delivered when data is available to read.
TEST_P(EventSourcePosixTest, PipeDataDeliversPollIn) {
  int pipe_fds[2];
  ASSERT_EQ(pipe(pipe_fds), 0) << "pipe() failed: " << strerror(errno);
  int read_fd = pipe_fds[0];
  int write_fd = pipe_fds[1];

  struct CallbackState {
    std::atomic<int> call_count{0};
    std::atomic<iree_async_poll_events_t> last_events{0};
    int read_fd = -1;
  };
  CallbackState state;
  state.read_fd = read_fd;

  iree_async_event_source_callback_t callback = {
      [](void* user_data, iree_async_event_source_t* source,
         iree_async_poll_events_t events) {
        auto* state = static_cast<CallbackState*>(user_data);
        state->call_count++;
        state->last_events.store(events);
        // Drain the pipe to prevent re-triggering.
        char buffer[64];
        ssize_t result = read(state->read_fd, buffer, sizeof(buffer));
        IREE_ASSERT(result >= 0 || errno == EAGAIN);
      },
      &state};

  iree_async_event_source_t* source = nullptr;
  iree_async_primitive_t handle = iree_async_primitive_from_fd(read_fd);
  IREE_ASSERT_OK(iree_async_proactor_register_event_source(proactor_, handle,
                                                           callback, &source));

  // Write data to the pipe - this makes the read end readable.
  const char* msg = "test";
  ssize_t written = write(write_fd, msg, strlen(msg));
  ASSERT_EQ(written, (ssize_t)strlen(msg));

  // Poll until callback fires.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(200);
  while (state.call_count.load() == 0 && iree_time_now() < deadline) {
    PollOnce();
  }

  ASSERT_GE(state.call_count.load(), 1) << "Callback should have fired";

  // Verify IN event was delivered.
  iree_async_poll_events_t events = state.last_events.load();
  EXPECT_TRUE(events & IREE_ASYNC_POLL_EVENT_IN)
      << "Expected POLLIN, got events=0x" << std::hex << events;

  // Cleanup.
  iree_async_proactor_unregister_event_source(proactor_, source);
  close(read_fd);
  close(write_fd);
}

CTS_REGISTER_TEST_SUITE(EventSourcePosixTest);

}  // namespace iree::async::cts
