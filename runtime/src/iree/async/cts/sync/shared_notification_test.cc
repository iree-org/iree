// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for shared (cross-process) notification operations.
//
// Shared notifications use a caller-provided epoch counter in (simulated)
// shared memory and caller-provided wake/signal primitives. These tests
// simulate cross-process semantics by heap-allocating an epoch and creating
// shared notifications with caller-created eventfd/pipe fds (POSIX) or
// Events (Windows).
//
// The key behavioral differences from local notifications:
//   - epoch_ptr points to caller-provided memory, not the inline epoch field
//   - Destroy does not close the wake/signal primitives (caller owns them)
//   - On Linux: futex calls omit FUTEX_PRIVATE_FLAG (physical page hashing)
//   - On macOS: sync waiters use poll() instead of condvar (process-local)
//   - On Windows: WakeByAddress works cross-process natively

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"

#if defined(IREE_PLATFORM_WINDOWS)
// Windows: Event objects for wake/signal primitives.
#else
// POSIX: eventfd (Linux) or pipe (macOS) for wake/signal primitives.
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#if defined(IREE_PLATFORM_LINUX)
#include <sys/eventfd.h>
#endif  // IREE_PLATFORM_LINUX
#endif  // IREE_PLATFORM_WINDOWS

namespace iree::async::cts {

class SharedNotificationTest : public CtsTestBase<> {
 protected:
  // Represents the shared state between two "processes" (simulated).
  // In real cross-process usage, the epoch would be in mmap'd shared memory
  // and the primitives would be inherited or passed via IPC.
  struct SharedState {
    iree_atomic_int32_t epoch;

    // POSIX: eventfd (Linux) or pipe pair (macOS).
    // Windows: Event HANDLEs.
#if defined(IREE_PLATFORM_WINDOWS)
    HANDLE wake_event;
    HANDLE signal_event;
#elif defined(IREE_PLATFORM_LINUX)
    int eventfd;
#else
    int pipe_fds[2];  // [0]=read (wake), [1]=write (signal)
#endif
  };

  // Creates the shared state: initializes epoch and creates platform
  // wake/signal primitives.
  iree_status_t CreateSharedState(SharedState* state) {
    iree_atomic_store(&state->epoch, 0, iree_memory_order_release);

#if defined(IREE_PLATFORM_WINDOWS)
    state->wake_event = CreateEventW(NULL, /*bManualReset=*/FALSE,
                                     /*bInitialState=*/FALSE, NULL);
    if (!state->wake_event) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "CreateEvent failed for wake_event");
    }
    state->signal_event = CreateEventW(NULL, /*bManualReset=*/FALSE,
                                       /*bInitialState=*/FALSE, NULL);
    if (!state->signal_event) {
      CloseHandle(state->wake_event);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "CreateEvent failed for signal_event");
    }
#elif defined(IREE_PLATFORM_LINUX)
    state->eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (state->eventfd < 0) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "eventfd creation failed");
    }
#else
    if (pipe(state->pipe_fds) < 0) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "pipe creation failed");
    }
    for (int i = 0; i < 2; ++i) {
      int current_flags = fcntl(state->pipe_fds[i], F_GETFL);
      if (current_flags >= 0) {
        fcntl(state->pipe_fds[i], F_SETFL, current_flags | O_NONBLOCK);
      }
      fcntl(state->pipe_fds[i], F_SETFD, FD_CLOEXEC);
    }
#endif
    return iree_ok_status();
  }

  // Destroys the shared state: closes platform primitives.
  void DestroySharedState(SharedState* state) {
#if defined(IREE_PLATFORM_WINDOWS)
    if (state->signal_event) CloseHandle(state->signal_event);
    if (state->wake_event) CloseHandle(state->wake_event);
#elif defined(IREE_PLATFORM_LINUX)
    if (state->eventfd >= 0) close(state->eventfd);
#else
    if (state->pipe_fds[0] >= 0) close(state->pipe_fds[0]);
    if (state->pipe_fds[1] >= 0) close(state->pipe_fds[1]);
#endif
  }

  // Fills shared notification options from the shared state.
  iree_async_notification_shared_options_t MakeSharedOptions(
      SharedState* state) {
    iree_async_notification_shared_options_t options;
    memset(&options, 0, sizeof(options));
    options.epoch_address = &state->epoch;

#if defined(IREE_PLATFORM_WINDOWS)
    options.wake_primitive.type = IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE;
    options.wake_primitive.value.win32_handle = (uintptr_t)state->wake_event;
    options.signal_primitive.type = IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE;
    options.signal_primitive.value.win32_handle =
        (uintptr_t)state->signal_event;
#elif defined(IREE_PLATFORM_LINUX)
    // Linux eventfd: same fd for both wake (POLLIN) and signal (write).
    options.wake_primitive = iree_async_primitive_from_fd(state->eventfd);
    options.signal_primitive = iree_async_primitive_from_fd(state->eventfd);
#else
    // macOS pipe: read end for wake, write end for signal.
    options.wake_primitive = iree_async_primitive_from_fd(state->pipe_fds[0]);
    options.signal_primitive = iree_async_primitive_from_fd(state->pipe_fds[1]);
#endif
    return options;
  }
};

// Signal via shared notification, verify epoch_ptr advances.
TEST_P(SharedNotificationTest, SharedEpochSignalAndQuery) {
  SharedState state;
  IREE_ASSERT_OK(CreateSharedState(&state));

  auto options = MakeSharedOptions(&state);
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification));

  // Initial epoch should be 0.
  EXPECT_EQ(iree_async_notification_query_epoch(notification), 0u);

  // Signal and verify epoch advances.
  iree_async_notification_signal(notification, 1);
  EXPECT_EQ(iree_async_notification_query_epoch(notification), 1u);

  iree_async_notification_signal(notification, 1);
  EXPECT_EQ(iree_async_notification_query_epoch(notification), 2u);

  // Verify the shared epoch in "shared memory" matches.
  EXPECT_EQ((uint32_t)iree_atomic_load(&state.epoch, iree_memory_order_acquire),
            2u);

  iree_async_notification_release(notification);
  DestroySharedState(&state);
}

// Synchronous wait on shared notification, signal from another thread.
TEST_P(SharedNotificationTest, SharedEpochSyncWait) {
  SharedState state;
  IREE_ASSERT_OK(CreateSharedState(&state));

  auto options = MakeSharedOptions(&state);
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification));

  std::atomic<bool> wait_started{false};
  std::atomic<bool> wait_result{false};

  std::thread waiter([&]() {
    wait_started.store(true, std::memory_order_release);
    bool result =
        iree_async_notification_wait(notification, iree_make_timeout_ms(5000));
    wait_result.store(result, std::memory_order_release);
  });

  while (!wait_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }
  iree_wait_until(iree_time_now() + iree_make_duration_ms(10));

  iree_async_notification_signal(notification, 1);

  waiter.join();
  EXPECT_TRUE(wait_result.load(std::memory_order_acquire));

  iree_async_notification_release(notification);
  DestroySharedState(&state);
}

// Async NOTIFICATION_WAIT completes when shared epoch advances.
TEST_P(SharedNotificationTest, SharedEpochAsyncWait) {
  SharedState state;
  IREE_ASSERT_OK(CreateSharedState(&state));

  auto options = MakeSharedOptions(&state);
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification));

  CompletionTracker tracker;
  iree_async_notification_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;
  wait_op.notification = notification;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  std::thread signaler(
      [notification]() { iree_async_notification_signal(notification, 1); });

  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(5000));

  signaler.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_notification_release(notification);
  DestroySharedState(&state);
}

// Destroy shared notification, verify caller's fds are still valid.
TEST_P(SharedNotificationTest, DestroyDoesNotClosePrimitives) {
  SharedState state;
  IREE_ASSERT_OK(CreateSharedState(&state));

  auto options = MakeSharedOptions(&state);
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification));

  // Destroy the notification.
  iree_async_notification_release(notification);

  // Verify caller's primitives are still valid by writing/reading them.
#if defined(IREE_PLATFORM_WINDOWS)
  // Signal the wake event and verify it was signaled.
  EXPECT_TRUE(SetEvent(state.wake_event));
  DWORD wait_result = WaitForSingleObject(state.wake_event, 0);
  // Event was set, so wait should succeed (auto-reset already consumed it
  // in SetEvent, but the wait at 0ms may or may not see it depending on
  // timing). The key check is that SetEvent didn't fail with
  // ERROR_INVALID_HANDLE.
  (void)wait_result;
#elif defined(IREE_PLATFORM_LINUX)
  // Write to eventfd and read back.
  uint64_t write_value = 1;
  ssize_t write_result =
      write(state.eventfd, &write_value, sizeof(write_value));
  EXPECT_EQ(write_result, (ssize_t)sizeof(write_value))
      << "eventfd should still be writable after notification destroy";
  uint64_t read_value = 0;
  ssize_t read_result = read(state.eventfd, &read_value, sizeof(read_value));
  EXPECT_EQ(read_result, (ssize_t)sizeof(read_value))
      << "eventfd should still be readable after notification destroy";
  EXPECT_EQ(read_value, 1u);
#else
  // macOS pipe: write to signal end, read from wake end.
  uint8_t write_byte = 42;
  ssize_t write_result =
      write(state.pipe_fds[1], &write_byte, sizeof(write_byte));
  EXPECT_EQ(write_result, (ssize_t)sizeof(write_byte))
      << "pipe should still be writable after notification destroy";
  uint8_t read_byte = 0;
  ssize_t read_result = read(state.pipe_fds[0], &read_byte, sizeof(read_byte));
  EXPECT_EQ(read_result, (ssize_t)sizeof(read_byte))
      << "pipe should still be readable after notification destroy";
  EXPECT_EQ(read_byte, 42u);
#endif

  DestroySharedState(&state);
}

// Repeated signal/wait cycles through shared epoch.
TEST_P(SharedNotificationTest, MultipleCycles) {
  SharedState state;
  IREE_ASSERT_OK(CreateSharedState(&state));

  auto options = MakeSharedOptions(&state);
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification));

  std::atomic<int> cycles_completed{0};
  constexpr int kCycles = 5;

  std::thread worker([&]() {
    for (int i = 0; i < kCycles; ++i) {
      bool result = iree_async_notification_wait(notification,
                                                 iree_make_timeout_ms(1000));
      if (result) {
        cycles_completed.fetch_add(1, std::memory_order_acq_rel);
      }
    }
  });

  for (int i = 0; i < kCycles; ++i) {
    iree_wait_until(iree_time_now() + iree_make_duration_ms(20));
    iree_async_notification_signal(notification, 1);
  }

  worker.join();
  EXPECT_EQ(cycles_completed.load(std::memory_order_acquire), kCycles);

  iree_async_notification_release(notification);
  DestroySharedState(&state);
}

// Two shared notifications pointing at the same epoch: signaling one should
// advance the epoch observable by the other. This exercises the cross-process
// semantic (two proactors sharing one epoch) within a single process.
TEST_P(SharedNotificationTest, TwoNotificationsOneEpoch) {
  SharedState state;
  IREE_ASSERT_OK(CreateSharedState(&state));

  // Create two shared notifications pointing at the same epoch.
  // In real cross-process usage, each process creates one notification.
  auto options = MakeSharedOptions(&state);
  iree_async_notification_t* notification_a = nullptr;
  iree_async_notification_t* notification_b = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification_a));
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification_b));

  // Signal through notification_a.
  iree_async_notification_signal(notification_a, 1);

  // Both notifications should see the epoch advance.
  EXPECT_EQ(iree_async_notification_query_epoch(notification_a), 1u);
  EXPECT_EQ(iree_async_notification_query_epoch(notification_b), 1u);

  // Signal through notification_b.
  iree_async_notification_signal(notification_b, 1);

  EXPECT_EQ(iree_async_notification_query_epoch(notification_a), 2u);
  EXPECT_EQ(iree_async_notification_query_epoch(notification_b), 2u);

  // Shared memory epoch matches.
  EXPECT_EQ((uint32_t)iree_atomic_load(&state.epoch, iree_memory_order_acquire),
            2u);

  iree_async_notification_release(notification_a);
  iree_async_notification_release(notification_b);
  DestroySharedState(&state);
}

// Sync wait on one shared notification, signal from the other (same epoch).
TEST_P(SharedNotificationTest, CrossNotificationSyncWait) {
  SharedState state;
  IREE_ASSERT_OK(CreateSharedState(&state));

  auto options = MakeSharedOptions(&state);
  iree_async_notification_t* notification_waiter = nullptr;
  iree_async_notification_t* notification_signaler = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification_waiter));
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification_signaler));

  std::atomic<bool> wait_started{false};
  std::atomic<bool> wait_result{false};

  // Wait on notification_waiter.
  std::thread waiter([&]() {
    wait_started.store(true, std::memory_order_release);
    bool result = iree_async_notification_wait(notification_waiter,
                                               iree_make_timeout_ms(5000));
    wait_result.store(result, std::memory_order_release);
  });

  while (!wait_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }
  iree_wait_until(iree_time_now() + iree_make_duration_ms(10));

  // Signal through notification_signaler (same shared epoch).
  iree_async_notification_signal(notification_signaler, 1);

  waiter.join();
  EXPECT_TRUE(wait_result.load(std::memory_order_acquire));

  iree_async_notification_release(notification_waiter);
  iree_async_notification_release(notification_signaler);
  DestroySharedState(&state);
}

// Synchronous wait timeout on shared notification.
TEST_P(SharedNotificationTest, SharedSyncWaitTimeout) {
  SharedState state;
  IREE_ASSERT_OK(CreateSharedState(&state));

  auto options = MakeSharedOptions(&state);
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create_shared(proactor_, &options,
                                                       &notification));

  iree_time_t start = iree_time_now();
  bool result =
      iree_async_notification_wait(notification, iree_make_timeout_ms(50));
  iree_time_t elapsed = iree_time_now() - start;

  EXPECT_FALSE(result);
  EXPECT_GE(elapsed, iree_make_duration_ms(10));

  iree_async_notification_release(notification);
  DestroySharedState(&state);
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(SharedNotificationTest,
                                  /*required=*/{"shared_notification"},
                                  /*excluded=*/{});

}  // namespace iree::async::cts
