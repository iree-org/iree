// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/wait_handle.h"

#if !defined(IREE_WAIT_HANDLE_DISABLED)

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

// We don't want to wait too long in here but when we are testing that timeouts
// work as expected we do have to sometimes wait. These are set to hopefully
// reduce flakes and not hang a build bot forever if something is broken :)
constexpr iree_duration_t kShortTimeoutNS = 1000000ull;     // 1ms
constexpr iree_duration_t kLongTimeoutNS = 60000000000ull;  // 1min

//===----------------------------------------------------------------------===//
// IREE_WAIT_PRIMITIVE_TYPE_EVENT_FD
//===----------------------------------------------------------------------===//

#if defined(IREE_HAVE_WAIT_TYPE_EVENTFD)

// TODO(benvanik): tests wrapping external eventfds.

#endif  // IREE_HAVE_WAIT_TYPE_EVENTFD

//===----------------------------------------------------------------------===//
// IREE_WAIT_PRIMITIVE_TYPE_SYNC_FILE
//===----------------------------------------------------------------------===//

#if defined(IREE_HAVE_WAIT_TYPE_SYNC_FILE)

// TODO(benvanik): tests wrapping external sync files.

#endif  // IREE_HAVE_WAIT_TYPE_SYNC_FILE

//===----------------------------------------------------------------------===//
// IREE_WAIT_PRIMITIVE_TYPE_PIPE
//===----------------------------------------------------------------------===//

#if defined(IREE_HAVE_WAIT_TYPE_PIPE)

// TODO(benvanik): tests wrapping external pipes.

#endif  // IREE_HAVE_WAIT_TYPE_PIPE

//===----------------------------------------------------------------------===//
// IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE
//===----------------------------------------------------------------------===//

#if defined(IREE_HAVE_WAIT_TYPE_WIN32_HANDLE)

// TODO(benvanik): tests wrapping external win32 handles.

#endif  // IREE_HAVE_WAIT_TYPE_WIN32_HANDLE

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//
// NOTE: this is testing the user-visible behavior of iree_event_t and the use
// of functions like iree_wait_one is not exhaustive as that is tested
// elsewhere.

// Tests that we don't leak.
TEST(Event, Lifetime) {
  iree_event_t event;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &event));
  iree_event_deinitialize(&event);
}

TEST(Event, WaitOneInitialFalse) {
  iree_event_t event;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &event));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_wait_one(&event, IREE_TIME_INFINITE_PAST));
  iree_event_deinitialize(&event);
}

TEST(Event, WaitOneInitialTrue) {
  iree_event_t event;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &event));
  IREE_EXPECT_OK(iree_wait_one(&event, IREE_TIME_INFINITE_PAST));
  iree_event_deinitialize(&event);
}

// Tests an event that was wrapped from an immediate primitive.
// These are used to neuter events in lists/sets and should be no-ops.
TEST(Event, ImmediateEvent) {
  iree_event_t event;
  iree_wait_handle_wrap_primitive(IREE_WAIT_PRIMITIVE_TYPE_NONE, {0}, &event);
  iree_event_set(&event);
  IREE_EXPECT_OK(iree_wait_one(&event, IREE_TIME_INFINITE_PAST));
  iree_event_reset(&event);
  IREE_EXPECT_OK(iree_wait_one(&event, IREE_TIME_INFINITE_PAST));
}

TEST(Event, SetWait) {
  iree_event_t event;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &event));

  // Initially unset.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_wait_one(&event, IREE_TIME_INFINITE_PAST));

  // Set and wait.
  iree_event_set(&event);
  IREE_EXPECT_OK(iree_wait_one(&event, IREE_TIME_INFINITE_PAST));

  // Set should be sticky until reset manually.
  IREE_EXPECT_OK(iree_wait_one(&event, IREE_TIME_INFINITE_PAST));

  // Resetting should unsignal the event.
  iree_event_reset(&event);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_wait_one(&event, IREE_TIME_INFINITE_PAST));

  iree_event_deinitialize(&event);
}

// Tests that we can use set/reset and that certain behavior (such as sets
// without intervening resets) is allowed. Note that this does not wait and is
// just testing the client behavior; it's possible to implement these such that
// a set while another set is pending fails and we want to verify that here.
TEST(Event, SetReset) {
  iree_event_t event;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &event));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_wait_one(&event, IREE_TIME_INFINITE_PAST));

  iree_event_set(&event);
  IREE_EXPECT_OK(iree_wait_one(&event, IREE_TIME_INFINITE_PAST));
  iree_event_set(&event);
  IREE_EXPECT_OK(iree_wait_one(&event, IREE_TIME_INFINITE_PAST));

  iree_event_reset(&event);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_wait_one(&event, IREE_TIME_INFINITE_PAST));
  iree_event_reset(&event);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_wait_one(&event, IREE_TIME_INFINITE_PAST));

  iree_event_set(&event);
  IREE_EXPECT_OK(iree_wait_one(&event, IREE_TIME_INFINITE_PAST));
  iree_event_set(&event);
  IREE_EXPECT_OK(iree_wait_one(&event, IREE_TIME_INFINITE_PAST));

  iree_event_deinitialize(&event);
}

TEST(Event, BlockingBehavior) {
  iree_event_t main_to_thread;
  IREE_ASSERT_OK(
      iree_event_initialize(/*initial_state=*/false, &main_to_thread));
  iree_event_t thread_to_main;
  IREE_ASSERT_OK(
      iree_event_initialize(/*initial_state=*/false, &thread_to_main));

  // Spinup a thread to signal the event.
  // Note that it waits on the main_to_thread event until we get further along.
  std::atomic<bool> did_run_thread{false};
  std::thread thread([&]() {
    // Wait for main thread to signal (below).
    IREE_ASSERT_OK(iree_wait_one(&main_to_thread, IREE_TIME_INFINITE_FUTURE));

    // Set something so we know this ran at all.
    did_run_thread.store(true);

    // Notify the caller thread.
    iree_event_set(&thread_to_main);
  });

  // The thread may take some time to spin up; it must wait for us to allow it
  // to run its body though so we should be fine here.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ASSERT_FALSE(did_run_thread.load());

  // Allow the thread to continue and wait for it to exit.
  iree_event_set(&main_to_thread);
  IREE_ASSERT_OK(iree_wait_one(&thread_to_main, IREE_TIME_INFINITE_FUTURE));
  ASSERT_TRUE(did_run_thread.load());

  thread.join();
  iree_event_deinitialize(&main_to_thread);
  iree_event_deinitialize(&thread_to_main);
}

// Tests using an iree_event_t as a wait source for waiting.
TEST(Event, WaitSourceBlocking) {
  iree_event_t event;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &event));
  iree_wait_source_t wait_source = iree_event_await(&event);

  // Initially unset.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_source_wait_one(wait_source, iree_immediate_timeout()));

  // Set and wait.
  iree_event_set(&event);
  IREE_EXPECT_OK(
      iree_wait_source_wait_one(wait_source, iree_immediate_timeout()));

  // Set should be sticky until reset manually.
  IREE_EXPECT_OK(
      iree_wait_source_wait_one(wait_source, iree_immediate_timeout()));

  // Resetting should unsignal the event.
  iree_event_reset(&event);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_source_wait_one(wait_source, iree_immediate_timeout()));

  iree_event_deinitialize(&event);
}

//===----------------------------------------------------------------------===//
// iree_wait_one
//===----------------------------------------------------------------------===//

// Tests iree_wait_one when polling (deadline_ns = IREE_TIME_INFINITE_PAST).
TEST(WaitOne, Polling) {
  iree_event_t ev_unset, ev_set;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set));

  // Polling (don't block even if unset).
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_wait_one(&ev_unset, IREE_TIME_INFINITE_PAST));
  IREE_ASSERT_OK(iree_wait_one(&ev_set, IREE_TIME_INFINITE_PAST));

  iree_event_deinitialize(&ev_unset);
  iree_event_deinitialize(&ev_set);
}

// Tests iree_wait_one with timeouts (deadline_ns = non-zero).
TEST(WaitOne, Timeout) {
  iree_event_t ev_unset, ev_set;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set));

  // Force a timeout by waiting on an event that'll never get set.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_one(&ev_unset, iree_time_now() + kShortTimeoutNS));

  // Ensure we return immediately when waiting on a set value (and not wait
  // 100 years because we messed up our math).
  IREE_ASSERT_OK(iree_wait_one(&ev_set, iree_time_now() + kLongTimeoutNS));

  iree_event_deinitialize(&ev_unset);
  iree_event_deinitialize(&ev_set);
}

// Tests iree_wait_one when blocking (deadline_ns = IREE_TIME_INFINITE_FUTURE).
TEST(WaitOne, Blocking) {
  iree_event_t thread_to_main;
  IREE_ASSERT_OK(
      iree_event_initialize(/*initial_state=*/false, &thread_to_main));

  // Wait forever (no timeout).
  // We approximate that by forking off a thread to signal our local event. We
  // can assume that a moderate wait is enough to verify the forever behavior as
  // otherwise we are probably just messing up the math and will timeout.
  std::thread thread([&]() {
    // Notify the caller thread after sleeping (to ensure it's not polling).
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    iree_event_set(&thread_to_main);
  });
  IREE_ASSERT_OK(iree_wait_one(&thread_to_main, IREE_TIME_INFINITE_FUTURE));

  thread.join();
  iree_event_deinitialize(&thread_to_main);
}

}  // namespace
}  // namespace iree

#endif  // !IREE_WAIT_HANDLE_DISABLED
