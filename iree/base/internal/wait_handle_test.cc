// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/base/internal/wait_handle.h"

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
  bool did_run_thread = false;
  std::thread thread([&]() {
    // Wait for main thread to signal (below).
    IREE_ASSERT_OK(iree_wait_one(&main_to_thread, IREE_TIME_INFINITE_FUTURE));

    // Set something so we know this ran at all.
    did_run_thread = true;

    // Notify the caller thread.
    iree_event_set(&thread_to_main);
  });

  // The thread may take some time to spin up; it must wait for us to allow it
  // to run its body though so we should be fine here.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ASSERT_FALSE(did_run_thread);

  // Allow the thread to continue and wait for it to exit.
  iree_event_set(&main_to_thread);
  IREE_ASSERT_OK(iree_wait_one(&thread_to_main, IREE_TIME_INFINITE_FUTURE));
  ASSERT_TRUE(did_run_thread);

  thread.join();
  iree_event_deinitialize(&main_to_thread);
  iree_event_deinitialize(&thread_to_main);
}

//===----------------------------------------------------------------------===//
// iree_wait_set_t
//===----------------------------------------------------------------------===//

// Tests basic usage of the wait set API without waiting.
TEST(WaitSet, Lifetime) {
  iree_event_t event;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &event));

  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, event));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, event));
  iree_wait_set_erase(wait_set, event);
  iree_wait_set_clear(wait_set);
  iree_wait_set_free(wait_set);

  iree_event_deinitialize(&event);
}

TEST(WaitSet, UnreasonableCapacity) {
  iree_wait_set_t* wait_set = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_wait_set_allocate(1 * 1024 * 1024, iree_allocator_system(),
                             &wait_set));
}

// Tests that inserting the same handles multiple times is tracked correctly.
TEST(WaitSet, Deduplication) {
  iree_event_t ev_unset, ev_dupe;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_dupe));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  // We want to test for duplication on ev_dupe here so ensure it's added.
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_dupe));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_dupe));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_dupe));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset));

  // Wait should succeed immediately because ev_dupe is set (and our wake handle
  // should be ev_dupe).
  iree_wait_handle_t wake_handle;
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0,
            memcmp(&ev_dupe.value, &wake_handle.value, sizeof(ev_dupe.value)));

  // Erase the events one at a time and ensure we still get the expected number
  // of waits on ev_dupe.
  iree_wait_set_erase(wait_set, wake_handle);
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0,
            memcmp(&ev_dupe.value, &wake_handle.value, sizeof(ev_dupe.value)));
  iree_wait_set_erase(wait_set, wake_handle);
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0,
            memcmp(&ev_dupe.value, &wake_handle.value, sizeof(ev_dupe.value)));
  iree_wait_set_erase(wait_set, wake_handle);

  // Now there should just be ev_unset present in the set and a poll will fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset);
  iree_event_deinitialize(&ev_dupe);
}

// Tests that clear handles things right in the face of dupes.
TEST(WaitSet, Clear) {
  iree_event_t ev_unset, ev_dupe;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_dupe));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  // We want to test for duplication o n ev_dupe here.
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_dupe));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_dupe));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_dupe));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset));

  // Wait should succeed immediately because ev_dupe is set (and our wake handle
  // should be ev_dupe).
  iree_wait_handle_t wake_handle;
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0,
            memcmp(&ev_dupe.value, &wake_handle.value, sizeof(ev_dupe.value)));

  // Erase all events from the set.
  iree_wait_set_clear(wait_set);

  // No more events remaining; should pass immediately.
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset);
  iree_event_deinitialize(&ev_dupe);
}

// Tests iree_wait_all when polling (deadline_ns = IREE_TIME_INFINITE_PAST).
TEST(WaitSet, WaitAllPolling) {
  iree_event_t ev_unset_0, ev_unset_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_1));
  iree_event_t ev_set_0, ev_set_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_1));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  // Polls when empty should never block.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_all(wait_set, IREE_TIME_INFINITE_PAST));

  // Polls with only unset handles should never block.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_wait_all(wait_set, IREE_TIME_INFINITE_PAST));

  // Polls with only set handles should return immediately.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_1));
  IREE_ASSERT_OK(iree_wait_all(wait_set, IREE_TIME_INFINITE_PAST));

  // Polls with mixed set/unset should never succeed.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_1));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_wait_all(wait_set, IREE_TIME_INFINITE_PAST));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset_0);
  iree_event_deinitialize(&ev_unset_1);
  iree_event_deinitialize(&ev_set_0);
  iree_event_deinitialize(&ev_set_1);
}

// Tests iree_wait_all with timeouts (deadline_ns = non-zero).
TEST(WaitSet, WaitAllTimeout) {
  iree_event_t ev_unset_0, ev_unset_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_1));
  iree_event_t ev_set_0, ev_set_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_1));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  // Timeouts when empty should never block.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_all(wait_set, iree_time_now() + kShortTimeoutNS));

  // Timeouts with only unset handles should block (and then expire).
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));
  constexpr iree_duration_t kShortTimeoutNS = 1000000ull;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_all(wait_set, iree_time_now() + kShortTimeoutNS));

  // Timeouts with only set handles should return immediately.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_1));
  IREE_ASSERT_OK(iree_wait_all(wait_set, iree_time_now() + kShortTimeoutNS));

  // Timeouts with mixed set/unset should never succeed.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_1));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_all(wait_set, iree_time_now() + kShortTimeoutNS));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset_0);
  iree_event_deinitialize(&ev_unset_1);
  iree_event_deinitialize(&ev_set_0);
  iree_event_deinitialize(&ev_set_1);
}

// Tests iree_wait_all when blocking (deadline_ns = IREE_TIME_INFINITE_FUTURE).
TEST(WaitSet, WaitAllBlocking) {
  iree_event_t thread_to_main;
  IREE_ASSERT_OK(
      iree_event_initialize(/*initial_state=*/false, &thread_to_main));
  iree_event_t ev_set_0, ev_set_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_1));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  // Throw in some other set handles so that we are multi-waiting for just the
  // thread_to_main event to be set.
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_1));

  // Wait forever (no timeout).
  // We approximate that by forking off a thread to signal our local event. We
  // can assume that a moderate wait is enough to verify the forever behavior as
  // otherwise we are probably just messing up the math and will timeout.
  std::thread thread([&]() {
    // Notify the caller thread after sleeping (to ensure it's not polling).
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    iree_event_set(&thread_to_main);
  });
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, thread_to_main));
  IREE_ASSERT_OK(iree_wait_all(wait_set, IREE_TIME_INFINITE_FUTURE));

  thread.join();
  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&thread_to_main);
  iree_event_deinitialize(&ev_set_0);
  iree_event_deinitialize(&ev_set_1);
}

// Tests iree_wait_all when one or more handles are duplicated.
TEST(WaitSet, WaitAllDuplicates) {
  iree_event_t ev_set;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));

  // Wait should succeed immediately because ev_set is set.
  IREE_ASSERT_OK(iree_wait_all(wait_set, IREE_TIME_INFINITE_PAST));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_set);
}

// Tests iree_wait_any; note that this is only focused on testing the wait.
TEST(WaitSet, WaitAny) {
  iree_event_t ev_unset, ev_set;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));

  // Wait should succeed immediately because ev_set is set (and our wake handle
  // should be ev_set).
  iree_wait_handle_t wake_handle;
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0, memcmp(&ev_set.value, &wake_handle.value, sizeof(ev_set.value)));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset);
  iree_event_deinitialize(&ev_set);
}

// Tests iree_wait_any when polling (deadline_ns = IREE_TIME_INFINITE_PAST).
TEST(WaitSet, WaitAnyPolling) {
  iree_event_t ev_unset_0, ev_unset_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_1));
  iree_event_t ev_set_0, ev_set_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_1));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  iree_wait_handle_t empty_handle;
  memset(&empty_handle, 0, sizeof(empty_handle));

  // Polls when empty should never block and return an empty wake handle.
  // This is so that if the caller touches the wake_handle they at least have
  // initialized memory.
  iree_wait_set_clear(wait_set);
  iree_wait_handle_t wake_handle;
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0, memcmp(&empty_handle, &wake_handle, sizeof(empty_handle)));

  // Polls with only unset handles should never block.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0, memcmp(&empty_handle, &wake_handle, sizeof(empty_handle)));

  // Polls with only set handles should return immediately.
  // Note that which handle is returned is not specified.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_1));
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_TRUE(
      0 ==
          memcmp(&ev_set_0.value, &wake_handle.value, sizeof(ev_set_0.value)) ||
      0 == memcmp(&ev_set_1.value, &wake_handle.value, sizeof(ev_set_1.value)));

  // Polls with mixed set/unset should return immediately.
  // Note that which handle is returned is not specified but we know it should
  // at least be one of the signaled ones.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_1));
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_TRUE(
      0 ==
          memcmp(&ev_set_0.value, &wake_handle.value, sizeof(ev_set_0.value)) ||
      0 == memcmp(&ev_set_1.value, &wake_handle.value, sizeof(ev_set_1.value)));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset_0);
  iree_event_deinitialize(&ev_unset_1);
  iree_event_deinitialize(&ev_set_0);
  iree_event_deinitialize(&ev_set_1);
}

// Tests iree_wait_any with timeouts (deadline_ns = non-zero).
TEST(WaitSet, WaitAnyTimeout) {
  iree_event_t ev_unset_0, ev_unset_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_1));
  iree_event_t ev_set_0, ev_set_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set_1));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  iree_wait_handle_t empty_handle;
  memset(&empty_handle, 0, sizeof(empty_handle));

  // Timeouts when empty should never block.
  iree_wait_set_clear(wait_set);
  iree_wait_handle_t wake_handle;
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, iree_time_now() + kShortTimeoutNS, &wake_handle));
  EXPECT_EQ(0, memcmp(&empty_handle, &wake_handle, sizeof(empty_handle)));

  // Timeouts with only unset handles should block (and then expire).
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));
  constexpr iree_duration_t kShortTimeoutNS = 1000000ull;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_any(wait_set, iree_time_now() + kShortTimeoutNS, &wake_handle));
  EXPECT_EQ(0, memcmp(&empty_handle, &wake_handle, sizeof(empty_handle)));

  // Timeouts with only set handles should return immediately and have one of
  // the set handles as the wake handle.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_1));
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, iree_time_now() + kShortTimeoutNS, &wake_handle));
  EXPECT_TRUE(
      0 ==
          memcmp(&ev_set_0.value, &wake_handle.value, sizeof(ev_set_0.value)) ||
      0 == memcmp(&ev_set_1.value, &wake_handle.value, sizeof(ev_set_1.value)));

  // Timeouts with mixed set/unset should return immediately and have one of the
  // set handles as the wake handle.
  iree_wait_set_clear(wait_set);
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set_1));
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, iree_time_now() + kShortTimeoutNS, &wake_handle));
  EXPECT_TRUE(
      0 ==
          memcmp(&ev_set_0.value, &wake_handle.value, sizeof(ev_set_0.value)) ||
      0 == memcmp(&ev_set_1.value, &wake_handle.value, sizeof(ev_set_1.value)));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset_0);
  iree_event_deinitialize(&ev_unset_1);
  iree_event_deinitialize(&ev_set_0);
  iree_event_deinitialize(&ev_set_1);
}

// Tests iree_wait_any when blocking (deadline_ns = IREE_TIME_INFINITE_FUTURE).
TEST(WaitSet, WaitAnyBlocking) {
  iree_event_t thread_to_main;
  IREE_ASSERT_OK(
      iree_event_initialize(/*initial_state=*/false, &thread_to_main));
  iree_event_t ev_unset_0, ev_unset_1;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_1));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  // Throw in some unset handles so that we are multi-waiting for just the
  // thread_to_main event to be set.
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));

  // Wait forever (no timeout).
  // We approximate that by forking off a thread to signal our local event. We
  // can assume that a moderate wait is enough to verify the forever behavior as
  // otherwise we are probably just messing up the math and will timeout.
  std::thread thread([&]() {
    // Notify the caller thread after sleeping (to ensure it's not polling).
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    iree_event_set(&thread_to_main);
  });
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, thread_to_main));
  iree_wait_handle_t wake_handle;
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_FUTURE, &wake_handle));
  EXPECT_EQ(0, memcmp(&thread_to_main.value, &wake_handle.value,
                      sizeof(thread_to_main.value)));

  thread.join();
  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&thread_to_main);
  iree_event_deinitialize(&ev_unset_0);
  iree_event_deinitialize(&ev_unset_1);
}

// Tests that an iree_wait_any followed by an iree_wait_set_erase properly
// chooses the right handle to erase.
TEST(WaitSet, WaitAnyErase) {
  iree_event_t ev_unset_0, ev_unset_1;
  iree_event_t ev_set;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_0));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset_1));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_0));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset_1));

  // Wait should succeed immediately because ev_set is set (and our wake handle
  // should be ev_set).
  iree_wait_handle_t wake_handle;
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0, memcmp(&ev_set.value, &wake_handle.value, sizeof(ev_set.value)));

  // Erase the woken handle.
  // NOTE: to get the behavior we want to test we must pass wake_handle here and
  // not the ev_set value.
  iree_wait_set_erase(wait_set, wake_handle);

  // Try to wait again; this time we should timeout because only ev_unset_*
  // remains in the set.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset_0);
  iree_event_deinitialize(&ev_unset_1);
  iree_event_deinitialize(&ev_set);
}

// Tests that an iree_wait_any followed by an iree_wait_set_erase properly
// chooses the right handle to erase (the tail one).
TEST(WaitSet, WaitAnyEraseTail) {
  iree_event_t ev_unset, ev_set;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));

  // Wait should succeed immediately because ev_set is set (and our wake handle
  // should be ev_set).
  iree_wait_handle_t wake_handle;
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0, memcmp(&ev_set.value, &wake_handle.value, sizeof(ev_set.value)));

  // Erase the woken handle.
  // NOTE: to get the behavior we want to test we must pass wake_handle here and
  // not the ev_set value.
  iree_wait_set_erase(wait_set, wake_handle);

  // Try to wait again; this time we should timeout because only ev_unset
  // remains in the set.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset);
  iree_event_deinitialize(&ev_set);
}

// Tests that an iree_wait_any followed by an iree_wait_set_erase without using
// the wake_handle still erases the correct handle.
TEST(WaitSet, WaitAnyEraseSplit) {
  iree_event_t ev_unset, ev_set;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev_unset));
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/true, &ev_set));
  iree_wait_set_t* wait_set = NULL;
  IREE_ASSERT_OK(
      iree_wait_set_allocate(128, iree_allocator_system(), &wait_set));

  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_unset));
  IREE_ASSERT_OK(iree_wait_set_insert(wait_set, ev_set));

  // Wait should succeed immediately because ev_set is set (and our wake handle
  // should be ev_set).
  iree_wait_handle_t wake_handle;
  IREE_ASSERT_OK(
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));
  EXPECT_EQ(0, memcmp(&ev_set.value, &wake_handle.value, sizeof(ev_set.value)));

  // Erase the woken handle *WITHOUT* using the wake_handle.
  iree_wait_set_erase(wait_set, ev_set);

  // Try to wait again; this time we should timeout because only ev_unset
  // remains in the set.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_wait_any(wait_set, IREE_TIME_INFINITE_PAST, &wake_handle));

  iree_wait_set_free(wait_set);
  iree_event_deinitialize(&ev_unset);
  iree_event_deinitialize(&ev_set);
}

// Tests iree_wait_one when polling (deadline_ns = IREE_TIME_INFINITE_PAST).
TEST(WaitSet, WaitOnePolling) {
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
TEST(WaitSet, WaitOneTimeout) {
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
TEST(WaitSet, WaitOneBlocking) {
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
