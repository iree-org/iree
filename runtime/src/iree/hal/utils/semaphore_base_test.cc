// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/semaphore_base.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace {

using ::iree::testing::status::IsOkAndHolds;
using ::iree::testing::status::StatusIs;
using ::testing::Eq;

namespace {
extern const iree_hal_semaphore_vtable_t test_semaphore_vtable;
}  // namespace

struct TestSemaphore {
  iree_hal_semaphore_t base;
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;
  uint64_t current_value;
  iree_status_t failure_status;
  iree_notification_t notification;

  static TestSemaphore* Create(uint64_t initial_value,
                               iree_allocator_t host_allocator) {
    TestSemaphore* semaphore = nullptr;
    IREE_CHECK_OK(iree_allocator_malloc(host_allocator, sizeof(*semaphore),
                                        (void**)&semaphore));
    iree_hal_semaphore_initialize(&test_semaphore_vtable, &semaphore->base);
    semaphore->host_allocator = host_allocator;
    iree_slim_mutex_initialize(&semaphore->mutex);
    iree_notification_initialize(&semaphore->notification);
    return semaphore;
  }

  static TestSemaphore* Cast(iree_hal_semaphore_t* base_semaphore) {
    return reinterpret_cast<TestSemaphore*>(base_semaphore);
  }

  static void Destroy(iree_hal_semaphore_t* base_semaphore) {
    auto* semaphore = Cast(base_semaphore);
    iree_status_ignore(semaphore->failure_status);
    iree_notification_deinitialize(&semaphore->notification);
    iree_slim_mutex_deinitialize(&semaphore->mutex);
    iree_hal_semaphore_deinitialize(&semaphore->base);
    iree_allocator_free(semaphore->host_allocator, semaphore);
  }

  static iree_status_t Query(iree_hal_semaphore_t* base_semaphore,
                             uint64_t* out_value) {
    auto* semaphore = Cast(base_semaphore);
    iree_slim_mutex_lock(&semaphore->mutex);
    iree_status_t status = iree_status_clone(semaphore->failure_status);
    *out_value = semaphore->current_value;
    iree_slim_mutex_unlock(&semaphore->mutex);
    return status;
  }

  static iree_status_t Signal(iree_hal_semaphore_t* base_semaphore,
                              uint64_t new_value) {
    auto* semaphore = Cast(base_semaphore);
    iree_slim_mutex_lock(&semaphore->mutex);
    iree_status_t status = iree_ok_status();
    semaphore->current_value = new_value;
    iree_slim_mutex_unlock(&semaphore->mutex);
    iree_hal_semaphore_notify(&semaphore->base, new_value, IREE_STATUS_OK);
    iree_notification_post(&semaphore->notification, IREE_ALL_WAITERS);
    return status;
  }

  static void Fail(iree_hal_semaphore_t* base_semaphore, iree_status_t status) {
    auto* semaphore = Cast(base_semaphore);
    iree_slim_mutex_lock(&semaphore->mutex);
    iree_status_ignore(semaphore->failure_status);
    semaphore->failure_status = status;
    const iree_status_code_t status_code = iree_status_code(status);
    iree_slim_mutex_unlock(&semaphore->mutex);
    iree_hal_semaphore_notify(&semaphore->base, 0, status_code);
    iree_notification_post(&semaphore->notification, IREE_ALL_WAITERS);
  }

  static iree_status_t Wait(iree_hal_semaphore_t* base_semaphore,
                            uint64_t value, iree_timeout_t timeout) {
    auto* semaphore = Cast(base_semaphore);
    struct notify_state_t {
      TestSemaphore* semaphore;
      uint64_t value;
    } notify_state = {semaphore, value};
    iree_notification_await(
        &semaphore->notification,
        [](void* user_data) -> bool {
          auto* state = reinterpret_cast<notify_state_t*>(user_data);
          iree_slim_mutex_lock(&state->semaphore->mutex);
          bool is_signaled =
              state->semaphore->current_value >= state->value ||
              !iree_status_is_ok(state->semaphore->failure_status);
          iree_slim_mutex_unlock(&state->semaphore->mutex);
          return is_signaled;
        },
        (void*)&notify_state, timeout);
    return iree_ok_status();
  }

  constexpr operator iree_hal_semaphore_t*() noexcept { return &base; }
};

namespace {
const iree_hal_semaphore_vtable_t test_semaphore_vtable = {
    /*.destroy=*/TestSemaphore::Destroy,
    /*.query=*/TestSemaphore::Query,
    /*.signal=*/TestSemaphore::Signal,
    /*.fail=*/TestSemaphore::Fail,
    /*.wait=*/TestSemaphore::Wait,
};
}  // namespace

struct CallbackState {
  std::atomic<int> callback_count = {0};
  std::atomic<uint64_t> value = {0ull};
  std::atomic<iree_status_code_t> status_code = {IREE_STATUS_OK};
  std::atomic<iree_status_code_t> callback_result = {IREE_STATUS_OK};
};

static iree_status_t SemaphoreTimepointHandler(void* user_data,
                                               iree_hal_semaphore_t* semaphore,
                                               uint64_t value,
                                               iree_status_code_t status_code) {
  auto* state = reinterpret_cast<CallbackState*>(user_data);
  ++state->callback_count;
  state->value = value;
  state->status_code = status_code;
  return iree_status_from_code(state->callback_result);
}

static iree_hal_semaphore_callback_t MakeCallback(CallbackState* state) {
  return {
      SemaphoreTimepointHandler,
      state,
  };
}

struct TrackingSemaphoreTest : public ::testing::Test {
  // We could check the allocator to ensure all memory is freed if we wanted to
  // reduce the reliance on asan.
  iree_allocator_t host_allocator = iree_allocator_default();

  void SetUp() override {}

  void TearDown() override {}
};

// Tests the lifetime of an unsignaled semaphore.
TEST_F(TrackingSemaphoreTest, Unsignaled) {
  auto* semaphore = TestSemaphore::Create(0ull, host_allocator);
  iree_hal_semaphore_release(*semaphore);
}

// Tests the lifetime of a signaled semaphore with no timepoints.
TEST_F(TrackingSemaphoreTest, Signaled) {
  auto* semaphore = TestSemaphore::Create(0ull, host_allocator);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(*semaphore, 1ull));
  iree_hal_semaphore_release(*semaphore);
}

// Tests acquiring timepoints that are already resolved.
TEST_F(TrackingSemaphoreTest, AcquireResolvedTimepoint) {
  auto* semaphore = TestSemaphore::Create(0ull, host_allocator);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(*semaphore, 2ull));

  CallbackState state;
  iree_hal_semaphore_timepoint_t timepoint;
  iree_hal_semaphore_acquire_timepoint(*semaphore, 1ull,
                                       iree_infinite_timeout(),
                                       MakeCallback(&state), &timepoint);
  ASSERT_EQ(state.callback_count, 0);

  // Callback happens here:
  iree_hal_semaphore_poll(*semaphore);

  ASSERT_EQ(state.callback_count, 1);
  ASSERT_EQ(state.status_code, IREE_STATUS_OK);
  ASSERT_EQ(state.value, 2ull);

  iree_hal_semaphore_release(*semaphore);
}

// Tests acquiring timepoints that are already failed.
TEST_F(TrackingSemaphoreTest, AcquireFailedTimepoint) {
  auto* semaphore = TestSemaphore::Create(0ull, host_allocator);

  iree_hal_semaphore_fail(*semaphore,
                          iree_make_status(IREE_STATUS_DATA_LOSS, "whoops"));

  CallbackState state;
  iree_hal_semaphore_timepoint_t timepoint;
  iree_hal_semaphore_acquire_timepoint(*semaphore, 1ull,
                                       iree_infinite_timeout(),
                                       MakeCallback(&state), &timepoint);
  ASSERT_EQ(state.callback_count, 0);

  // Callback happens here:
  iree_hal_semaphore_poll(*semaphore);

  ASSERT_EQ(state.callback_count, 1);
  ASSERT_EQ(state.status_code, IREE_STATUS_DATA_LOSS);

  iree_hal_semaphore_release(*semaphore);
}

// Tests acquiring timepoints that are resolved synchronously from the host.
TEST_F(TrackingSemaphoreTest, AcquireUnresolvedSyncTimepoint) {
  auto* semaphore = TestSemaphore::Create(0ull, host_allocator);

  CallbackState state;
  iree_hal_semaphore_timepoint_t timepoint;
  iree_hal_semaphore_acquire_timepoint(*semaphore, 1ull,
                                       iree_infinite_timeout(),
                                       MakeCallback(&state), &timepoint);
  ASSERT_EQ(state.callback_count, 0);

  // Callback does not happen as timepoint has not been reached:
  iree_hal_semaphore_poll(*semaphore);
  ASSERT_EQ(state.callback_count, 0);

  // Callback happens here:
  IREE_ASSERT_OK(iree_hal_semaphore_signal(*semaphore, 2ull));

  ASSERT_EQ(state.callback_count, 1);
  ASSERT_EQ(state.status_code, IREE_STATUS_OK);
  ASSERT_EQ(state.value, 2ull);

  iree_hal_semaphore_release(*semaphore);
}

// Tests acquiring timepoints that are rejected synchronously from the host.
TEST_F(TrackingSemaphoreTest, AcquireUnrejectedSyncTimepoint) {
  auto* semaphore = TestSemaphore::Create(0ull, host_allocator);

  CallbackState state;
  iree_hal_semaphore_timepoint_t timepoint;
  iree_hal_semaphore_acquire_timepoint(*semaphore, 1ull,
                                       iree_infinite_timeout(),
                                       MakeCallback(&state), &timepoint);
  ASSERT_EQ(state.callback_count, 0);

  // Callback does not happen as timepoint has not been reached:
  iree_hal_semaphore_poll(*semaphore);
  ASSERT_EQ(state.callback_count, 0);

  // Callback happens here:
  iree_hal_semaphore_fail(*semaphore,
                          iree_make_status(IREE_STATUS_DATA_LOSS, "whoops"));

  ASSERT_EQ(state.callback_count, 1);
  ASSERT_EQ(state.status_code, IREE_STATUS_DATA_LOSS);

  iree_hal_semaphore_release(*semaphore);
}

// Tests acquiring timepoints that are resolved asynchronously from the host.
TEST_F(TrackingSemaphoreTest, AcquireUnresolvedAsyncTimepoint) {
  iree_event_t ev0;
  IREE_ASSERT_OK(iree_event_initialize(/*initial_state=*/false, &ev0));
  auto* semaphore = TestSemaphore::Create(0ull, host_allocator);

  CallbackState state;
  iree_hal_semaphore_timepoint_t timepoint;
  iree_hal_semaphore_acquire_timepoint(*semaphore, 1ull,
                                       iree_infinite_timeout(),
                                       MakeCallback(&state), &timepoint);
  ASSERT_EQ(state.callback_count, 0);

  // Callback does not happen as timepoint has not been reached:
  iree_hal_semaphore_poll(*semaphore);
  ASSERT_EQ(state.callback_count, 0);

  // Thread acting as an external signaler.
  std::thread thread([&]() {
    // Block until the main thread catches up.
    IREE_ASSERT_OK(iree_wait_one(&ev0, IREE_TIME_INFINITE_FUTURE));

    // Callback happens here:
    IREE_ASSERT_OK(iree_hal_semaphore_signal(*semaphore, 2ull));
  });

  // Unblock the thread and have it signal.
  ASSERT_EQ(state.callback_count, 0);
  iree_event_set(&ev0);

  // Wait for the semaphore to be signaled.
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(*semaphore, 1ull, iree_infinite_timeout()));

  // Should have been called back on the thread.
  ASSERT_EQ(state.callback_count, 1);
  ASSERT_EQ(state.status_code, IREE_STATUS_OK);
  ASSERT_EQ(state.value, 2ull);

  thread.join();
  iree_hal_semaphore_release(*semaphore);
  iree_event_deinitialize(&ev0);
}

// Tests cancelling timepoints before they are resolved.
TEST_F(TrackingSemaphoreTest, CancelUnresolvedTimepoint) {
  auto* semaphore = TestSemaphore::Create(0ull, host_allocator);

  CallbackState state;
  iree_hal_semaphore_timepoint_t timepoint;
  iree_hal_semaphore_acquire_timepoint(*semaphore, 1ull,
                                       iree_infinite_timeout(),
                                       MakeCallback(&state), &timepoint);
  ASSERT_EQ(state.callback_count, 0);

  // Callback does not happen as timepoint has not been reached:
  iree_hal_semaphore_poll(*semaphore);
  ASSERT_EQ(state.callback_count, 0);

  // Cancel before signal.
  iree_hal_semaphore_cancel_timepoint(*semaphore, &timepoint);
  ASSERT_EQ(state.callback_count, 0);

  // Callback should not happens here because we cancelled:
  IREE_ASSERT_OK(iree_hal_semaphore_signal(*semaphore, 2ull));
  ASSERT_EQ(state.callback_count, 0);

  iree_hal_semaphore_release(*semaphore);
}

// Tests cancelling timepoints after they are resolved. Should be a no-op.
TEST_F(TrackingSemaphoreTest, CancelResolvedTimepoint) {
  auto* semaphore = TestSemaphore::Create(0ull, host_allocator);

  CallbackState state;
  iree_hal_semaphore_timepoint_t timepoint;
  iree_hal_semaphore_acquire_timepoint(*semaphore, 1ull,
                                       iree_infinite_timeout(),
                                       MakeCallback(&state), &timepoint);
  ASSERT_EQ(state.callback_count, 0);

  // Callback does not happen as timepoint has not been reached:
  iree_hal_semaphore_poll(*semaphore);
  ASSERT_EQ(state.callback_count, 0);

  // Callback happens here:
  IREE_ASSERT_OK(iree_hal_semaphore_signal(*semaphore, 2ull));

  ASSERT_EQ(state.callback_count, 1);
  ASSERT_EQ(state.status_code, IREE_STATUS_OK);
  ASSERT_EQ(state.value, 2ull);

  // Cancel after signal, which should be a no-op.
  iree_hal_semaphore_cancel_timepoint(*semaphore, &timepoint);
  ASSERT_EQ(state.callback_count, 1);

  iree_hal_semaphore_release(*semaphore);
}

}  // namespace
}  // namespace hal
}  // namespace iree
