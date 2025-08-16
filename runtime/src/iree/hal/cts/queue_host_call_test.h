// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_QUEUE_HOST_CALL_TEST_H_
#define IREE_HAL_CTS_QUEUE_HOST_CALL_TEST_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;
using ::testing::ContainerEq;

struct SemaphoreList {
  SemaphoreList() = default;
  SemaphoreList(iree_hal_device_t* device, std::vector<uint64_t> initial_values,
                std::vector<uint64_t> desired_values) {
    for (size_t i = 0; i < initial_values.size(); ++i) {
      iree_hal_semaphore_t* semaphore = NULL;
      IREE_EXPECT_OK(iree_hal_semaphore_create(
          device, initial_values[i], IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
      semaphores.push_back(semaphore);
    }
    payload_values = desired_values;
    assert(semaphores.size() == payload_values.size());
  }

  // Copy constructor that retains semaphores.
  SemaphoreList(const iree_hal_semaphore_list_t& list) {
    semaphores.reserve(list.count);
    payload_values.reserve(list.count);
    for (iree_host_size_t i = 0; i < list.count; ++i) {
      semaphores.push_back(list.semaphores[i]);
      payload_values.push_back(list.payload_values[i]);
    }
    // Retain all semaphores.
    iree_hal_semaphore_list_retain(*this);
  }

  // Copy constructor from another SemaphoreList.
  SemaphoreList(const SemaphoreList& other) {
    semaphores = other.semaphores;
    payload_values = other.payload_values;
    // Retain all semaphores.
    iree_hal_semaphore_list_retain(*this);
  }

  // Copy assignment.
  SemaphoreList& operator=(const SemaphoreList& other) {
    if (this != &other) {
      // Release old semaphores.
      iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
      // Copy new ones.
      semaphores = other.semaphores;
      payload_values = other.payload_values;
      // Retain new semaphores.
      iree_hal_semaphore_list_retain(*this);
    }
    return *this;
  }

  SemaphoreList(SemaphoreList&& other) noexcept
      : semaphores(std::move(other.semaphores)),
        payload_values(std::move(other.payload_values)) {
    other.semaphores.clear();
    other.payload_values.clear();
  }

  SemaphoreList& operator=(SemaphoreList&& other) noexcept {
    if (this != &other) {
      iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
      semaphores = std::move(other.semaphores);
      payload_values = std::move(other.payload_values);
      other.semaphores.clear();
      other.payload_values.clear();
    }
    return *this;
  }

  ~SemaphoreList() {
    iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
  }

  operator iree_hal_semaphore_list_t() {
    iree_hal_semaphore_list_t list;
    list.count = semaphores.size();
    list.semaphores = semaphores.data();
    list.payload_values = payload_values.data();
    return list;
  }

  std::vector<iree_hal_semaphore_t*> semaphores;
  std::vector<uint64_t> payload_values;
};

class QueueHostCallTest : public CTSTestBase<> {};

// Enqueues a host call on a wait condition that will not be satisfied until
// after the enqueue request completes. This ensures that host calls properly
// park themselves and get rescheduled as their dependencies resolve.
TEST_F(QueueHostCallTest, EnqueueBeforeSignal) {
  IREE_TRACE_SCOPE();

  struct state_t {
    std::atomic<int> did_call;
    std::atomic<uint64_t> args[4];
  } state = {0};
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t* context) {
        auto* state = (state_t*)user_data;
        ++state->did_call;
        memcpy(state->args, args, sizeof(state->args));
        return iree_ok_status();
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  EXPECT_EQ(state.did_call, 0);

  // NOTE: we do this before issuing the host call so we can still function in
  // synchronous contexts.
  std::thread waker([&]() {
    EXPECT_EQ(state.did_call, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(state.did_call, 0);
    IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));
  });

  uint64_t args[4] = {10, 20, 30, UINT64_MAX};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000)));

  EXPECT_EQ(state.did_call, 1);
  EXPECT_EQ(state.args[0], args[0]);
  EXPECT_EQ(state.args[1], args[1]);
  EXPECT_EQ(state.args[2], args[2]);
  EXPECT_EQ(state.args[3], args[3]);

  waker.join();
}

// Tests that a host call with no wait semaphores gets called ASAP.
// The call may not be immediate but should execute without waiting.
TEST_F(QueueHostCallTest, NoWaitSemaphores) {
  IREE_TRACE_SCOPE();

  struct state_t {
    std::atomic<int> did_call;
    std::atomic<uint64_t> args[4];
  } state = {0};
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t* context) {
        auto* state = (state_t*)user_data;
        ++state->did_call;
        memcpy(state->args, args, sizeof(state->args));
        return iree_ok_status();
      },
      &state);

  // Empty wait list - should execute ASAP.
  SemaphoreList wait_semaphore_list;
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  uint64_t args[4] = {100, 200, 300, 400};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  // Wait for completion - the host call should complete quickly.
  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000)));

  EXPECT_EQ(state.did_call, 1);
  EXPECT_EQ(state.args[0], args[0]);
  EXPECT_EQ(state.args[1], args[1]);
  EXPECT_EQ(state.args[2], args[2]);
  EXPECT_EQ(state.args[3], args[3]);
}

// Tests that NON_BLOCKING flag causes signal_semaphore_list to be omitted.
// The callback should not receive the semaphores and they should be signaled
// before the callback returns.
TEST_F(QueueHostCallTest, NonBlockingFlag) {
  IREE_TRACE_SCOPE();

  struct state_t {
    std::atomic<int> did_call;
    std::atomic<bool> received_semaphores;
    SemaphoreList sideband_semaphore_list;
  } state = {0, false, {device_, {0}, {1}}};
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t* context) {
        IREE_TRACE_SCOPE_NAMED("callback");
        auto* state = (state_t*)user_data;
        ++state->did_call;
        // With NON_BLOCKING flag, signal_semaphore_list should be empty.
        state->received_semaphores =
            !iree_hal_semaphore_list_is_empty(context->signal_semaphore_list);
        // The enqueuing thread should have been signaled already, but we need
        // to make sure it made it to at least here for the test to not be
        // flakey (since in NON_BLOCKING we may still not have executed this
        // callback by the time any waiters have executed).
        IREE_EXPECT_OK(
            iree_hal_semaphore_list_signal(state->sideband_semaphore_list));
        return iree_ok_status();
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  // Signal the wait semaphore so the call can proceed.
  IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));

  uint64_t args[4] = {1, 2, 3, 4};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING));

  // Wait for the signal semaphores - they should be signaled prior to the
  // callback having executed, but it's hard to verify that. Instead we just
  // wait for the signal and then wait again to join the thread.
  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000)));
  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(state.sideband_semaphore_list,
                                              iree_make_timeout_ms(5000)));

  EXPECT_EQ(state.did_call, 1);
  EXPECT_FALSE(state.received_semaphores)
      << "Callback should not receive semaphores with NON_BLOCKING flag";
}

// Tests async callback that clones signal_semaphore_list and signals from a
// thread. The semaphores should not be signaled until the spawned thread runs.
TEST_F(QueueHostCallTest, AsyncCallback) {
  IREE_TRACE_SCOPE();

  struct state_t {
    std::atomic<int> did_call;
    std::thread* signal_thread;
    SemaphoreList* cloned_list;
    std::atomic<bool> thread_started;
    std::atomic<bool> thread_completed;
  } state = {0, nullptr, nullptr, false, false};
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t* context) {
        auto* state = (state_t*)user_data;
        ++state->did_call;

        // Clone the signal semaphore list for async completion using the copy
        // constructor.
        auto& list = context->signal_semaphore_list;

        if (list.count > 0) {
          // Use the SemaphoreList copy constructor that retains semaphores.
          state->cloned_list = new SemaphoreList(list);

          // Launch thread to signal after a delay.
          state->signal_thread = new std::thread([state]() {
            IREE_TRACE_SCOPE_NAMED("signal_thread");
            state->thread_started = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            state->thread_completed = true;

            // Signal all semaphores.
            iree_hal_semaphore_list_signal(*state->cloned_list);

            // Clean up the cloned list.
            delete state->cloned_list;
            state->cloned_list = nullptr;
          });
        }

        // Notify that we are an asynchronous operation.
        return iree_status_from_code(IREE_STATUS_DEFERRED);
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0, 0}, {1, 2});

  // Signal wait semaphore to let the call proceed.
  IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));

  uint64_t args[4] = {5, 6, 7, 8};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  // Now wait for the semaphores to be signaled by the thread.
  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000)));

  EXPECT_EQ(state.did_call, 1);
  EXPECT_TRUE(state.thread_started);
  EXPECT_TRUE(state.thread_completed);

  // Clean up thread.
  if (state.signal_thread) {
    state.signal_thread->join();
    delete state.signal_thread;
  }
}

// Tests that a callback returning an error signals semaphores with error state.
TEST_F(QueueHostCallTest, CallbackReturnsError) {
  IREE_TRACE_SCOPE();

  struct state_t {
    std::atomic<int> did_call;
  } state = {0};
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t* context) {
        auto* state = (state_t*)user_data;
        ++state->did_call;
        // Return an error - this should cause signal semaphores to fail.
        return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "test error from callback");
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0, 0}, {1, 2});

  // Signal wait semaphore to let the call proceed.
  IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));

  uint64_t args[4] = {9, 10, 11, 12};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  // Wait for semaphores - this should fail because the callback returned an
  // error.
  EXPECT_THAT(Status(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                                  iree_make_timeout_ms(5000))),
              StatusIs(StatusCode::kAborted));

  // Query individual semaphores to verify they're in error state.
  uint64_t value0 = 0;
  EXPECT_THAT(Status(iree_hal_semaphore_query(
                  signal_semaphore_list.semaphores[0], &value0)),
              StatusIs(StatusCode::kPermissionDenied));
  uint64_t value1 = 0;
  EXPECT_THAT(Status(iree_hal_semaphore_query(
                  signal_semaphore_list.semaphores[1], &value1)),
              StatusIs(StatusCode::kPermissionDenied));

  EXPECT_EQ(state.did_call, 1);
}

// Tests that a callback returning an error after waiting for dependencies
// properly signals semaphores with error state.
TEST_F(QueueHostCallTest, CallbackReturnsErrorAfterWait) {
  IREE_TRACE_SCOPE();

  struct state_t {
    std::atomic<int> did_call;
    std::atomic<bool> wait_completed;
  } state = {0, false};
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t* context) {
        auto* state = (state_t*)user_data;
        ++state->did_call;
        state->wait_completed = true;
        // Return an error which should cause signal semaphores to fail.
        return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "test error after waiting");
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0, 0}, {1, 2});

  // Verify the callback hasn't been called yet.
  EXPECT_EQ(state.did_call, 0);
  EXPECT_FALSE(state.wait_completed);

  // Start a thread that will signal the wait semaphore after a delay.
  // NOTE: we do this before issuing the host call so we can still function in
  // synchronous contexts.
  std::thread waker([&]() {
    EXPECT_EQ(state.did_call, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(state.did_call, 0);
    EXPECT_FALSE(state.wait_completed);
    // Signal the wait semaphore to unblock the host call.
    IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));
  });

  uint64_t args[4] = {13, 14, 15, 16};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  // Wait for signal semaphores - this should fail because the callback
  // returned an error after waiting.
  EXPECT_THAT(Status(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                                  iree_make_timeout_ms(5000))),
              StatusIs(StatusCode::kAborted));

  // Verify the callback was called after waiting.
  EXPECT_EQ(state.did_call, 1);
  EXPECT_TRUE(state.wait_completed);

  // Query individual semaphores to verify they're in error state.
  uint64_t value0 = 0;
  EXPECT_THAT(Status(iree_hal_semaphore_query(
                  signal_semaphore_list.semaphores[0], &value0)),
              StatusIs(StatusCode::kPermissionDenied));
  uint64_t value1 = 0;
  EXPECT_THAT(Status(iree_hal_semaphore_query(
                  signal_semaphore_list.semaphores[1], &value1)),
              StatusIs(StatusCode::kPermissionDenied));

  waker.join();
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_QUEUE_HOST_CALL_TEST_H_
