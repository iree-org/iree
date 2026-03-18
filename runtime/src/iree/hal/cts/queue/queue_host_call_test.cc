// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

class QueueHostCallTest : public CtsTestBase<> {};

// Enqueues a host call on a wait condition that will not be satisfied until
// after the enqueue request completes.
TEST_P(QueueHostCallTest, EnqueueBeforeSignal) {
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
        for (size_t i = 0; i < 4; ++i) state->args[i].store(args[i]);
        return iree_ok_status();
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  EXPECT_EQ(state.did_call, 0);

  // Signal the wait semaphore from a background thread. The sleep gives the
  // main thread time to enter the (potentially blocking) queue_host_call
  // before the signal fires — necessary for synchronous backends where the
  // call blocks inline waiting for the wait semaphore.
  std::thread waker([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));
  });

  uint64_t args[4] = {10, 20, 30, UINT64_MAX};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_EQ(state.did_call, 1);
  EXPECT_EQ(state.args[0], args[0]);
  EXPECT_EQ(state.args[1], args[1]);
  EXPECT_EQ(state.args[2], args[2]);
  EXPECT_EQ(state.args[3], args[3]);

  waker.join();
}

// Tests that a host call with no wait semaphores gets called ASAP.
TEST_P(QueueHostCallTest, NoWaitSemaphores) {
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
        for (size_t i = 0; i < 4; ++i) state->args[i].store(args[i]);
        return iree_ok_status();
      },
      &state);

  SemaphoreList wait_semaphore_list;
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  uint64_t args[4] = {100, 200, 300, 400};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_EQ(state.did_call, 1);
  EXPECT_EQ(state.args[0], args[0]);
  EXPECT_EQ(state.args[1], args[1]);
  EXPECT_EQ(state.args[2], args[2]);
  EXPECT_EQ(state.args[3], args[3]);
}

// Tests that NON_BLOCKING flag causes signal_semaphore_list to be omitted
// from the callback context.
TEST_P(QueueHostCallTest, NonBlockingFlag) {
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
        state->received_semaphores =
            !iree_hal_semaphore_list_is_empty(context->signal_semaphore_list);
        IREE_EXPECT_OK(
            iree_hal_semaphore_list_signal(state->sideband_semaphore_list));
        return iree_ok_status();
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));

  uint64_t args[4] = {1, 2, 3, 4};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING));

  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(state.sideband_semaphore_list,
                                              iree_make_timeout_ms(5000),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_EQ(state.did_call, 1);
  EXPECT_FALSE(state.received_semaphores)
      << "Callback should not receive semaphores with NON_BLOCKING flag";
}

// Tests async callback that clones signal_semaphore_list and signals from a
// thread.
TEST_P(QueueHostCallTest, AsyncCallback) {
  IREE_TRACE_SCOPE();

  struct state_t {
    std::atomic<int> did_call;
    // Atomic because the assignment happens after the std::thread constructor
    // (which calls pthread_create), so the happens-before from thread creation
    // does not cover the pointer store. Without atomic, TSAN correctly flags
    // the main thread's post-wait read as a race with the callback's write.
    std::atomic<std::thread*> signal_thread;
    SemaphoreList* cloned_list;
    std::atomic<bool> thread_started;
    std::atomic<bool> thread_completed;
  } state = {0, {nullptr}, nullptr, false, false};
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t* context) {
        auto* state = (state_t*)user_data;
        ++state->did_call;

        auto& list = context->signal_semaphore_list;
        if (list.count > 0) {
          state->cloned_list = new SemaphoreList(list);
          state->signal_thread = new std::thread([state]() {
            IREE_TRACE_SCOPE_NAMED("signal_thread");
            state->thread_started = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            state->thread_completed = true;
            iree_hal_semaphore_list_signal(*state->cloned_list);
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

  IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));

  uint64_t args[4] = {5, 6, 7, 8};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_EQ(state.did_call, 1);
  EXPECT_TRUE(state.thread_started);
  EXPECT_TRUE(state.thread_completed);

  std::thread* signal_thread = state.signal_thread.load();
  if (signal_thread) {
    signal_thread->join();
    delete signal_thread;
  }
}

// Tests that a callback returning an error signals semaphores with error state.
TEST_P(QueueHostCallTest, CallbackReturnsError) {
  IREE_TRACE_SCOPE();

  struct state_t {
    std::atomic<int> did_call;
  } state = {0};
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t* context) {
        auto* state = (state_t*)user_data;
        ++state->did_call;
        return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "test error from callback");
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0, 0}, {1, 2});

  IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));

  uint64_t args[4] = {9, 10, 11, 12};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  // multi_wait returns the actual failure code from the first failed semaphore.
  EXPECT_THAT(Status(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                                  iree_make_timeout_ms(5000),
                                                  IREE_ASYNC_WAIT_FLAG_NONE)),
              StatusIs(StatusCode::kPermissionDenied));

  // iree_hal_semaphore_list_fail iterates semaphores non-atomically: it fails
  // each semaphore and triggers its timepoint notifications before moving to
  // the next. The list wait above returns as soon as the first semaphore is
  // failed, so we must wait on each semaphore individually to synchronize
  // with the completion of the full failure iteration before querying state.
  for (size_t i = 0; i < signal_semaphore_list.semaphores.size(); ++i) {
    EXPECT_THAT(Status(iree_hal_semaphore_wait(
                    signal_semaphore_list.semaphores[i],
                    signal_semaphore_list.payload_values[i],
                    iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE)),
                StatusIs(StatusCode::kPermissionDenied));
  }

  // All signal semaphores must now be in failed state with the original
  // callback error code.
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
TEST_P(QueueHostCallTest, CallbackReturnsErrorAfterWait) {
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
        return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "test error after waiting");
      },
      &state);

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0, 0}, {1, 2});

  EXPECT_EQ(state.did_call, 0);
  EXPECT_FALSE(state.wait_completed);

  // Same pattern as EnqueueBeforeSignal: sleep to let the main thread enter
  // the blocking queue_host_call before signaling the wait semaphore.
  std::thread waker([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));
  });

  uint64_t args[4] = {13, 14, 15, 16};
  IREE_EXPECT_OK(iree_hal_device_queue_host_call(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  EXPECT_THAT(Status(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                                  iree_make_timeout_ms(5000),
                                                  IREE_ASYNC_WAIT_FLAG_NONE)),
              StatusIs(StatusCode::kPermissionDenied));

  EXPECT_EQ(state.did_call, 1);
  EXPECT_TRUE(state.wait_completed);

  // See CallbackReturnsError for why per-semaphore waits are needed.
  for (size_t i = 0; i < signal_semaphore_list.semaphores.size(); ++i) {
    EXPECT_THAT(Status(iree_hal_semaphore_wait(
                    signal_semaphore_list.semaphores[i],
                    signal_semaphore_list.payload_values[i],
                    iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE)),
                StatusIs(StatusCode::kPermissionDenied));
  }

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

CTS_REGISTER_TEST_SUITE(QueueHostCallTest);

}  // namespace iree::hal::cts
