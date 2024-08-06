// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_SEMAPHORE_TEST_H_
#define IREE_HAL_CTS_SEMAPHORE_TEST_H_

#include <chrono>
#include <cstdint>
#include <thread>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {

using namespace std::chrono_literals;

class SemaphoreTest : public CTSTestBase<> {};

// Tests that a semaphore that is unused properly cleans itself up.
TEST_F(SemaphoreTest, NoOp) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 123ull, &semaphore));

  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(123ull, value);

  iree_hal_semaphore_release(semaphore);
}

// Tests that a semaphore will accept new values as it is signaled.
TEST_F(SemaphoreTest, NormalSignaling) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 2ull, &semaphore));

  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(2ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 3ull));
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(3ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 40ull));
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(40ull, value);

  iree_hal_semaphore_release(semaphore);
}

// Note: Behavior is undefined when signaling with decreasing values, so we
// can't reliably test it across backends. Some backends may return errors,
// while others may accept the new, decreasing, values.

// Tests semaphore failure handling.
TEST_F(SemaphoreTest, Failure) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 2ull, &semaphore));

  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 3ull));
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(3ull, value);

  iree_hal_semaphore_fail(semaphore,
                          iree_status_from_code(IREE_STATUS_UNKNOWN));
  EXPECT_TRUE(
      iree_status_is_unknown(iree_hal_semaphore_query(semaphore, &value)));

  // Signaling again is undefined behavior. Some backends may return a sticky
  // failure status while others may silently process new signal values.

  iree_hal_semaphore_release(semaphore);
}

// Tests waiting on no semaphores.
TEST_F(SemaphoreTest, EmptyWait) {
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, iree_hal_semaphore_list_empty(),
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ALL, iree_hal_semaphore_list_empty(),
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, iree_hal_semaphore_list_empty(),
      iree_make_timeout_ns(IREE_DURATION_INFINITE)));
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ALL, iree_hal_semaphore_list_empty(),
      iree_make_timeout_ns(IREE_DURATION_INFINITE)));
}

// Tests waiting on a semaphore that has already been signaled.
TEST_F(SemaphoreTest, WaitAlreadySignaled) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 2ull, &semaphore));

  // Test both previous and current values.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 1ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 2ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 1ull, iree_make_timeout_ns(IREE_DURATION_INFINITE)));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 2ull, iree_make_timeout_ns(IREE_DURATION_INFINITE)));

  iree_hal_semaphore_release(semaphore);
}

// Tests waiting on a semaphore that has not been signaled.
TEST_F(SemaphoreTest, WaitUnsignaled) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 2ull, &semaphore));

  // NOTE: we don't actually block here because otherwise we'd lock up.
  // Result status is undefined - some backends may return DeadlineExceededError
  // while others may return success.
  IREE_IGNORE_ERROR(iree_hal_semaphore_wait(
      semaphore, 3ull, iree_make_deadline(IREE_TIME_INFINITE_PAST)));

  iree_hal_semaphore_release(semaphore);
}

// Tests waiting on a semaphore that has signals past the desired value.
TEST_F(SemaphoreTest, WaitLaterSignaledBeyond) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 2ull, &semaphore));

  std::thread thread([&]() {
    // Wait for a short period before signaling.
    std::this_thread::sleep_for(100ms);
    // Signal beyond the desired value.
    IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 10ull));
  });

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 3ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  thread.join();

  iree_hal_semaphore_release(semaphore);
}

// Waiting on a failed semaphore is undefined behavior. Some backends may
// return UnknownError while others may succeed.

// Tests IREE_HAL_WAIT_MODE_ALL when not all are signaled.
TEST_F(SemaphoreTest, WaitAllButNotAllSignaled) {
  iree_hal_semaphore_t* semaphore_a = NULL;
  iree_hal_semaphore_t* semaphore_b = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &semaphore_a));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 1ull, &semaphore_b));

  iree_hal_semaphore_list_t semaphore_list;
  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  semaphore_list.count = IREE_ARRAYSIZE(semaphore_ptrs);
  semaphore_list.semaphores = semaphore_ptrs;
  uint64_t payload_values[] = {1ull, 1ull};
  semaphore_list.payload_values = payload_values;

  // NOTE: we don't actually block here because otherwise we'd lock up.
  // Result status is undefined - some backends may return DeadlineExceededError
  // while others may return success.
  IREE_IGNORE_ERROR(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ALL, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_PAST)));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests IREE_HAL_WAIT_MODE_ALL when all are signaled.
TEST_F(SemaphoreTest, WaitAllAndAllSignaled) {
  iree_hal_semaphore_t* semaphore_a = NULL;
  iree_hal_semaphore_t* semaphore_b = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 1ull, &semaphore_a));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 1ull, &semaphore_b));

  iree_hal_semaphore_list_t semaphore_list;
  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  semaphore_list.count = IREE_ARRAYSIZE(semaphore_ptrs);
  semaphore_list.semaphores = semaphore_ptrs;
  uint64_t payload_values[] = {1ull, 1ull};
  semaphore_list.payload_values = payload_values;

  // NOTE: we don't actually block here because otherwise we'd lock up.
  // Result status is undefined - some backends may return DeadlineExceededError
  // while others may return success.
  IREE_IGNORE_ERROR(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ALL, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests IREE_HAL_WAIT_MODE_ANY.
TEST_F(SemaphoreTest, WaitAnyAlreadySignaled) {
  iree_hal_semaphore_t* semaphore_a = NULL;
  iree_hal_semaphore_t* semaphore_b = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &semaphore_a));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 1ull, &semaphore_b));

  iree_hal_semaphore_list_t semaphore_list;
  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  semaphore_list.count = IREE_ARRAYSIZE(semaphore_ptrs);
  semaphore_list.semaphores = semaphore_ptrs;
  uint64_t payload_values[] = {1ull, 1ull};
  semaphore_list.payload_values = payload_values;

  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

TEST_F(SemaphoreTest, WaitAnyLaterSignaled) {
  iree_hal_semaphore_t* semaphore_a = NULL;
  iree_hal_semaphore_t* semaphore_b = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &semaphore_a));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &semaphore_b));

  iree_hal_semaphore_list_t semaphore_list;
  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  semaphore_list.count = IREE_ARRAYSIZE(semaphore_ptrs);
  semaphore_list.semaphores = semaphore_ptrs;
  uint64_t payload_values[] = {1ull, 1ull};
  semaphore_list.payload_values = payload_values;

  std::thread thread([&]() {
    // Wait for a short period before signaling.
    std::this_thread::sleep_for(100ms);
    IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_b, 1ull));
  });

  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  thread.join();

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests threading behavior by ping-ponging between the test main thread and
// a little thread.
TEST_F(SemaphoreTest, PingPong) {
  iree_hal_semaphore_t* a2b = NULL;
  iree_hal_semaphore_t* b2a = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &a2b));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &b2a));
  std::thread thread([&]() {
    // Should advance right past this because the value is already set.
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        a2b, 0ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(b2a, 1ull));
    // Jump ahead (blocking at first).
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        a2b, 4ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  });
  // Block until thread signals.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      b2a, 1ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(a2b, 4ull));
  thread.join();

  iree_hal_semaphore_release(a2b);
  iree_hal_semaphore_release(b2a);
}

// Waiting the same value multiple times.
TEST_F(SemaphoreTest, WaitOnTheSameValueMultipleTimes) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();
  std::thread thread(
      [&]() { IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 1)); });

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore, 1);

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore, 1);

  thread.join();

  iree_hal_semaphore_release(semaphore);
}

// Waiting for a finite amount of time.
TEST_F(SemaphoreTest, WaitForFiniteTime) {
  auto generic_test_fn = [this](auto wait_fn) {
    iree_hal_semaphore_t* semaphore = this->CreateSemaphore();

    // Wait before signaling and make sure the semaphore value has not changed.
    IREE_ASSERT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED, wait_fn(semaphore));
    CheckSemaphoreValue(semaphore, 0);

    std::thread signaling_thread(
        [&]() { IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 1)); });

    // The semaphore must advance at some point.
    while (true) {
      iree_status_t status = wait_fn(semaphore);
      // The semaphore either timed out or has advanced.
      IREE_ASSERT_TRUE(iree_status_is_ok(status) ||
                       iree_status_is_deadline_exceeded(status));
      if (iree_status_is_deadline_exceeded(status)) continue;
      CheckSemaphoreValue(semaphore, 1);
      break;
    }

    signaling_thread.join();
    iree_hal_semaphore_release(semaphore);
  };

  // Immediate timeout.
  generic_test_fn([](iree_hal_semaphore_t* semaphore) {
    return iree_hal_semaphore_wait(semaphore, 1, iree_immediate_timeout());
  });

  // Absolute timeout.
  generic_test_fn([](iree_hal_semaphore_t* semaphore) {
    return iree_hal_semaphore_wait(semaphore, 1,
                                   iree_make_deadline(iree_time_now() + 1));
  });

  // Relative timeout.
  generic_test_fn([](iree_hal_semaphore_t* semaphore) {
    return iree_hal_semaphore_wait(semaphore, 1, iree_make_timeout_ns(1));
  });
}

// Wait on all semaphores on multiple places simultaneously.
TEST_F(SemaphoreTest, SimultaneousMultiWaitAll) {
  iree_hal_semaphore_t* semaphore1 = this->CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = this->CreateSemaphore();

  iree_hal_semaphore_t* semaphore_array[] = {semaphore1, semaphore2};
  uint64_t payload_array[] = {1, 1};
  iree_hal_semaphore_list_t semaphore_list = {
      IREE_ARRAYSIZE(semaphore_array),
      semaphore_array,
      payload_array,
  };

  std::thread wait_thread1([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        semaphore_list, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  });

  std::thread wait_thread2([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        semaphore_list, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  });

  std::thread signal_thread([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_list_signal(semaphore_list));
  });

  wait_thread1.join();
  wait_thread2.join();
  signal_thread.join();

  CheckSemaphoreValue(semaphore1, 1);
  CheckSemaphoreValue(semaphore2, 1);

  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
}

// Wait on a semaphore that is then failed.
TEST_F(SemaphoreTest, FailThenWait) {
  iree_hal_semaphore_t* semaphore = this->CreateSemaphore();

  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "FailThenWait test.");
  iree_hal_semaphore_fail(semaphore, iree_status_clone(status));

  iree_status_t wait_status = iree_hal_semaphore_wait(
      semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE));
  EXPECT_EQ(iree_status_code(wait_status), IREE_STATUS_ABORTED);
  uint64_t value = 1234;
  iree_status_t query_status = iree_hal_semaphore_query(semaphore, &value);
  EXPECT_EQ(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  CheckStatusContains(query_status, status);

  iree_hal_semaphore_release(semaphore);
  iree_status_ignore(status);
  iree_status_ignore(wait_status);
  iree_status_ignore(query_status);
}

// Wait on a semaphore that is then failed.
TEST_F(SemaphoreTest, WaitThenFail) {
  iree_hal_semaphore_t* semaphore = this->CreateSemaphore();

  // It is possible that the order becomes fail than wait.
  // We assume that it is less likely since starting the thread takes time.
  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "WaitThenFail test.");
  std::thread signal_thread(
      [&]() { iree_hal_semaphore_fail(semaphore, iree_status_clone(status)); });

  iree_status_t wait_status = iree_hal_semaphore_wait(
      semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE));
  EXPECT_EQ(iree_status_code(wait_status), IREE_STATUS_ABORTED);
  uint64_t value = 1234;
  iree_status_t query_status = iree_hal_semaphore_query(semaphore, &value);
  EXPECT_EQ(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  CheckStatusContains(query_status, status);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore);
  iree_status_ignore(status);
  iree_status_ignore(wait_status);
  iree_status_ignore(query_status);
}

// Wait 2 semaphores then fail one of them.
TEST_F(SemaphoreTest, MultiWaitThenFail) {
  iree_hal_semaphore_t* semaphore1 = this->CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = this->CreateSemaphore();

  // It is possible that the order becomes fail than wait.
  // We assume that it is less likely since starting the thread takes time.
  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "MultiWaitThenFail test.");
  std::thread signal_thread([&]() {
    iree_hal_semaphore_fail(semaphore1, iree_status_clone(status));
  });

  iree_hal_semaphore_t* semaphore_array[] = {semaphore1, semaphore2};
  uint64_t payload_array[] = {1, 1};
  iree_hal_semaphore_list_t semaphore_list = {
      IREE_ARRAYSIZE(semaphore_array),
      semaphore_array,
      payload_array,
  };
  iree_status_t wait_status = iree_hal_semaphore_list_wait(
      semaphore_list, iree_make_deadline(IREE_TIME_INFINITE_FUTURE));
  EXPECT_EQ(iree_status_code(wait_status), IREE_STATUS_ABORTED);
  uint64_t value = 1234;
  iree_status_t semaphore1_query_status =
      iree_hal_semaphore_query(semaphore1, &value);
  EXPECT_EQ(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  CheckStatusContains(semaphore1_query_status, status);

  // semaphore2 must not have changed.
  uint64_t semaphore2_value = 1234;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore2, &semaphore2_value));
  EXPECT_EQ(semaphore2_value, 0);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_status_ignore(status);
  iree_status_ignore(wait_status);
  iree_status_ignore(semaphore1_query_status);
}

// Wait 2 semaphores using iree_hal_device_wait_semaphores then fail
// one of them.
TEST_F(SemaphoreTest, DeviceMultiWaitThenFail) {
  iree_hal_semaphore_t* semaphore1 = this->CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = this->CreateSemaphore();

  // It is possible that the order becomes fail than wait.
  // We assume that it is less likely since starting the thread takes time.
  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "DeviceMultiWaitThenFail test.");
  std::thread signal_thread([&]() {
    iree_hal_semaphore_fail(semaphore1, iree_status_clone(status));
  });

  iree_hal_semaphore_t* semaphore_array[] = {semaphore1, semaphore2};
  uint64_t payload_array[] = {1, 1};
  iree_hal_semaphore_list_t semaphore_list = {
      IREE_ARRAYSIZE(semaphore_array),
      semaphore_array,
      payload_array,
  };
  iree_status_t wait_status = iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE));
  EXPECT_EQ(iree_status_code(wait_status), IREE_STATUS_ABORTED);
  uint64_t value = 1234;
  iree_status_t semaphore1_query_status =
      iree_hal_semaphore_query(semaphore1, &value);
  EXPECT_EQ(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  CheckStatusContains(semaphore1_query_status, status);

  // semaphore2 must not have changed.
  uint64_t semaphore2_value = 1234;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore2, &semaphore2_value));
  EXPECT_EQ(semaphore2_value, 0);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_status_ignore(status);
  iree_status_ignore(wait_status);
  iree_status_ignore(semaphore1_query_status);
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_SEMAPHORE_TEST_H_
