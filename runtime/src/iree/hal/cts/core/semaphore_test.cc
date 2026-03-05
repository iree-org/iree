// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <chrono>
#include <thread>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using namespace std::chrono_literals;

class SemaphoreTest : public CtsTestBase<> {};

// Tests that a semaphore that is unused properly cleans itself up.
TEST_P(SemaphoreTest, NoOp) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 123ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(123ull, value);

  iree_hal_semaphore_release(semaphore);
}

// Tests that a semaphore will accept new values as it is signaled.
TEST_P(SemaphoreTest, NormalSignaling) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 2ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

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

// Tests semaphore failure handling.
TEST_P(SemaphoreTest, Failure) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 2ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 3ull));
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(3ull, value);

  iree_hal_semaphore_fail(semaphore,
                          iree_status_from_code(IREE_STATUS_UNKNOWN));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNKNOWN,
                        iree_hal_semaphore_query(semaphore, &value));

  iree_hal_semaphore_release(semaphore);
}

// Tests waiting on no semaphores.
TEST_P(SemaphoreTest, EmptyWait) {
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, iree_hal_semaphore_list_empty(),
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ALL, iree_hal_semaphore_list_empty(),
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));

  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, iree_hal_semaphore_list_empty(),
      iree_make_timeout_ns(IREE_DURATION_INFINITE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ALL, iree_hal_semaphore_list_empty(),
      iree_make_timeout_ns(IREE_DURATION_INFINITE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
}

// Tests waiting on a semaphore that has already been signaled.
TEST_P(SemaphoreTest, WaitAlreadySignaled) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 2ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

  // Test both previous and current values.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 1ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 2ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 1ull, iree_make_timeout_ns(IREE_DURATION_INFINITE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 2ull, iree_make_timeout_ns(IREE_DURATION_INFINITE),
      IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_semaphore_release(semaphore);
}

// Tests waiting on a semaphore that has not been signaled.
TEST_P(SemaphoreTest, WaitUnsignaled) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 2ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

  // Semaphore is at 2, waiting for 3 with an immediate deadline must fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_hal_semaphore_wait(semaphore, 3ull,
                              iree_make_deadline(IREE_TIME_INFINITE_PAST),
                              IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_semaphore_release(semaphore);
}

// Tests waiting on a semaphore that signals past the desired value.
TEST_P(SemaphoreTest, WaitLaterSignaledBeyond) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 2ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

  std::thread thread([&]() {
    std::this_thread::sleep_for(100ms);
    // Signal beyond the desired value.
    IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 10ull));
  });

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 3ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  thread.join();

  iree_hal_semaphore_release(semaphore);
}

// Tests IREE_HAL_WAIT_MODE_ALL when not all are signaled.
TEST_P(SemaphoreTest, WaitAllButNotAllSignaled) {
  iree_hal_semaphore_t* semaphore_a = NULL;
  iree_hal_semaphore_t* semaphore_b = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_a));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 1ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_b));

  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  uint64_t payload_values[] = {1ull, 1ull};
  iree_hal_semaphore_list_t semaphore_list = {IREE_ARRAYSIZE(semaphore_ptrs),
                                              semaphore_ptrs, payload_values};

  // semaphore_a is at 0 but needs 1: WAIT_ALL with immediate deadline must
  // fail.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_hal_device_wait_semaphores(
                            device_, IREE_HAL_WAIT_MODE_ALL, semaphore_list,
                            iree_make_deadline(IREE_TIME_INFINITE_PAST),
                            IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests IREE_HAL_WAIT_MODE_ALL when all are signaled.
TEST_P(SemaphoreTest, WaitAllAndAllSignaled) {
  iree_hal_semaphore_t* semaphore_a = NULL;
  iree_hal_semaphore_t* semaphore_b = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 1ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_a));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 1ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_b));

  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  uint64_t payload_values[] = {1ull, 1ull};
  iree_hal_semaphore_list_t semaphore_list = {IREE_ARRAYSIZE(semaphore_ptrs),
                                              semaphore_ptrs, payload_values};

  // Both semaphores already at their target values: must succeed.
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ALL, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests IREE_HAL_WAIT_MODE_ANY.
TEST_P(SemaphoreTest, WaitAnyAlreadySignaled) {
  iree_hal_semaphore_t* semaphore_a = NULL;
  iree_hal_semaphore_t* semaphore_b = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_a));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 1ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_b));

  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  uint64_t payload_values[] = {1ull, 1ull};
  iree_hal_semaphore_list_t semaphore_list = {IREE_ARRAYSIZE(semaphore_ptrs),
                                              semaphore_ptrs, payload_values};

  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

TEST_P(SemaphoreTest, WaitAnyLaterSignaled) {
  iree_hal_semaphore_t* semaphore_a = NULL;
  iree_hal_semaphore_t* semaphore_b = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_a));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_b));

  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  uint64_t payload_values[] = {1ull, 1ull};
  iree_hal_semaphore_list_t semaphore_list = {IREE_ARRAYSIZE(semaphore_ptrs),
                                              semaphore_ptrs, payload_values};

  std::thread thread([&]() {
    std::this_thread::sleep_for(100ms);
    IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_b, 1ull));
  });

  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  thread.join();

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests threading behavior by ping-ponging between the test main thread and
// a little thread.
TEST_P(SemaphoreTest, PingPong) {
  iree_hal_semaphore_t* a2b = NULL;
  iree_hal_semaphore_t* b2a = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &a2b));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &b2a));
  std::thread thread([&]() {
    // Should advance right past this because the value is already set.
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        a2b, 0ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
        IREE_HAL_WAIT_FLAG_DEFAULT));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(b2a, 1ull));
    // Jump ahead (blocking at first).
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        a2b, 4ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
        IREE_HAL_WAIT_FLAG_DEFAULT));
  });
  // Block until thread signals.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      b2a, 1ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(a2b, 4ull));
  thread.join();

  iree_hal_semaphore_release(a2b);
  iree_hal_semaphore_release(b2a);
}

// Waiting the same value multiple times.
TEST_P(SemaphoreTest, WaitOnTheSameValueMultipleTimes) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();
  std::thread thread(
      [&]() { IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 1)); });

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore, 1);

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore, 1);

  thread.join();

  iree_hal_semaphore_release(semaphore);
}

// Waiting for a finite amount of time.
TEST_P(SemaphoreTest, WaitForFiniteTime) {
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
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        continue;
      }
      IREE_ASSERT_OK(status);
      CheckSemaphoreValue(semaphore, 1);
      break;
    }

    signaling_thread.join();
    iree_hal_semaphore_release(semaphore);
  };

  // Immediate timeout.
  generic_test_fn([](iree_hal_semaphore_t* semaphore) {
    return iree_hal_semaphore_wait(semaphore, 1, iree_immediate_timeout(),
                                   IREE_HAL_WAIT_FLAG_DEFAULT);
  });

  // Absolute timeout.
  generic_test_fn([](iree_hal_semaphore_t* semaphore) {
    return iree_hal_semaphore_wait(semaphore, 1,
                                   iree_make_deadline(iree_time_now() + 1),
                                   IREE_HAL_WAIT_FLAG_DEFAULT);
  });

  // Relative timeout.
  generic_test_fn([](iree_hal_semaphore_t* semaphore) {
    return iree_hal_semaphore_wait(semaphore, 1, iree_make_timeout_ns(1),
                                   IREE_HAL_WAIT_FLAG_DEFAULT);
  });
}

// Wait on all semaphores on multiple places simultaneously.
TEST_P(SemaphoreTest, SimultaneousMultiWaitAll) {
  iree_hal_semaphore_t* semaphore1 = this->CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = this->CreateSemaphore();

  iree_hal_semaphore_t* semaphore_array[] = {semaphore1, semaphore2};
  uint64_t payload_array[] = {1, 1};
  iree_hal_semaphore_list_t semaphore_list = {IREE_ARRAYSIZE(semaphore_array),
                                              semaphore_array, payload_array};

  std::thread wait_thread1([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        semaphore_list, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
        IREE_HAL_WAIT_FLAG_DEFAULT));
  });

  std::thread wait_thread2([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        semaphore_list, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
        IREE_HAL_WAIT_FLAG_DEFAULT));
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
TEST_P(SemaphoreTest, FailThenWait) {
  iree_hal_semaphore_t* semaphore = this->CreateSemaphore();

  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "FailThenWait test.");
  iree_hal_semaphore_fail(semaphore, iree_status_clone(status));

  iree_status_t wait_status = iree_hal_semaphore_wait(
      semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, wait_status);
  uint64_t value = 1234;
  iree_status_t query_status = iree_hal_semaphore_query(semaphore, &value);
  EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  IREE_EXPECT_STATUS_IS(iree_status_code(status), query_status);
  iree_status_ignore(status);

  iree_hal_semaphore_release(semaphore);
}

// Wait on a semaphore that is then failed.
TEST_P(SemaphoreTest, WaitThenFail) {
  iree_hal_semaphore_t* semaphore = this->CreateSemaphore();

  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "WaitThenFail test.");
  std::thread signal_thread(
      [&]() { iree_hal_semaphore_fail(semaphore, iree_status_clone(status)); });

  iree_status_t wait_status = iree_hal_semaphore_wait(
      semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, wait_status);
  uint64_t value = 1234;
  iree_status_t query_status = iree_hal_semaphore_query(semaphore, &value);
  EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  IREE_EXPECT_STATUS_IS(iree_status_code(status), query_status);
  iree_status_ignore(status);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore);
}

// Wait 2 semaphores then fail one of them.
TEST_P(SemaphoreTest, MultiWaitThenFail) {
  iree_hal_semaphore_t* semaphore1 = this->CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = this->CreateSemaphore();

  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "MultiWaitThenFail test.");
  std::thread signal_thread([&]() {
    iree_hal_semaphore_fail(semaphore1, iree_status_clone(status));
  });

  iree_hal_semaphore_t* semaphore_array[] = {semaphore1, semaphore2};
  uint64_t payload_array[] = {1, 1};
  iree_hal_semaphore_list_t semaphore_list = {IREE_ARRAYSIZE(semaphore_array),
                                              semaphore_array, payload_array};
  iree_status_t wait_status = iree_hal_semaphore_list_wait(
      semaphore_list, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, wait_status);
  uint64_t value = 1234;
  iree_status_t semaphore1_query_status =
      iree_hal_semaphore_query(semaphore1, &value);
  EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  IREE_EXPECT_STATUS_IS(iree_status_code(status), semaphore1_query_status);
  iree_status_ignore(status);

  // semaphore2 must not have changed.
  uint64_t semaphore2_value = 1234;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore2, &semaphore2_value));
  EXPECT_EQ(semaphore2_value, 0);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
}

// Wait 2 semaphores using iree_hal_device_wait_semaphores then fail one.
TEST_P(SemaphoreTest, DeviceMultiWaitThenFail) {
  iree_hal_semaphore_t* semaphore1 = this->CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = this->CreateSemaphore();

  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "DeviceMultiWaitThenFail test.");
  std::thread signal_thread([&]() {
    iree_hal_semaphore_fail(semaphore1, iree_status_clone(status));
  });

  iree_hal_semaphore_t* semaphore_array[] = {semaphore1, semaphore2};
  uint64_t payload_array[] = {1, 1};
  iree_hal_semaphore_list_t semaphore_list = {IREE_ARRAYSIZE(semaphore_array),
                                              semaphore_array, payload_array};
  iree_status_t wait_status = iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_HAL_WAIT_FLAG_DEFAULT);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, wait_status);
  uint64_t value = 1234;
  iree_status_t semaphore1_query_status =
      iree_hal_semaphore_query(semaphore1, &value);
  EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  IREE_EXPECT_STATUS_IS(iree_status_code(status), semaphore1_query_status);
  iree_status_ignore(status);

  // semaphore2 must not have changed.
  uint64_t semaphore2_value = 1234;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore2, &semaphore2_value));
  EXPECT_EQ(semaphore2_value, 0);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
}

// Tests that failure status codes are preserved through the query round-trip.
// Validates the failure encoding mechanism: drivers encode failure status in
// the semaphore value via iree_hal_status_as_semaphore_failure(), and the HAL
// dispatch layer decodes it back via iree_hal_semaphore_failure_as_status().
TEST_P(SemaphoreTest, FailurePreservesStatusCode) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();

  // Use DATA_LOSS specifically — distinct from UNKNOWN (Failure test) and
  // CANCELLED (FailThenWait test). This exercises the encoding path with a
  // full status (message + backtrace), not just a bare status code.
  iree_hal_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_DATA_LOSS, "device fault"));

  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_semaphore_query(semaphore, &value));
  EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);

  iree_hal_semaphore_release(semaphore);
}

// Tests that failing an already-failed semaphore does not crash and preserves
// the original failure status. The second failure must be silently dropped.
TEST_P(SemaphoreTest, DoubleFailurePreservesFirst) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();

  iree_hal_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_DATA_LOSS, "first failure"));
  iree_hal_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_CANCELLED, "second failure"));

  // The first failure status (DATA_LOSS) must be preserved.
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_semaphore_query(semaphore, &value));
  EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);

  iree_hal_semaphore_release(semaphore);
}

// Tests that iree_hal_semaphore_export_timepoint dispatches through the vtable
// without crashing. Verifies the vtable slot is populated and callable.
TEST_P(SemaphoreTest, ExportTimepointReturnsError) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 1ull));

  iree_hal_external_timepoint_t external_timepoint;
  memset(&external_timepoint, 0, sizeof(external_timepoint));
  iree_status_t status = iree_hal_semaphore_export_timepoint(
      semaphore, 1ull, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE,
      IREE_HAL_EXTERNAL_TIMEPOINT_FLAG_NONE, &external_timepoint);
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);

  iree_hal_semaphore_release(semaphore);
}

// Tests that iree_hal_semaphore_import_timepoint dispatches through the vtable
// without crashing. Verifies the vtable slot is populated and callable.
TEST_P(SemaphoreTest, ImportTimepointReturnsError) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();

  iree_hal_external_timepoint_t external_timepoint;
  memset(&external_timepoint, 0, sizeof(external_timepoint));
  external_timepoint.type = IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_WAIT_PRIMITIVE;
  iree_status_t status = iree_hal_semaphore_import_timepoint(
      semaphore, 1ull, IREE_HAL_QUEUE_AFFINITY_ANY, external_timepoint);
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);

  iree_hal_semaphore_release(semaphore);
}

CTS_REGISTER_TEST_SUITE(SemaphoreTest);

}  // namespace iree::hal::cts
