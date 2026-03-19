// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Single-threaded semaphore tests. Tests that require std::thread (cross-thread
// signaling, ping-pong, timed waits with background signalers) are in
// semaphore_thread_test.cc.

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

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
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 3ull, /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(3ull, value);
  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(semaphore, 40ull, /*frontier=*/NULL));
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

  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 3ull, /*frontier=*/NULL));
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
      device_, IREE_ASYNC_WAIT_MODE_ANY, iree_hal_semaphore_list_empty(),
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_ASYNC_WAIT_MODE_ALL, iree_hal_semaphore_list_empty(),
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_ASYNC_WAIT_MODE_ANY, iree_hal_semaphore_list_empty(),
      iree_make_timeout_ns(IREE_DURATION_INFINITE), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_ASYNC_WAIT_MODE_ALL, iree_hal_semaphore_list_empty(),
      iree_make_timeout_ns(IREE_DURATION_INFINITE), IREE_ASYNC_WAIT_FLAG_NONE));
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
      IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 2ull, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 1ull, iree_make_timeout_ns(IREE_DURATION_INFINITE),
      IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 2ull, iree_make_timeout_ns(IREE_DURATION_INFINITE),
      IREE_ASYNC_WAIT_FLAG_NONE));

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
                              IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
}

// Tests IREE_ASYNC_WAIT_MODE_ALL when not all are signaled.
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
                            device_, IREE_ASYNC_WAIT_MODE_ALL, semaphore_list,
                            iree_make_deadline(IREE_TIME_INFINITE_PAST),
                            IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests IREE_ASYNC_WAIT_MODE_ALL when all are signaled.
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
      device_, IREE_ASYNC_WAIT_MODE_ALL, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests IREE_ASYNC_WAIT_MODE_ANY.
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
      device_, IREE_ASYNC_WAIT_MODE_ANY, semaphore_list,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Wait on a semaphore that is then failed.
TEST_P(SemaphoreTest, FailThenWait) {
  iree_hal_semaphore_t* semaphore = this->CreateSemaphore();

  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "FailThenWait test.");
  iree_hal_semaphore_fail(semaphore, iree_status_clone(status));

  iree_status_t wait_status = iree_hal_semaphore_wait(
      semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE),
      IREE_ASYNC_WAIT_FLAG_NONE);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, wait_status);
  uint64_t value = 1234;
  iree_status_t query_status = iree_hal_semaphore_query(semaphore, &value);
  EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  IREE_EXPECT_STATUS_IS(iree_status_code(status), query_status);
  iree_status_ignore(status);

  iree_hal_semaphore_release(semaphore);
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

CTS_REGISTER_TEST_SUITE(SemaphoreTest);

}  // namespace iree::hal::cts
