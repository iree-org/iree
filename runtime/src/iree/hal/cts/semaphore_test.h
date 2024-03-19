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

namespace iree {
namespace hal {
namespace cts {
using namespace std::chrono_literals;

class semaphore_test : public CtsTestBase {};

// Tests that a semaphore that is unused properly cleans itself up.
TEST_P(semaphore_test, NoOp) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 123ull, &semaphore));

  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(123ull, value);

  iree_hal_semaphore_release(semaphore);
}

// Tests that a semaphore will accept new values as it is signaled.
TEST_P(semaphore_test, NormalSignaling) {
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
TEST_P(semaphore_test, Failure) {
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
TEST_P(semaphore_test, EmptyWait) {
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
TEST_P(semaphore_test, WaitAlreadySignaled) {
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
TEST_P(semaphore_test, WaitUnsignaled) {
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
TEST_P(semaphore_test, WaitLaterSignaledBeyond) {
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
TEST_P(semaphore_test, WaitAllButNotAllSignaled) {
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
TEST_P(semaphore_test, WaitAllAndAllSignaled) {
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
TEST_P(semaphore_test, WaitAnyAlreadySignaled) {
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

TEST_P(semaphore_test, WaitAnyLaterSignaled) {
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
TEST_P(semaphore_test, PingPong) {
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

// TODO: test waiting the same value multiple times.
// TODO: test waiting for a finite amount of time.

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_SEMAPHORE_TEST_H_
