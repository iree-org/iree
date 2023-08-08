// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_SEMAPHORE_SUBMISSION_TEST_H_
#define IREE_HAL_CTS_SEMAPHORE_SUBMISSION_TEST_H_

#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class semaphore_submission_test : public CtsTestBase {};

TEST_P(semaphore_submission_test, SubmitWithNoCommandBuffers) {
  // No waits, one signal which we immediately wait on after submit.
  iree_hal_semaphore_t* signal_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &signal_semaphore));
  uint64_t signal_payload_values[] = {1ull};
  iree_hal_semaphore_list_t signal_semaphores = {
      1,
      &signal_semaphore,
      signal_payload_values,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_execute(device_,
                                               /*queue_affinity=*/0,
                                               iree_hal_semaphore_list_empty(),
                                               signal_semaphores, 0, NULL));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(signal_semaphore, 1ull, iree_infinite_timeout()));

  iree_hal_semaphore_release(signal_semaphore);
}

TEST_P(semaphore_submission_test, SubmitAndSignal) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // No waits, one signal which we immediately wait on after submit.
  iree_hal_semaphore_t* signal_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &signal_semaphore));
  uint64_t signal_payload_values[] = {1ull};
  iree_hal_semaphore_list_t signal_semaphores = {
      1,
      &signal_semaphore,
      signal_payload_values,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_,
      /*queue_affinity=*/0, iree_hal_semaphore_list_empty(), signal_semaphores,
      1, &command_buffer));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(signal_semaphore, 1ull, iree_infinite_timeout()));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_P(semaphore_submission_test, SubmitWithWait) {
  // Empty command buffer.
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // One wait and one signal semaphore.
  iree_hal_semaphore_t* wait_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &wait_semaphore));
  uint64_t wait_payload_values[] = {1ull};
  iree_hal_semaphore_list_t wait_semaphores = {
      1,
      &wait_semaphore,
      wait_payload_values,
  };
  iree_hal_semaphore_t* signal_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 100ull, &signal_semaphore));
  uint64_t signal_payload_values[] = {101ull};
  iree_hal_semaphore_list_t signal_semaphores = {
      1,
      &signal_semaphore,
      signal_payload_values,
  };

  IREE_ASSERT_OK(
      iree_hal_device_queue_execute(device_,
                                    /*queue_affinity=*/0, wait_semaphores,
                                    signal_semaphores, 1, &command_buffer));

  // Work shouldn't start until the wait semaphore reaches its payload value.
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(signal_semaphore, &value));
  EXPECT_EQ(100ull, value);

  // Signal the wait semaphore, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore, 1ull));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal_semaphore, 101ull,
                                         iree_infinite_timeout()));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_P(semaphore_submission_test, SubmitWithMultipleSemaphores) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_semaphore_t* wait_semaphore_1 = NULL;
  iree_hal_semaphore_t* wait_semaphore_2 = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &wait_semaphore_1));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &wait_semaphore_2));
  iree_hal_semaphore_t* wait_semaphore_ptrs[] = {wait_semaphore_1,
                                                 wait_semaphore_2};
  uint64_t wait_payload_values[] = {1ull, 1ull};
  iree_hal_semaphore_list_t wait_semaphores = {
      IREE_ARRAYSIZE(wait_semaphore_ptrs),
      wait_semaphore_ptrs,
      wait_payload_values,
  };

  iree_hal_semaphore_t* signal_semaphore_1 = NULL;
  iree_hal_semaphore_t* signal_semaphore_2 = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &signal_semaphore_1));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &signal_semaphore_2));
  iree_hal_semaphore_t* signal_semaphore_ptrs[] = {signal_semaphore_1,
                                                   signal_semaphore_2};
  uint64_t signal_payload_values[] = {1ull, 1ull};
  iree_hal_semaphore_list_t signal_semaphores = {
      IREE_ARRAYSIZE(signal_semaphore_ptrs),
      signal_semaphore_ptrs,
      signal_payload_values,
  };

  IREE_ASSERT_OK(
      iree_hal_device_queue_execute(device_,
                                    /*queue_affinity=*/0, wait_semaphores,
                                    signal_semaphores, 1, &command_buffer));

  // Work shouldn't start until all wait semaphores reach their payload values.
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(signal_semaphore_1, &value));
  EXPECT_EQ(0ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_query(signal_semaphore_2, &value));
  EXPECT_EQ(0ull, value);

  // Signal the wait semaphores, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore_1, 1ull));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore_2, 1ull));

  IREE_ASSERT_OK(
      iree_hal_semaphore_list_wait(signal_semaphores, iree_infinite_timeout()));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(wait_semaphore_1);
  iree_hal_semaphore_release(wait_semaphore_2);
  iree_hal_semaphore_release(signal_semaphore_1);
  iree_hal_semaphore_release(signal_semaphore_2);
}

// TODO: test device -> device synchronization: submit two batches with a
// semaphore singal -> wait dependency.
//
// TODO: test device -> device synchronization: submit multiple batches with
// multiple later batches waiting on the same signaling from a former batch.
//
// TODO: test device -> device synchronization: submit multiple batches with
// a former batch signaling a value greater than all other batches' (different)
// wait values.

// TODO: test host + device -> device synchronization: submit two batches
// with a later batch waiting on both a host and device singal to proceed.

// TODO: test device -> host + device synchronization: submit two batches
// with a former batch signaling to enable both host and device to proceed.

// TODO: test signaling a larger value before/after enqueuing waiting a smaller
// value to the device.

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_SEMAPHORE_SUBMISSION_TEST_H_
