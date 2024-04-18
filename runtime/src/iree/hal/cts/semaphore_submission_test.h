// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_SEMAPHORE_SUBMISSION_TEST_H_
#define IREE_HAL_CTS_SEMAPHORE_SUBMISSION_TEST_H_

#include <cstdint>
#include <thread>

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
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

// Tests we can wait on both host and device semaphore to singal.
TEST_P(semaphore_submission_test, WaitAllHostAndDeviceSemaphores) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // Create two semaphores, one for the host thread to wait on, and one for the
  // device to wait on.
  iree_hal_semaphore_t* host_wait_semaphore = NULL;
  iree_hal_semaphore_t* device_wait_semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &host_wait_semaphore));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &device_wait_semaphore));

  // Create two semaphores, one for the host thread to signal, and one for the
  // device to signal.
  iree_hal_semaphore_t* host_signal_semaphore = NULL;
  iree_hal_semaphore_t* device_signal_semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &host_signal_semaphore));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &device_signal_semaphore));

  // Prepare device wait and signal semaphore lists.
  uint64_t device_wait_payload_values[] = {1ull};
  iree_hal_semaphore_list_t device_wait_semaphores = {
      /*count=*/1, &device_wait_semaphore, device_wait_payload_values};
  uint64_t device_signal_payload_values[] = {1ull};
  iree_hal_semaphore_list_t device_signal_semaphores = {
      /*count=*/1, &device_signal_semaphore, device_signal_payload_values};

  // Dispatch the device command buffer to have it wait.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, device_wait_semaphores,
      device_signal_semaphores, 1, &command_buffer));

  // Start another thread and have it wait.
  std::thread thread([&]() {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(host_wait_semaphore, 1ull,
                                iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(host_signal_semaphore, 1ull));
  });

  // Work shouldn't start until all wait semaphores reach their payload values.
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(host_signal_semaphore, &value));
  EXPECT_EQ(0ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_query(device_signal_semaphore, &value));
  EXPECT_EQ(0ull, value);

  // Signal the wait semaphores for both host thread and device, work should
  // begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_wait_semaphore, 1ull));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(device_wait_semaphore, 1ull));

  // Now wait on both to complete.
  iree_hal_semaphore_t* main_semaphore_ptrs[] = {host_signal_semaphore,
                                                 device_signal_semaphore};
  uint64_t main_payload_values[] = {1ull, 1ull};
  iree_hal_semaphore_list_t main_wait_semaphores = {
      IREE_ARRAYSIZE(main_semaphore_ptrs), main_semaphore_ptrs,
      main_payload_values};
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ALL, main_wait_semaphores,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  thread.join();

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(host_wait_semaphore);
  iree_hal_semaphore_release(device_wait_semaphore);
  iree_hal_semaphore_release(host_signal_semaphore);
  iree_hal_semaphore_release(device_signal_semaphore);
}

// Tests that we can wait on any host and device semaphore to singal,
// and device signals.
TEST_P(semaphore_submission_test,
       WaitAnyHostAndDeviceSemaphoresAndDeviceSignals) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // Create two semaphores, one for the host thread to wait on, and one for the
  // device to wait on.
  iree_hal_semaphore_t* host_wait_semaphore = NULL;
  iree_hal_semaphore_t* device_wait_semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &host_wait_semaphore));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &device_wait_semaphore));

  // Create two semaphores, one for the host thread to signal, and one for the
  // device to signal.
  iree_hal_semaphore_t* host_signal_semaphore = NULL;
  iree_hal_semaphore_t* device_signal_semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &host_signal_semaphore));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &device_signal_semaphore));

  // Prepare device wait and signal semaphore lists.
  uint64_t device_wait_payload_values[] = {1ull};
  iree_hal_semaphore_list_t device_wait_semaphores = {
      /*count=*/1, &device_wait_semaphore, device_wait_payload_values};
  uint64_t device_signal_payload_values[] = {1ull};
  iree_hal_semaphore_list_t device_signal_semaphores = {
      /*count=*/1, &device_signal_semaphore, device_signal_payload_values};

  // Dispatch the device command buffer to have it wait.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, device_wait_semaphores,
      device_signal_semaphores, 1, &command_buffer));

  // Start another thread and have it wait.
  std::thread thread([&]() {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(host_wait_semaphore, 1ull,
                                iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(host_signal_semaphore, 1ull));
  });

  // Work shouldn't start until all wait semaphores reach their payload values.
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(host_signal_semaphore, &value));
  EXPECT_EQ(0ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_query(device_signal_semaphore, &value));
  EXPECT_EQ(0ull, value);

  // Signal the wait semaphores for the device, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(device_wait_semaphore, 1ull));

  // Now wait on any to complete.
  iree_hal_semaphore_t* main_semaphore_ptrs[] = {host_signal_semaphore,
                                                 device_signal_semaphore};
  uint64_t main_payload_values[] = {1ull, 1ull};
  iree_hal_semaphore_list_t main_wait_semaphores = {
      IREE_ARRAYSIZE(main_semaphore_ptrs), main_semaphore_ptrs,
      main_payload_values};
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, main_wait_semaphores,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  // Check that the device has signaled but the host thread hasn't.
  IREE_ASSERT_OK(iree_hal_semaphore_query(host_signal_semaphore, &value));
  EXPECT_EQ(0ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_query(device_signal_semaphore, &value));
  EXPECT_EQ(1ull, value);

  // Now let the host thread to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_wait_semaphore, 1ull));
  thread.join();

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(host_wait_semaphore);
  iree_hal_semaphore_release(device_wait_semaphore);
  iree_hal_semaphore_release(host_signal_semaphore);
  iree_hal_semaphore_release(device_signal_semaphore);
}

// Tests we can wait on any host and device semaphore to singal,
// and host signals.
TEST_P(semaphore_submission_test,
       WaitAnyHostAndDeviceSemaphoresAndHostSignals) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // Create two semaphores, one for the host thread to wait on, and one for the
  // device to wait on.
  iree_hal_semaphore_t* host_wait_semaphore = NULL;
  iree_hal_semaphore_t* device_wait_semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &host_wait_semaphore));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &device_wait_semaphore));

  // Create two semaphores, one for the host thread to signal, and one for the
  // device to signal.
  iree_hal_semaphore_t* host_signal_semaphore = NULL;
  iree_hal_semaphore_t* device_signal_semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &host_signal_semaphore));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, 0ull, &device_signal_semaphore));

  // Prepare device wait and signal semaphore lists.
  uint64_t device_wait_payload_values[] = {1ull};
  iree_hal_semaphore_list_t device_wait_semaphores = {
      /*count=*/1, &device_wait_semaphore, device_wait_payload_values};
  uint64_t device_signal_payload_values[] = {1ull};
  iree_hal_semaphore_list_t device_signal_semaphores = {
      /*count=*/1, &device_signal_semaphore, device_signal_payload_values};

  // Dispatch the device command buffer to have it wait.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, device_wait_semaphores,
      device_signal_semaphores, 1, &command_buffer));

  // Start another thread and have it wait.
  std::thread thread([&]() {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(host_wait_semaphore, 1ull,
                                iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(host_signal_semaphore, 1ull));
  });

  // Work shouldn't start until all wait semaphores reach their payload values.
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(host_signal_semaphore, &value));
  EXPECT_EQ(0ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_query(device_signal_semaphore, &value));
  EXPECT_EQ(0ull, value);

  // Signal the wait semaphores for the host thread, work should begin and
  // complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_wait_semaphore, 1ull));

  // Now wait on any to complete.
  iree_hal_semaphore_t* main_semaphore_ptrs[] = {host_signal_semaphore,
                                                 device_signal_semaphore};
  uint64_t main_payload_values[] = {1ull, 1ull};
  iree_hal_semaphore_list_t main_wait_semaphores = {
      IREE_ARRAYSIZE(main_semaphore_ptrs), main_semaphore_ptrs,
      main_payload_values};
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, main_wait_semaphores,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  thread.join();

  // Check that the host thread has signaled but the device hasn't.
  IREE_ASSERT_OK(iree_hal_semaphore_query(host_signal_semaphore, &value));
  EXPECT_EQ(1ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_query(device_signal_semaphore, &value));
  EXPECT_EQ(0ull, value);

  // Signal and wait for the device to complete too.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(device_wait_semaphore, 1ull));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(device_signal_semaphore, 1ull,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(host_wait_semaphore);
  iree_hal_semaphore_release(device_wait_semaphore);
  iree_hal_semaphore_release(host_signal_semaphore);
  iree_hal_semaphore_release(device_signal_semaphore);
}

// Test device -> device synchronization: submit two batches with a
// semaphore signal -> wait dependency.
TEST_P(semaphore_submission_test, IntermediateSemaphoreBetweenDeviceBatches) {
  // The signaling relationship is
  // command_buffer1 -> semaphore1 -> command_buffer2 -> semaphore2

  // Create first command buffer.
  iree_hal_command_buffer_t* command_buffer1 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer1));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer1));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer1));

  // Create second command buffer.
  iree_hal_command_buffer_t* command_buffer2 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer2));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer2));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer2));

  // Semaphore to signal when command_buffer1 is done and to wait to
  // start executing command_buffer2.
  iree_hal_semaphore_t* semaphore1 = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0, &semaphore1));
  uint64_t semaphore1_value = 1;
  iree_hal_semaphore_list_t semaphore1_list = {/*count=*/1, &semaphore1,
                                               &semaphore1_value};

  // Semaphore to signal when all work (command_buffer2) is done.
  iree_hal_semaphore_t* semaphore2 = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0, &semaphore2));
  uint64_t semaphore2_value = 1;
  iree_hal_semaphore_list_t semaphore2_list = {/*count=*/1, &semaphore2,
                                               &semaphore2_value};

  // Dispatch the second command buffer.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/semaphore1_list,
      /*signal_semaphore_list=*/semaphore2_list, 1, &command_buffer2));

  // Make sure that the intermediate and second semaphores have not advanced
  // since only command_buffer2 is queued.
  uint64_t semaphore2_value_after_queueing_command_buffer2;
  IREE_ASSERT_OK(iree_hal_semaphore_query(
      semaphore2, &semaphore2_value_after_queueing_command_buffer2));
  EXPECT_EQ(static_cast<uint64_t>(0),
            semaphore2_value_after_queueing_command_buffer2);
  uint64_t semaphore1_value_after_queueing_command_buffer2;
  IREE_ASSERT_OK(iree_hal_semaphore_query(
      semaphore1, &semaphore1_value_after_queueing_command_buffer2));
  EXPECT_EQ(static_cast<uint64_t>(0),
            semaphore1_value_after_queueing_command_buffer2);

  // Submit the first command buffer.
  iree_hal_semaphore_list_t command_buffer1_wait_semaphore_list = {
      /*count=*/0, nullptr, nullptr};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer1_wait_semaphore_list,
      /*signal_semaphore_list=*/semaphore1_list, 1, &command_buffer1));

  // Wait on the intermediate semaphore and check its value.
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore1, semaphore1_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  uint64_t semaphore1_value_after_command_buffer1_has_done_executing;
  IREE_ASSERT_OK(iree_hal_semaphore_query(
      semaphore1, &semaphore1_value_after_command_buffer1_has_done_executing));
  uint64_t expected_semaphore1_value = semaphore1_value;
  EXPECT_EQ(semaphore1_value,
            semaphore1_value_after_command_buffer1_has_done_executing);

  // Wait on the second semaphore and check its value.
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore2, semaphore2_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  uint64_t semaphore2_value_after_command_buffer2_has_done_executing;
  IREE_ASSERT_OK(iree_hal_semaphore_query(
      semaphore2, &semaphore2_value_after_command_buffer2_has_done_executing));
  uint64_t expected_semaphore2_value = semaphore2_value;
  EXPECT_EQ(expected_semaphore2_value,
            semaphore2_value_after_command_buffer2_has_done_executing);

  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
}

// Test device -> device synchronization: submit multiple batches with
// multiple later batches waiting on the same signaling from a former batch.
TEST_P(semaphore_submission_test, TwoBatchesWaitingOn1FormerBatchAmongst2) {
  // The signaling-wait relation is:
  //                  command_buffer11  command_buffer12
  //                         ↓
  //                     semaphore11
  //                     ↙        ↘
  //      command_buffer21         command_buffer22
  //             ↓                        ↓
  //        semaphore21              semaphore22

  iree_hal_command_buffer_t* command_buffer11 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer12 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer21 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer22 = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore11 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore21 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore22 = CreateSemaphore();

  // All semaphores start from value 0 and reach 1.
  uint64_t semaphore_signal_wait_value = 1;
  iree_hal_semaphore_list_t semaphore11_list = {/*count=*/1, &semaphore11,
                                                &semaphore_signal_wait_value};
  iree_hal_semaphore_list_t semaphore21_list = {/*count=*/1, &semaphore21,
                                                &semaphore_signal_wait_value};
  iree_hal_semaphore_list_t semaphore22_list = {/*count=*/1, &semaphore22,
                                                &semaphore_signal_wait_value};
  iree_hal_semaphore_list_t empty_semaphore_list{/*count=*/0, nullptr, nullptr};

  // We submit the command buffers in reverse order.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/semaphore11_list,
      /*signal_semaphore_list=*/semaphore22_list, 1, &command_buffer22));
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/semaphore11_list,
      /*signal_semaphore_list=*/semaphore21_list, 1, &command_buffer21));
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/empty_semaphore_list,
      /*signal_semaphore_list=*/empty_semaphore_list, 1, &command_buffer12));

  // Assert that semaphores have not advance since we have not yet submitted
  // command_buffer11.
  CheckSemaphoreValue(semaphore11, 0);
  CheckSemaphoreValue(semaphore21, 0);
  CheckSemaphoreValue(semaphore22, 0);

  // Submit command_buffer11.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/empty_semaphore_list,
      /*signal_semaphore_list=*/semaphore11_list, 1, &command_buffer11));

  // Wait and check that semaphore values have advanced.
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore21, semaphore_signal_wait_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore21, semaphore_signal_wait_value);
  // semaphore11 must have also advanced because semaphore21 has advanced.
  CheckSemaphoreValue(semaphore11, semaphore_signal_wait_value);

  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore22, semaphore_signal_wait_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore22, semaphore_signal_wait_value);

  iree_hal_semaphore_release(semaphore11);
  iree_hal_semaphore_release(semaphore21);
  iree_hal_semaphore_release(semaphore22);
  iree_hal_command_buffer_release(command_buffer11);
  iree_hal_command_buffer_release(command_buffer12);
  iree_hal_command_buffer_release(command_buffer21);
  iree_hal_command_buffer_release(command_buffer22);
}

// Test device -> device synchronization: submit multiple batches with
// a former batch signaling a value greater than all other batches' (different)
// wait values.
TEST_P(semaphore_submission_test, TwoBatchesWaitingOnDifferentSemaphoreValues) {
  // The signal-wait relation is
  //
  //          command_buffer11
  //                 ↓
  //           signal value 3
  //                 ↓
  //            semaphore11
  //           ↙           ↘
  //  wait value 1       wait value 2
  //        ↓                  ↓
  // command_buffer21   command_buffer22
  //        ↓                  ↓
  //    semaphore21       semaphore22

  iree_hal_command_buffer_t* command_buffer11 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer21 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer22 = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore11 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore21 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore22 = CreateSemaphore();

  // Command buffer wait/signal lists.
  iree_hal_semaphore_list_t command_buffer11_semaphore_wait_list = {
      /*count=*/0, nullptr, nullptr};
  uint64_t command_buffer11_semaphore11_signal_value = 3;
  iree_hal_semaphore_list_t command_buffer11_semaphore_signal_list = {
      /*count=*/1, &semaphore11, &command_buffer11_semaphore11_signal_value};
  uint64_t command_buffer21_semaphore11_wait_value = 1;
  iree_hal_semaphore_list_t command_buffer21_semaphore_wait_list = {
      /*count=*/1, &semaphore11, &command_buffer21_semaphore11_wait_value};
  uint64_t command_buffer22_semaphore11_wait_value = 2;
  iree_hal_semaphore_list_t command_buffer22_semaphore_wait_list = {
      /*count=*/1, &semaphore11, &command_buffer22_semaphore11_wait_value};
  uint64_t semaphore2x_signal_value = 1;
  iree_hal_semaphore_list_t command_buffer21_signal_list = {
      /*count=*/1, &semaphore21, &semaphore2x_signal_value};
  iree_hal_semaphore_list_t command_buffer22_signal_list = {
      /*count=*/1, &semaphore22, &semaphore2x_signal_value};

  // We submit the command buffers in reverse order.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer22_semaphore_wait_list,
      /*signal_semaphore_list=*/command_buffer22_signal_list, 1,
      &command_buffer22));
  // We submit the command buffers in reverse order.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer21_semaphore_wait_list,
      /*signal_semaphore_list=*/command_buffer21_signal_list, 1,
      &command_buffer21));

  // Semaphores have not advance since we have not yet submitted
  // command_buffer11.
  CheckSemaphoreValue(semaphore11, 0);
  CheckSemaphoreValue(semaphore21, 0);
  CheckSemaphoreValue(semaphore22, 0);

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer11_semaphore_wait_list,
      /*signal_semaphore_list=*/command_buffer11_semaphore_signal_list, 1,
      &command_buffer11));

  // Wait and check that semaphore values have advanced.
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore21, semaphore2x_signal_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore21, semaphore2x_signal_value);
  // semaphore11 must have advanced, because semaphore22 has advanced already.
  CheckSemaphoreValue(semaphore11, command_buffer11_semaphore11_signal_value);

  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore22, semaphore2x_signal_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore22, semaphore2x_signal_value);

  iree_hal_semaphore_release(semaphore11);
  iree_hal_semaphore_release(semaphore21);
  iree_hal_semaphore_release(semaphore22);
  iree_hal_command_buffer_release(command_buffer11);
  iree_hal_command_buffer_release(command_buffer21);
  iree_hal_command_buffer_release(command_buffer22);
}

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
