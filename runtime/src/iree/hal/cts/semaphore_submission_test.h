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

namespace iree::hal::cts {

class SemaphoreSubmissionTest : public CTSTestBase<> {};

TEST_F(SemaphoreSubmissionTest, SubmitWithNoCommandBuffers) {
  // No waits, one signal which we immediately wait on after submit.
  iree_hal_semaphore_t* signal_semaphore = CreateSemaphore();
  uint64_t signal_payload_values[] = {1};
  iree_hal_semaphore_list_t signal_semaphores = {
      1,
      &signal_semaphore,
      signal_payload_values,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_barrier(device_,
                                               /*queue_affinity=*/0,
                                               iree_hal_semaphore_list_empty(),
                                               signal_semaphores));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(signal_semaphore, 1, iree_infinite_timeout()));

  iree_hal_semaphore_release(signal_semaphore);
}

TEST_F(SemaphoreSubmissionTest, SubmitAndSignal) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  // No waits, one signal which we immediately wait on after submit.
  iree_hal_semaphore_t* signal_semaphore = CreateSemaphore();
  uint64_t signal_payload_values[] = {1};
  iree_hal_semaphore_list_t signal_semaphores = {
      1,
      &signal_semaphore,
      signal_payload_values,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_,
      /*queue_affinity=*/0, iree_hal_semaphore_list_empty(), signal_semaphores,
      1, &command_buffer, /*binding_tables=*/NULL));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(signal_semaphore, 1, iree_infinite_timeout()));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_F(SemaphoreSubmissionTest, SubmitWithWait) {
  // Empty command buffer.
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  // One wait and one signal semaphore.
  iree_hal_semaphore_t* wait_semaphore = CreateSemaphore();
  uint64_t wait_payload_values[] = {1};
  iree_hal_semaphore_list_t wait_semaphores = {
      1,
      &wait_semaphore,
      wait_payload_values,
  };
  iree_hal_semaphore_t* signal_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 100, IREE_HAL_SEMAPHORE_FLAG_NONE, &signal_semaphore));
  uint64_t signal_payload_values[] = {101};
  iree_hal_semaphore_list_t signal_semaphores = {
      1,
      &signal_semaphore,
      signal_payload_values,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_,
      /*queue_affinity=*/0, wait_semaphores, signal_semaphores, 1,
      &command_buffer, /*binding_tables=*/NULL));

  // Work shouldn't start until the wait semaphore reaches its payload value.
  CheckSemaphoreValue(signal_semaphore, 100);

  // Signal the wait semaphore, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore, 1));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(signal_semaphore, 101, iree_infinite_timeout()));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_F(SemaphoreSubmissionTest, SubmitWithMultipleSemaphores) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  iree_hal_semaphore_t* wait_semaphore_1 = CreateSemaphore();
  iree_hal_semaphore_t* wait_semaphore_2 = CreateSemaphore();
  iree_hal_semaphore_t* wait_semaphore_ptrs[] = {wait_semaphore_1,
                                                 wait_semaphore_2};
  uint64_t wait_payload_values[] = {1, 1};
  iree_hal_semaphore_list_t wait_semaphores = {
      IREE_ARRAYSIZE(wait_semaphore_ptrs),
      wait_semaphore_ptrs,
      wait_payload_values,
  };

  iree_hal_semaphore_t* signal_semaphore_1 = CreateSemaphore();
  iree_hal_semaphore_t* signal_semaphore_2 = CreateSemaphore();
  iree_hal_semaphore_t* signal_semaphore_ptrs[] = {signal_semaphore_1,
                                                   signal_semaphore_2};
  uint64_t signal_payload_values[] = {1, 1};
  iree_hal_semaphore_list_t signal_semaphores = {
      IREE_ARRAYSIZE(signal_semaphore_ptrs),
      signal_semaphore_ptrs,
      signal_payload_values,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_,
      /*queue_affinity=*/0, wait_semaphores, signal_semaphores, 1,
      &command_buffer, /*binding_tables=*/NULL));

  // Work shouldn't start until all wait semaphores reach their payload values.
  CheckSemaphoreValue(signal_semaphore_1, 0);
  CheckSemaphoreValue(signal_semaphore_2, 0);

  // Signal the wait semaphores, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore_1, 1));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore_2, 1));

  IREE_ASSERT_OK(
      iree_hal_semaphore_list_wait(signal_semaphores, iree_infinite_timeout()));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(wait_semaphore_1);
  iree_hal_semaphore_release(wait_semaphore_2);
  iree_hal_semaphore_release(signal_semaphore_1);
  iree_hal_semaphore_release(signal_semaphore_2);
}

// Tests we can wait on both host and device semaphore to singal.
TEST_F(SemaphoreSubmissionTest, WaitAllHostAndDeviceSemaphores) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  // Create two semaphores, one for the host thread to wait on, and one for the
  // device to wait on.
  iree_hal_semaphore_t* host_wait_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_wait_semaphore = CreateSemaphore();

  // Create two semaphores, one for the host thread to signal, and one for the
  // device to signal.
  iree_hal_semaphore_t* host_signal_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_signal_semaphore = CreateSemaphore();

  // Prepare device wait and signal semaphore lists.
  uint64_t device_wait_payload_values[] = {1};
  iree_hal_semaphore_list_t device_wait_semaphores = {
      /*count=*/1, &device_wait_semaphore, device_wait_payload_values};
  uint64_t device_signal_payload_values[] = {1};
  iree_hal_semaphore_list_t device_signal_semaphores = {
      /*count=*/1, &device_signal_semaphore, device_signal_payload_values};

  // Dispatch the device command buffer to have it wait.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, device_wait_semaphores,
      device_signal_semaphores, 1, &command_buffer, /*binding_tables=*/NULL));

  // Start another thread and have it wait.
  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        host_wait_semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(host_signal_semaphore, 1));
  });

  // Work shouldn't start until all wait semaphores reach their payload values.
  CheckSemaphoreValue(host_signal_semaphore, 0);
  CheckSemaphoreValue(device_signal_semaphore, 0);

  // Signal the wait semaphores for both host thread and device, work should
  // begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_wait_semaphore, 1));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(device_wait_semaphore, 1));

  // Now wait on both to complete.
  iree_hal_semaphore_t* main_semaphore_ptrs[] = {host_signal_semaphore,
                                                 device_signal_semaphore};
  uint64_t main_payload_values[] = {1, 1};
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
TEST_F(SemaphoreSubmissionTest,
       WaitAnyHostAndDeviceSemaphoresAndDeviceSignals) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  // Create two semaphores, one for the host thread to wait on, and one for the
  // device to wait on.
  iree_hal_semaphore_t* host_wait_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_wait_semaphore = CreateSemaphore();

  // Create two semaphores, one for the host thread to signal, and one for the
  // device to signal.
  iree_hal_semaphore_t* host_signal_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_signal_semaphore = CreateSemaphore();

  // Prepare device wait and signal semaphore lists.
  uint64_t device_wait_payload_values[] = {1};
  iree_hal_semaphore_list_t device_wait_semaphores = {
      /*count=*/1, &device_wait_semaphore, device_wait_payload_values};
  uint64_t device_signal_payload_values[] = {1};
  iree_hal_semaphore_list_t device_signal_semaphores = {
      /*count=*/1, &device_signal_semaphore, device_signal_payload_values};

  // Dispatch the device command buffer to have it wait.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, device_wait_semaphores,
      device_signal_semaphores, 1, &command_buffer, /*binding_tables=*/NULL));

  // Start another thread and have it wait.
  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        host_wait_semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(host_signal_semaphore, 1));
  });

  // Work shouldn't start until all wait semaphores reach their payload values.
  CheckSemaphoreValue(host_signal_semaphore, 0);
  CheckSemaphoreValue(device_signal_semaphore, 0);

  // Signal the wait semaphores for the device, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(device_wait_semaphore, 1));

  // Now wait on any to complete.
  iree_hal_semaphore_t* main_semaphore_ptrs[] = {host_signal_semaphore,
                                                 device_signal_semaphore};
  uint64_t main_payload_values[] = {1, 1};
  iree_hal_semaphore_list_t main_wait_semaphores = {
      IREE_ARRAYSIZE(main_semaphore_ptrs), main_semaphore_ptrs,
      main_payload_values};
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, main_wait_semaphores,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  // Check that the device has signaled but the host thread hasn't.
  CheckSemaphoreValue(host_signal_semaphore, 0);
  CheckSemaphoreValue(device_signal_semaphore, 1);

  // Now let the host thread to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_wait_semaphore, 1));
  thread.join();

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(host_wait_semaphore);
  iree_hal_semaphore_release(device_wait_semaphore);
  iree_hal_semaphore_release(host_signal_semaphore);
  iree_hal_semaphore_release(device_signal_semaphore);
}

// Tests we can wait on any host and device semaphore to singal,
// and host signals.
TEST_F(SemaphoreSubmissionTest, WaitAnyHostAndDeviceSemaphoresAndHostSignals) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  // Create two semaphores, one for the host thread to wait on, and one for the
  // device to wait on.
  iree_hal_semaphore_t* host_wait_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_wait_semaphore = CreateSemaphore();

  // Create two semaphores, one for the host thread to signal, and one for the
  // device to signal.
  iree_hal_semaphore_t* host_signal_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_signal_semaphore = CreateSemaphore();

  // Prepare device wait and signal semaphore lists.
  uint64_t device_wait_payload_values[] = {1};
  iree_hal_semaphore_list_t device_wait_semaphores = {
      /*count=*/1, &device_wait_semaphore, device_wait_payload_values};
  uint64_t device_signal_payload_values[] = {1};
  iree_hal_semaphore_list_t device_signal_semaphores = {
      /*count=*/1, &device_signal_semaphore, device_signal_payload_values};

  // Dispatch the device command buffer to have it wait.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, device_wait_semaphores,
      device_signal_semaphores, 1, &command_buffer, /*binding_tables=*/NULL));

  // Start another thread and have it wait.
  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        host_wait_semaphore, 1, iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(host_signal_semaphore, 1));
  });

  // Work shouldn't start until all wait semaphores reach their payload values.
  CheckSemaphoreValue(host_signal_semaphore, 0);
  CheckSemaphoreValue(device_signal_semaphore, 0);

  // Signal the wait semaphores for the host thread, work should begin and
  // complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_wait_semaphore, 1));

  // Now wait on any to complete.
  iree_hal_semaphore_t* main_semaphore_ptrs[] = {host_signal_semaphore,
                                                 device_signal_semaphore};
  uint64_t main_payload_values[] = {1, 1};
  iree_hal_semaphore_list_t main_wait_semaphores = {
      IREE_ARRAYSIZE(main_semaphore_ptrs), main_semaphore_ptrs,
      main_payload_values};
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, main_wait_semaphores,
      iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  thread.join();

  // Check that the host thread has signaled but the device hasn't.
  CheckSemaphoreValue(host_signal_semaphore, 1);
  CheckSemaphoreValue(device_signal_semaphore, 0);

  // Signal and wait for the device to complete too.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(device_wait_semaphore, 1));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(device_signal_semaphore, 1,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(host_wait_semaphore);
  iree_hal_semaphore_release(device_wait_semaphore);
  iree_hal_semaphore_release(host_signal_semaphore);
  iree_hal_semaphore_release(device_signal_semaphore);
}

// Test device -> device synchronization: submit two batches with a
// semaphore signal -> wait dependency.
TEST_F(SemaphoreSubmissionTest, IntermediateSemaphoreBetweenDeviceBatches) {
  // The signaling relationship is
  // command_buffer1 -> semaphore1 -> command_buffer2 -> semaphore2

  iree_hal_command_buffer_t* command_buffer1 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer2 = CreateEmptyCommandBuffer();

  // Semaphore to signal when command_buffer1 is done and to wait to
  // start executing command_buffer2.
  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  uint64_t semaphore_signal_value = 1;
  iree_hal_semaphore_list_t semaphore1_list = {/*count=*/1, &semaphore1,
                                               &semaphore_signal_value};

  // Semaphore to signal when all work (command_buffer2) is done.
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();
  iree_hal_semaphore_list_t semaphore2_list = {/*count=*/1, &semaphore2,
                                               &semaphore_signal_value};

  // Dispatch the second command buffer.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/semaphore1_list,
      /*signal_semaphore_list=*/semaphore2_list, 1, &command_buffer2,
      /*binding_tables=*/NULL));

  // Make sure that the intermediate and second semaphores have not advanced
  // since only command_buffer2 is queued.
  CheckSemaphoreValue(semaphore1, 0);
  CheckSemaphoreValue(semaphore2, 0);

  // Submit the first command buffer.
  iree_hal_semaphore_list_t command_buffer1_wait_semaphore_list = {
      /*count=*/0, nullptr, nullptr};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer1_wait_semaphore_list,
      /*signal_semaphore_list=*/semaphore1_list, 1, &command_buffer1,
      /*binding_tables=*/NULL));

  // Wait on the intermediate semaphore and check its value.
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore1, semaphore_signal_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore1, semaphore_signal_value);

  // Wait on the second semaphore and check its value.
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore2, semaphore_signal_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore2, semaphore_signal_value);

  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
}

// Test device -> device synchronization: submit multiple batches with
// multiple later batches waiting on the same signaling from a former batch.
TEST_F(SemaphoreSubmissionTest, TwoBatchesWaitingOn1FormerBatchAmongst2) {
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
      /*signal_semaphore_list=*/semaphore22_list, 1, &command_buffer22,
      /*binding_tables=*/NULL));
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/semaphore11_list,
      /*signal_semaphore_list=*/semaphore21_list, 1, &command_buffer21,
      /*binding_tables=*/NULL));
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/empty_semaphore_list,
      /*signal_semaphore_list=*/empty_semaphore_list, 1, &command_buffer12,
      /*binding_tables=*/NULL));

  // Assert that semaphores have not advance since we have not yet submitted
  // command_buffer11.
  CheckSemaphoreValue(semaphore11, 0);
  CheckSemaphoreValue(semaphore21, 0);
  CheckSemaphoreValue(semaphore22, 0);

  // Submit command_buffer11.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/empty_semaphore_list,
      /*signal_semaphore_list=*/semaphore11_list, 1, &command_buffer11,
      /*binding_tables=*/NULL));

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
TEST_F(SemaphoreSubmissionTest, TwoBatchesWaitingOnDifferentSemaphoreValues) {
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
      &command_buffer22, /*binding_tables=*/NULL));
  // We submit the command buffers in reverse order.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer21_semaphore_wait_list,
      /*signal_semaphore_list=*/command_buffer21_signal_list, 1,
      &command_buffer21, /*binding_tables=*/NULL));

  // Semaphores have not advance since we have not yet submitted
  // command_buffer11.
  CheckSemaphoreValue(semaphore11, 0);
  CheckSemaphoreValue(semaphore21, 0);
  CheckSemaphoreValue(semaphore22, 0);

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer11_semaphore_wait_list,
      /*signal_semaphore_list=*/command_buffer11_semaphore_signal_list, 1,
      &command_buffer11, /*binding_tables=*/NULL));

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

// Test host + device -> device synchronization: submit two batches
// with a later batch waiting on both a host and device signal to proceed.
TEST_F(SemaphoreSubmissionTest, BatchWaitingOnAnotherAndHostSignal) {
  // Signal/wait relation:
  //
  // command_buffer1
  //        ↓
  //    semaphore1   semaphore2
  //        ↓       ↙
  // command_buffer2
  //        ↓
  //    semaphore3

  iree_hal_command_buffer_t* command_buffer1 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer2 = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore3 = CreateSemaphore();

  uint64_t semaphore_signal_value = 1;

  // Submit command_buffer2.
  iree_hal_semaphore_t* command_buffer2_wait_semaphore_array[] = {semaphore1,
                                                                  semaphore2};
  uint64_t command_buffer2_wait_value_array[] = {semaphore_signal_value,
                                                 semaphore_signal_value};
  iree_hal_semaphore_list_t command_buffer2_wait_list = {
      /*count=*/2, command_buffer2_wait_semaphore_array,
      command_buffer2_wait_value_array};
  iree_hal_semaphore_list_t command_buffer2_signal_list = {
      /*count=*/1, &semaphore3, &semaphore_signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer2_wait_list,
      /*signal_semaphore_list=*/command_buffer2_signal_list, 1,
      &command_buffer2, /*binding_tables=*/NULL));

  // semaphore3 must not have advanced, because it depends on semaphore1 and
  // semaphore2, which have not been signaled yet.
  CheckSemaphoreValue(semaphore3, 0);

  // Submit command_buffer1.
  iree_hal_semaphore_list_t command_buffer1_wait_list = {/*count=*/0, nullptr,
                                                         nullptr};
  iree_hal_semaphore_list_t command_buffer1_signal_list = {
      /*count=*/1, &semaphore1, &semaphore_signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer1_wait_list,
      /*signal_semaphore_list=*/command_buffer1_signal_list, 1,
      &command_buffer1, /*binding_tables=*/NULL));

  // semaphore3 must not have advanced still, because it depends on semaphore2,
  // which has not been signaled yet.
  CheckSemaphoreValue(semaphore3, 0);

  std::thread signal_thread([&]() {
    IREE_ASSERT_OK(
        iree_hal_semaphore_signal(semaphore2, semaphore_signal_value));
  });

  // Wait and check that semaphore3 has advanced.
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore3, semaphore_signal_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore3, semaphore_signal_value);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_semaphore_release(semaphore3);
  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
}

// Test device -> host + device synchronization: submit two batches
// with a former batch signaling to enable both host and device to proceed.
TEST_F(SemaphoreSubmissionTest, DeviceBatchSignalAnotherAndHost) {
  // Signal-wait relation:
  //
  //         command_buffer1
  //         ↙             ↘
  //    semaphore11    semaphore12
  //        ↓               ↓
  // command_buffer2   wait on host
  //        ↓
  //    semaphore2
  //        ↓
  //   wait on host

  iree_hal_command_buffer_t* command_buffer1 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer2 = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore11 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore12 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();

  uint64_t signal_value = 1;

  // Submit command_buffer2.
  iree_hal_semaphore_list_t command_buffer2_wait_list = {
      /*count=*/1, &semaphore11, &signal_value};
  iree_hal_semaphore_list_t command_buffer2_signal_list = {
      /*count=*/1, &semaphore2, &signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer2_wait_list,
      /*signal_semaphore_list=*/command_buffer2_signal_list, 1,
      &command_buffer2, /*binding_tables=*/NULL));

  // Semaphores have not advance since we have not yet submitted
  // command_buffer1.
  CheckSemaphoreValue(semaphore11, 0);
  CheckSemaphoreValue(semaphore12, 0);
  CheckSemaphoreValue(semaphore2, 0);

  // Wait in parallel for all semaphores.
  std::thread thread11([&]() {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(semaphore11, signal_value,
                                iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  });
  std::thread thread12([&]() {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(semaphore12, signal_value,
                                iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  });
  std::thread thread2([&]() {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(semaphore2, signal_value,
                                iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  });

  // Submit command_buffer1.
  iree_hal_semaphore_list_t command_buffer1_wait_list = {/*count=*/0, nullptr,
                                                         nullptr};
  iree_hal_semaphore_t* command_buffer1_signal_semaphore_array[] = {
      semaphore11, semaphore12};
  uint64_t command_buffer1_signal_value_array[] = {signal_value, signal_value};
  iree_hal_semaphore_list_t command_buffer1_signal_list = {
      /*count=*/2, command_buffer1_signal_semaphore_array,
      command_buffer1_signal_value_array};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer1_wait_list,
      /*signal_semaphore_list=*/command_buffer1_signal_list, 1,
      &command_buffer1, /*binding_tables=*/NULL));

  thread11.join();
  thread12.join();
  thread2.join();

  // Check that semaphores have advanced.
  CheckSemaphoreValue(semaphore11, signal_value);
  CheckSemaphoreValue(semaphore12, signal_value);
  CheckSemaphoreValue(semaphore2, signal_value);

  iree_hal_semaphore_release(semaphore11);
  iree_hal_semaphore_release(semaphore12);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
}

// Test signaling a larger value before enqueuing waiting a smaller
// value to the device.
TEST_F(SemaphoreSubmissionTest, BatchWaitingOnSmallerValueAfterSignaled) {
  // signal-wait relation:
  //
  //   signal value 2
  //         ↓
  //     semaphore1
  //         ↓
  //    wait value 1
  //         ↓
  //  command_buffer
  //         ↓
  //     semaphore2

  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();

  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore1, 2));

  // Submit the command buffer.
  uint64_t semaphore1_wait_value = 1;
  iree_hal_semaphore_list_t command_buffer_wait_list = {
      /*count=*/1, &semaphore1, &semaphore1_wait_value};
  uint64_t semaphore2_signal_value = 1;
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1, &semaphore2, &semaphore2_signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer_wait_list,
      /*signal_semaphore_list=*/command_buffer_signal_list, 1, &command_buffer,
      /*binding_tables=*/NULL));

  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore2, semaphore2_signal_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore2, semaphore2_signal_value);

  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_command_buffer_release(command_buffer);
}

// Test signaling a larger value after enqueuing waiting a smaller
// value to the device.
TEST_F(SemaphoreSubmissionTest, BatchWaitingOnSmallerValueBeforeSignaled) {
  // signal-wait relation:
  //
  //   signal value 2
  //         ↓
  //     semaphore1
  //         ↓
  //    wait value 1
  //         ↓
  //  command_buffer
  //         ↓
  //     semaphore2

  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();

  // Submit the command buffer.
  uint64_t semaphore1_wait_value = 1;
  iree_hal_semaphore_list_t command_buffer_wait_list = {
      /*count=*/1, &semaphore1, &semaphore1_wait_value};
  uint64_t semaphore2_signal_value = 1;
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1, &semaphore2, &semaphore2_signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer_wait_list,
      /*signal_semaphore_list=*/command_buffer_signal_list, 1, &command_buffer,
      /*binding_tables=*/NULL));

  std::thread signal_thread(
      [&]() { IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore1, 2)); });

  // We don't explicitly make sure that the signal has been submitted.
  // In the majority of cases by time the signaling thread starts executing,
  // the waiting would have begun.
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(semaphore2, semaphore2_signal_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE)));
  CheckSemaphoreValue(semaphore2, semaphore2_signal_value);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_command_buffer_release(command_buffer);
}

// Submit an batch and check that the wait semaphore fails when the signal
// semaphore fails.
TEST_F(SemaphoreSubmissionTest, PropagateFailSignal) {
  // signal-wait relation:
  //
  //     semaphore1
  //         ↓
  //  command_buffer
  //         ↓
  //     semaphore2

  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();

  // Submit the command buffer.
  uint64_t semaphore1_wait_value = 1;
  iree_hal_semaphore_list_t command_buffer_wait_list = {
      /*count=*/1, &semaphore1, &semaphore1_wait_value};
  uint64_t semaphore2_signal_value = 1;
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1, &semaphore2, &semaphore2_signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer_wait_list,
      /*signal_semaphore_list=*/command_buffer_signal_list, 1, &command_buffer,
      /*binding_tables=*/NULL));

  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "PropagateFailSignal test.");
  std::thread signal_thread([&]() {
    iree_hal_semaphore_fail(semaphore1, iree_status_clone(status));
  });

  iree_status_t wait_status =
      iree_hal_semaphore_wait(semaphore2, semaphore2_signal_value,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE));
  EXPECT_EQ(iree_status_code(wait_status), IREE_STATUS_ABORTED);
  uint64_t value = 1234;
  iree_status_t query_status = iree_hal_semaphore_query(semaphore2, &value);
  EXPECT_EQ(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  CheckStatusContains(query_status, status);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_command_buffer_release(command_buffer);
  iree_status_ignore(status);
  iree_status_ignore(wait_status);
  iree_status_ignore(query_status);
}

// Submit an invalid dispatch and check that the wait semaphore fails.
TEST_F(SemaphoreSubmissionTest, PropagateDispatchFailure) {
  // signal-wait relation:
  //
  //     semaphore1
  //         ↓
  //  command_buffer
  //         ↓
  //     semaphore2

  // TODO (sogartar):
  // I tried to add a kernel that stores into a null pointer or
  // traps(aborts), but with HIP that causes the whole executable to abort,
  // which is not what we want.
  // We want a failure of the kernel launch or when waiting on the stream for
  // the kernel to complete.
  // This needs to be "soft" failure that result in a returned error from the
  // underlying API call.
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_SEMAPHORE_SUBMISSION_TEST_H_
