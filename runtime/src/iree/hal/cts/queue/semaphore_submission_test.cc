// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <thread>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class SemaphoreSubmissionTest : public CtsTestBase<> {};

TEST_P(SemaphoreSubmissionTest, SubmitWithNoCommandBuffers) {
  // No waits, one signal which we immediately wait on after submit.
  iree_hal_semaphore_t* signal_semaphore = CreateSemaphore();
  uint64_t signal_payload_values[] = {1};
  iree_hal_semaphore_list_t signal_semaphores = {
      1,
      &signal_semaphore,
      signal_payload_values,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_barrier(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal_semaphore, 1,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_semaphore_release(signal_semaphore);
}

TEST_P(SemaphoreSubmissionTest, SubmitAndSignal) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  iree_hal_semaphore_t* signal_semaphore = CreateSemaphore();
  uint64_t signal_payload_values[] = {1};
  iree_hal_semaphore_list_t signal_semaphores = {
      1,
      &signal_semaphore,
      signal_payload_values,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_semaphores, command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal_semaphore, 1,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_P(SemaphoreSubmissionTest, SubmitWithWait) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  iree_hal_semaphore_t* wait_semaphore = CreateSemaphore();
  uint64_t wait_payload_values[] = {1};
  iree_hal_semaphore_list_t wait_semaphores = {
      1,
      &wait_semaphore,
      wait_payload_values,
  };
  iree_hal_semaphore_t* signal_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, 100ull,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  uint64_t signal_payload_values[] = {101};
  iree_hal_semaphore_list_t signal_semaphores = {
      1,
      &signal_semaphore,
      signal_payload_values,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores, signal_semaphores,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));

  // Work shouldn't start until the wait semaphore reaches its payload value.
  CheckSemaphoreValue(signal_semaphore, 100);

  // Signal the wait semaphore, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore, 1));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal_semaphore, 101,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_P(SemaphoreSubmissionTest, SubmitWithMultipleSemaphores) {
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
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores, signal_semaphores,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));

  // Work shouldn't start until all wait semaphores reach their payload values.
  CheckSemaphoreValue(signal_semaphore_1, 0);
  CheckSemaphoreValue(signal_semaphore_2, 0);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore_1, 1));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore_2, 1));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_semaphores, iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(wait_semaphore_1);
  iree_hal_semaphore_release(wait_semaphore_2);
  iree_hal_semaphore_release(signal_semaphore_1);
  iree_hal_semaphore_release(signal_semaphore_2);
}

// Tests waiting on both host and device semaphores to signal.
TEST_P(SemaphoreSubmissionTest, WaitAllHostAndDeviceSemaphores) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  iree_hal_semaphore_t* host_wait_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_wait_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* host_signal_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_signal_semaphore = CreateSemaphore();

  uint64_t device_wait_payload_values[] = {1};
  iree_hal_semaphore_list_t device_wait_semaphores = {
      /*count=*/1, &device_wait_semaphore, device_wait_payload_values};
  uint64_t device_signal_payload_values[] = {1};
  iree_hal_semaphore_list_t device_signal_semaphores = {
      /*count=*/1, &device_signal_semaphore, device_signal_payload_values};

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, device_wait_semaphores,
      device_signal_semaphores, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  // Start another thread that waits on the host semaphore then signals.
  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(host_wait_semaphore, 1,
                                           iree_infinite_timeout(),
                                           IREE_HAL_WAIT_FLAG_DEFAULT));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(host_signal_semaphore, 1));
  });

  CheckSemaphoreValue(host_signal_semaphore, 0);
  CheckSemaphoreValue(device_signal_semaphore, 0);

  // Signal both wait semaphores, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_wait_semaphore, 1));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(device_wait_semaphore, 1));

  // Wait for both to complete.
  iree_hal_semaphore_t* main_semaphore_ptrs[] = {host_signal_semaphore,
                                                 device_signal_semaphore};
  uint64_t main_payload_values[] = {1, 1};
  iree_hal_semaphore_list_t main_wait_semaphores = {
      IREE_ARRAYSIZE(main_semaphore_ptrs), main_semaphore_ptrs,
      main_payload_values};
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ALL, main_wait_semaphores,
      iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));
  thread.join();

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(host_wait_semaphore);
  iree_hal_semaphore_release(device_wait_semaphore);
  iree_hal_semaphore_release(host_signal_semaphore);
  iree_hal_semaphore_release(device_signal_semaphore);
}

// Tests wait-any with host and device semaphores where the device signals
// first.
TEST_P(SemaphoreSubmissionTest,
       WaitAnyHostAndDeviceSemaphoresAndDeviceSignals) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  iree_hal_semaphore_t* host_wait_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_wait_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* host_signal_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_signal_semaphore = CreateSemaphore();

  uint64_t device_wait_payload_values[] = {1};
  iree_hal_semaphore_list_t device_wait_semaphores = {
      /*count=*/1, &device_wait_semaphore, device_wait_payload_values};
  uint64_t device_signal_payload_values[] = {1};
  iree_hal_semaphore_list_t device_signal_semaphores = {
      /*count=*/1, &device_signal_semaphore, device_signal_payload_values};

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, device_wait_semaphores,
      device_signal_semaphores, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(host_wait_semaphore, 1,
                                           iree_infinite_timeout(),
                                           IREE_HAL_WAIT_FLAG_DEFAULT));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(host_signal_semaphore, 1));
  });

  CheckSemaphoreValue(host_signal_semaphore, 0);
  CheckSemaphoreValue(device_signal_semaphore, 0);

  // Only signal the device wait semaphore.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(device_wait_semaphore, 1));

  // Wait-any: should complete when device signals (host still blocked).
  iree_hal_semaphore_t* main_semaphore_ptrs[] = {host_signal_semaphore,
                                                 device_signal_semaphore};
  uint64_t main_payload_values[] = {1, 1};
  iree_hal_semaphore_list_t main_wait_semaphores = {
      IREE_ARRAYSIZE(main_semaphore_ptrs), main_semaphore_ptrs,
      main_payload_values};
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, main_wait_semaphores,
      iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));

  // Device has signaled but host thread hasn't.
  CheckSemaphoreValue(host_signal_semaphore, 0);
  CheckSemaphoreValue(device_signal_semaphore, 1);

  // Let the host thread complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_wait_semaphore, 1));
  thread.join();

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(host_wait_semaphore);
  iree_hal_semaphore_release(device_wait_semaphore);
  iree_hal_semaphore_release(host_signal_semaphore);
  iree_hal_semaphore_release(device_signal_semaphore);
}

// Tests wait-any with host and device semaphores where the host signals first.
TEST_P(SemaphoreSubmissionTest, WaitAnyHostAndDeviceSemaphoresAndHostSignals) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();

  iree_hal_semaphore_t* host_wait_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_wait_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* host_signal_semaphore = CreateSemaphore();
  iree_hal_semaphore_t* device_signal_semaphore = CreateSemaphore();

  uint64_t device_wait_payload_values[] = {1};
  iree_hal_semaphore_list_t device_wait_semaphores = {
      /*count=*/1, &device_wait_semaphore, device_wait_payload_values};
  uint64_t device_signal_payload_values[] = {1};
  iree_hal_semaphore_list_t device_signal_semaphores = {
      /*count=*/1, &device_signal_semaphore, device_signal_payload_values};

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, device_wait_semaphores,
      device_signal_semaphores, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  std::thread thread([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(host_wait_semaphore, 1,
                                           iree_infinite_timeout(),
                                           IREE_HAL_WAIT_FLAG_DEFAULT));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(host_signal_semaphore, 1));
  });

  CheckSemaphoreValue(host_signal_semaphore, 0);
  CheckSemaphoreValue(device_signal_semaphore, 0);

  // Only signal the host wait semaphore.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_wait_semaphore, 1));

  // Wait-any: should complete when host signals (device still blocked).
  iree_hal_semaphore_t* main_semaphore_ptrs[] = {host_signal_semaphore,
                                                 device_signal_semaphore};
  uint64_t main_payload_values[] = {1, 1};
  iree_hal_semaphore_list_t main_wait_semaphores = {
      IREE_ARRAYSIZE(main_semaphore_ptrs), main_semaphore_ptrs,
      main_payload_values};
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores(
      device_, IREE_HAL_WAIT_MODE_ANY, main_wait_semaphores,
      iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));
  thread.join();

  // Host has signaled but device hasn't.
  CheckSemaphoreValue(host_signal_semaphore, 1);
  CheckSemaphoreValue(device_signal_semaphore, 0);

  // Signal and wait for device to complete too.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(device_wait_semaphore, 1));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(device_signal_semaphore, 1,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(host_wait_semaphore);
  iree_hal_semaphore_release(device_wait_semaphore);
  iree_hal_semaphore_release(host_signal_semaphore);
  iree_hal_semaphore_release(device_signal_semaphore);
}

// Device -> device synchronization: submit two batches with a semaphore
// signal -> wait dependency.
TEST_P(SemaphoreSubmissionTest, IntermediateSemaphoreBetweenDeviceBatches) {
  // command_buffer1 -> semaphore1 -> command_buffer2 -> semaphore2

  iree_hal_command_buffer_t* command_buffer1 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer2 = CreateEmptyCommandBuffer();

  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();
  uint64_t semaphore_signal_value = 1;
  iree_hal_semaphore_list_t semaphore1_list = {/*count=*/1, &semaphore1,
                                               &semaphore_signal_value};
  iree_hal_semaphore_list_t semaphore2_list = {/*count=*/1, &semaphore2,
                                               &semaphore_signal_value};

  // Submit command_buffer2 first (waits on semaphore1, signals semaphore2).
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/semaphore1_list,
      /*signal_semaphore_list=*/semaphore2_list, command_buffer2,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  // Neither semaphore should have advanced yet.
  CheckSemaphoreValue(semaphore1, 0);
  CheckSemaphoreValue(semaphore2, 0);

  // Submit command_buffer1 (no waits, signals semaphore1).
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*signal_semaphore_list=*/semaphore1_list, command_buffer1,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  // Wait on the intermediate semaphore.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore1, semaphore_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore1, semaphore_signal_value);

  // Wait on the final semaphore.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore2, semaphore_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore2, semaphore_signal_value);

  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
}

// Device -> device synchronization: multiple later batches wait on the same
// signal from a former batch.
//                  command_buffer11  command_buffer12
//                         |
//                     semaphore11
//                     /        \
//      command_buffer21         command_buffer22
//             |                        |
//        semaphore21              semaphore22
TEST_P(SemaphoreSubmissionTest, TwoBatchesWaitingOn1FormerBatchAmongst2) {
  iree_hal_command_buffer_t* command_buffer11 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer12 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer21 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer22 = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore11 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore21 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore22 = CreateSemaphore();

  uint64_t semaphore_signal_wait_value = 1;
  iree_hal_semaphore_list_t semaphore11_list = {/*count=*/1, &semaphore11,
                                                &semaphore_signal_wait_value};
  iree_hal_semaphore_list_t semaphore21_list = {/*count=*/1, &semaphore21,
                                                &semaphore_signal_wait_value};
  iree_hal_semaphore_list_t semaphore22_list = {/*count=*/1, &semaphore22,
                                                &semaphore_signal_wait_value};

  // Submit in reverse order.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/semaphore11_list,
      /*signal_semaphore_list=*/semaphore22_list, command_buffer22,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/semaphore11_list,
      /*signal_semaphore_list=*/semaphore21_list, command_buffer21,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*signal_semaphore_list=*/iree_hal_semaphore_list_empty(),
      command_buffer12, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));

  // No semaphores should have advanced yet.
  CheckSemaphoreValue(semaphore11, 0);
  CheckSemaphoreValue(semaphore21, 0);
  CheckSemaphoreValue(semaphore22, 0);

  // Submit the head of the DAG.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*signal_semaphore_list=*/semaphore11_list, command_buffer11,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  // Wait and verify all semaphores advance.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore21, semaphore_signal_wait_value, iree_infinite_timeout(),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore21, semaphore_signal_wait_value);
  CheckSemaphoreValue(semaphore11, semaphore_signal_wait_value);

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore22, semaphore_signal_wait_value, iree_infinite_timeout(),
      IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore22, semaphore_signal_wait_value);

  iree_hal_semaphore_release(semaphore11);
  iree_hal_semaphore_release(semaphore21);
  iree_hal_semaphore_release(semaphore22);
  iree_hal_command_buffer_release(command_buffer11);
  iree_hal_command_buffer_release(command_buffer12);
  iree_hal_command_buffer_release(command_buffer21);
  iree_hal_command_buffer_release(command_buffer22);
}

// A former batch signals a value greater than the different wait values
// of two later batches.
//          command_buffer11
//                 |
//           signal value 3
//                 |
//            semaphore11
//           /           \
//  wait value 1       wait value 2
//        |                  |
// command_buffer21   command_buffer22
//        |                  |
//    semaphore21       semaphore22
TEST_P(SemaphoreSubmissionTest, TwoBatchesWaitingOnDifferentSemaphoreValues) {
  iree_hal_command_buffer_t* command_buffer11 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer21 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer22 = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore11 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore21 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore22 = CreateSemaphore();

  uint64_t command_buffer11_semaphore11_signal_value = 3;
  iree_hal_semaphore_list_t command_buffer11_semaphore_wait_list =
      iree_hal_semaphore_list_empty();
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

  // Submit in reverse order.
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer22_semaphore_wait_list,
      /*signal_semaphore_list=*/command_buffer22_signal_list, command_buffer22,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer21_semaphore_wait_list,
      /*signal_semaphore_list=*/command_buffer21_signal_list, command_buffer21,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  CheckSemaphoreValue(semaphore11, 0);
  CheckSemaphoreValue(semaphore21, 0);
  CheckSemaphoreValue(semaphore22, 0);

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer11_semaphore_wait_list,
      /*signal_semaphore_list=*/command_buffer11_semaphore_signal_list,
      command_buffer11, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore21, semaphore2x_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore21, semaphore2x_signal_value);
  CheckSemaphoreValue(semaphore11, command_buffer11_semaphore11_signal_value);

  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore22, semaphore2x_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore22, semaphore2x_signal_value);

  iree_hal_semaphore_release(semaphore11);
  iree_hal_semaphore_release(semaphore21);
  iree_hal_semaphore_release(semaphore22);
  iree_hal_command_buffer_release(command_buffer11);
  iree_hal_command_buffer_release(command_buffer21);
  iree_hal_command_buffer_release(command_buffer22);
}

// Host + device -> device synchronization: a later batch waits on both a
// host signal and a device signal.
// command_buffer1
//        |
//    semaphore1   semaphore2
//        |       /
// command_buffer2
//        |
//    semaphore3
TEST_P(SemaphoreSubmissionTest, BatchWaitingOnAnotherAndHostSignal) {
  iree_hal_command_buffer_t* command_buffer1 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer2 = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore3 = CreateSemaphore();

  uint64_t semaphore_signal_value = 1;

  // Submit command_buffer2 (waits on semaphore1 + semaphore2).
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
      /*signal_semaphore_list=*/command_buffer2_signal_list, command_buffer2,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  CheckSemaphoreValue(semaphore3, 0);

  // Submit command_buffer1 (no waits, signals semaphore1).
  iree_hal_semaphore_list_t command_buffer1_signal_list = {
      /*count=*/1, &semaphore1, &semaphore_signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*signal_semaphore_list=*/command_buffer1_signal_list, command_buffer1,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  // semaphore3 still shouldn't have advanced (semaphore2 not signaled).
  CheckSemaphoreValue(semaphore3, 0);

  // Signal semaphore2 from a host thread.
  std::thread signal_thread([&]() {
    IREE_ASSERT_OK(
        iree_hal_semaphore_signal(semaphore2, semaphore_signal_value));
  });

  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore3, semaphore_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore3, semaphore_signal_value);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_semaphore_release(semaphore3);
  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
}

// Device -> host + device synchronization: a former batch signals to enable
// both host and device to proceed.
//         command_buffer1
//         /             \
//    semaphore11    semaphore12
//        |               |
// command_buffer2   wait on host
//        |
//    semaphore2
//        |
//   wait on host
TEST_P(SemaphoreSubmissionTest, DeviceBatchSignalAnotherAndHost) {
  iree_hal_command_buffer_t* command_buffer1 = CreateEmptyCommandBuffer();
  iree_hal_command_buffer_t* command_buffer2 = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore11 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore12 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();

  uint64_t signal_value = 1;

  // Submit command_buffer2 (waits on semaphore11, signals semaphore2).
  iree_hal_semaphore_list_t command_buffer2_wait_list = {
      /*count=*/1, &semaphore11, &signal_value};
  iree_hal_semaphore_list_t command_buffer2_signal_list = {
      /*count=*/1, &semaphore2, &signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer2_wait_list,
      /*signal_semaphore_list=*/command_buffer2_signal_list, command_buffer2,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  CheckSemaphoreValue(semaphore11, 0);
  CheckSemaphoreValue(semaphore12, 0);
  CheckSemaphoreValue(semaphore2, 0);

  // Wait in parallel for all semaphores from host threads.
  std::thread thread11([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore11, signal_value,
                                           iree_infinite_timeout(),
                                           IREE_HAL_WAIT_FLAG_DEFAULT));
  });
  std::thread thread12([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore12, signal_value,
                                           iree_infinite_timeout(),
                                           IREE_HAL_WAIT_FLAG_DEFAULT));
  });
  std::thread thread2([&]() {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore2, signal_value,
                                           iree_infinite_timeout(),
                                           IREE_HAL_WAIT_FLAG_DEFAULT));
  });

  // Submit command_buffer1 (no waits, signals semaphore11 + semaphore12).
  iree_hal_semaphore_t* command_buffer1_signal_semaphore_array[] = {
      semaphore11, semaphore12};
  uint64_t command_buffer1_signal_value_array[] = {signal_value, signal_value};
  iree_hal_semaphore_list_t command_buffer1_signal_list = {
      /*count=*/2, command_buffer1_signal_semaphore_array,
      command_buffer1_signal_value_array};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*signal_semaphore_list=*/command_buffer1_signal_list, command_buffer1,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  thread11.join();
  thread12.join();
  thread2.join();

  CheckSemaphoreValue(semaphore11, signal_value);
  CheckSemaphoreValue(semaphore12, signal_value);
  CheckSemaphoreValue(semaphore2, signal_value);

  iree_hal_semaphore_release(semaphore11);
  iree_hal_semaphore_release(semaphore12);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_command_buffer_release(command_buffer1);
  iree_hal_command_buffer_release(command_buffer2);
}

// Signal a larger value before enqueuing a wait on a smaller value.
TEST_P(SemaphoreSubmissionTest, BatchWaitingOnSmallerValueAfterSignaled) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();

  // Signal value 2 before submitting the wait for value 1.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore1, 2));

  uint64_t semaphore1_wait_value = 1;
  iree_hal_semaphore_list_t command_buffer_wait_list = {
      /*count=*/1, &semaphore1, &semaphore1_wait_value};
  uint64_t semaphore2_signal_value = 1;
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1, &semaphore2, &semaphore2_signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer_wait_list,
      /*signal_semaphore_list=*/command_buffer_signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore2, semaphore2_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore2, semaphore2_signal_value);

  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_command_buffer_release(command_buffer);
}

// Signal a larger value after enqueuing a wait on a smaller value.
TEST_P(SemaphoreSubmissionTest, BatchWaitingOnSmallerValueBeforeSignaled) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();

  uint64_t semaphore1_wait_value = 1;
  iree_hal_semaphore_list_t command_buffer_wait_list = {
      /*count=*/1, &semaphore1, &semaphore1_wait_value};
  uint64_t semaphore2_signal_value = 1;
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1, &semaphore2, &semaphore2_signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer_wait_list,
      /*signal_semaphore_list=*/command_buffer_signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  // Signal value 2 from another thread (wait was for value 1).
  std::thread signal_thread(
      [&]() { IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore1, 2)); });

  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore2, semaphore2_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_HAL_WAIT_FLAG_DEFAULT));
  CheckSemaphoreValue(semaphore2, semaphore2_signal_value);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_command_buffer_release(command_buffer);
}

// Fail a wait semaphore and check that the signal semaphore also fails.
TEST_P(SemaphoreSubmissionTest, PropagateFailSignal) {
  iree_hal_command_buffer_t* command_buffer = CreateEmptyCommandBuffer();
  iree_hal_semaphore_t* semaphore1 = CreateSemaphore();
  iree_hal_semaphore_t* semaphore2 = CreateSemaphore();

  uint64_t semaphore1_wait_value = 1;
  iree_hal_semaphore_list_t command_buffer_wait_list = {
      /*count=*/1, &semaphore1, &semaphore1_wait_value};
  uint64_t semaphore2_signal_value = 1;
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1, &semaphore2, &semaphore2_signal_value};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/command_buffer_wait_list,
      /*signal_semaphore_list=*/command_buffer_signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  iree_status_t status =
      iree_make_status(IREE_STATUS_CANCELLED, "PropagateFailSignal test.");
  std::thread signal_thread([&]() {
    iree_hal_semaphore_fail(semaphore1, iree_status_clone(status));
  });

  iree_status_t wait_status = iree_hal_semaphore_wait(
      semaphore2, semaphore2_signal_value, iree_infinite_timeout(),
      IREE_HAL_WAIT_FLAG_DEFAULT);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, wait_status);
  uint64_t value = 1234;
  iree_status_t query_status = iree_hal_semaphore_query(semaphore2, &value);
  EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  CheckStatusContains(query_status, status);

  signal_thread.join();
  iree_hal_semaphore_release(semaphore1);
  iree_hal_semaphore_release(semaphore2);
  iree_hal_command_buffer_release(command_buffer);
}

// Requires "async_queue" because most tests submit work that waits on
// semaphores signaled after queue_execute returns. Synchronous drivers
// (local_sync) would deadlock on the inline wait.
CTS_REGISTER_TEST_SUITE_WITH_TAGS(SemaphoreSubmissionTest, {"async_queue"}, {});

}  // namespace iree::hal::cts
