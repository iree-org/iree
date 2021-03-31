// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace cts {

class SemaphoreSubmissionTest : public CtsTestBase {
 public:
  // Disable cuda backend for this test as semaphores are not implemented yet.
  SemaphoreSubmissionTest() { declareUnavailableDriver("cuda"); }
};

TEST_P(SemaphoreSubmissionTest, SubmitWithNoCommandBuffers) {
  // No waits, one signal which we immediately wait on after submit.
  iree_hal_submission_batch_t submission_batch;
  submission_batch.wait_semaphores.count = 0;
  submission_batch.wait_semaphores.semaphores = NULL;
  submission_batch.wait_semaphores.payload_values = NULL;
  submission_batch.command_buffer_count = 0;
  submission_batch.command_buffers = NULL;
  iree_hal_semaphore_t* signal_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphore_ptrs[] = {signal_semaphore};
  submission_batch.signal_semaphores.count =
      IREE_ARRAYSIZE(signal_semaphore_ptrs);
  submission_batch.signal_semaphores.semaphores = signal_semaphore_ptrs;
  uint64_t payload_values[] = {1ull};
  submission_batch.signal_semaphores.payload_values = payload_values;

  IREE_ASSERT_OK(
      iree_hal_device_queue_submit(device_, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                   /*queue_affinity=*/0,
                                   /*batch_count=*/1, &submission_batch));
  IREE_ASSERT_OK(iree_hal_semaphore_wait_with_deadline(
      signal_semaphore, 1ull, IREE_TIME_INFINITE_FUTURE));

  iree_hal_semaphore_release(signal_semaphore);
}

TEST_P(SemaphoreSubmissionTest, SubmitAndSignal) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // No waits, one signal which we immediately wait on after submit.
  iree_hal_submission_batch_t submission_batch;
  submission_batch.wait_semaphores.count = 0;
  submission_batch.wait_semaphores.semaphores = NULL;
  submission_batch.wait_semaphores.payload_values = NULL;
  submission_batch.command_buffer_count = 1;
  submission_batch.command_buffers = &command_buffer;
  iree_hal_semaphore_t* signal_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &signal_semaphore));
  iree_hal_semaphore_t* signal_semaphore_ptrs[] = {signal_semaphore};
  submission_batch.signal_semaphores.count =
      IREE_ARRAYSIZE(signal_semaphore_ptrs);
  submission_batch.signal_semaphores.semaphores = signal_semaphore_ptrs;
  uint64_t payload_values[] = {1ull};
  submission_batch.signal_semaphores.payload_values = payload_values;

  IREE_ASSERT_OK(
      iree_hal_device_queue_submit(device_, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                   /*queue_affinity=*/0,
                                   /*batch_count=*/1, &submission_batch));
  IREE_ASSERT_OK(iree_hal_semaphore_wait_with_deadline(
      signal_semaphore, 1ull, IREE_TIME_INFINITE_FUTURE));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_P(SemaphoreSubmissionTest, SubmitWithWait) {
  // Empty command buffer.
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // One wait and one signal semaphore.
  iree_hal_submission_batch_t submission_batch;
  iree_hal_semaphore_t* wait_semaphore;
  iree_hal_semaphore_t* signal_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &wait_semaphore));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 100ull, &signal_semaphore));
  iree_hal_semaphore_t* wait_semaphore_ptrs[] = {wait_semaphore};
  iree_hal_semaphore_t* signal_semaphore_ptrs[] = {signal_semaphore};
  uint64_t wait_payload_values[] = {1ull};
  uint64_t signal_payload_values[] = {101ull};
  submission_batch.wait_semaphores.count = IREE_ARRAYSIZE(wait_semaphore_ptrs);
  submission_batch.wait_semaphores.semaphores = wait_semaphore_ptrs;
  submission_batch.wait_semaphores.payload_values = wait_payload_values;
  submission_batch.command_buffer_count = 1;
  submission_batch.command_buffers = &command_buffer;
  submission_batch.signal_semaphores.count =
      IREE_ARRAYSIZE(signal_semaphore_ptrs);
  submission_batch.signal_semaphores.semaphores = signal_semaphore_ptrs;
  submission_batch.signal_semaphores.payload_values = signal_payload_values;

  IREE_ASSERT_OK(
      iree_hal_device_queue_submit(device_, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                   /*queue_affinity=*/0,
                                   /*batch_count=*/1, &submission_batch));

  // Work shouldn't start until the wait semaphore reaches its payload value.
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(signal_semaphore, &value));
  EXPECT_EQ(100ull, value);

  // Signal the wait semaphore, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore, 1ull));
  IREE_ASSERT_OK(iree_hal_semaphore_wait_with_deadline(
      signal_semaphore, 101ull, IREE_TIME_INFINITE_FUTURE));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_semaphore_release(signal_semaphore);
}

TEST_P(SemaphoreSubmissionTest, SubmitWithMultipleSemaphores) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_submission_batch_t submission_batch;
  iree_hal_semaphore_t* wait_semaphore_1;
  iree_hal_semaphore_t* wait_semaphore_2;
  iree_hal_semaphore_t* signal_semaphore_1;
  iree_hal_semaphore_t* signal_semaphore_2;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &wait_semaphore_1));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &wait_semaphore_2));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &signal_semaphore_1));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &signal_semaphore_2));
  iree_hal_semaphore_t* wait_semaphore_ptrs[] = {wait_semaphore_1,
                                                 wait_semaphore_2};
  iree_hal_semaphore_t* signal_semaphore_ptrs[] = {signal_semaphore_1,
                                                   signal_semaphore_2};
  uint64_t wait_payload_values[] = {1ull, 1ull};
  uint64_t signal_payload_values[] = {1ull, 1ull};
  submission_batch.wait_semaphores.count = IREE_ARRAYSIZE(wait_semaphore_ptrs);
  submission_batch.wait_semaphores.semaphores = wait_semaphore_ptrs;
  submission_batch.wait_semaphores.payload_values = wait_payload_values;
  submission_batch.command_buffer_count = 1;
  submission_batch.command_buffers = &command_buffer;
  submission_batch.signal_semaphores.count =
      IREE_ARRAYSIZE(signal_semaphore_ptrs);
  submission_batch.signal_semaphores.semaphores = signal_semaphore_ptrs;
  submission_batch.signal_semaphores.payload_values = signal_payload_values;

  IREE_ASSERT_OK(
      iree_hal_device_queue_submit(device_, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                   /*queue_affinity=*/0,
                                   /*batch_count=*/1, &submission_batch));

  // Work shouldn't start until all wait semaphores reach their payload values.
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(signal_semaphore_1, &value));
  EXPECT_EQ(0ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_query(signal_semaphore_2, &value));
  EXPECT_EQ(0ull, value);

  // Signal the wait semaphores, work should begin and complete.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore_1, 1ull));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore_2, 1ull));

  iree_hal_semaphore_list_t signal_semaphore_list;
  signal_semaphore_list.count = IREE_ARRAYSIZE(signal_semaphore_ptrs);
  signal_semaphore_list.semaphores = signal_semaphore_ptrs;
  uint64_t payload_values[] = {1ull, 1ull};
  signal_semaphore_list.payload_values = payload_values;
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores_with_deadline(
      device_, IREE_HAL_WAIT_MODE_ALL, &signal_semaphore_list,
      IREE_TIME_INFINITE_FUTURE));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_semaphore_release(wait_semaphore_1);
  iree_hal_semaphore_release(wait_semaphore_2);
  iree_hal_semaphore_release(signal_semaphore_1);
  iree_hal_semaphore_release(signal_semaphore_2);
}

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, SemaphoreSubmissionTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
