// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_EVENT_TEST_H_
#define IREE_HAL_CTS_EVENT_TEST_H_

#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class event_test : public CtsTestBase {};

TEST_P(event_test, Create) {
  iree_hal_event_t* event = NULL;
  IREE_ASSERT_OK(iree_hal_event_create(device_, &event));
  iree_hal_event_release(event);
}

TEST_P(event_test, SignalAndReset) {
  iree_hal_event_t* event = NULL;
  IREE_ASSERT_OK(iree_hal_event_create(device_, &event));

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_signal_event(
      command_buffer, event, IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS));
  IREE_ASSERT_OK(iree_hal_command_buffer_reset_event(
      command_buffer, event, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                            command_buffer));

  iree_hal_event_release(event);
  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(event_test, SubmitWithChainedCommandBuffers) {
  iree_hal_event_t* event = NULL;
  IREE_ASSERT_OK(iree_hal_event_create(device_, &event));

  iree_hal_command_buffer_t* command_buffer_1 = NULL;
  iree_hal_command_buffer_t* command_buffer_2 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer_1));
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer_2));

  // First command buffer signals the event when it completes.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer_1));
  IREE_ASSERT_OK(iree_hal_command_buffer_signal_event(
      command_buffer_1, event, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer_1));

  // Second command buffer waits on the event before starting.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer_2));
  const iree_hal_event_t* event_pts[] = {event};
  // TODO(scotttodd): verify execution stage usage (check Vulkan spec)
  IREE_ASSERT_OK(iree_hal_command_buffer_wait_events(
      command_buffer_2, IREE_ARRAYSIZE(event_pts), event_pts,
      /*source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      /*target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      /*memory_barrier_count=*/0,
      /*memory_barriers=*/NULL, /*buffer_barrier_count=*/0,
      /*buffer_barriers=*/NULL));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer_2));

  // No wait semaphores, one signal which we immediately wait on after submit.
  iree_hal_submission_batch_t submission_batch;
  submission_batch.wait_semaphores.count = 0;
  submission_batch.wait_semaphores.semaphores = NULL;
  submission_batch.wait_semaphores.payload_values = NULL;
  iree_hal_command_buffer_t* command_buffer_ptrs[] = {command_buffer_1,
                                                      command_buffer_2};
  submission_batch.command_buffer_count = IREE_ARRAYSIZE(command_buffer_ptrs);
  submission_batch.command_buffers = command_buffer_ptrs;
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
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(signal_semaphore, 1ull, iree_infinite_timeout()));

  iree_hal_command_buffer_release(command_buffer_1);
  iree_hal_command_buffer_release(command_buffer_2);
  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_event_release(event);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_EVENT_TEST_H_
