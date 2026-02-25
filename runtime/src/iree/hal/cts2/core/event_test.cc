// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts2/util/test_base.h"

namespace iree::hal::cts {

class EventTest : public CtsTestBase<> {};

TEST_P(EventTest, Create) {
  iree_hal_event_t* event = NULL;
  IREE_ASSERT_OK(iree_hal_event_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                                       IREE_HAL_EVENT_FLAG_NONE, &event));
  iree_hal_event_release(event);
}

TEST_P(EventTest, SignalAndReset) {
  iree_hal_event_t* event = NULL;
  IREE_ASSERT_OK(iree_hal_event_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                                       IREE_HAL_EVENT_FLAG_NONE, &event));

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_signal_event(
      command_buffer, event, IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS));
  IREE_ASSERT_OK(iree_hal_command_buffer_reset_event(
      command_buffer, event, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  iree_hal_event_release(event);
  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(EventTest, SubmitWithChainedCommandBuffers) {
  iree_hal_event_t* event = NULL;
  IREE_ASSERT_OK(iree_hal_event_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                                       IREE_HAL_EVENT_FLAG_NONE, &event));

  iree_hal_command_buffer_t* command_buffer_1 = NULL;
  iree_hal_command_buffer_t* command_buffer_2 = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer_1));
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer_2));

  // First command buffer signals the event when it completes.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer_1));
  IREE_ASSERT_OK(iree_hal_command_buffer_signal_event(
      command_buffer_1, event, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer_1));

  // Second command buffer waits on the event before starting.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer_2));
  const iree_hal_event_t* event_ptrs[] = {event};
  IREE_ASSERT_OK(iree_hal_command_buffer_wait_events(
      command_buffer_2, IREE_ARRAYSIZE(event_ptrs), event_ptrs,
      /*source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      /*target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      /*memory_barrier_count=*/0,
      /*memory_barriers=*/NULL, /*buffer_barrier_count=*/0,
      /*buffer_barriers=*/NULL));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer_2));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer_1));
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer_2));

  iree_hal_command_buffer_release(command_buffer_1);
  iree_hal_command_buffer_release(command_buffer_2);
  iree_hal_event_release(event);
}

CTS_REGISTER_TEST_SUITE(EventTest);

}  // namespace iree::hal::cts
