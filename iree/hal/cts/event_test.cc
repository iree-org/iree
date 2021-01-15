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
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class EventTest : public CtsTestBase {};

TEST_P(EventTest, Create) {
  iree_hal_event_t* event;
  IREE_ASSERT_OK(iree_hal_event_create(device_, &event));
  iree_hal_event_release(event);
}

TEST_P(EventTest, SignalAndReset) {
  iree_hal_event_t* event;
  IREE_ASSERT_OK(iree_hal_event_create(device_, &event));

  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_signal_event(
      command_buffer, event, IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS));
  IREE_ASSERT_OK(iree_hal_command_buffer_reset_event(
      command_buffer, event, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_event_release(event);
}

// TODO(scotttodd): iree_hal_command_buffer_wait_events

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, EventTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
