// Copyright 2019 Google LLC
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

#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/driver_registry.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace cts {

class CommandBufferTest : public CtsTestBase {};

TEST_P(CommandBufferTest, CreateCommandBuffer) {
  ASSERT_OK_AND_ASSIGN(auto command_buffer, device_->CreateCommandBuffer(
                                                CommandBufferMode::kOneShot,
                                                CommandCategory::kDispatch));

  EXPECT_TRUE((command_buffer->mode() & CommandBufferMode::kOneShot) ==
              CommandBufferMode::kOneShot);
  EXPECT_TRUE((command_buffer->command_categories() &
               CommandCategory::kDispatch) == CommandCategory::kDispatch);
  EXPECT_FALSE(command_buffer->is_recording());
}

// TODO(scotttodd): Begin, End, UpdateBuffer, CopyBuffer, Dispatch, Sync, etc.

INSTANTIATE_TEST_SUITE_P(AllDrivers, CommandBufferTest,
                         ::testing::ValuesIn(DriverRegistry::shared_registry()
                                                 ->EnumerateAvailableDrivers()),
                         GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
