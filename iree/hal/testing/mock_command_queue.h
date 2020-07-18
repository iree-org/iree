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

#ifndef IREE_HAL_TESTING_MOCK_COMMAND_QUEUE_H_
#define IREE_HAL_TESTING_MOCK_COMMAND_QUEUE_H_

#include "iree/hal/command_queue.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace testing {

class MockCommandQueue : public ::testing::StrictMock<CommandQueue> {
 public:
  MockCommandQueue(std::string name,
                   CommandCategoryBitfield supported_categories)
      : ::testing::StrictMock<CommandQueue>(std::move(name),
                                            supported_categories) {}

  MOCK_METHOD(Status, Submit, (absl::Span<const SubmissionBatch> batches),
              (override));

  MOCK_METHOD(Status, WaitIdle, (Time deadline_ns), (override));
};

}  // namespace testing
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_TESTING_MOCK_COMMAND_QUEUE_H_
