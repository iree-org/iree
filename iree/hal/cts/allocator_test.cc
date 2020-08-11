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
#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class AllocatorTest : public CtsTestBase {
 protected:
  virtual void SetUp() {
    CtsTestBase::SetUp();

    if (!device_) {
      return;
    }

    allocator_ = device_->allocator();
  }

  Allocator* allocator_ = nullptr;
};

TEST_P(AllocatorTest, CanAllocate) {
  EXPECT_TRUE(allocator_->CanAllocate(
      MemoryType::kHostLocal | MemoryType::kDeviceVisible,
      BufferUsage::kMapping, 1024));
  EXPECT_TRUE(allocator_->CanAllocate(
      MemoryType::kHostVisible | MemoryType::kDeviceLocal,
      BufferUsage::kMapping, 1024));

  // TODO(scotttodd): Minimum memory types and buffer usages necessary for use
  // TODO(scotttodd): Test upper limits of memory size for allocations (1GB+)?
}

TEST_P(AllocatorTest, Allocate) {
  MemoryType memory_type = MemoryType::kHostLocal | MemoryType::kDeviceVisible;
  BufferUsage usage = BufferUsage::kMapping;
  size_t allocation_size = 1024;

  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer, allocator_->Allocate(memory_type, usage, allocation_size));

  EXPECT_EQ(allocator_, buffer->allocator());
  // At a mimimum, the requested memory type should be respected.
  // Additional bits may be optionally set depending on the allocator.
  EXPECT_TRUE((buffer->memory_type() & memory_type) == memory_type);
  EXPECT_TRUE((buffer->usage() & usage) == usage);
  EXPECT_GE(buffer->allocation_size(), allocation_size);  // Larger is okay.
}

INSTANTIATE_TEST_SUITE_P(AllDrivers, AllocatorTest,
                         ::testing::ValuesIn(DriverRegistry::shared_registry()
                                                 ->EnumerateAvailableDrivers()),
                         GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
