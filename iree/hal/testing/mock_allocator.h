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

#ifndef IREE_HAL_TESTING_MOCK_ALLOCATOR_H_
#define IREE_HAL_TESTING_MOCK_ALLOCATOR_H_

#include "gmock/gmock.h"
#include "iree/hal/allocator.h"

namespace iree {
namespace hal {
namespace testing {

class MockAllocator : public ::testing::StrictMock<Allocator> {
 public:
  MockAllocator() : ::testing::StrictMock<Allocator>() {}

  MOCK_CONST_METHOD4(CanUseBufferLike,
                     bool(Allocator* source_allocator,
                          MemoryTypeBitfield memory_type,
                          BufferUsageBitfield buffer_usage,
                          BufferUsageBitfield intended_usage));

  MOCK_CONST_METHOD3(CanAllocate, bool(MemoryTypeBitfield memory_type,
                                       BufferUsageBitfield buffer_usage,
                                       size_t allocation_size));

  MOCK_METHOD3(Allocate,
               StatusOr<ref_ptr<Buffer>>(MemoryTypeBitfield memory_type,
                                         BufferUsageBitfield buffer_usage,
                                         size_t allocation_size));

  MOCK_METHOD5(WrapMutable,
               StatusOr<ref_ptr<Buffer>>(MemoryTypeBitfield memory_type,
                                         MemoryAccessBitfield allowed_access,
                                         BufferUsageBitfield buffer_usage,
                                         void* data, size_t data_length));
};

}  // namespace testing
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_TESTING_MOCK_ALLOCATOR_H_
