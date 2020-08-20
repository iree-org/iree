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

#include "iree/hal/deferred_buffer.h"

#include "absl/memory/memory.h"
#include "iree/hal/heap_buffer.h"
#include "iree/hal/testing/mock_allocator.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace {

using ::iree::hal::testing::MockAllocator;
using ::testing::_;
using ::testing::Return;

// Tests properties of unbound buffers.
TEST(DeferredBufferTest, Unbound) {
  MockAllocator allocator;
  auto deferred_buffer = absl::make_unique<DeferredBuffer>(
      &allocator, MemoryType::kHostLocal, MemoryAccess::kAll, BufferUsage::kAll,
      100);
  EXPECT_EQ(&allocator, deferred_buffer->allocator());
  EXPECT_EQ(deferred_buffer.get(), deferred_buffer->allocated_buffer());
  EXPECT_EQ(0, deferred_buffer->allocation_size());
  EXPECT_EQ(0, deferred_buffer->byte_offset());
  EXPECT_EQ(100, deferred_buffer->byte_length());
}

// Tests that binding verifies allocators are compatible.
TEST(DeferredBufferTest, AllocatorCheck) {
  MockAllocator allocator;
  auto deferred_buffer = absl::make_unique<DeferredBuffer>(
      &allocator, MemoryType::kHostLocal, MemoryAccess::kAll, BufferUsage::kAll,
      100);
  auto real_buffer =
      HeapBuffer::Allocate(MemoryType::kHostLocal, BufferUsage::kAll, 256);
  EXPECT_CALL(
      allocator,
      CanUseBufferLike(real_buffer->allocator(), real_buffer->memory_type(),
                       real_buffer->usage(), BufferUsage::kAll))
      .WillOnce(Return(false));
  EXPECT_TRUE(IsInvalidArgument(
      deferred_buffer->BindAllocation(std::move(real_buffer), 0, 100)));
}

// Tests that binding verifies allocation sizes.
TEST(DeferredBufferTest, SizeCheck) {
  MockAllocator allocator;
  auto deferred_buffer = absl::make_unique<DeferredBuffer>(
      &allocator, MemoryType::kHostLocal, MemoryAccess::kAll, BufferUsage::kAll,
      100);
  auto real_buffer =
      HeapBuffer::Allocate(MemoryType::kHostLocal, BufferUsage::kAll, 256);
  EXPECT_CALL(allocator, CanUseBufferLike(_, _, _, _))
      .WillRepeatedly(Return(true));

  IREE_EXPECT_OK(
      deferred_buffer->BindAllocation(add_ref(real_buffer), 10, 100));
  EXPECT_EQ(256, deferred_buffer->allocation_size());
  EXPECT_EQ(10, deferred_buffer->byte_offset());
  EXPECT_EQ(100, deferred_buffer->byte_length());
  IREE_EXPECT_OK(
      deferred_buffer->BindAllocation(add_ref(real_buffer), 10, kWholeBuffer));
  EXPECT_EQ(256, deferred_buffer->allocation_size());
  EXPECT_EQ(10, deferred_buffer->byte_offset());
  EXPECT_EQ(100, deferred_buffer->byte_length());

  EXPECT_TRUE(IsOutOfRange(
      deferred_buffer->BindAllocation(add_ref(real_buffer), 200, 100)));
  EXPECT_TRUE(IsOutOfRange(deferred_buffer->BindAllocation(add_ref(real_buffer),
                                                           200, kWholeBuffer)));
  EXPECT_TRUE(IsOutOfRange(
      deferred_buffer->BindAllocation(add_ref(real_buffer), 10, 10)));
}

// Tests resizing buffers after they have been allocated.
TEST(DeferredBufferTest, Resizing) {
  MockAllocator allocator;
  auto deferred_buffer = absl::make_unique<DeferredBuffer>(
      &allocator, MemoryType::kHostLocal, MemoryAccess::kAll, BufferUsage::kAll,
      100);
  auto real_buffer =
      HeapBuffer::Allocate(MemoryType::kHostLocal, BufferUsage::kAll, 256);
  EXPECT_CALL(allocator, CanUseBufferLike(_, _, _, _))
      .WillRepeatedly(Return(true));

  // Grow.
  EXPECT_EQ(100, deferred_buffer->byte_length());
  IREE_EXPECT_OK(deferred_buffer->GrowByteLength(150));
  EXPECT_EQ(150, deferred_buffer->byte_length());

  // Shrinking should fail.
  EXPECT_TRUE(IsInvalidArgument(deferred_buffer->GrowByteLength(5)));

  // Growing should fail if bound.
  IREE_EXPECT_OK(
      deferred_buffer->BindAllocation(std::move(real_buffer), 0, 150));
  EXPECT_TRUE(IsFailedPrecondition(deferred_buffer->GrowByteLength(100)));
}

// Tests binding and rebinding behavior.
TEST(DeferredBufferTest, Rebinding) {
  MockAllocator allocator;
  auto deferred_buffer = absl::make_unique<DeferredBuffer>(
      &allocator, MemoryType::kHostLocal, MemoryAccess::kAll, BufferUsage::kAll,
      100);
  auto real_buffer =
      HeapBuffer::Allocate(MemoryType::kHostLocal, BufferUsage::kAll, 256);
  EXPECT_CALL(allocator, CanUseBufferLike(_, _, _, _))
      .WillRepeatedly(Return(true));

  // Safe to reset when not bound.
  deferred_buffer->ResetAllocation();
  EXPECT_EQ(deferred_buffer.get(), deferred_buffer->allocated_buffer());
  EXPECT_EQ(0, deferred_buffer->allocation_size());

  IREE_EXPECT_OK(deferred_buffer->BindAllocation(add_ref(real_buffer), 0, 100));
  EXPECT_EQ(real_buffer.get(), deferred_buffer->allocated_buffer());
  EXPECT_EQ(256, deferred_buffer->allocation_size());
  deferred_buffer->ResetAllocation();
  EXPECT_EQ(deferred_buffer.get(), deferred_buffer->allocated_buffer());
  EXPECT_EQ(0, deferred_buffer->allocation_size());
  IREE_EXPECT_OK(deferred_buffer->BindAllocation(add_ref(real_buffer), 0, 100));
  EXPECT_EQ(real_buffer.get(), deferred_buffer->allocated_buffer());
  EXPECT_EQ(256, deferred_buffer->allocation_size());
}

// Tests normal usage of bound buffers.
TEST(DeferredBufferTest, BoundUsage) {
  MockAllocator allocator;
  auto deferred_buffer = absl::make_unique<DeferredBuffer>(
      &allocator, MemoryType::kHostLocal, MemoryAccess::kAll, BufferUsage::kAll,
      100);
  auto real_buffer =
      HeapBuffer::Allocate(MemoryType::kHostLocal, BufferUsage::kAll, 256);
  EXPECT_CALL(allocator, CanUseBufferLike(_, _, _, _))
      .WillRepeatedly(Return(true));
  IREE_EXPECT_OK(
      deferred_buffer->BindAllocation(std::move(real_buffer), 0, 100));

  EXPECT_FALSE(deferred_buffer->DebugString().empty());
  EXPECT_FALSE(deferred_buffer->DebugStringShort().empty());

  IREE_EXPECT_OK(deferred_buffer->Fill8(0, 10, 0xFF));
}

// Tests that unbound buffers fail to perform any buffer actions.
TEST(DeferredBufferTest, UnboundUsage) {
  MockAllocator allocator;
  auto deferred_buffer = absl::make_unique<DeferredBuffer>(
      &allocator, MemoryType::kHostLocal, MemoryAccess::kAll, BufferUsage::kAll,
      100);
  EXPECT_FALSE(deferred_buffer->DebugString().empty());
  EXPECT_FALSE(deferred_buffer->DebugStringShort().empty());

  EXPECT_TRUE(IsFailedPrecondition(deferred_buffer->Fill8(0, 10, 0xFF)));
}

}  // namespace
}  // namespace hal
}  // namespace iree
