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

// Tests for the shared buffer functionality and host heap buffers.
// This does not test device-specific buffer implementations; see the device
// code for associated tests.

#include "iree/hal/buffer.h"

#include <vector>

#include "iree/hal/heap_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Not;

TEST(BufferTest, Allocate) {
  auto buffer =
      HeapBuffer::Allocate(BufferUsage::kTransfer | BufferUsage::kMapping, 14);
  EXPECT_NE(nullptr, buffer->allocator());
  EXPECT_EQ(MemoryAccess::kAll, buffer->allowed_access());
  EXPECT_EQ(MemoryType::kHostLocal, buffer->memory_type());
  EXPECT_EQ(BufferUsage::kTransfer | BufferUsage::kMapping, buffer->usage());

  // We don't currently do any padding on the host.
  // Other implementations may differ.
  EXPECT_GE(14, buffer->allocation_size());
  EXPECT_EQ(0, buffer->byte_offset());
  EXPECT_EQ(14, buffer->byte_length());

  // Data should be zeroed by default.
  std::vector<uint8_t> zero_data(buffer->allocation_size());
  std::vector<uint8_t> actual_data(buffer->allocation_size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(zero_data));
}

TEST(BufferTest, AllocateZeroLength) {
  auto buffer =
      HeapBuffer::Allocate(BufferUsage::kTransfer | BufferUsage::kMapping, 0);
  EXPECT_NE(nullptr, buffer->allocator());
  EXPECT_EQ(MemoryType::kHostLocal, buffer->memory_type());
  EXPECT_EQ(BufferUsage::kTransfer | BufferUsage::kMapping, buffer->usage());
  EXPECT_EQ(0, buffer->allocation_size());
}

TEST(BufferTest, AllocateCopy) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  EXPECT_NE(nullptr, buffer->allocator());
  EXPECT_GE(src_data.size(), buffer->allocation_size());

  // Data should have been copied.
  std::vector<uint8_t> actual_data(src_data.size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(src_data));

  // Modify the source data and ensure it is not reflected in the buffer.
  src_data[0] = 0x88;
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Not(Eq(src_data)));
}

TEST(BufferTest, AllocateCopyZeroLength) {
  std::vector<uint8_t> src_data;
  auto buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  EXPECT_NE(nullptr, buffer->allocator());
  EXPECT_EQ(0, buffer->allocation_size());
}

TEST(BufferTest, AllocateCopyTyped) {
  std::vector<int32_t> src_data = {0, 1, 2, 3};
  auto buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               absl::MakeConstSpan(src_data));
  EXPECT_NE(nullptr, buffer->allocator());
  EXPECT_EQ(MemoryType::kHostLocal, buffer->memory_type());
  EXPECT_EQ(BufferUsage::kTransfer | BufferUsage::kMapping, buffer->usage());
  EXPECT_GE(src_data.size() * sizeof(int32_t), buffer->allocation_size());

  // Data should have been copied.
  std::vector<int32_t> actual_data(src_data.size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(),
                                  actual_data.size() * sizeof(int32_t)));
  EXPECT_THAT(actual_data, Eq(src_data));
}

TEST(BufferTest, WrapConstant) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer = HeapBuffer::Wrap(MemoryType::kHostLocal,
                                 BufferUsage::kTransfer | BufferUsage::kMapping,
                                 absl::MakeConstSpan(src_data));
  EXPECT_EQ(MemoryType::kHostLocal, buffer->memory_type());
  EXPECT_EQ(BufferUsage::kTransfer | BufferUsage::kMapping, buffer->usage());
  EXPECT_EQ(src_data.size(), buffer->allocation_size());

  // src_data and buffer should match after the wrapping.
  std::vector<uint8_t> actual_data(src_data.size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(src_data));

  // Modify the source data directly.
  src_data[0] = 123;
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(src_data));

  // Attempts to modify the buffer should fail.
  std::vector<uint8_t> new_data = {3, 2, 1, 0};
  EXPECT_TRUE(IsPermissionDenied(
      buffer->WriteData(0, new_data.data(), new_data.size())));
}

TEST(BufferTest, WrapMutable) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer = HeapBuffer::WrapMutable(
      MemoryType::kHostLocal, MemoryAccess::kAll,
      BufferUsage::kTransfer | BufferUsage::kMapping, absl::MakeSpan(src_data));
  EXPECT_EQ(MemoryType::kHostLocal, buffer->memory_type());
  EXPECT_EQ(BufferUsage::kTransfer | BufferUsage::kMapping, buffer->usage());
  EXPECT_EQ(src_data.size(), buffer->allocation_size());

  // src_data and buffer should match after the wrapping.
  std::vector<uint8_t> actual_data(src_data.size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(src_data));

  // Modify the source data directly.
  src_data[0] = 123;
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(src_data));

  // Modify the source data via the Buffer and ensure reflected in src_data.
  std::vector<uint8_t> new_data = {3, 2, 1, 0};
  IREE_EXPECT_OK(buffer->WriteData(0, new_data.data(), new_data.size()));
  EXPECT_THAT(src_data, Eq(new_data));
}

TEST(BufferTest, WrapExternal) {
  // This is not fully supported yet, but does let us verify that the validation
  // of memory types is working.
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer = HeapBuffer::Wrap(MemoryType::kDeviceLocal, BufferUsage::kAll,
                                 absl::MakeConstSpan(src_data));
  EXPECT_EQ(MemoryType::kDeviceLocal, buffer->memory_type());

  // Should fail (for now) as the buffer is not host visible.
  EXPECT_TRUE(IsPermissionDenied(buffer->Fill8(0, kWholeBuffer, 0x99u)));
}

TEST(BufferTest, DoesOverlap) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto parent_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());

  // A buffer should overlap with itself.
  EXPECT_FALSE(Buffer::DoesOverlap(parent_buffer.get(), 0, 1,
                                   parent_buffer.get(), 1, 1));
  EXPECT_TRUE(Buffer::DoesOverlap(parent_buffer.get(), 0, 1,
                                  parent_buffer.get(), 0, 1));

  // Zero length buffers never overlap.
  EXPECT_FALSE(Buffer::DoesOverlap(parent_buffer.get(), 1, 1,
                                   parent_buffer.get(), 1, 0));

  // Subspans should offset within their allocation.
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer_0,
                            Buffer::Subspan(parent_buffer, 1, 2));
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer_1,
                            Buffer::Subspan(parent_buffer, 2, 2));
  EXPECT_FALSE(Buffer::DoesOverlap(subspan_buffer_0.get(), 0, 1,
                                   subspan_buffer_1.get(), 0, 1));
  EXPECT_TRUE(Buffer::DoesOverlap(subspan_buffer_0.get(), 1, 1,
                                  subspan_buffer_1.get(), 0, 1));

  // Mixing subspans and normal buffers.
  EXPECT_FALSE(Buffer::DoesOverlap(parent_buffer.get(), 0, 1,
                                   subspan_buffer_0.get(), 0, 1));
  EXPECT_TRUE(Buffer::DoesOverlap(parent_buffer.get(), 1, 2,
                                  subspan_buffer_0.get(), 1, 1));

  // Independent buffers should not be able to overlap.
  auto other_buffer = HeapBuffer::Allocate(BufferUsage::kAll, 128);
  EXPECT_FALSE(Buffer::DoesOverlap(parent_buffer.get(), 0, kWholeBuffer,
                                   other_buffer.get(), 0, kWholeBuffer));
}

TEST(BufferTest, Subspan) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto parent_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(parent_buffer);

  // Create a subspan of the buffer.
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer,
                            Buffer::Subspan(parent_buffer, 1, 2));
  ASSERT_TRUE(subspan_buffer);
  EXPECT_EQ(1, subspan_buffer->byte_offset());
  EXPECT_EQ(2, subspan_buffer->byte_length());

  // Modifications to either buffer should appear in the other.
  IREE_EXPECT_OK(subspan_buffer->Fill8(1, kWholeBuffer, 0xFFu));
  std::vector<uint8_t> actual_data(src_data.size());
  IREE_EXPECT_OK(
      parent_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 1, 0xFF, 3));

  // Subspans should be able to create subspans.
  // NOTE: offset is from the original buffer.
  IREE_ASSERT_OK_AND_ASSIGN(auto subsubspan_buffer,
                            Buffer::Subspan(subspan_buffer, 1, 1));
  ASSERT_TRUE(subsubspan_buffer);
  EXPECT_EQ(2, subsubspan_buffer->byte_offset());
  EXPECT_EQ(1, subsubspan_buffer->byte_length());

  // Zero length subspans are fine.
  IREE_ASSERT_OK_AND_ASSIGN(auto zero_subspan_buffer,
                            Buffer::Subspan(parent_buffer, 0, 0));
  ASSERT_TRUE(zero_subspan_buffer);
  EXPECT_EQ(0, zero_subspan_buffer->byte_offset());
  EXPECT_EQ(0, zero_subspan_buffer->byte_length());

  // Subspan with kWholeBuffer should get the remaining size (or zero).
  IREE_ASSERT_OK_AND_ASSIGN(auto whole_subspan_buffer,
                            Buffer::Subspan(parent_buffer, 1, kWholeBuffer));
  ASSERT_TRUE(whole_subspan_buffer);
  EXPECT_EQ(1, whole_subspan_buffer->byte_offset());
  EXPECT_EQ(3, whole_subspan_buffer->byte_length());

  // Zero length subspans are fine.
  IREE_ASSERT_OK(Buffer::Subspan(subspan_buffer, 2, 0));
  IREE_ASSERT_OK(Buffer::Subspan(subspan_buffer, 2, kWholeBuffer));
}

TEST(BufferTest, SubspanIdentity) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto parent_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());

  // Asking for a subspan of the entire buffer should return the same buffer.
  // Mostly an optimization.
  EXPECT_EQ(parent_buffer.get(),
            Buffer::Subspan(parent_buffer, 0, kWholeBuffer).value().get());
  EXPECT_EQ(parent_buffer.get(),
            Buffer::Subspan(parent_buffer, 0, 4).value().get());
}

TEST(BufferTest, SubspanOutOfRange) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto parent_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(parent_buffer);

  // Create a subspan of the buffer.
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer,
                            Buffer::Subspan(parent_buffer, 1, 2));
  ASSERT_TRUE(subspan_buffer);
  EXPECT_EQ(1, subspan_buffer->byte_offset());
  EXPECT_EQ(2, subspan_buffer->byte_length());

  // Try to make subspans from invalid ranges.
  EXPECT_TRUE(IsOutOfRange(Buffer::Subspan(parent_buffer, 5, 0).status()));
  EXPECT_TRUE(
      IsOutOfRange(Buffer::Subspan(parent_buffer, 5, kWholeBuffer).status()));
  EXPECT_TRUE(IsOutOfRange(Buffer::Subspan(parent_buffer, 4, 1).status()));
  EXPECT_TRUE(IsOutOfRange(Buffer::Subspan(parent_buffer, 0, 123).status()));
  EXPECT_TRUE(IsOutOfRange(Buffer::Subspan(subspan_buffer, 1, 2).status()));
  EXPECT_TRUE(IsOutOfRange(Buffer::Subspan(subspan_buffer, 0, 44).status()));
}

TEST(BufferTest, Fill8) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 5);
  ASSERT_TRUE(buffer);

  // Data should be zeroed by default.
  std::vector<uint8_t> actual_data(buffer->allocation_size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 0, 0, 0, 0));

  // Fill with a sentinel.
  IREE_EXPECT_OK(buffer->Fill8(0, buffer->allocation_size(), 0x33u));

  // Verify data.
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x33, 0x33, 0x33, 0x33, 0x33));

  // Zero fills are fine.
  IREE_EXPECT_OK(buffer->Fill8(0, 0, 0x44u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x33, 0x33, 0x33, 0x33, 0x33));

  // Fill the remaining parts of the buffer by using kWholeBuffer.
  IREE_EXPECT_OK(buffer->Fill8(2, kWholeBuffer, 0x55u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x33, 0x33, 0x55, 0x55, 0x55));

  // Fill a small region of the buffer.
  IREE_EXPECT_OK(buffer->Fill8(1, 1, 0x66u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x33, 0x66, 0x55, 0x55, 0x55));

  // Whole buffer helper.
  IREE_EXPECT_OK(buffer->Fill8(0x99u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x99, 0x99, 0x99, 0x99, 0x99));
}

TEST(BufferTest, Fill8OutOfRange) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 5);
  ASSERT_TRUE(buffer);

  // Fill with a sentinel.
  IREE_EXPECT_OK(buffer->Fill8(0, buffer->allocation_size(), 0x33u));

  // Try to fill with invalid ranges.
  EXPECT_TRUE(IsOutOfRange(buffer->Fill8(1, 444, 0x44u)));
  EXPECT_TRUE(IsOutOfRange(buffer->Fill8(123, 444, 0x44u)));
  EXPECT_TRUE(IsOutOfRange(buffer->Fill8(123, 1, 0x44u)));
  EXPECT_TRUE(IsOutOfRange(buffer->Fill8(1, 444, 0x44u)));

  // Ensure nothing happened with the bad ranges.
  std::vector<uint8_t> actual_data(buffer->allocation_size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x33, 0x33, 0x33, 0x33, 0x33));
}

TEST(BufferTest, Fill8BadMode) {
  // Fail to fill buffers not supporting mapping.
  auto nonmapping_buffer = HeapBuffer::Allocate(BufferUsage::kTransfer, 4);
  EXPECT_TRUE(
      IsPermissionDenied(nonmapping_buffer->Fill8(0, kWholeBuffer, 0x99u)));

  // Fail to fill constant buffers.
  std::vector<uint8_t> const_data = {1, 2, 3};
  auto constant_buffer =
      HeapBuffer::Wrap(MemoryType::kHostLocal, BufferUsage::kMapping,
                       absl::MakeConstSpan(const_data));
  EXPECT_TRUE(
      IsPermissionDenied(constant_buffer->Fill8(0, kWholeBuffer, 0x99u)));
}

TEST(BufferTest, Fill8Subspan) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 5);
  ASSERT_TRUE(buffer);

  // Test on subspan.
  std::vector<uint8_t> actual_data(buffer->allocation_size());
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer, Buffer::Subspan(buffer, 1, 3));
  IREE_EXPECT_OK(subspan_buffer->Fill8(2, kWholeBuffer, 0xDDu));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 0, 0, 0xDD, 0));
}

TEST(BufferTest, Fill16) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 9);
  ASSERT_TRUE(buffer);

  // Data should be zeroed by default.
  std::vector<uint8_t> actual_data(buffer->allocation_size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0));

  // Fill with a sentinel.
  IREE_EXPECT_OK(buffer->Fill16(0, 4, 0x1122u));

  // Verify data.
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x22, 0x11, 0x22, 0x11, 0, 0, 0, 0, 0));

  // Zero fills are fine.
  IREE_EXPECT_OK(buffer->Fill16(0, 0, 0x5566u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x22, 0x11, 0x22, 0x11, 0, 0, 0, 0, 0));

  // Fill the remaining parts of the buffer by using kWholeBuffer.
  auto aligned_buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 8);
  IREE_EXPECT_OK(aligned_buffer->Fill16(4, kWholeBuffer, 0x5566u));
  std::vector<uint8_t> aligned_actual_data(aligned_buffer->allocation_size());
  IREE_EXPECT_OK(aligned_buffer->ReadData(0, aligned_actual_data.data(),
                                          aligned_actual_data.size()));
  EXPECT_THAT(aligned_actual_data,
              ElementsAre(0, 0, 0, 0, 0x66, 0x55, 0x66, 0x55));

  // Whole buffer helper.
  IREE_EXPECT_OK(aligned_buffer->Fill16(0x5566u));
  IREE_EXPECT_OK(aligned_buffer->ReadData(0, aligned_actual_data.data(),
                                          aligned_actual_data.size()));
  EXPECT_THAT(aligned_actual_data,
              ElementsAre(0x66, 0x55, 0x66, 0x55, 0x66, 0x55, 0x66, 0x55));
}

TEST(BufferTest, Fill16OutOfRange) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 9);
  ASSERT_TRUE(buffer);

  // Try to fill with invalid ranges.
  EXPECT_TRUE(IsOutOfRange(buffer->Fill16(4, 444, 0x5566u)));
  EXPECT_TRUE(IsOutOfRange(buffer->Fill16(128, 444, 0x5566u)));
  EXPECT_TRUE(IsOutOfRange(buffer->Fill16(128, 4, 0x5566u)));
  EXPECT_TRUE(IsOutOfRange(buffer->Fill16(4, 444, 0x5566u)));
}

TEST(BufferTest, Fill16Unaligned) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 9);
  ASSERT_TRUE(buffer);

  // Try to fill with unaligned ranges.
  EXPECT_TRUE(IsInvalidArgument(buffer->Fill16(1, 4, 0x5566u)));
  EXPECT_TRUE(IsInvalidArgument(buffer->Fill16(0, 5, 0x5566u)));
}

TEST(BufferTest, Fill16BadMode) {
  // Fail to fill buffers not supporting mapping.
  auto nonmapping_buffer = HeapBuffer::Allocate(BufferUsage::kTransfer, 4);
  EXPECT_TRUE(
      IsPermissionDenied(nonmapping_buffer->Fill16(0, kWholeBuffer, 0x99AAu)));

  // Fail to fill constant buffers.
  std::vector<uint8_t> const_data = {1, 2, 3};
  auto constant_buffer =
      HeapBuffer::Wrap(MemoryType::kHostLocal, BufferUsage::kMapping,
                       absl::MakeConstSpan(const_data));
  EXPECT_TRUE(
      IsPermissionDenied(constant_buffer->Fill16(0, kWholeBuffer, 0x99AAu)));
}

TEST(BufferTest, Fill16Subspan) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 9);
  ASSERT_TRUE(buffer);

  // Fill with a sentinel.
  IREE_EXPECT_OK(buffer->Fill16(0, 4, 0x1122u));

  // Test on subspan.
  std::vector<uint8_t> actual_data(buffer->allocation_size());
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer, Buffer::Subspan(buffer, 2, 4));
  IREE_EXPECT_OK(subspan_buffer->Fill16(2, kWholeBuffer, 0xAABBu));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data,
              ElementsAre(0x22, 0x11, 0x22, 0x11, 0xBB, 0xAA, 0, 0, 0));
}

TEST(BufferTest, Fill32) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 9);
  ASSERT_TRUE(buffer);

  // Data should be zeroed by default.
  std::vector<uint8_t> actual_data(buffer->allocation_size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0));

  // Fill with a sentinel.
  IREE_EXPECT_OK(buffer->Fill32(0, 8, 0x11223344u));

  // Verify data.
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data,
              ElementsAre(0x44, 0x33, 0x22, 0x11, 0x44, 0x33, 0x22, 0x11, 0));

  // Zero fills are fine.
  IREE_EXPECT_OK(buffer->Fill32(0, 0, 0x55667788u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data,
              ElementsAre(0x44, 0x33, 0x22, 0x11, 0x44, 0x33, 0x22, 0x11, 0));

  // Fill the remaining parts of the buffer by using kWholeBuffer.
  auto aligned_buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 8);
  IREE_EXPECT_OK(aligned_buffer->Fill32(4, kWholeBuffer, 0x55667788u));
  std::vector<uint8_t> aligned_actual_data(aligned_buffer->allocation_size());
  IREE_EXPECT_OK(aligned_buffer->ReadData(0, aligned_actual_data.data(),
                                          aligned_actual_data.size()));
  EXPECT_THAT(aligned_actual_data,
              ElementsAre(0, 0, 0, 0, 0x88, 0x77, 0x66, 0x55));

  // Whole buffer helper.
  IREE_EXPECT_OK(aligned_buffer->Fill32(0x55667788u));
  IREE_EXPECT_OK(aligned_buffer->ReadData(0, aligned_actual_data.data(),
                                          aligned_actual_data.size()));
  EXPECT_THAT(aligned_actual_data,
              ElementsAre(0x88, 0x77, 0x66, 0x55, 0x88, 0x77, 0x66, 0x55));
}

TEST(BufferTest, Fill32OutOfRange) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 9);
  ASSERT_TRUE(buffer);

  // Try to fill with invalid ranges.
  EXPECT_TRUE(IsOutOfRange(buffer->Fill32(4, 444, 0x55667788u)));
  EXPECT_TRUE(IsOutOfRange(buffer->Fill32(128, 444, 0x55667788u)));
  EXPECT_TRUE(IsOutOfRange(buffer->Fill32(128, 4, 0x55667788u)));
  EXPECT_TRUE(IsOutOfRange(buffer->Fill32(4, 444, 0x55667788u)));
}

TEST(BufferTest, Fill32Unaligned) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 9);
  ASSERT_TRUE(buffer);

  // Try to fill with unaligned ranges.
  EXPECT_TRUE(IsInvalidArgument(buffer->Fill32(1, 4, 0x55667788u)));
  EXPECT_TRUE(IsInvalidArgument(buffer->Fill32(0, 5, 0x55667788u)));
}

TEST(BufferTest, Fill32BadMode) {
  // Fail to fill buffers not supporting mapping.
  auto nonmapping_buffer = HeapBuffer::Allocate(BufferUsage::kTransfer, 4);
  EXPECT_TRUE(IsPermissionDenied(
      nonmapping_buffer->Fill32(0, kWholeBuffer, 0x99AABBCCu)));

  // Fail to fill constant buffers.
  std::vector<uint8_t> const_data = {1, 2, 3};
  auto constant_buffer =
      HeapBuffer::Wrap(MemoryType::kHostLocal, BufferUsage::kMapping,
                       absl::MakeConstSpan(const_data));
  EXPECT_TRUE(IsPermissionDenied(
      constant_buffer->Fill32(0, kWholeBuffer, 0x99AABBCCu)));
}

TEST(BufferTest, Fill32Subspan) {
  auto buffer = HeapBuffer::Allocate(BufferUsage::kMapping, 9);
  ASSERT_TRUE(buffer);

  // Fill with a sentinel.
  IREE_EXPECT_OK(buffer->Fill32(0, 8, 0x11223344u));

  // Test on subspan.
  std::vector<uint8_t> actual_data(buffer->allocation_size());
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer, Buffer::Subspan(buffer, 4, 4));
  IREE_EXPECT_OK(subspan_buffer->Fill32(0, kWholeBuffer, 0xAABBCCDDu));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data,
              ElementsAre(0x44, 0x33, 0x22, 0x11, 0xDD, 0xCC, 0xBB, 0xAA, 0));
}

TEST(BufferTest, ReadData) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Read the data back.
  std::vector<uint8_t> actual_data(src_data.size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(src_data));

  // Reading zero bytes is valid.
  std::vector<uint8_t> zero_data(0);
  IREE_EXPECT_OK(buffer->ReadData(1, zero_data.data(), 0));

  // Read a portion of the data.
  std::vector<uint8_t> partial_data(2);
  IREE_EXPECT_OK(buffer->ReadData(1, partial_data.data(), 2));
  EXPECT_THAT(partial_data, ElementsAre(1, 2));
}

TEST(BufferTest, ReadDataOutOfRange) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Try to read out of range.
  std::vector<uint8_t> partial_data(2);
  EXPECT_TRUE(IsOutOfRange(buffer->ReadData(0, partial_data.data(), 444)));
  EXPECT_TRUE(IsOutOfRange(buffer->ReadData(1230, partial_data.data(), 444)));
  EXPECT_TRUE(IsOutOfRange(buffer->ReadData(1230, partial_data.data(), 1)));
  EXPECT_TRUE(IsInvalidArgument(
      buffer->ReadData(0, partial_data.data(), kWholeBuffer)));
}

TEST(BufferTest, ReadDataBadMode) {
  // Fail to read buffers not supporting mapping.
  std::vector<uint8_t> actual_data(1);
  auto nonmapping_buffer = HeapBuffer::Allocate(BufferUsage::kTransfer, 4);
  EXPECT_TRUE(IsPermissionDenied(
      nonmapping_buffer->ReadData(0, actual_data.data(), 1)));
}

TEST(BufferTest, ReadDataSubspan) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Test on subspan.
  std::vector<uint8_t> subspan_data(1);
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer, Buffer::Subspan(buffer, 1, 2));
  IREE_EXPECT_OK(subspan_buffer->ReadData(1, subspan_data.data(), 1));
  EXPECT_THAT(subspan_data, ElementsAre(2));
}

TEST(BufferTest, WriteData) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Read the data back - should still match.
  std::vector<uint8_t> actual_data(src_data.size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(src_data));

  // Write over the entire buffer.
  std::vector<uint8_t> new_data = {10, 20, 30, 40};
  IREE_EXPECT_OK(buffer->WriteData(0, new_data.data(), new_data.size()));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(new_data));

  // Writing zero bytes is valid.
  std::vector<uint8_t> zero_data;
  IREE_EXPECT_OK(buffer->WriteData(0, zero_data.data(), 0));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(new_data));

  // Write over a portion of the buffer.
  std::vector<uint8_t> partial_data = {99};
  IREE_EXPECT_OK(
      buffer->WriteData(1, partial_data.data(), partial_data.size()));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(10, 99, 30, 40));
}

TEST(BufferTest, WriteDataOutOfRange) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Try to write out of range.
  std::vector<uint8_t> partial_data = {99};
  EXPECT_TRUE(IsOutOfRange(buffer->WriteData(0, partial_data.data(), 444)));
  EXPECT_TRUE(IsOutOfRange(buffer->WriteData(1230, partial_data.data(), 444)));
  EXPECT_TRUE(IsOutOfRange(buffer->WriteData(1230, partial_data.data(), 1)));
  EXPECT_TRUE(IsInvalidArgument(
      buffer->WriteData(0, partial_data.data(), kWholeBuffer)));
}

TEST(BufferTest, WriteDataBadMode) {
  std::vector<uint8_t> actual_data(4);

  // Fail to write buffers not supporting mapping.
  auto nonmapping_buffer = HeapBuffer::Allocate(BufferUsage::kTransfer, 4);
  EXPECT_TRUE(IsPermissionDenied(
      nonmapping_buffer->WriteData(0, actual_data.data(), 1)));

  // Fail to write to constant buffers.
  std::vector<uint8_t> const_data = {1, 2, 3};
  auto constant_buffer =
      HeapBuffer::Wrap(MemoryType::kHostLocal, BufferUsage::kTransfer,
                       absl::MakeConstSpan(const_data));
  EXPECT_TRUE(
      IsPermissionDenied(constant_buffer->WriteData(0, actual_data.data(), 2)));
}

TEST(BufferTest, WriteDataSubspan) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kTransfer | BufferUsage::kMapping,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Test on subspan.
  std::vector<uint8_t> subspan_data = {0xAA};
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer, Buffer::Subspan(buffer, 1, 2));
  IREE_EXPECT_OK(subspan_buffer->WriteData(1, subspan_data.data(), 1));
  std::vector<uint8_t> actual_data(src_data.size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 1, 0xAA, 3));
}

TEST(BufferTest, CopyData) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto src_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kMapping | BufferUsage::kTransfer,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(src_buffer);
  std::vector<uint8_t> dst_data = {0, 1, 2, 3, 4};
  auto dst_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kMapping | BufferUsage::kTransfer,
                               dst_data.data(), dst_data.size());
  ASSERT_TRUE(dst_buffer);

  // Copy of length 0 should not change the dest buffer.
  IREE_EXPECT_OK(dst_buffer->CopyData(0, src_buffer.get(), 0, 0));
  std::vector<uint8_t> actual_data(dst_data.size());
  IREE_EXPECT_OK(
      dst_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, Eq(dst_data));

  // Copy a subrange of the buffer.
  IREE_EXPECT_OK(dst_buffer->CopyData(1, src_buffer.get(), 2, 2));
  IREE_EXPECT_OK(
      dst_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 2, 3, 3, 4));

  // Copy the entire buffer using kWholeBuffer. This will adjust sizes
  // to ensure that the min buffer is taken. We test both src and dst buffer
  // offset/length calculations (note that some may end up as 0 copies).
  IREE_EXPECT_OK(dst_buffer->CopyData(3, src_buffer.get(), 0, kWholeBuffer));
  IREE_EXPECT_OK(
      dst_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 2, 3, 0, 1));
  IREE_EXPECT_OK(dst_buffer->CopyData(0, src_buffer.get(), 2, kWholeBuffer));
  IREE_EXPECT_OK(
      dst_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(2, 3, 3, 0, 1));
  IREE_EXPECT_OK(dst_buffer->CopyData(0, src_buffer.get(), 3, kWholeBuffer));
  IREE_EXPECT_OK(
      dst_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(3, 3, 3, 0, 1));
  IREE_EXPECT_OK(dst_buffer->CopyData(4, src_buffer.get(), 0, kWholeBuffer));
  IREE_EXPECT_OK(
      dst_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(3, 3, 3, 0, 0));
}

TEST(BufferTest, CopyDataOutOfRange) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto src_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kMapping | BufferUsage::kTransfer,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(src_buffer);
  std::vector<uint8_t> dst_data = {0, 1, 2, 3, 4};
  auto dst_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kMapping | BufferUsage::kTransfer,
                               dst_data.data(), dst_data.size());
  ASSERT_TRUE(dst_buffer);

  // Try to copy out of range of source and dest.
  EXPECT_TRUE(IsOutOfRange(dst_buffer->CopyData(123, src_buffer.get(), 0, 1)));
  EXPECT_TRUE(IsOutOfRange(dst_buffer->CopyData(4, src_buffer.get(), 0, 4)));
  EXPECT_TRUE(IsOutOfRange(dst_buffer->CopyData(0, src_buffer.get(), 123, 1)));
  EXPECT_TRUE(IsOutOfRange(dst_buffer->CopyData(0, src_buffer.get(), 0, 123)));
  EXPECT_TRUE(
      IsOutOfRange(dst_buffer->CopyData(123, src_buffer.get(), 123, 123)));
  EXPECT_TRUE(IsOutOfRange(dst_buffer->CopyData(0, src_buffer.get(), 123, 0)));
}

TEST(BufferTest, CopyDataOverlapping) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto src_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kMapping | BufferUsage::kTransfer,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(src_buffer);
  std::vector<uint8_t> dst_data = {0, 1, 2, 3, 4};
  auto dst_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kMapping | BufferUsage::kTransfer,
                               dst_data.data(), dst_data.size());
  ASSERT_TRUE(dst_buffer);

  // Test overlap. Non-overlapping regions should be fine, otherwise fail.
  std::vector<uint8_t> actual_data(dst_data.size());
  IREE_EXPECT_OK(dst_buffer->CopyData(0, dst_buffer.get(), 4, 1));
  IREE_EXPECT_OK(
      dst_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(4, 1, 2, 3, 4));
  EXPECT_TRUE(
      IsInvalidArgument(dst_buffer->CopyData(2, dst_buffer.get(), 0, 3)));
  EXPECT_TRUE(
      IsInvalidArgument(dst_buffer->CopyData(0, dst_buffer.get(), 0, 3)));
}

TEST(BufferTest, CopyDataBadMode) {
  // Both source and target buffers must support mapping.
  auto nonmapping_src_buffer = HeapBuffer::Allocate(BufferUsage::kTransfer, 4);
  auto nonmapping_dst_buffer = HeapBuffer::Allocate(BufferUsage::kTransfer, 4);
  EXPECT_TRUE(IsPermissionDenied(nonmapping_dst_buffer->CopyData(
      0, nonmapping_src_buffer.get(), 0, kWholeBuffer)));
  EXPECT_TRUE(IsPermissionDenied(nonmapping_src_buffer->CopyData(
      0, nonmapping_dst_buffer.get(), 0, kWholeBuffer)));

  // Fail to copy into to constant buffers.
  std::vector<uint8_t> const_data = {1, 2, 3};
  auto constant_buffer =
      HeapBuffer::Wrap(MemoryType::kHostLocal, BufferUsage::kTransfer,
                       absl::MakeConstSpan(const_data));
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto src_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kMapping | BufferUsage::kTransfer,
                               src_data.data(), src_data.size());
  EXPECT_TRUE(IsPermissionDenied(
      constant_buffer->CopyData(0, src_buffer.get(), 0, kWholeBuffer)));
}

TEST(BufferTest, CopyDataSubspan) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  auto src_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kMapping | BufferUsage::kTransfer,
                               src_data.data(), src_data.size());
  ASSERT_TRUE(src_buffer);
  std::vector<uint8_t> dst_data = {0, 1, 2, 3, 4};
  auto dst_buffer =
      HeapBuffer::AllocateCopy(BufferUsage::kMapping | BufferUsage::kTransfer,
                               dst_data.data(), dst_data.size());
  ASSERT_TRUE(dst_buffer);

  // Test on subspan.
  std::vector<uint8_t> actual_data(dst_data.size());
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_src_buffer,
                            Buffer::Subspan(src_buffer, 1, 3));
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_dst_buffer,
                            Buffer::Subspan(dst_buffer, 2, 3));
  IREE_EXPECT_OK(
      subspan_dst_buffer->CopyData(1, subspan_src_buffer.get(), 1, 2));
  IREE_EXPECT_OK(
      dst_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 1, 2, 2, 3));
}

// NOTE: more tests related specifically to MappedMemory are in
// buffer_mapping_test.cc. This tests the MapMemory operation and enough to
// ensure the memory was mapped to the correct range and the HostBuffer and
// SubspanBuffer work as intended for basic usage.
TEST(BufferTest, MapMemory) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  auto buffer = HeapBuffer::AllocateCopy(
      BufferUsage::kTransfer | BufferUsage::kMapping, MemoryAccess::kRead,
      src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // 0-length mappings are valid.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto mapping, buffer->MapMemory<uint8_t>(MemoryAccess::kRead, 0, 0));
  EXPECT_TRUE(mapping.empty());
  EXPECT_EQ(0, mapping.size());
  EXPECT_EQ(0, mapping.byte_length());
  EXPECT_NE(nullptr, mapping.data());
  IREE_ASSERT_OK_AND_ASSIGN(auto span, mapping.Subspan());
  EXPECT_TRUE(span.empty());
  mapping.reset();

  // Map the whole buffer for reading.
  IREE_ASSERT_OK_AND_ASSIGN(mapping, buffer->MapMemory<uint8_t>(
                                         MemoryAccess::kRead, 0, kWholeBuffer));
  EXPECT_EQ(src_data.size(), mapping.size());
  IREE_ASSERT_OK_AND_ASSIGN(span, mapping.Subspan());
  EXPECT_THAT(span, ElementsAre(0, 1, 2, 3, 4, 5, 6));
  mapping.reset();

  // Map a portion of the buffer for reading.
  IREE_ASSERT_OK_AND_ASSIGN(
      mapping, buffer->MapMemory<uint8_t>(MemoryAccess::kRead, 1, 2));
  EXPECT_EQ(2, mapping.size());
  IREE_ASSERT_OK_AND_ASSIGN(span, mapping.Subspan());
  EXPECT_THAT(span, ElementsAre(1, 2));
  mapping.reset();
}

TEST(BufferTest, MapMemoryNonByte) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  auto buffer = HeapBuffer::AllocateCopy(
      BufferUsage::kTransfer | BufferUsage::kMapping, MemoryAccess::kRead,
      src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Map the buffer as non-byte values.
  // Note that we'll round down to the number of valid elements at the
  // alignment.
  IREE_ASSERT_OK_AND_ASSIGN(auto mapping16,
                            buffer->MapMemory<uint16_t>(MemoryAccess::kRead));
  EXPECT_EQ(3, mapping16.size());
  EXPECT_LE(6, mapping16.byte_length());
  IREE_ASSERT_OK_AND_ASSIGN(auto span16, mapping16.Subspan());
  EXPECT_THAT(span16, ElementsAre(0x0100, 0x0302, 0x0504));
  mapping16.reset();
}

TEST(BufferTest, MapMemoryOutOfRange) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  auto buffer = HeapBuffer::AllocateCopy(
      BufferUsage::kTransfer | BufferUsage::kMapping, MemoryAccess::kRead,
      src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Test invalid mapping ranges.
  EXPECT_TRUE(IsOutOfRange(
      buffer->MapMemory<uint16_t>(MemoryAccess::kRead, 0, 123).status()));
  EXPECT_TRUE(IsOutOfRange(
      buffer->MapMemory<uint16_t>(MemoryAccess::kRead, 5, 1231).status()));
  EXPECT_TRUE(IsOutOfRange(
      buffer->MapMemory<uint16_t>(MemoryAccess::kRead, 6, kWholeBuffer)
          .status()));
  EXPECT_TRUE(IsOutOfRange(
      buffer->MapMemory<uint16_t>(MemoryAccess::kRead, 1236, 1).status()));
}

TEST(BufferTest, MapMemoryBadMode) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  auto read_buffer = HeapBuffer::AllocateCopy(
      BufferUsage::kTransfer | BufferUsage::kMapping, MemoryAccess::kRead,
      src_data.data(), src_data.size());
  ASSERT_TRUE(read_buffer);

  // Test mapping the read-only buffer for writing.
  EXPECT_TRUE(IsPermissionDenied(
      read_buffer->MapMemory<uint8_t>(MemoryAccess::kWrite).status()));
  EXPECT_TRUE(IsPermissionDenied(
      read_buffer->MapMemory<uint8_t>(MemoryAccess::kDiscardWrite).status()));
  EXPECT_TRUE(IsPermissionDenied(
      read_buffer
          ->MapMemory<uint8_t>(MemoryAccess::kRead | MemoryAccess::kDiscard)
          .status()));
  EXPECT_TRUE(IsInvalidArgument(
      read_buffer->MapMemory<uint8_t>(MemoryAccess::kNone).status()));
}

TEST(BufferTest, MapMemoryWrite) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  auto buffer = HeapBuffer::AllocateCopy(
      BufferUsage::kTransfer | BufferUsage::kMapping, MemoryAccess::kAll,
      src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Map and modify the data. We should see it when we read back.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto mapping, buffer->MapMemory<uint8_t>(MemoryAccess::kWrite, 1, 2));
  auto mutable_data = mapping.mutable_data();
  mutable_data[0] = 0xAA;
  mutable_data[1] = 0xBB;
  mapping.reset();
  std::vector<uint8_t> actual_data(src_data.size());
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 0xAA, 0xBB, 3, 4, 5, 6));
}

TEST(BufferTest, MapMemoryDiscard) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  auto buffer = HeapBuffer::AllocateCopy(
      BufferUsage::kTransfer | BufferUsage::kMapping, MemoryAccess::kAll,
      src_data.data(), src_data.size());
  ASSERT_TRUE(buffer);

  // Map for discard. Note that we can't really rely on the value of the data
  // so we just trust that it's been discarded. It's a hint, anyway. We can be
  // sure that the data we didn't want to discard is the same though.
  std::vector<uint8_t> actual_data(src_data.size());
  IREE_ASSERT_OK_AND_ASSIGN(
      auto mapping,
      buffer->MapMemory<uint8_t>(MemoryAccess::kDiscardWrite, 1, 2));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, _, _, 3, 4, 5, 6));
  mapping.reset();
}

TEST(BufferTest, MapMemorySubspan) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  auto parent_buffer = HeapBuffer::AllocateCopy(
      BufferUsage::kTransfer | BufferUsage::kMapping, MemoryAccess::kAll,
      src_data.data(), src_data.size());
  ASSERT_TRUE(parent_buffer);
  IREE_ASSERT_OK_AND_ASSIGN(auto subspan_buffer,
                            Buffer::Subspan(parent_buffer, 1, 3));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto mapping,
      subspan_buffer->MapMemory<uint8_t>(MemoryAccess::kDiscardWrite, 1, 2));
  auto* mutable_data = mapping.mutable_data();
  mutable_data[0] = 0xCC;
  mutable_data[1] = 0xDD;
  mapping.reset();

  std::vector<uint8_t> actual_data(src_data.size());
  IREE_EXPECT_OK(
      parent_buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0, 1, 0xCC, 0xDD, 4, 5, 6));

  // Just here to make coverage happy; they are currently no-ops on the host.
  // buffer_mapping_test.cc contains tests that ensure they are called
  // correctly.
  std::vector<uint8_t> external_data = {0, 1, 2, 3, 4};
  auto external_buffer = HeapBuffer::WrapMutable(
      MemoryType::kHostVisible | MemoryType::kHostCached, MemoryAccess::kAll,
      BufferUsage::kAll, absl::MakeSpan(external_data));
  IREE_ASSERT_OK_AND_ASSIGN(auto external_subspan_buffer,
                            Buffer::Subspan(external_buffer, 0, 1));
  IREE_ASSERT_OK_AND_ASSIGN(
      mapping, external_subspan_buffer->MapMemory<uint8_t>(MemoryAccess::kAll));
  IREE_EXPECT_OK(mapping.Invalidate());
  IREE_EXPECT_OK(mapping.Flush());
}

}  // namespace
}  // namespace hal
}  // namespace iree
