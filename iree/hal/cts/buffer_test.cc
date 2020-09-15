// Copyright 2020 Google LLC
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

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;

// Note: this file only covers hal::Buffer APIs that can be overridden by
// subclasses. Errors caught by hal::Buffer's common validations are not
// covered as they are already tested in iree/hal/buffer_test.cc.

class BufferTest : public CtsTestBase {};

TEST_P(BufferTest, Allocate) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer, device_->allocator()->Allocate(
                       MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                       BufferUsage::kTransfer | BufferUsage::kMapping, 14));

  EXPECT_NE(nullptr, buffer->allocator());
  EXPECT_EQ(MemoryAccess::kAll, buffer->allowed_access());
  EXPECT_EQ(MemoryType::kHostLocal | MemoryType::kDeviceVisible,
            buffer->memory_type());
  EXPECT_EQ(BufferUsage::kTransfer | BufferUsage::kMapping, buffer->usage());

  EXPECT_LE(14, buffer->allocation_size());
  EXPECT_EQ(0, buffer->byte_offset());
  EXPECT_EQ(14, buffer->byte_length());
}

TEST_P(BufferTest, AllocateZeroLength) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer, device_->allocator()->Allocate(
                       MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                       BufferUsage::kTransfer | BufferUsage::kMapping, 0));
  EXPECT_LE(0, buffer->allocation_size());
}

TEST_P(BufferTest, Fill8) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer, device_->allocator()->Allocate(
                       MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                       BufferUsage::kTransfer | BufferUsage::kMapping, 5));

  std::vector<uint8_t> actual_data(buffer->allocation_size());

  // Fill with a sentinel.
  IREE_EXPECT_OK(buffer->Fill8(0, buffer->allocation_size(), 0x33u));
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

TEST_P(BufferTest, Fill16) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer, device_->allocator()->Allocate(
                       MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                       BufferUsage::kTransfer | BufferUsage::kMapping, 9));

  std::vector<uint8_t> actual_data(buffer->allocation_size());

  // Fill with a sentinel.
  IREE_EXPECT_OK(buffer->Fill16(0, 4, 0x1122u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x22, 0x11, 0x22, 0x11, 0, 0, 0, 0, 0));

  // Zero fills are fine.
  IREE_EXPECT_OK(buffer->Fill16(0, 0, 0x5566u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data, ElementsAre(0x22, 0x11, 0x22, 0x11, 0, 0, 0, 0, 0));

  // Fill the remaining parts of the buffer by using kWholeBuffer.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto aligned_buffer,
      device_->allocator()->Allocate(
          MemoryType::kHostLocal | MemoryType::kDeviceVisible,
          BufferUsage::kTransfer | BufferUsage::kMapping, 8));
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

TEST_P(BufferTest, Fill32) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer, device_->allocator()->Allocate(
                       MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                       BufferUsage::kTransfer | BufferUsage::kMapping, 9));

  std::vector<uint8_t> actual_data(buffer->allocation_size());

  // Fill with a sentinel.
  IREE_EXPECT_OK(buffer->Fill32(0, 8, 0x11223344u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data,
              ElementsAre(0x44, 0x33, 0x22, 0x11, 0x44, 0x33, 0x22, 0x11, 0));

  // Zero fills are fine.
  IREE_EXPECT_OK(buffer->Fill32(0, 0, 0x55667788u));
  IREE_EXPECT_OK(buffer->ReadData(0, actual_data.data(), actual_data.size()));
  EXPECT_THAT(actual_data,
              ElementsAre(0x44, 0x33, 0x22, 0x11, 0x44, 0x33, 0x22, 0x11, 0));

  // Fill the remaining parts of the buffer by using kWholeBuffer.
  IREE_ASSERT_OK_AND_ASSIGN(
      auto aligned_buffer,
      device_->allocator()->Allocate(
          MemoryType::kHostLocal | MemoryType::kDeviceVisible,
          BufferUsage::kTransfer | BufferUsage::kMapping, 8));
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

TEST_P(BufferTest, ReadWriteData) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer, device_->allocator()->Allocate(
                       MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                       BufferUsage::kTransfer | BufferUsage::kMapping, 4));

  std::vector<uint8_t> actual_data(4);

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

TEST_P(BufferTest, CopyData) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto src_buffer,
      device_->allocator()->Allocate(
          MemoryType::kHostLocal | MemoryType::kDeviceVisible,
          BufferUsage::kTransfer | BufferUsage::kMapping, src_data.size()));
  IREE_EXPECT_OK(src_buffer->WriteData(0, src_data.data(), src_data.size()));

  std::vector<uint8_t> dst_data = {0, 1, 2, 3, 4};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto dst_buffer,
      device_->allocator()->Allocate(
          MemoryType::kHostLocal | MemoryType::kDeviceVisible,
          BufferUsage::kTransfer | BufferUsage::kMapping, dst_data.size()));
  IREE_EXPECT_OK(dst_buffer->WriteData(0, dst_data.data(), dst_data.size()));

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

TEST_P(BufferTest, MapMemory) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      device_->allocator()->Allocate(
          MemoryType::kHostLocal | MemoryType::kDeviceVisible,
          BufferUsage::kTransfer | BufferUsage::kMapping, src_data.size()));
  IREE_EXPECT_OK(buffer->WriteData(0, src_data.data(), src_data.size()));

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

TEST_P(BufferTest, MapMemoryNonByte) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      device_->allocator()->Allocate(
          MemoryType::kHostLocal | MemoryType::kDeviceVisible,
          BufferUsage::kTransfer | BufferUsage::kMapping, src_data.size()));
  IREE_EXPECT_OK(buffer->WriteData(0, src_data.data(), src_data.size()));

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

TEST_P(BufferTest, MapMemoryWrite) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      device_->allocator()->Allocate(
          MemoryType::kHostLocal | MemoryType::kDeviceVisible,
          BufferUsage::kTransfer | BufferUsage::kMapping, src_data.size()));
  IREE_EXPECT_OK(buffer->WriteData(0, src_data.data(), src_data.size()));

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

TEST_P(BufferTest, MapMemoryDiscard) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      device_->allocator()->Allocate(
          MemoryType::kHostLocal | MemoryType::kDeviceVisible,
          BufferUsage::kTransfer | BufferUsage::kMapping, src_data.size()));
  IREE_EXPECT_OK(buffer->WriteData(0, src_data.data(), src_data.size()));

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

TEST_P(BufferTest, MapMemorySubspan) {
  std::vector<uint8_t> src_data = {0, 1, 2, 3, 4, 5, 6};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto parent_buffer,
      device_->allocator()->Allocate(
          MemoryType::kHostLocal | MemoryType::kDeviceVisible,
          BufferUsage::kTransfer | BufferUsage::kMapping, src_data.size()));
  IREE_EXPECT_OK(parent_buffer->WriteData(0, src_data.data(), src_data.size()));

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
}

INSTANTIATE_TEST_SUITE_P(AllDrivers, BufferTest,
                         ::testing::ValuesIn(DriverRegistry::shared_registry()
                                                 ->EnumerateAvailableDrivers()),
                         GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
