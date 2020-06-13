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

// Tests for the MemoryMapping RAII wrapper.
// This uses a mock buffer implementation such that it is only testing
// MemoryMapping and not any real underlying memory mapping behavior.

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/types/span.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/hal/buffer.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
class Allocator;

namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

static void* const kValidPtr = reinterpret_cast<void*>(0xBEEFCAFEF00D1234ull);

class MockBuffer : public Buffer {
 public:
  using MappingMode = Buffer::MappingMode;

  MockBuffer(Allocator* allocator, MemoryTypeBitfield memory_type,
             MemoryAccessBitfield allowed_access, BufferUsageBitfield usage,
             device_size_t allocation_size)
      : Buffer(allocator, memory_type, allowed_access, usage, allocation_size,
               0, allocation_size) {}

  MOCK_METHOD(Status, FillImpl,
              (device_size_t byte_offset, device_size_t byte_length,
               const void* pattern, device_size_t pattern_length),
              (override));

  MOCK_METHOD(Status, ReadDataImpl,
              (device_size_t source_offset, void* data,
               device_size_t data_length),
              (override));
  MOCK_METHOD(Status, WriteDataImpl,
              (device_size_t target_offset, const void* data,
               device_size_t data_length),
              (override));
  MOCK_METHOD(Status, CopyDataImpl,
              (device_size_t target_offset, Buffer* source_buffer,
               device_size_t source_offset, device_size_t data_length),
              (override));

  MOCK_METHOD(Status, MapMemoryImpl,
              (MappingMode mapping_mode, MemoryAccessBitfield memory_access,
               device_size_t local_byte_offset, device_size_t local_byte_length,
               void** out_data),
              (override));
  MOCK_METHOD(Status, UnmapMemoryImpl,
              (device_size_t local_byte_offset, device_size_t local_byte_length,
               void* data),
              (override));
  MOCK_METHOD(Status, InvalidateMappedMemoryImpl,
              (device_size_t local_byte_offset,
               device_size_t local_byte_length),
              (override));
  MOCK_METHOD(Status, FlushMappedMemoryImpl,
              (device_size_t local_byte_offset,
               device_size_t local_byte_length),
              (override));
};

TEST(MemoryMappingTest, MapWholeBuffer) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kAll,
                           BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mapping,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kRead));
  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mapping.reset();
}

TEST(MemoryMappingTest, MapPartialBuffer) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kAll,
                           BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead, 4, 12, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mapping,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kRead, 4, 12));
  EXPECT_CALL(*buffer, UnmapMemoryImpl(4, 12, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mapping.reset();
}

TEST(MemoryMappingTest, EmptyHandle) {
  MappedMemory<uint8_t> mm_a;
  MappedMemory<uint8_t> mm_b;
  mm_a = std::move(mm_b);
  EXPECT_EQ(nullptr, mm_a.buffer());
  EXPECT_EQ(0, mm_a.byte_offset());
  EXPECT_EQ(0, mm_a.byte_length());
  EXPECT_TRUE(mm_a.empty());
  EXPECT_EQ(0, mm_a.size());
  EXPECT_EQ(nullptr, mm_a.data());
  EXPECT_EQ(nullptr, mm_a.mutable_data());
  EXPECT_TRUE(IsFailedPrecondition(mm_a.Subspan().status()));
  EXPECT_TRUE(IsFailedPrecondition(mm_a.MutableSubspan().status()));
  EXPECT_TRUE(IsFailedPrecondition(mm_a.Invalidate()));
  EXPECT_TRUE(IsFailedPrecondition(mm_a.Flush()));
  mm_a.reset();
}

TEST(MemoryMappingTest, MoveHandle) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kAll,
                           BufferUsage::kAll, 128);

  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_a,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kRead));

  // Should be able to move the handle around without having any calls.
  auto mm_b = std::move(mm_a);
  mm_a = std::move(mm_b);
  mm_b = std::move(mm_a);

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_b.reset();
}

TEST(MemoryMappingTest, ReadOnlyAccess) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kRead,
                           BufferUsage::kAll, 128);

  // Should succeed to map for reading.
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_r,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kRead));

  // Non-mutable access is fine.
  EXPECT_EQ(kValidPtr, mm_r.data());
  ASSERT_OK_AND_ASSIGN(auto span, mm_r.Subspan());
  (void)span;

  // Read-only mappings should not be able to get mutable access.
  EXPECT_EQ(nullptr, mm_r.mutable_data());
  EXPECT_TRUE(IsPermissionDenied(mm_r.MutableSubspan().status()));

  // Read-only mappings should not be able to call Flush.
  EXPECT_TRUE(IsPermissionDenied(mm_r.Flush()));

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_r.reset();

  // Should fail to map for writing.
  EXPECT_TRUE(IsPermissionDenied(
      buffer->MapMemory<uint8_t>(MemoryAccess::kWrite).status()));
}

TEST(MemoryMappingTest, ReadWriteAccess) {
  auto buffer = make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal,
                                     MemoryAccess::kRead | MemoryAccess::kWrite,
                                     BufferUsage::kAll, 128);

  // Should succeed to map for reading and/or writing.
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead | MemoryAccess::kWrite,
                                     0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(
      auto mm_rw,
      buffer->MapMemory<uint8_t>(MemoryAccess::kRead | MemoryAccess::kWrite));

  // Everything valid.
  EXPECT_EQ(kValidPtr, mm_rw.data());
  ASSERT_OK_AND_ASSIGN(auto span, mm_rw.Subspan());
  EXPECT_EQ(kValidPtr, mm_rw.mutable_data());
  ASSERT_OK_AND_ASSIGN(span, mm_rw.MutableSubspan());

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_rw.reset();

  // Should fail to map for discard.
  EXPECT_TRUE(IsPermissionDenied(
      buffer->MapMemory<uint8_t>(MemoryAccess::kDiscardWrite).status()));
}

TEST(MemoryMappingTest, WriteOnlyAccess) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal,
                           MemoryAccess::kWrite, BufferUsage::kAll, 128);

  // Should succeed to map for writing.
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kWrite, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_w,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));

  // Mutable access is valid.
  EXPECT_EQ(kValidPtr, mm_w.mutable_data());
  ASSERT_OK_AND_ASSIGN(auto span, mm_w.MutableSubspan());
  (void)span;

  // Write-only mappings should not be able to get non-mutable access.
  EXPECT_EQ(nullptr, mm_w.data());
  EXPECT_TRUE(IsPermissionDenied(mm_w.Subspan().status()));

  // Write-only mappings should not be able to call Invalidate.
  EXPECT_TRUE(IsPermissionDenied(mm_w.Invalidate()));

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_w.reset();

  // Should fail to map for reading.
  EXPECT_TRUE(IsPermissionDenied(
      buffer->MapMemory<uint8_t>(MemoryAccess::kRead).status()));

  // Should fail to map for discard.
  EXPECT_TRUE(IsPermissionDenied(
      buffer->MapMemory<uint8_t>(MemoryAccess::kDiscardWrite).status()));
}

TEST(MemoryMappingTest, WriteDiscardAccess) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal,
                           MemoryAccess::kDiscardWrite, BufferUsage::kAll, 128);

  // Should succeed to map for writing with discard.
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kDiscardWrite, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_dw,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kDiscardWrite));
  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_dw.reset();

  // Should also be ok to map for just writing.
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kWrite, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_w,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));
  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_w.reset();

  // Should fail to map for reading.
  EXPECT_TRUE(IsPermissionDenied(
      buffer->MapMemory<uint8_t>(MemoryAccess::kRead).status()));
}

TEST(MemoryMappingTest, Subspan) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kAll,
                           BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_r,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kRead));

  // Request some valid ranges and ensure the byte offsets are correct.
  ASSERT_OK_AND_ASSIGN(auto ss, mm_r.Subspan());
  EXPECT_EQ(kValidPtr, ss.data());
  EXPECT_EQ(128, ss.size());
  ASSERT_OK_AND_ASSIGN(ss, mm_r.Subspan(100, 2));
  EXPECT_EQ(static_cast<const uint8_t*>(kValidPtr) + 100, ss.data());
  EXPECT_EQ(2, ss.size());
  ASSERT_OK_AND_ASSIGN(ss, mm_r.Subspan(100, kWholeBuffer));
  EXPECT_EQ(static_cast<const uint8_t*>(kValidPtr) + 100, ss.data());
  EXPECT_EQ(28, ss.size());

  // Zero length ranges are fine.
  ASSERT_OK_AND_ASSIGN(ss, mm_r.Subspan(0, 0));
  EXPECT_EQ(kValidPtr, ss.data());
  EXPECT_TRUE(ss.empty());
  ASSERT_OK_AND_ASSIGN(ss, mm_r.Subspan(128, 0));
  EXPECT_EQ(static_cast<const uint8_t*>(kValidPtr) + 128, ss.data());
  EXPECT_TRUE(ss.empty());
  ASSERT_OK_AND_ASSIGN(ss, mm_r.Subspan(128, kWholeBuffer));
  EXPECT_EQ(static_cast<const uint8_t*>(kValidPtr) + 128, ss.data());
  EXPECT_TRUE(ss.empty());

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_r.reset();
}

TEST(MemoryMappingTest, SubspanOutOfRange) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kAll,
                           BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_r,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kRead));

  // Try some invalid ranges that would overrun the span.
  EXPECT_TRUE(IsOutOfRange(mm_r.Subspan(1234, 0).status()));
  EXPECT_TRUE(IsOutOfRange(mm_r.Subspan(1234, 2).status()));
  EXPECT_TRUE(IsOutOfRange(mm_r.Subspan(1234, kWholeBuffer).status()));
  EXPECT_TRUE(IsOutOfRange(mm_r.Subspan(100, 1234).status()));

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_r.reset();
}

TEST(MemoryMappingTest, MutableSubspan) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kAll,
                           BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kWrite, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_w,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));

  // Request some valid ranges and ensure the byte offsets are correct.
  ASSERT_OK_AND_ASSIGN(auto ss, mm_w.MutableSubspan());
  EXPECT_EQ(kValidPtr, ss.data());
  EXPECT_EQ(128, ss.size());
  ASSERT_OK_AND_ASSIGN(ss, mm_w.MutableSubspan(100, 2));
  EXPECT_EQ(static_cast<const uint8_t*>(kValidPtr) + 100, ss.data());
  EXPECT_EQ(2, ss.size());
  ASSERT_OK_AND_ASSIGN(ss, mm_w.MutableSubspan(100, kWholeBuffer));
  EXPECT_EQ(static_cast<const uint8_t*>(kValidPtr) + 100, ss.data());
  EXPECT_EQ(28, ss.size());

  // Zero length ranges are fine.
  ASSERT_OK_AND_ASSIGN(ss, mm_w.MutableSubspan(0, 0));
  EXPECT_EQ(kValidPtr, ss.data());
  EXPECT_TRUE(ss.empty());
  ASSERT_OK_AND_ASSIGN(ss, mm_w.MutableSubspan(128, 0));
  EXPECT_EQ(static_cast<const uint8_t*>(kValidPtr) + 128, ss.data());
  EXPECT_TRUE(ss.empty());
  ASSERT_OK_AND_ASSIGN(ss, mm_w.MutableSubspan(128, kWholeBuffer));
  EXPECT_EQ(static_cast<const uint8_t*>(kValidPtr) + 128, ss.data());
  EXPECT_TRUE(ss.empty());

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_w.reset();
}

TEST(MemoryMappingTest, MutableSubspanOutOfRange) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kAll,
                           BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kWrite, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_w,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));

  // Try some invalid ranges that would overrun the span.
  EXPECT_TRUE(IsOutOfRange(mm_w.MutableSubspan(1234, 0).status()));
  EXPECT_TRUE(IsOutOfRange(mm_w.MutableSubspan(1234, 2).status()));
  EXPECT_TRUE(IsOutOfRange(mm_w.MutableSubspan(1234, kWholeBuffer).status()));
  EXPECT_TRUE(IsOutOfRange(mm_w.MutableSubspan(100, 1234).status()));

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_w.reset();
}

TEST(MemoryMappingTest, ElementOperator) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kAll,
                           BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_r,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kRead));

  // Just verify we are getting the expected pointer back.
  EXPECT_EQ(kValidPtr, &mm_r[0]);

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_r.reset();
}

TEST(MemoryMappingTest, Invalidate) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostVisible,
                           MemoryAccess::kAll, BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_r,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kRead));

  // Invalidate a few ways.
  EXPECT_CALL(*buffer, InvalidateMappedMemoryImpl(0, 128))
      .WillOnce(Return(OkStatus()));
  EXPECT_OK(mm_r.Invalidate());
  EXPECT_CALL(*buffer, InvalidateMappedMemoryImpl(100, 2))
      .WillOnce(Return(OkStatus()));
  EXPECT_OK(mm_r.Invalidate(100, 2));
  EXPECT_CALL(*buffer, InvalidateMappedMemoryImpl(100, 28))
      .WillOnce(Return(OkStatus()));
  EXPECT_OK(mm_r.Invalidate(100, kWholeBuffer));

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_r.reset();
}

TEST(MemoryMappingTest, InvalidateOutOfRange) {
  auto buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostVisible,
                           MemoryAccess::kAll, BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kRead, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_r,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kRead));

  // Try to invalidate invalid ranges.
  EXPECT_TRUE(IsOutOfRange(mm_r.Invalidate(1234, 0)));
  EXPECT_TRUE(IsOutOfRange(mm_r.Invalidate(1234, 12345)));
  EXPECT_TRUE(IsOutOfRange(mm_r.Invalidate(1234, kWholeBuffer)));
  EXPECT_TRUE(IsOutOfRange(mm_r.Invalidate(1, 1234)));

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_r.reset();
}

TEST(MemoryMappingTest, InvalidateBadMode) {
  // Invalidate is not required on coherent memory.
  auto coherent_buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostLocal, MemoryAccess::kAll,
                           BufferUsage::kAll, 128);
  EXPECT_CALL(*coherent_buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                              MemoryAccess::kRead, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(
      auto mm_r, coherent_buffer->MapMemory<uint8_t>(MemoryAccess::kRead));
  EXPECT_TRUE(IsPermissionDenied(mm_r.Invalidate()));
  EXPECT_CALL(*coherent_buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_r.reset();
}

TEST(MemoryMappingTest, Flush) {
  auto buffer = make_ref<MockBuffer>(
      nullptr, MemoryType::kHostVisible | MemoryType::kHostCached,
      MemoryAccess::kAll, BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kWrite, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_w,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));

  // Flush a few ways.
  EXPECT_CALL(*buffer, FlushMappedMemoryImpl(0, 128))
      .WillOnce(Return(OkStatus()));
  EXPECT_OK(mm_w.Flush());
  EXPECT_CALL(*buffer, FlushMappedMemoryImpl(100, 2))
      .WillOnce(Return(OkStatus()));
  EXPECT_OK(mm_w.Flush(100, 2));
  EXPECT_CALL(*buffer, FlushMappedMemoryImpl(100, 28))
      .WillOnce(Return(OkStatus()));
  EXPECT_OK(mm_w.Flush(100, kWholeBuffer));

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_w.reset();
}

TEST(MemoryMappingTest, FlushOutOfRange) {
  auto buffer = make_ref<MockBuffer>(
      nullptr, MemoryType::kHostVisible | MemoryType::kHostCached,
      MemoryAccess::kAll, BufferUsage::kAll, 128);
  EXPECT_CALL(*buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                     MemoryAccess::kWrite, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(auto mm_w,
                       buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));

  // Try to flush invalid ranges.
  EXPECT_TRUE(IsOutOfRange(mm_w.Flush(1234, 0)));
  EXPECT_TRUE(IsOutOfRange(mm_w.Flush(1234, 12345)));
  EXPECT_TRUE(IsOutOfRange(mm_w.Flush(1234, kWholeBuffer)));
  EXPECT_TRUE(IsOutOfRange(mm_w.Flush(1, 1234)));

  EXPECT_CALL(*buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_w.reset();
}

TEST(MemoryMappingTest, FlushBadMode) {
  // Flush is not required on uncached memory.
  auto uncached_buffer =
      make_ref<MockBuffer>(nullptr, MemoryType::kHostVisible,
                           MemoryAccess::kAll, BufferUsage::kAll, 128);
  EXPECT_CALL(*uncached_buffer, MapMemoryImpl(MockBuffer::MappingMode::kScoped,
                                              MemoryAccess::kWrite, 0, 128, _))
      .WillOnce(DoAll(SetArgPointee<4>(kValidPtr), Return(OkStatus())));
  ASSERT_OK_AND_ASSIGN(
      auto mm_w, uncached_buffer->MapMemory<uint8_t>(MemoryAccess::kWrite));
  EXPECT_TRUE(IsPermissionDenied(mm_w.Flush()));
  EXPECT_CALL(*uncached_buffer, UnmapMemoryImpl(0, 128, kValidPtr))
      .WillOnce(Return(OkStatus()));
  mm_w.reset();
}

}  // namespace
}  // namespace hal
}  // namespace iree
