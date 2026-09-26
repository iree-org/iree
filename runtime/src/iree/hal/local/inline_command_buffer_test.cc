// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/inline_command_buffer.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::local {
namespace {

struct FakeBuffer {
  iree_hal_buffer_t base;
  uint8_t storage[256] = {};
  int map_count = 0;
  int unmap_count = 0;
  int flush_count = 0;
  int invalidate_count = 0;
  iree_hal_mapping_mode_t mapping_mode = 0;
  iree_hal_memory_access_t memory_access = 0;
  iree_device_size_t mapped_offset = 0;
  iree_device_size_t mapped_length = 0;
  iree_device_size_t flush_offset = 0;
  iree_device_size_t flush_length = 0;
  iree_device_size_t invalidate_offset = 0;
  iree_device_size_t invalidate_length = 0;
  iree_status_code_t map_status = IREE_STATUS_OK;
  iree_status_code_t unmap_status = IREE_STATUS_OK;
  iree_status_code_t flush_status = IREE_STATUS_OK;
  iree_status_code_t invalidate_status = IREE_STATUS_OK;
};

static void IREE_API_PTR Destroy(iree_hal_buffer_t* /*buffer*/) {
  // The fixture owns both the buffer and its storage; neither may be freed here.
}

static iree_status_t IREE_API_PTR MapRange(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_buffer_mapping_t* mapping) {
  auto* buffer = reinterpret_cast<FakeBuffer*>(base_buffer);
  ++buffer->map_count;
  buffer->mapping_mode = mapping_mode;
  buffer->memory_access = memory_access;
  buffer->mapped_offset = byte_offset;
  buffer->mapped_length = byte_length;
  if (buffer->map_status != IREE_STATUS_OK) {
    return iree_status_from_code(buffer->map_status);
  }
  // Preserve the HAL-populated mapping metadata and provide real storage so
  // iree_hal_buffer_unmap_range dispatches to our unmap callback.
  mapping->contents =
      iree_make_byte_span(buffer->storage + byte_offset, byte_length);
  return iree_ok_status();
}

static iree_status_t IREE_API_PTR UnmapRange(
    iree_hal_buffer_t* base_buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_buffer_mapping_t* mapping) {
  auto* buffer = reinterpret_cast<FakeBuffer*>(base_buffer);
  ++buffer->unmap_count;
  EXPECT_EQ(buffer->mapped_offset, byte_offset);
  EXPECT_EQ(buffer->mapped_length, byte_length);
  EXPECT_EQ(buffer->storage + byte_offset, mapping->contents.data);
  return iree_status_from_code(buffer->unmap_status);
}

static iree_status_t IREE_API_PTR InvalidateRange(
    iree_hal_buffer_t* base_buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length) {
  auto* buffer = reinterpret_cast<FakeBuffer*>(base_buffer);
  ++buffer->invalidate_count;
  buffer->invalidate_offset = byte_offset;
  buffer->invalidate_length = byte_length;
  EXPECT_EQ(buffer->map_count, 1);
  EXPECT_EQ(buffer->unmap_count, 0);
  return iree_status_from_code(buffer->invalidate_status);
}

static iree_status_t IREE_API_PTR FlushRange(
    iree_hal_buffer_t* base_buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length) {
  auto* buffer = reinterpret_cast<FakeBuffer*>(base_buffer);
  ++buffer->flush_count;
  buffer->flush_offset = byte_offset;
  buffer->flush_length = byte_length;
  EXPECT_EQ(buffer->map_count, 1);
  EXPECT_EQ(buffer->unmap_count, 0);
  return iree_status_from_code(buffer->flush_status);
}

static const iree_hal_buffer_vtable_t kFakeBufferVTable = {
    /*.recycle=*/iree_hal_buffer_recycle,
    /*.destroy=*/Destroy,
    /*.map_range=*/MapRange,
    /*.unmap_range=*/UnmapRange,
    /*.invalidate_range=*/InvalidateRange,
    /*.flush_range=*/FlushRange,
};

class InlineCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_buffer_initialize(
        /*placement=*/{}, /*allocated_buffer=*/&buffer_.base,
        /*allocation_size=*/sizeof(buffer_.storage), /*byte_offset=*/0,
        /*byte_length=*/sizeof(buffer_.storage), IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED | IREE_HAL_BUFFER_USAGE_TRANSFER,
        &kFakeBufferVTable, &buffer_.base);
    IREE_ASSERT_OK(iree_hal_inline_command_buffer_create(
        /*device_allocator=*/nullptr,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
            IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
            IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, iree_allocator_system(), &command_buffer_));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer_));
  }

  void TearDown() override {
    if (command_buffer_) {
      IREE_EXPECT_OK(iree_hal_command_buffer_end(command_buffer_));
      iree_hal_command_buffer_release(command_buffer_);
    }
    iree_hal_buffer_release(&buffer_.base);
  }

  void ExpectCounts(int map, int flush, int invalidate, int unmap) {
    EXPECT_EQ(buffer_.map_count, map);
    EXPECT_EQ(buffer_.flush_count, flush);
    EXPECT_EQ(buffer_.invalidate_count, invalidate);
    EXPECT_EQ(buffer_.unmap_count, unmap);
  }

  FakeBuffer buffer_ = {};
  iree_hal_command_buffer_t* command_buffer_ = nullptr;
};

TEST_F(InlineCommandBufferTest, FlushMapsCorrectRangeAndAccess) {
  IREE_ASSERT_OK(iree_hal_command_buffer_flush_buffer(
      command_buffer_, iree_hal_make_buffer_ref(&buffer_.base, 37, 59)));
  // All callbacks must have completed before the command returns, without end
  // or submission of the command buffer.
  ExpectCounts(/*map=*/1, /*flush=*/1, /*invalidate=*/0, /*unmap=*/1);
  EXPECT_EQ(buffer_.mapping_mode, IREE_HAL_MAPPING_MODE_SCOPED);
  EXPECT_EQ(buffer_.memory_access, IREE_HAL_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(buffer_.mapped_offset, 37);
  EXPECT_EQ(buffer_.mapped_length, 59);
  EXPECT_EQ(buffer_.flush_offset, 37);
  EXPECT_EQ(buffer_.flush_length, 59);
}

TEST_F(InlineCommandBufferTest, InvalidateMapsCorrectRangeAndAccess) {
  IREE_ASSERT_OK(iree_hal_command_buffer_invalidate_buffer(
      command_buffer_, iree_hal_make_buffer_ref(&buffer_.base, 41, 61)));
  // Check synchronous execution before ending or submitting the command buffer.
  ExpectCounts(/*map=*/1, /*flush=*/0, /*invalidate=*/1, /*unmap=*/1);
  EXPECT_EQ(buffer_.mapping_mode, IREE_HAL_MAPPING_MODE_SCOPED);
  EXPECT_EQ(buffer_.memory_access, IREE_HAL_MEMORY_ACCESS_READ);
  EXPECT_EQ(buffer_.mapped_offset, 41);
  EXPECT_EQ(buffer_.mapped_length, 61);
  EXPECT_EQ(buffer_.invalidate_offset, 41);
  EXPECT_EQ(buffer_.invalidate_length, 61);
}

TEST_F(InlineCommandBufferTest, FlushUsesMappedRange) {
  IREE_ASSERT_OK(iree_hal_command_buffer_flush_buffer(
      command_buffer_,
      iree_hal_make_buffer_ref(&buffer_.base, 53, IREE_HAL_WHOLE_BUFFER)));
  // Mapping-relative 0/WHOLE_BUFFER must resolve to the mapped tail, not the
  // entire allocation or an untranslated WHOLE_BUFFER sentinel.
  ExpectCounts(/*map=*/1, /*flush=*/1, /*invalidate=*/0, /*unmap=*/1);
  EXPECT_EQ(buffer_.mapped_offset, 53);
  EXPECT_EQ(buffer_.mapped_length, sizeof(buffer_.storage) - 53);
  EXPECT_EQ(buffer_.flush_offset, buffer_.mapped_offset);
  EXPECT_EQ(buffer_.flush_length, buffer_.mapped_length);
}

TEST_F(InlineCommandBufferTest, InvalidateUsesMappedRange) {
  IREE_ASSERT_OK(iree_hal_command_buffer_invalidate_buffer(
      command_buffer_,
      iree_hal_make_buffer_ref(&buffer_.base, 67, IREE_HAL_WHOLE_BUFFER)));
  // The invalidate helper must also translate mapping-relative 0/WHOLE_BUFFER.
  ExpectCounts(/*map=*/1, /*flush=*/0, /*invalidate=*/1, /*unmap=*/1);
  EXPECT_EQ(buffer_.mapped_offset, 67);
  EXPECT_EQ(buffer_.mapped_length, sizeof(buffer_.storage) - 67);
  EXPECT_EQ(buffer_.invalidate_offset, buffer_.mapped_offset);
  EXPECT_EQ(buffer_.invalidate_length, buffer_.mapped_length);
}

TEST_F(InlineCommandBufferTest, FlushErrorPropagatesAndStillUnmaps) {
  buffer_.flush_status = IREE_STATUS_ABORTED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_command_buffer_flush_buffer(
          command_buffer_, iree_hal_make_buffer_ref(&buffer_.base, 37, 59)));
  ExpectCounts(/*map=*/1, /*flush=*/1, /*invalidate=*/0, /*unmap=*/1);
}

TEST_F(InlineCommandBufferTest, InvalidateErrorPropagatesAndStillUnmaps) {
  buffer_.invalidate_status = IREE_STATUS_DATA_LOSS;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_command_buffer_invalidate_buffer(
          command_buffer_, iree_hal_make_buffer_ref(&buffer_.base, 41, 61)));
  ExpectCounts(/*map=*/1, /*flush=*/0, /*invalidate=*/1, /*unmap=*/1);
}

TEST_F(InlineCommandBufferTest, FlushMapErrorPropagatesWithoutFurtherCallbacks) {
  buffer_.map_status = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_command_buffer_flush_buffer(
          command_buffer_, iree_hal_make_buffer_ref(&buffer_.base, 37, 59)));
  ExpectCounts(/*map=*/1, /*flush=*/0, /*invalidate=*/0, /*unmap=*/0);
}

TEST_F(InlineCommandBufferTest,
       InvalidateMapErrorPropagatesWithoutFurtherCallbacks) {
  buffer_.map_status = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_command_buffer_invalidate_buffer(
          command_buffer_, iree_hal_make_buffer_ref(&buffer_.base, 41, 61)));
  ExpectCounts(/*map=*/1, /*flush=*/0, /*invalidate=*/0, /*unmap=*/0);
}

TEST_F(InlineCommandBufferTest, FlushUnmapErrorPropagates) {
  buffer_.unmap_status = IREE_STATUS_INTERNAL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_hal_command_buffer_flush_buffer(
          command_buffer_, iree_hal_make_buffer_ref(&buffer_.base, 37, 59)));
  ExpectCounts(/*map=*/1, /*flush=*/1, /*invalidate=*/0, /*unmap=*/1);
}

TEST_F(InlineCommandBufferTest, InvalidateUnmapErrorPropagates) {
  buffer_.unmap_status = IREE_STATUS_INTERNAL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_hal_command_buffer_invalidate_buffer(
          command_buffer_, iree_hal_make_buffer_ref(&buffer_.base, 41, 61)));
  ExpectCounts(/*map=*/1, /*flush=*/0, /*invalidate=*/1, /*unmap=*/1);
}

}  // namespace
}  // namespace iree::hal::local
