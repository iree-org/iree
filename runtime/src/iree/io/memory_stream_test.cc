// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/memory_stream.h"

#include <array>
#include <string>
#include <string_view>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::Eq;

TEST(MemoryStreamTest, Wrap) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE, iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  EXPECT_EQ(iree_io_stream_mode(stream), IREE_IO_STREAM_MODE_READABLE);
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_EQ(iree_io_stream_length(stream), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, WrapEmpty) {
  uint8_t data[1] = {0};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE, iree_make_byte_span(data, 0),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  EXPECT_EQ(iree_io_stream_mode(stream), IREE_IO_STREAM_MODE_READABLE);
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_EQ(iree_io_stream_length(stream), 0);
  EXPECT_TRUE(iree_io_stream_is_eos(stream));

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, WrapReleaseCallback) {
  int callback_count = 0;
  iree_io_memory_stream_release_callback_t release_callback = {
      +[](void* user_data, iree_io_stream_t* stream) {
        int* callback_count = (int*)user_data;
        ++(*callback_count);
      },
      &callback_count,
  };

  ASSERT_EQ(callback_count, 0);

  uint8_t data[1] = {0};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE, iree_make_byte_span(data, sizeof(data)),
      release_callback, iree_allocator_system(), &stream));
  ASSERT_EQ(callback_count, 0);

  iree_io_stream_release(stream);
  ASSERT_EQ(callback_count, 1);
}

TEST(MemoryStreamTest, SeekSet) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE, iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_EQ(iree_io_stream_length(stream), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // No-op seek to origin.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Seek to end-of-stream.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                                     iree_io_stream_length(stream)));
  EXPECT_EQ(iree_io_stream_offset(stream), iree_io_stream_length(stream));
  EXPECT_TRUE(iree_io_stream_is_eos(stream));

  // Seek to absolute offset 1.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 1));
  EXPECT_EQ(iree_io_stream_offset(stream), 1);
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Seek to absolute offset length-1 (last valid byte).
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 4));
  EXPECT_EQ(iree_io_stream_offset(stream), 4);
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Try seeking out of bounds.
  EXPECT_THAT(Status(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, -1)),
              StatusIs(StatusCode::kOutOfRange));
  EXPECT_THAT(Status(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 6)),
              StatusIs(StatusCode::kOutOfRange));

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, SeekFromCurrent) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE, iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_EQ(iree_io_stream_length(stream), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Seek to end-of-stream by jumping the full length.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_CURRENT,
                                     iree_io_stream_length(stream)));
  EXPECT_EQ(iree_io_stream_offset(stream), iree_io_stream_length(stream));
  EXPECT_TRUE(iree_io_stream_is_eos(stream));

  // Reset back to origin by seeking back the full length.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_CURRENT,
                                     -iree_io_stream_length(stream)));
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Seek forward to absolute position 1.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_CURRENT, 1));
  EXPECT_EQ(iree_io_stream_offset(stream), 1);
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // No-op seek to current location (absolute 1).
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_CURRENT, 0));
  EXPECT_EQ(iree_io_stream_offset(stream), 1);
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Seek to absolute offset length-1 (last valid byte) - here (5-1) - 1 = 3.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_CURRENT, 3));
  EXPECT_EQ(iree_io_stream_offset(stream), 4);
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Seek forward 1 to absolute end-of-stream.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_CURRENT, 1));
  EXPECT_EQ(iree_io_stream_offset(stream), iree_io_stream_length(stream));
  EXPECT_TRUE(iree_io_stream_is_eos(stream));

  // Reset back to origin.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));

  // Try seeking out of bounds.
  EXPECT_THAT(Status(iree_io_stream_seek(
                  stream, IREE_IO_STREAM_SEEK_FROM_CURRENT, -100)),
              StatusIs(StatusCode::kOutOfRange));
  EXPECT_THAT(Status(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 600)),
              StatusIs(StatusCode::kOutOfRange));

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, SeekFromEnd) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE, iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_EQ(iree_io_stream_length(stream), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Jump to end-of-stream.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_EQ(iree_io_stream_offset(stream), iree_io_stream_length(stream));
  EXPECT_TRUE(iree_io_stream_is_eos(stream));

  // Reset back to origin by seeking back the full length.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END,
                                     -iree_io_stream_length(stream)));
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Seek to absolute offset length-1 (last valid byte) - here 5 - 1 = 4.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, -1));
  EXPECT_EQ(iree_io_stream_offset(stream), 4);
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Reset back to origin.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));

  // Try seeking out of bounds.
  EXPECT_THAT(
      Status(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, -100)),
      StatusIs(StatusCode::kOutOfRange));
  EXPECT_THAT(
      Status(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, 600)),
      StatusIs(StatusCode::kOutOfRange));

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, SeekToAlignment) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE, iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_EQ(iree_io_stream_length(stream), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  // Alignment must be a power of two.
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream, 3)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream, 63)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream, -2)),
              StatusIs(StatusCode::kInvalidArgument));

  // Alignment at 0 should always be ok.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream, 0));
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream, 1));
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream, 2));
  EXPECT_EQ(iree_io_stream_offset(stream), 0);

  // Seek forward to an unaligned absolute offset 1.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 1));
  EXPECT_EQ(iree_io_stream_offset(stream), 1);

  // Seek forward to alignment 2, which should be absolute offset 2.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream, 2));
  EXPECT_EQ(iree_io_stream_offset(stream), 2);

  // Alignment that matches the current offset (2) should be a no-op.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream, 2));
  EXPECT_EQ(iree_io_stream_offset(stream), 2);

  // Align up from an aligned value.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream, 4));
  EXPECT_EQ(iree_io_stream_offset(stream), 4);

  // Try aligning off the end of the stream.
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream, 16)),
              StatusIs(StatusCode::kOutOfRange));

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, ReadUpTo) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE, iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_EQ(iree_io_stream_length(stream), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  uint8_t read_buffer[64] = {0xDD};
  iree_host_size_t read_length = 0;

  // Reads of zero length should no-op.
  IREE_EXPECT_OK(iree_io_stream_read(stream, 0, read_buffer, &read_length));
  EXPECT_EQ(read_length, 0);
  EXPECT_EQ(iree_io_stream_offset(stream), 0);

  // Reads should advance the stream offset.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream, 1, read_buffer, &read_length));
  EXPECT_EQ(read_length, 1);
  EXPECT_EQ(iree_io_stream_offset(stream), 1);
  EXPECT_EQ(read_buffer[0], 0);
  EXPECT_EQ(read_buffer[1], 0xDD);

  // Read another chunk of 2 bytes.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream, 2, read_buffer, &read_length));
  EXPECT_EQ(read_length, 2);
  EXPECT_EQ(iree_io_stream_offset(stream), 3);
  EXPECT_EQ(read_buffer[0], 1);
  EXPECT_EQ(read_buffer[1], 2);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Read up to the end of the stream (2 bytes remaining) by reading over.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream, sizeof(read_buffer), read_buffer,
                                     &read_length));
  EXPECT_EQ(read_length, 2);
  EXPECT_EQ(iree_io_stream_offset(stream), iree_io_stream_length(stream));
  EXPECT_TRUE(iree_io_stream_is_eos(stream));
  EXPECT_EQ(read_buffer[0], 3);
  EXPECT_EQ(read_buffer[1], 4);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Reading from the end of the stream should be a no-op.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream, sizeof(read_buffer), read_buffer,
                                     &read_length));
  EXPECT_EQ(read_length, 0);
  EXPECT_EQ(iree_io_stream_offset(stream), iree_io_stream_length(stream));
  EXPECT_TRUE(iree_io_stream_is_eos(stream));
  EXPECT_EQ(read_buffer[0], 0xDD);

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, ReadExact) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE, iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_EQ(iree_io_stream_length(stream), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream));

  uint8_t read_buffer[64] = {0xDD};

  // Reads of zero length should no-op.
  IREE_EXPECT_OK(iree_io_stream_read(stream, 0, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream), 0);

  // Reads should advance the stream offset.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream, 1, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream), 1);
  EXPECT_EQ(read_buffer[0], 0);
  EXPECT_EQ(read_buffer[1], 0xDD);

  // Read another chunk of 2 bytes.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream, 2, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream), 3);
  EXPECT_EQ(read_buffer[0], 1);
  EXPECT_EQ(read_buffer[1], 2);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Read up to the end of the stream (2 bytes remaining) by reading over.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream, 2, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream), iree_io_stream_length(stream));
  EXPECT_TRUE(iree_io_stream_is_eos(stream));
  EXPECT_EQ(read_buffer[0], 3);
  EXPECT_EQ(read_buffer[1], 4);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Reading from the end of the stream fails with no read length arg.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  EXPECT_THAT(Status(iree_io_stream_read(stream, sizeof(read_buffer),
                                         read_buffer, NULL)),
              StatusIs(StatusCode::kOutOfRange));

  // Reset back to the origin and try reading off the end.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  EXPECT_THAT(Status(iree_io_stream_read(stream, sizeof(read_buffer),
                                         read_buffer, NULL)),
              StatusIs(StatusCode::kOutOfRange));
  EXPECT_EQ(iree_io_stream_offset(stream), 0);

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, Write) {
  uint8_t data[5] = {0xDD};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_WRITABLE, iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  const uint8_t write_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Writes of zero length should be a no-op.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_write(stream, 0, write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream), 0);
  EXPECT_EQ(data[0], 0xDD);

  // Writes should advance the stream.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_write(stream, 1, write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream), 1);
  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 0xDD);

  // Write 2 more bytes and ensure only those are mutated.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_write(stream, 2, &write_buffer[1]));
  EXPECT_EQ(iree_io_stream_offset(stream), 1 + 2);
  EXPECT_EQ(data[0], 0xDD);
  EXPECT_EQ(data[1], 1);
  EXPECT_EQ(data[2], 2);
  EXPECT_EQ(data[3], 0xDD);

  // Writes off the end of the stream should fail.
  EXPECT_THAT(
      Status(iree_io_stream_write(stream, sizeof(write_buffer), write_buffer)),
      StatusIs(StatusCode::kOutOfRange));

  // Seek to the end of the stream and try to write 0 bytes (should be a no-op).
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_TRUE(iree_io_stream_is_eos(stream));
  IREE_EXPECT_OK(iree_io_stream_write(stream, 0, write_buffer));
  EXPECT_TRUE(iree_io_stream_is_eos(stream));

  // Writing off the end of the stream should fail.
  EXPECT_THAT(Status(iree_io_stream_write(stream, 1, write_buffer)),
              StatusIs(StatusCode::kOutOfRange));

  // Overwrite the entire contents of the storage.
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_EXPECT_OK(iree_io_stream_write(stream, sizeof(data), write_buffer));
  EXPECT_THAT(data,
              ElementsAre(write_buffer[0], write_buffer[1], write_buffer[2],
                          write_buffer[3], write_buffer[4]));

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, Fill) {
  uint8_t data[16] = {0xDD};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_WRITABLE, iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Fill patterns must be 1,2,4,8 bytes.
  EXPECT_THAT(Status(iree_io_stream_fill(stream, 1, pattern, 3)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_io_stream_fill(stream, 1, pattern, 9)),
              StatusIs(StatusCode::kInvalidArgument));

  // Fills are bounds checked.
  EXPECT_THAT(Status(iree_io_stream_fill(stream, 100, pattern, 1)),
              StatusIs(StatusCode::kOutOfRange));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_THAT(Status(iree_io_stream_fill(stream, 1, pattern, 1)),
              StatusIs(StatusCode::kOutOfRange));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, -1));
  EXPECT_THAT(Status(iree_io_stream_fill(stream, 2, pattern, 1)),
              StatusIs(StatusCode::kOutOfRange));

  // Fill with pattern size 1.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream, 3, pattern, 1));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, -2));
  IREE_EXPECT_OK(iree_io_stream_fill(stream, 2, pattern, 1));
  EXPECT_THAT(data,
              ElementsAre(0xDD, 0x80, 0x80, 0x80, 0xDD, 0xDD, 0xDD, 0xDD, 0xDD,
                          0xDD, 0xDD, 0xDD, 0xDD, 0xDD, 0x80, 0x80));

  // Fill with pattern size 2.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream, 3, pattern, 2));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, -4));
  IREE_EXPECT_OK(iree_io_stream_fill(stream, 2, pattern, 2));
  EXPECT_THAT(data,
              ElementsAre(0xDD, 0x80, 0x90, 0x80, 0x90, 0x80, 0x90, 0xDD, 0xDD,
                          0xDD, 0xDD, 0xDD, 0x80, 0x90, 0x80, 0x90));

  // Fill with pattern size 4.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream, 2, pattern, 4));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, -4));
  IREE_EXPECT_OK(iree_io_stream_fill(stream, 1, pattern, 4));
  EXPECT_THAT(data,
              ElementsAre(0xDD, 0x80, 0x90, 0xA0, 0xB0, 0x80, 0x90, 0xA0, 0xB0,
                          0xDD, 0xDD, 0xDD, 0x80, 0x90, 0xA0, 0xB0));

  // Fill with pattern size 8.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream, 1, pattern, 8));
  EXPECT_THAT(data,
              ElementsAre(0xDD, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0,
                          0xDD, 0xDD, 0xDD, 0xDD, 0xDD, 0xDD, 0xDD));
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, -8));
  IREE_EXPECT_OK(iree_io_stream_fill(stream, 1, pattern, 8));
  EXPECT_THAT(data,
              ElementsAre(0xDD, 0xDD, 0xDD, 0xDD, 0xDD, 0xDD, 0xDD, 0xDD, 0x80,
                          0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0));

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, MapRead) {
  uint8_t data[5] = {0xDD};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_MAPPABLE,
      iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  iree_const_byte_span_t span = iree_const_byte_span_empty();

  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_map_read(stream, 2, &span));
  EXPECT_EQ(span.data, &data[1]);
  EXPECT_EQ(span.data_length, 2);

  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, -1));
  IREE_EXPECT_OK(iree_io_stream_map_read(stream, 1, &span));
  EXPECT_EQ(span.data, &data[4]);
  EXPECT_EQ(span.data_length, 1);

  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  EXPECT_THAT(Status(iree_io_stream_map_read(stream, 100, &span)),
              StatusIs(StatusCode::kOutOfRange));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_THAT(Status(iree_io_stream_map_read(stream, 1, &span)),
              StatusIs(StatusCode::kOutOfRange));

  iree_io_stream_release(stream);
}

TEST(MemoryStreamTest, MapWrite) {
  uint8_t data[5] = {0xDD};
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_MAPPABLE,
      iree_make_byte_span(data, sizeof(data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &stream));

  iree_byte_span_t span = iree_byte_span_empty();

  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_map_write(stream, 2, &span));
  EXPECT_EQ(span.data, &data[1]);
  EXPECT_EQ(span.data_length, 2);

  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, -1));
  IREE_EXPECT_OK(iree_io_stream_map_write(stream, 1, &span));
  EXPECT_EQ(span.data, &data[4]);
  EXPECT_EQ(span.data_length, 1);

  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  EXPECT_THAT(Status(iree_io_stream_map_write(stream, 100, &span)),
              StatusIs(StatusCode::kOutOfRange));
  IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_THAT(Status(iree_io_stream_map_write(stream, 1, &span)),
              StatusIs(StatusCode::kOutOfRange));

  iree_io_stream_release(stream);
}

// Tests copies that fit within a single block.
TEST(MemoryStreamTest, Copy) {
  uint8_t source_data[5] = {0xA0, 0xB0, 0xC0, 0xD0, 0xE0};
  iree_io_stream_t* source_stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE,
      iree_make_byte_span(source_data, sizeof(source_data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &source_stream));

  uint8_t target_data[5] = {0xDD};
  iree_io_stream_t* target_stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_WRITABLE,
      iree_make_byte_span(target_data, sizeof(target_data)),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &target_stream));

  // Bounds checks length.
  EXPECT_THAT(Status(iree_io_stream_copy(source_stream, target_stream, 100)),
              StatusIs(StatusCode::kOutOfRange));

  // Bounds check source.
  IREE_EXPECT_OK(
      iree_io_stream_seek(source_stream, IREE_IO_STREAM_SEEK_FROM_END, 0));
  IREE_EXPECT_OK(
      iree_io_stream_seek(target_stream, IREE_IO_STREAM_SEEK_SET, 0));
  EXPECT_THAT(Status(iree_io_stream_copy(source_stream, target_stream, 1)),
              StatusIs(StatusCode::kOutOfRange));

  // Bounds check target.
  IREE_EXPECT_OK(
      iree_io_stream_seek(source_stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_EXPECT_OK(
      iree_io_stream_seek(target_stream, IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_THAT(Status(iree_io_stream_copy(source_stream, target_stream, 1)),
              StatusIs(StatusCode::kOutOfRange));

  // Copy entire contents.
  memset(target_data, 0xDD, sizeof(target_data));
  IREE_EXPECT_OK(
      iree_io_stream_seek(source_stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_EXPECT_OK(
      iree_io_stream_seek(target_stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_EXPECT_OK(
      iree_io_stream_copy(source_stream, target_stream, sizeof(target_data)));
  EXPECT_THAT(target_data, ElementsAreArray(source_data));

  // Copy an interior subrange.
  memset(target_data, 0xDD, sizeof(target_data));
  IREE_EXPECT_OK(
      iree_io_stream_seek(source_stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_EXPECT_OK(
      iree_io_stream_seek(target_stream, IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_copy(source_stream, target_stream, 2));
  EXPECT_THAT(target_data, ElementsAre(0xDD, 0xA0, 0xB0, 0xDD, 0xDD));

  // Copy to up to the end.
  memset(target_data, 0xDD, sizeof(target_data));
  IREE_EXPECT_OK(
      iree_io_stream_seek(source_stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_EXPECT_OK(
      iree_io_stream_seek(target_stream, IREE_IO_STREAM_SEEK_FROM_END, -2));
  IREE_EXPECT_OK(iree_io_stream_copy(source_stream, target_stream, 2));
  EXPECT_THAT(target_data, ElementsAre(0xDD, 0xDD, 0xDD, 0xA0, 0xB0));

  iree_io_stream_release(source_stream);
  iree_io_stream_release(target_stream);
}

// Tests a copy that should trip into the multi-block code path.
TEST(MemoryStreamTest, CopyLarge) {
  std::vector<uint8_t> source_data(1 * 1024 * 1024);
  for (size_t i = 0; i < source_data.size(); ++i) {
    source_data[i] = (uint8_t)i;
  }
  iree_io_stream_t* source_stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE,
      iree_make_byte_span(source_data.data(), source_data.size()),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &source_stream));

  std::vector<uint8_t> target_data(1 * 1024 * 1024);
  iree_io_stream_t* target_stream = NULL;
  IREE_ASSERT_OK(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_WRITABLE,
      iree_make_byte_span(target_data.data(), target_data.size()),
      iree_io_memory_stream_release_callback_null(), iree_allocator_system(),
      &target_stream));

  // Copy an interior subrange.
  memset(target_data.data(), 0xDD, target_data.size());
  IREE_EXPECT_OK(
      iree_io_stream_seek(source_stream, IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(
      iree_io_stream_seek(target_stream, IREE_IO_STREAM_SEEK_SET, 2));
  IREE_EXPECT_OK(iree_io_stream_copy(source_stream, target_stream,
                                     source_data.size() - 10));
  for (size_t i = 0; i < target_data.size(); ++i) {
    if (i < 2) {
      // Before the copy range.
      EXPECT_EQ(target_data[i], 0xDD);
    } else if (i < source_data.size() - 10 + 2) {
      // In the copy range.
      EXPECT_EQ(target_data[i], source_data[1 + i - 2]);
    } else {
      // After the copy range.
      EXPECT_EQ(target_data[i], 0xDD);
    }
  }

  iree_io_stream_release(source_stream);
  iree_io_stream_release(target_stream);
}

}  // namespace
