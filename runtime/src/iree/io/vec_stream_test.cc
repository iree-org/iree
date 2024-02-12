// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/vec_stream.h"

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

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

static StreamPtr CreateStream(iree_io_stream_mode_t mode,
                              size_t block_size = 1 * 1024) {
  iree_io_stream_t* stream = NULL;
  IREE_CHECK_OK(iree_io_vec_stream_create(mode, block_size,
                                          iree_allocator_system(), &stream));
  return StreamPtr(stream, iree_io_stream_release);
}

template <typename T, size_t N>
static StreamPtr CreateStreamWithContents(iree_io_stream_mode_t mode,
                                          T (&elements)[N],
                                          size_t block_size = 1 * 1024) {
  iree_io_stream_t* stream = NULL;
  IREE_CHECK_OK(iree_io_vec_stream_create(mode | IREE_IO_STREAM_MODE_WRITABLE,
                                          block_size, iree_allocator_system(),
                                          &stream));
  IREE_CHECK_OK(iree_io_stream_write(stream, sizeof(T) * N, elements));
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  return StreamPtr(stream, iree_io_stream_release);
}

TEST(VecStreamTest, Empty) {
  auto stream = CreateStream(IREE_IO_STREAM_MODE_READABLE);
  EXPECT_EQ(iree_io_stream_mode(stream.get()), IREE_IO_STREAM_MODE_READABLE);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 0);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, SeekSet) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  auto stream = CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data);

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // No-op seek to origin.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to end-of-stream.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET,
                                     iree_io_stream_length(stream.get())));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Seek to absolute offset 1.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to absolute offset length-1 (last valid byte).
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 4));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 4);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Try seeking out of bounds (off the front of the list).
  EXPECT_THAT(
      Status(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, -1)),
      StatusIs(StatusCode::kOutOfRange));

  // Seek off the end of the stream to extend it.
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 6));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 6);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 6);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, SeekFromCurrent) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  auto stream = CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data);

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to end-of-stream by jumping the full length.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(),
                                     IREE_IO_STREAM_SEEK_FROM_CURRENT,
                                     iree_io_stream_length(stream.get())));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Reset back to origin by seeking back the full length.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(),
                                     IREE_IO_STREAM_SEEK_FROM_CURRENT,
                                     -iree_io_stream_length(stream.get())));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek forward to absolute position 1.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // No-op seek to current location (absolute 1).
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 0));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to absolute offset length-1 (last valid byte) - here (5-1) - 1 = 3.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 3));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 4);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek forward 1 to absolute end-of-stream.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Reset back to origin.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  // Try seeking out of bounds.
  EXPECT_THAT(Status(iree_io_stream_seek(
                  stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, -100)),
              StatusIs(StatusCode::kOutOfRange));

  // Seek off the end of the stream to extend it.
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 600));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 600);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 600);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, SeekFromEnd) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  auto stream = CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data);

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Jump to end-of-stream.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Reset back to origin by seeking back the full length.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END,
                                     -iree_io_stream_length(stream.get())));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to absolute offset length-1 (last valid byte) - here 5 - 1 = 4.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 4);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Reset back to origin.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  // Try seeking out of bounds.
  EXPECT_THAT(Status(iree_io_stream_seek(stream.get(),
                                         IREE_IO_STREAM_SEEK_FROM_END, -100)),
              StatusIs(StatusCode::kOutOfRange));

  // Seek off the end of the stream to extend it.
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, 100));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 105);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 105);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, SeekToAlignment) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  auto stream = CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data);

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Alignment must be a power of two.
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream.get(), 3)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream.get(), 63)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream.get(), -2)),
              StatusIs(StatusCode::kInvalidArgument));

  // Alignment at 0 should always be ok.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 0));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);

  // Seek forward to an unaligned absolute offset 1.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);

  // Seek forward to alignment 2, which should be absolute offset 2.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 2);

  // Alignment that matches the current offset (2) should be a no-op.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 2);

  // Align up from an aligned value.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 4));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 4);

  // Align off the end of the stream to extend.
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 16));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 16);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, ReadUpTo) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  auto stream = CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data);

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  uint8_t read_buffer[64] = {0xDD};
  iree_host_size_t read_length = 0;

  // Reads of zero length should no-op.
  IREE_EXPECT_OK(
      iree_io_stream_read(stream.get(), 0, read_buffer, &read_length));
  EXPECT_EQ(read_length, 0);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);

  // Reads should advance the stream offset.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(
      iree_io_stream_read(stream.get(), 1, read_buffer, &read_length));
  EXPECT_EQ(read_length, 1);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_EQ(read_buffer[0], 0);
  EXPECT_EQ(read_buffer[1], 0xDD);

  // Read another chunk of 2 bytes.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(
      iree_io_stream_read(stream.get(), 2, read_buffer, &read_length));
  EXPECT_EQ(read_length, 2);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 3);
  EXPECT_EQ(read_buffer[0], 1);
  EXPECT_EQ(read_buffer[1], 2);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Read up to the end of the stream (2 bytes remaining) by reading over.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), sizeof(read_buffer),
                                     read_buffer, &read_length));
  EXPECT_EQ(read_length, 2);
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
  EXPECT_EQ(read_buffer[0], 3);
  EXPECT_EQ(read_buffer[1], 4);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Reading from the end of the stream should be a no-op.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), sizeof(read_buffer),
                                     read_buffer, &read_length));
  EXPECT_EQ(read_length, 0);
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
  EXPECT_EQ(read_buffer[0], 0xDD);
}

TEST(VecStreamTest, ReadExact) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  auto stream = CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data);

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  uint8_t read_buffer[64] = {0xDD};

  // Reads of zero length should no-op.
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), 0, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);

  // Reads should advance the stream offset.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), 1, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_EQ(read_buffer[0], 0);
  EXPECT_EQ(read_buffer[1], 0xDD);

  // Read another chunk of 2 bytes.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), 2, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 3);
  EXPECT_EQ(read_buffer[0], 1);
  EXPECT_EQ(read_buffer[1], 2);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Read up to the end of the stream (2 bytes remaining) by reading over.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), 2, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
  EXPECT_EQ(read_buffer[0], 3);
  EXPECT_EQ(read_buffer[1], 4);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Reading from the end of the stream fails with no read length arg.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  EXPECT_THAT(Status(iree_io_stream_read(stream.get(), sizeof(read_buffer),
                                         read_buffer, NULL)),
              StatusIs(StatusCode::kOutOfRange));

  // Reset back to the origin and try reading off the end.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  EXPECT_THAT(Status(iree_io_stream_read(stream.get(), sizeof(read_buffer),
                                         read_buffer, NULL)),
              StatusIs(StatusCode::kOutOfRange));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
}

TEST(VecStreamTest, Write) {
  auto stream =
      CreateStream(IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE);

  uint8_t data[5] = {0xDD};
  const uint8_t write_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Writes of zero length should be a no-op.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_write(stream.get(), 0, write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 0);
  EXPECT_EQ(data[0], 0xDD);

  // Writes should advance the stream.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_write(stream.get(), 1, write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 1);
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), 1, data, NULL));
  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 0xDD);

  // Write 2 more bytes and ensure only those are mutated.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_write(stream.get(), 2, &write_buffer[1]));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 2);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 1 + 2);
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), 3, data, NULL));
  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 1);
  EXPECT_EQ(data[2], 2);
  EXPECT_EQ(data[3], 0xDD);

  // Seek to the end of the stream and try to write 0 bytes (should be a no-op).
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
  IREE_EXPECT_OK(iree_io_stream_write(stream.get(), 0, write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 3);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 3);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Overwrite the entire contents of the storage.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_EXPECT_OK(
      iree_io_stream_write(stream.get(), sizeof(data), write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 5);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(write_buffer[0], write_buffer[1], write_buffer[2],
                          write_buffer[3], write_buffer[4]));
}

TEST(VecStreamTest, FillSizes) {
  auto stream =
      CreateStream(IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE);

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Fill patterns must be 1,2,4,8 bytes.
  EXPECT_THAT(Status(iree_io_stream_fill(stream.get(), 1, pattern, 3)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_io_stream_fill(stream.get(), 1, pattern, 9)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
}

TEST(VecStreamTest, Fill1) {
  auto stream =
      CreateStream(IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE);

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  // Fill with pattern size 1.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 3, pattern, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 3);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -2));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 2, pattern, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16 - 2 + 2);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x80));
}

TEST(VecStreamTest, Fill2) {
  auto stream =
      CreateStream(IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE);

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  // Fill with pattern size 2.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 3, pattern, 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 3 * 2);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -4));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 2, pattern, 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16 - 4 + 2 * 2);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x80, 0x90, 0x80, 0x90, 0x80, 0x90, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x80, 0x90, 0x80, 0x90));
}

TEST(VecStreamTest, Fill4) {
  auto stream =
      CreateStream(IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE);

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 2, pattern, 4));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 2 * 4);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -4));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 1, pattern, 4));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16 - 4 + 1 * 4);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x80, 0x90, 0xA0, 0xB0, 0x80, 0x90, 0xA0, 0xB0,
                          0x00, 0x00, 0x00, 0x80, 0x90, 0xA0, 0xB0));
}

TEST(VecStreamTest, Fill8Unaligned) {
  auto stream =
      CreateStream(IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE);

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 1, pattern, 8));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 1 * 8);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00));
}

TEST(VecStreamTest, Fill8End) {
  auto stream =
      CreateStream(IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE);

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -8));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 1, pattern, 8));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
                          0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0));
}

}  // namespace
