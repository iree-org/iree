// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <string>

#include "iree/io/file_contents.h"
#include "iree/io/stdio_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

#if IREE_FILE_IO_ENABLE

class WasiFileIoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_io_file_contents_write(
        path_.path_view(), iree_make_const_byte_span(kBytes, sizeof(kBytes)),
        iree_allocator_system()));
  }

  static constexpr uint8_t kBytes[] = {1, 0, 3, 4, 5};
  iree::testing::TempFilePath path_{"iree_wasi_file_io_test"};
};

TEST_F(WasiFileIoTest, ReadClonesContentsAndAddsTerminator) {
  iree_io_file_contents_t* contents = nullptr;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path_.path_view(), iree_allocator_system(), &contents));
  EXPECT_EQ(contents->buffer.data_length, sizeof(kBytes));
  EXPECT_EQ(memcmp(contents->buffer.data, kBytes, sizeof(kBytes)), 0);
  EXPECT_EQ(contents->buffer.data[sizeof(kBytes)], 0);
  EXPECT_TRUE(path_.Remove());
  EXPECT_EQ(memcmp(contents->buffer.data, kBytes, sizeof(kBytes)), 0);
  iree_io_file_contents_free(contents);
}

TEST_F(WasiFileIoTest, WriteOverwritesExistingContents) {
  const uint8_t replacement[] = {9, 8};
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path_.path_view(),
      iree_make_const_byte_span(replacement, sizeof(replacement)),
      iree_allocator_system()));
  iree_io_file_contents_t* contents = nullptr;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path_.path_view(), iree_allocator_system(), &contents));
  EXPECT_EQ(contents->buffer.data_length, sizeof(replacement));
  EXPECT_EQ(memcmp(contents->buffer.data, replacement, sizeof(replacement)), 0);
  iree_io_file_contents_free(contents);
}

TEST_F(WasiFileIoTest, EmptyContentsKeepTerminator) {
  IREE_ASSERT_OK(iree_io_file_contents_write(path_.path_view(),
                                             iree_const_byte_span_empty(),
                                             iree_allocator_system()));
  iree_io_file_contents_t* contents = nullptr;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path_.path_view(), iree_allocator_system(), &contents));
  EXPECT_EQ(contents->buffer.data_length, 0u);
  EXPECT_EQ(contents->buffer.data[0], 0);
  iree_io_file_contents_free(contents);
}

TEST_F(WasiFileIoTest, PreloadedHostMappingRetainsStorage) {
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_preload(IREE_IO_FILE_MODE_READ,
                                             path_.path_view(),
                                             iree_allocator_system(), &handle));
  iree_io_file_mapping_t* mapping = nullptr;
  IREE_ASSERT_OK(iree_io_file_map_view(handle, IREE_IO_FILE_ACCESS_READ, 1, 3,
                                       0, iree_allocator_system(), &mapping));
  iree_io_file_handle_release(handle);
  auto contents = iree_io_file_mapping_contents_ro(mapping);
  EXPECT_EQ(contents.data_length, 3u);
  EXPECT_EQ(memcmp(contents.data, kBytes + 1, 3), 0);
  iree_io_file_mapping_release(mapping);
}

TEST_F(WasiFileIoTest, DescriptorDuplicationIsRejectedWithoutTakingOwnership) {
  FILE* file = fopen(path_.path().c_str(), "rb");
  ASSERT_NE(file, nullptr);
  int fd = fileno(file);
  iree_io_file_handle_t* handle = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_io_file_handle_open_fd(IREE_IO_FILE_MODE_READ, fd,
                                  iree_allocator_system(), &handle));
  EXPECT_EQ(handle, nullptr);
  EXPECT_EQ(fgetc(file), kBytes[0]);
  rewind(file);
  iree_io_stream_t* stream = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_io_stdio_stream_open_fd(IREE_IO_STDIO_STREAM_MODE_READ, fd,
                                   iree_allocator_system(), &stream));
  EXPECT_EQ(stream, nullptr);
  EXPECT_EQ(fgetc(file), kBytes[0]);
  EXPECT_EQ(fclose(file), 0);
}

TEST_F(WasiFileIoTest, FileMappingIsRejected) {
  iree_io_file_contents_t* contents = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_io_file_contents_map(path_.path_view(), IREE_IO_FILE_ACCESS_READ,
                                iree_allocator_system(), &contents));
  EXPECT_EQ(contents, nullptr);
}

TEST_F(WasiFileIoTest, MissingReadAndWritePathsFail) {
  EXPECT_TRUE(path_.Remove());
  iree_io_file_contents_t* contents = nullptr;
  iree_status_t status = iree_io_file_contents_read(
      path_.path_view(), iree_allocator_system(), &contents);
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);
  EXPECT_EQ(contents, nullptr);
  std::string nested_path = path_.path() + "/missing";
  status = iree_io_file_contents_write(
      iree_make_cstring_view(nested_path.c_str()),
      iree_make_const_byte_span(kBytes, sizeof(kBytes)),
      iree_allocator_system());
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);
}

#else

TEST(WasiFileIoDisabled, ReadFailsVisibly) {
  iree_io_file_contents_t* contents = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_io_file_contents_read(IREE_SV("unopened"), iree_allocator_system(),
                                 &contents));
  EXPECT_EQ(contents, nullptr);
}

#endif  // IREE_FILE_IO_ENABLE

}  // namespace
