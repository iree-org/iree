// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/file_contents.h"

#include "iree/base/api.h"

#if IREE_FILE_IO_ENABLE

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace iree {
namespace io {
namespace {

using ::iree::testing::status::StatusIs;

std::string GetUniqueContents(const char* unique_name,
                              iree_host_size_t padded_size) {
  std::string str = std::string("Test with name ") + unique_name + "\n";
  if (str.size() < padded_size) str.resize(padded_size, 0);
  return str;
}

TEST(FileContents, ReadWriteContentsPreload) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  iree::testing::TempFilePath path("iree_file_contents_test");

  // File must not exist.
  ASSERT_FALSE(path.Exists());

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 32);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size()),
      iree_allocator_system()));

  // Read the contents from disk.
  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path.path_view(), iree_allocator_system(), &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read_contents);
}

TEST(FileContents, ReadWriteContentsMmap) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  iree::testing::TempFilePath path("iree_file_contents_test");

  // File must not exist.
  ASSERT_FALSE(path.Exists());

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 4096);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size()),
      iree_allocator_system()));

  // Read the contents from disk.
  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(
      iree_io_file_contents_map(path.path_view(), IREE_IO_FILE_ACCESS_READ,
                                iree_allocator_system(), &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read_contents);
}

// Tests that a file opened for reading can be opened again concurrently.
// Validates FILE_SHARE_READ behavior on Windows — without it, the second
// open fails with ERROR_SHARING_VIOLATION.
TEST(FileContents, ConcurrentReadOpens) {
  constexpr const char* kUniqueName = "ConcurrentReadOpens";
  auto path = GetUniquePath(kUniqueName);

  // Write a file to open.
  auto contents = GetUniqueContents(kUniqueName, 4096);
  IREE_ASSERT_OK(iree_io_file_contents_write(
      iree_make_string_view(path.data(), path.size()),
      iree_make_const_byte_span(contents.data(), contents.size()),
      iree_allocator_system()));

  // Open the file twice for reading — both should succeed.
  iree_io_file_contents_t* read1 = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_map(
      iree_make_string_view(path.data(), path.size()), IREE_IO_FILE_ACCESS_READ,
      iree_allocator_system(), &read1));

  iree_io_file_contents_t* read2 = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_map(
      iree_make_string_view(path.data(), path.size()), IREE_IO_FILE_ACCESS_READ,
      iree_allocator_system(), &read2));

  // Both should have the same contents.
  EXPECT_EQ(read1->const_buffer.data_length, read2->const_buffer.data_length);
  EXPECT_EQ(memcmp(read1->const_buffer.data, read2->const_buffer.data,
                   read1->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read2);
  iree_io_file_contents_free(read1);
}

}  // namespace
}  // namespace io
}  // namespace iree

#endif  // IREE_FILE_IO_ENABLE
