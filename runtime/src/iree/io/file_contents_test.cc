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

}  // namespace
}  // namespace io
}  // namespace iree

#endif  // IREE_FILE_IO_ENABLE
