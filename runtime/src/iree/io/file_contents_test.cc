// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/file_contents.h"

#include "iree/base/api.h"

#if IREE_FILE_IO_ENABLE

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace io {
namespace {

using ::iree::testing::status::StatusIs;

static std::uint64_t GetTrueRandomUint64() {
  std::random_device d;
  return (static_cast<std::uint64_t>(d()) << 32) | d();
}

static bool FileExists(const char* path) {
  struct stat stat_buf;
  return stat(path, &stat_buf) == 0 ? true : false;
}

static std::string GetUniquePath(const char* unique_name) {
  char* test_tmpdir = getenv("TEST_TMPDIR");
  if (!test_tmpdir) {
    test_tmpdir = getenv("TMPDIR");
  }
  if (!test_tmpdir) {
    test_tmpdir = getenv("TEMP");
  }
  if (!test_tmpdir) {
    std::cerr << "TEST_TMPDIR/TMPDIR/TEMP not defined\n";
    exit(1);
  }
  // This test might be running on multiple parallel processes, for example with
  //   ctest --repeat-until-fail 10 -j 10
  // It's hard to make this test completely race-free in that two processes
  // could generate the same "unique path" concurrently, and a serious attempt
  // will likely compromise on portability and/or complexity. Since this is only
  // test code and we only care to avoid intermittent test failures, let's try
  // just a random 64bit value.
  char unique_path[256];
  std::uint64_t random = GetTrueRandomUint64();
  snprintf(unique_path, sizeof unique_path, "%s/iree_test_%" PRIx64 "_%s",
           test_tmpdir, random, unique_name);
  return unique_path;
}

std::string GetUniqueContents(const char* unique_name,
                              iree_host_size_t padded_size) {
  std::string str = std::string("Test with name ") + unique_name + "\n";
  if (str.size() < padded_size) str.resize(padded_size, 0);
  return str;
}

TEST(FileContents, ReadWriteContentsPreload) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  auto path = GetUniquePath(kUniqueName);

  // File must not exist.
  ASSERT_FALSE(FileExists(path.c_str()));

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 32);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      iree_make_string_view(path.data(), path.size()),
      iree_make_const_byte_span(write_contents.data(), write_contents.size()),
      iree_allocator_system()));

  // Read the contents from disk.
  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      iree_make_string_view(path.data(), path.size()), iree_allocator_system(),
      &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read_contents);
}

TEST(FileContents, ReadWriteContentsMmap) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  auto path = GetUniquePath(kUniqueName);

  // File must not exist.
  ASSERT_FALSE(FileExists(path.c_str()));

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 4096);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      iree_make_string_view(path.data(), path.size()),
      iree_make_const_byte_span(write_contents.data(), write_contents.size()),
      iree_allocator_system()));

  // Read the contents from disk.
  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_map(
      iree_make_string_view(path.data(), path.size()), IREE_IO_FILE_ACCESS_READ,
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
