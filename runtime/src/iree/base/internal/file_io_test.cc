// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/file_io.h"

#include "iree/base/config.h"

#if IREE_FILE_IO_ENABLE

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace file_io {
namespace {

using ::iree::testing::status::StatusIs;

std::uint64_t GetTrueRandomUint64() {
  std::random_device d;
  return (static_cast<std::uint64_t>(d()) << 32) | d();
}

std::string GetUniquePath(const char* unique_name) {
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

TEST(FileIO, ReadWriteContentsPreload) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  auto path = GetUniquePath(kUniqueName);

  // File must not exist.
  iree_status_t status = iree_file_exists(path.c_str());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  iree_status_free(status);

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 32);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_file_write_contents(
      path.c_str(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size())));

  // Read the contents from disk.
  iree_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(
      iree_file_read_contents(path.c_str(), IREE_FILE_READ_FLAG_PRELOAD,
                              iree_allocator_default(), &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_file_contents_free(read_contents);
}

TEST(FileIO, ReadWriteContentsMmap) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  auto path = GetUniquePath(kUniqueName);

  // File must not exist.
  iree_status_t status = iree_file_exists(path.c_str());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  iree_status_free(status);

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 4096);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_file_write_contents(
      path.c_str(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size())));

  // Read the contents from disk.
  iree_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_file_read_contents(path.c_str(), IREE_FILE_READ_FLAG_MMAP,
                                         iree_allocator_default(),
                                         &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_file_contents_free(read_contents);
}

}  // namespace
}  // namespace file_io
}  // namespace iree

#endif  // IREE_FILE_IO_ENABLE
