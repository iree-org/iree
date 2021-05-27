// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/file_io.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace file_io {
namespace {

using ::iree::testing::status::StatusIs;

std::string GetUniquePath(const char* unique_name) {
  char* test_tmpdir = getenv("TEST_TMPDIR");
  if (!test_tmpdir) {
    test_tmpdir = getenv("TMPDIR");
  }
  if (!test_tmpdir) {
    test_tmpdir = getenv("TEMP");
  }
  IREE_CHECK(test_tmpdir) << "TEST_TMPDIR/TMPDIR/TEMP not defined";
  return test_tmpdir + std::string("/iree_test_") + unique_name;
}

std::string GetUniqueContents(const char* unique_name) {
  return std::string("Test with name ") + unique_name + "\n";
}

TEST(FileIO, ReadWriteContents) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  auto path = GetUniquePath(kUniqueName);

  // File must not exist.
  ASSERT_THAT(Status(iree_file_exists(path.c_str())),
              StatusIs(StatusCode::kNotFound));

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_file_write_contents(
      path.c_str(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size())));

  // Read the contents from disk.
  iree_byte_span_t read_contents;
  IREE_ASSERT_OK(iree_file_read_contents(path.c_str(), iree_allocator_system(),
                                         &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents.data,
                   read_contents.data_length),
            0);

  iree_allocator_free(iree_allocator_system(), read_contents.data);
}

}  // namespace
}  // namespace file_io
}  // namespace iree
