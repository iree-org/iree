// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/file_io.h"

#include "iree/base/config.h"

#if IREE_FILE_IO_ENABLE

#include <cstdlib>
#include <cstring>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

#include "iree/base/status_cc.h"
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
  if (!test_tmpdir) {
    std::cerr << "TEST_TMPDIR/TMPDIR/TEMP not defined\n";
    exit(1);
  }
  return test_tmpdir + std::string("/iree_test_") + unique_name;
}

std::string GetUniqueContents(const char* unique_name) {
  return std::string("Test with name ") + unique_name + "\n";
}

TEST(FileIO, ReadWriteContents) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  auto path = GetUniquePath(kUniqueName);

  // File must not exist.
  iree_status_t status = iree_file_exists(path.c_str());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  iree_status_free(status);

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_file_write_contents(
      path.c_str(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size())));

  // Read the contents from disk.
  iree_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_file_read_contents(path.c_str(), iree_allocator_system(),
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
