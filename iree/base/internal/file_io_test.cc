// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
