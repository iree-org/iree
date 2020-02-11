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

#include "iree/base/file_io.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "iree/base/file_path.h"
#include "iree/base/logging.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace file_io {
namespace {

using ::iree::testing::status::StatusIs;

std::string GetUniquePath(absl::string_view unique_name) {
  char* test_tmpdir = getenv("TEST_TMPDIR");
  CHECK(test_tmpdir) << "TEST_TMPDIR not defined";
  return file_path::JoinPaths(test_tmpdir,
                              absl::StrCat(unique_name, "_test.txt"));
}

std::string GetUniqueContents(absl::string_view unique_name) {
  return absl::StrCat("Test with name ", unique_name, "\n");
}

TEST(FileIo, GetSetContents) {
  std::string unique_name = "GetSetContents";
  auto path = GetUniquePath(unique_name);
  ASSERT_THAT(FileExists(path), StatusIs(StatusCode::kNotFound));
  auto to_write = GetUniqueContents(unique_name);

  ASSERT_OK(SetFileContents(path, to_write));
  ASSERT_OK_AND_ASSIGN(std::string read, GetFileContents(path));
  EXPECT_EQ(to_write, read);
}

TEST(FileIo, SetDeleteExists) {
  std::string unique_name = "SetDeleteExists";
  auto path = GetUniquePath(unique_name);
  ASSERT_THAT(FileExists(path), StatusIs(StatusCode::kNotFound));
  auto to_write = GetUniqueContents(unique_name);

  ASSERT_OK(SetFileContents(path, to_write));
  ASSERT_OK(FileExists(path));
  ASSERT_OK(DeleteFile(path));
  EXPECT_THAT(FileExists(path), StatusIs(StatusCode::kNotFound));
}

TEST(FileIo, MoveFile) {
  auto from_path = GetUniquePath("MoveFileFrom");
  auto to_path = GetUniquePath("MoveFileTo");
  ASSERT_THAT(FileExists(from_path), StatusIs(StatusCode::kNotFound));
  ASSERT_THAT(FileExists(to_path), StatusIs(StatusCode::kNotFound));
  auto to_write = GetUniqueContents("MoveFile");

  ASSERT_OK(SetFileContents(from_path, to_write));
  ASSERT_OK(FileExists(from_path));
  EXPECT_OK(MoveFile(from_path, to_path));
  EXPECT_THAT(FileExists(from_path), StatusIs(StatusCode::kNotFound));
  EXPECT_OK(FileExists(to_path));
  ASSERT_OK_AND_ASSIGN(std::string read, GetFileContents(to_path));
  EXPECT_EQ(to_write, read);
}

}  // namespace
}  // namespace file_io
}  // namespace iree
