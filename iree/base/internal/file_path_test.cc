// Copyright 2019 Google LLC
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

#include "iree/base/internal/file_path.h"

#include "iree/testing/gtest.h"

namespace iree {
namespace file_path {
namespace {

TEST(FilePathTest, JoinPathsEmpty) {
  EXPECT_EQ(JoinPaths("", ""), "");
  EXPECT_EQ(JoinPaths("", "bar"), "bar");
  EXPECT_EQ(JoinPaths("foo", ""), "foo");
}

TEST(FilePathTest, JoinPathsSlash) {
  EXPECT_EQ(JoinPaths("foo", "bar"), "foo/bar");
  EXPECT_EQ(JoinPaths("foo", "bar/"), "foo/bar/");
  EXPECT_EQ(JoinPaths("foo", "/bar"), "foo/bar");
  EXPECT_EQ(JoinPaths("foo", "/bar/"), "foo/bar/");

  EXPECT_EQ(JoinPaths("foo/", "bar"), "foo/bar");
  EXPECT_EQ(JoinPaths("foo/", "bar/"), "foo/bar/");
  EXPECT_EQ(JoinPaths("foo/", "/bar"), "foo/bar");
  EXPECT_EQ(JoinPaths("foo/", "/bar/"), "foo/bar/");

  EXPECT_EQ(JoinPaths("/foo", "bar"), "/foo/bar");
  EXPECT_EQ(JoinPaths("/foo", "bar/"), "/foo/bar/");
  EXPECT_EQ(JoinPaths("/foo", "/bar"), "/foo/bar");
  EXPECT_EQ(JoinPaths("/foo", "/bar/"), "/foo/bar/");

  EXPECT_EQ(JoinPaths("/foo/", "bar"), "/foo/bar");
  EXPECT_EQ(JoinPaths("/foo/", "bar/"), "/foo/bar/");
  EXPECT_EQ(JoinPaths("/foo/", "/bar"), "/foo/bar");
  EXPECT_EQ(JoinPaths("/foo/", "/bar/"), "/foo/bar/");
}

TEST(FilePathTest, JoinPathsDoubleSlash) {
  EXPECT_EQ(JoinPaths("foo//", "bar"), "foo//bar");
  EXPECT_EQ(JoinPaths("foo", "//bar"), "foo//bar");
}

TEST(FilePathTest, DirectoryNameEmpty) { EXPECT_EQ(DirectoryName(""), ""); }

TEST(FilePathTest, DirectoryNameAbsolute) {
  EXPECT_EQ(DirectoryName("/"), "/");
  EXPECT_EQ(DirectoryName("/foo"), "/");
  EXPECT_EQ(DirectoryName("/foo/"), "/foo");
  EXPECT_EQ(DirectoryName("/foo/bar"), "/foo");
  EXPECT_EQ(DirectoryName("/foo/bar/"), "/foo/bar");
}

TEST(FilePathTest, DirectoryNameRelative) {
  EXPECT_EQ(DirectoryName("foo"), "");
  EXPECT_EQ(DirectoryName("foo/"), "foo");
  EXPECT_EQ(DirectoryName("foo/bar"), "foo");
  EXPECT_EQ(DirectoryName("foo/bar/"), "foo/bar");
}

TEST(FilePathTest, DirectoryNameDoubleSlash) {
  EXPECT_EQ(DirectoryName("foo//"), "foo/");
}

TEST(FilePathTest, BasenameEmpty) { EXPECT_EQ(Basename(""), ""); }

TEST(FilePathTest, BasenameAbsolute) {
  EXPECT_EQ(Basename("/"), "");
  EXPECT_EQ(Basename("/foo"), "foo");
  EXPECT_EQ(Basename("/foo/"), "");
  EXPECT_EQ(Basename("/foo/bar"), "bar");
  EXPECT_EQ(Basename("/foo/bar/"), "");
}

TEST(FilePathTest, BasenameRelative) {
  EXPECT_EQ(Basename("foo"), "foo");
  EXPECT_EQ(Basename("foo/"), "");
  EXPECT_EQ(Basename("foo/bar"), "bar");
  EXPECT_EQ(Basename("foo/bar/"), "");
}

TEST(FilePathTest, BasenameDoubleSlash) { EXPECT_EQ(Basename("foo//"), ""); }

TEST(FilePathTest, Stem) {
  EXPECT_EQ(Stem(""), "");
  EXPECT_EQ(Stem("foo"), "foo");
  EXPECT_EQ(Stem("foo."), "foo");
  EXPECT_EQ(Stem("foo.bar"), "foo");
  EXPECT_EQ(Stem("foo.."), "foo.");
  EXPECT_EQ(Stem("foo..bar"), "foo.");
  EXPECT_EQ(Stem(".bar"), "");
  EXPECT_EQ(Stem("..bar"), ".");
}

TEST(FilePathTest, Extension) {
  EXPECT_EQ(Extension(""), "");
  EXPECT_EQ(Extension("foo"), "");
  EXPECT_EQ(Extension("foo."), "");
  EXPECT_EQ(Extension("foo.bar"), "bar");
  EXPECT_EQ(Extension("foo.."), "");
  EXPECT_EQ(Extension("foo..bar"), "bar");
  EXPECT_EQ(Extension(".bar"), "bar");
  EXPECT_EQ(Extension("..bar"), "bar");
}

}  // namespace
}  // namespace file_path
}  // namespace iree
