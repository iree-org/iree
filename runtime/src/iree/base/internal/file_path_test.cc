// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/file_path.h"

#include <string>

#include "iree/base/target_platform.h"
#include "iree/testing/gtest.h"

namespace {

#define _SV(str) iree_make_cstring_view(str)

#define EXPECT_SV_EQ(actual, expected) \
  EXPECT_TRUE(iree_string_view_equal(actual, expected))

TEST(FilePathTest, Canonicalize) {
  auto canonicalize = [](std::string value) {
    value.resize(
        iree_file_path_canonicalize((char*)value.data(), value.size()));
    return value;
  };
  EXPECT_EQ(canonicalize(""), "");
  EXPECT_EQ(canonicalize("a"), "a");
  EXPECT_EQ(canonicalize("ab"), "ab");

#if defined(IREE_PLATFORM_WINDOWS)
  EXPECT_EQ(canonicalize("/"), "\\");
  EXPECT_EQ(canonicalize("\\"), "\\");
  EXPECT_EQ(canonicalize("a/b"), "a\\b");
  EXPECT_EQ(canonicalize("a//b"), "a\\b");
  EXPECT_EQ(canonicalize("a////b"), "a\\b");
  EXPECT_EQ(canonicalize("a\\//b"), "a\\b");
  EXPECT_EQ(canonicalize("a\\\\b"), "a\\b");
  EXPECT_EQ(canonicalize("\\a"), "\\a");
  EXPECT_EQ(canonicalize("/a"), "\\a");
  EXPECT_EQ(canonicalize("//a"), "\\a");
  EXPECT_EQ(canonicalize("a/"), "a\\");
  EXPECT_EQ(canonicalize("a//"), "a\\");
#else
  EXPECT_EQ(canonicalize("/"), "/");
  EXPECT_EQ(canonicalize("a/b"), "a/b");
  EXPECT_EQ(canonicalize("a//b"), "a/b");
  EXPECT_EQ(canonicalize("a////b"), "a/b");
  EXPECT_EQ(canonicalize("/a"), "/a");
  EXPECT_EQ(canonicalize("//a"), "/a");
  EXPECT_EQ(canonicalize("a/"), "a/");
  EXPECT_EQ(canonicalize("a//"), "a/");
#endif  // IREE_PLATFORM_WINDOWS
}

static std::string JoinPaths(std::string lhs, std::string rhs) {
  char* result_str = NULL;
  IREE_IGNORE_ERROR(
      iree_file_path_join(iree_make_string_view(lhs.data(), lhs.size()),
                          iree_make_string_view(rhs.data(), rhs.size()),
                          iree_allocator_system(), &result_str));
  std::string result;
  result.resize(strlen(result_str));
  memcpy((char*)result.data(), result_str, result.size());
  iree_allocator_free(iree_allocator_system(), result_str);
  return result;
}

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

TEST(FilePathTest, DirnameEmpty) {
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("")), _SV(""));
}

TEST(FilePathTest, DirnameAbsolute) {
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/")), _SV("/"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/foo")), _SV("/"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/foo/")), _SV("/foo"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/foo/bar")), _SV("/foo"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/foo/bar/")), _SV("/foo/bar"));
}

TEST(FilePathTest, DirnameRelative) {
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo/")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo/bar")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo/bar/")), _SV("foo/bar"));
}

TEST(FilePathTest, DirnameDoubleSlash) {
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo//")), _SV("foo/"));
}

TEST(FilePathTest, BasenameEmpty) {
  EXPECT_SV_EQ(iree_file_path_basename(_SV("")), _SV(""));
}

TEST(FilePathTest, BasenameAbsolute) {
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/foo")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/foo/")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/foo/bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/foo/bar/")), _SV(""));
}

TEST(FilePathTest, BasenameRelative) {
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo/")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo/bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo/bar/")), _SV(""));
}

TEST(FilePathTest, BasenameDoubleSlash) {
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo//")), _SV(""));
}

TEST(FilePathTest, Stem) {
  EXPECT_SV_EQ(iree_file_path_stem(_SV("")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo.")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo.bar")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo..")), _SV("foo."));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo..bar")), _SV("foo."));
  EXPECT_SV_EQ(iree_file_path_stem(_SV(".bar")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("..bar")), _SV("."));
}

TEST(FilePathTest, Extension) {
  EXPECT_SV_EQ(iree_file_path_extension(_SV("")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo.")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo.bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo..")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo..bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_extension(_SV(".bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("..bar")), _SV("bar"));
}

}  // namespace
