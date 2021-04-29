// Copyright 2021 Google LLC
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

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

TEST(StringViewTest, StartsWith) {
  auto starts_with = [](const char* value, const char* prefix) -> bool {
    return iree_string_view_starts_with(iree_make_cstring_view(value),
                                        iree_make_cstring_view(prefix));
  };
  ASSERT_TRUE(starts_with("a", "a"));
  ASSERT_TRUE(starts_with("ab", "a"));
  ASSERT_TRUE(starts_with("ab", "ab"));
  ASSERT_TRUE(starts_with("abc", "ab"));
  ASSERT_TRUE(starts_with("abc", "abc"));
  ASSERT_FALSE(starts_with("abc", ""));
  ASSERT_FALSE(starts_with("", ""));
  ASSERT_FALSE(starts_with("", "a"));
  ASSERT_FALSE(starts_with("", "abc"));
  ASSERT_FALSE(starts_with("abc", "b"));
  ASSERT_FALSE(starts_with("abc", "bc"));
  ASSERT_FALSE(starts_with("a", "abc"));
}

TEST(StringViewTest, EndsWith) {
  auto ends_with = [](const char* value, const char* suffix) -> bool {
    return iree_string_view_ends_with(iree_make_cstring_view(value),
                                      iree_make_cstring_view(suffix));
  };
  ASSERT_TRUE(ends_with("a", "a"));
  ASSERT_TRUE(ends_with("ab", "b"));
  ASSERT_TRUE(ends_with("ab", "ab"));
  ASSERT_TRUE(ends_with("abc", "bc"));
  ASSERT_TRUE(ends_with("abc", "c"));
  ASSERT_FALSE(ends_with("abc", ""));
  ASSERT_FALSE(ends_with("", ""));
  ASSERT_FALSE(ends_with("", "a"));
  ASSERT_FALSE(ends_with("", "abc"));
  ASSERT_FALSE(ends_with("abc", "b"));
  ASSERT_FALSE(ends_with("abc", "ab"));
  ASSERT_FALSE(ends_with("a", "abc"));
}

TEST(StringViewTest, RemovePrefix) {
  auto remove_prefix = [](const char* value,
                          iree_host_size_t n) -> std::string {
    return ToString(
        iree_string_view_remove_prefix(iree_make_cstring_view(value), n));
  };
  ASSERT_EQ(remove_prefix("", 0), "");
  ASSERT_EQ(remove_prefix("", 1), "");
  ASSERT_EQ(remove_prefix("a", 10), "");
  ASSERT_EQ(remove_prefix("ab", 1), "b");
  ASSERT_EQ(remove_prefix("ab", 2), "");
  ASSERT_EQ(remove_prefix("abcdef", 2), "cdef");
}

TEST(StringViewTest, RemoveSuffix) {
  auto remove_suffix = [](const char* value,
                          iree_host_size_t n) -> std::string {
    return ToString(
        iree_string_view_remove_suffix(iree_make_cstring_view(value), n));
  };
  ASSERT_EQ(remove_suffix("", 0), "");
  ASSERT_EQ(remove_suffix("", 1), "");
  ASSERT_EQ(remove_suffix("a", 10), "");
  ASSERT_EQ(remove_suffix("ab", 1), "a");
  ASSERT_EQ(remove_suffix("ab", 2), "");
  ASSERT_EQ(remove_suffix("abcdef", 2), "abcd");
}

TEST(StringViewTest, Trim) {
  auto trim = [](const char* value) -> std::string {
    return ToString(iree_string_view_trim(iree_make_cstring_view(value)));
  };
  ASSERT_EQ(trim(""), "");
  ASSERT_EQ(trim("a"), "a");
  ASSERT_EQ(trim(" a"), "a");
  ASSERT_EQ(trim("a "), "a");
  ASSERT_EQ(trim("a b"), "a b");
  ASSERT_EQ(trim(" a b "), "a b");
  ASSERT_EQ(trim("\t\t\na b\n \t "), "a b");
  ASSERT_EQ(trim("\n"), "");
  ASSERT_EQ(trim("\r\n"), "");
}

}  // namespace
