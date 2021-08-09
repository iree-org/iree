// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

TEST(StringViewTest, Equal) {
  auto equal = [](const char* lhs, const char* rhs) -> bool {
    return iree_string_view_equal(iree_make_cstring_view(lhs),
                                  iree_make_cstring_view(rhs));
  };
  EXPECT_TRUE(equal("", ""));
  EXPECT_FALSE(equal("a", ""));
  EXPECT_FALSE(equal("", "a"));
  EXPECT_TRUE(equal("a", "a"));
  EXPECT_FALSE(equal("a", "ab"));
  EXPECT_FALSE(equal("b", "ab"));
  EXPECT_TRUE(equal("abc", "abc"));
  EXPECT_FALSE(equal("abc", "aBc"));
}

TEST(StringViewTest, FindChar) {
  auto find_char = [](const char* value, char c, iree_host_size_t pos) {
    return iree_string_view_find_char(iree_make_cstring_view(value), c, pos);
  };
  EXPECT_EQ(find_char("", 'x', 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_char("", 'x', 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_char("", 'x', IREE_STRING_VIEW_NPOS), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_char("x", 'x', 0), 0);
  EXPECT_EQ(find_char("x", 'x', 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_char("x", 'x', IREE_STRING_VIEW_NPOS), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_char("abc", 'x', 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_char("abc", 'x', 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_char("abc", 'x', IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_char("axbxc", 'x', 0), 1);
  EXPECT_EQ(find_char("axbxc", 'x', 1), 1);
  EXPECT_EQ(find_char("axbxc", 'x', 2), 3);
  EXPECT_EQ(find_char("axbxc", 'x', 3), 3);
  EXPECT_EQ(find_char("axbxc", 'x', 4), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_char("axbxc", 'x', IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
}

TEST(StringViewTest, FindFirstOf) {
  auto find_first_of = [](const char* value, const char* s,
                          iree_host_size_t pos) {
    return iree_string_view_find_first_of(iree_make_cstring_view(value),
                                          iree_make_cstring_view(s), pos);
  };
  EXPECT_EQ(find_first_of("", "", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("", "", 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("", "", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("", "x", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("", "x", 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("", "x", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("x", "x", 0), 0);
  EXPECT_EQ(find_first_of("x", "x", 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("x", "x", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("x", "", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("x", "", 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("x", "", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("abc", "x", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("abc", "x", 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("abc", "x", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("axbxc", "xy", 0), 1);
  EXPECT_EQ(find_first_of("axbxc", "xy", 1), 1);
  EXPECT_EQ(find_first_of("axbxc", "xy", 2), 3);
  EXPECT_EQ(find_first_of("axbxc", "xy", 3), 3);
  EXPECT_EQ(find_first_of("axbxc", "xy", 4), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("axbxc", "xy", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("aybxc", "xy", 0), 1);
  EXPECT_EQ(find_first_of("aybxc", "xy", 1), 1);
  EXPECT_EQ(find_first_of("aybxc", "xy", 2), 3);
  EXPECT_EQ(find_first_of("aybxc", "xy", 3), 3);
  EXPECT_EQ(find_first_of("aybxc", "xy", 4), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_first_of("aybxc", "xy", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
}

TEST(StringViewTest, FindLastOf) {
  auto find_last_of = [](const char* value, const char* s,
                         iree_host_size_t pos) {
    return iree_string_view_find_last_of(iree_make_cstring_view(value),
                                         iree_make_cstring_view(s), pos);
  };
  EXPECT_EQ(find_last_of("", "", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("", "", 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("", "", IREE_STRING_VIEW_NPOS), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("", "x", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("", "x", 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("", "x", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("x", "x", 0), 0);
  EXPECT_EQ(find_last_of("x", "x", 1), 0);
  EXPECT_EQ(find_last_of("x", "x", IREE_STRING_VIEW_NPOS), 0);
  EXPECT_EQ(find_last_of("x", "", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("x", "", 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("x", "", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("abc", "x", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("abc", "x", 1), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("abc", "x", IREE_STRING_VIEW_NPOS),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("axbxc", "xy", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("axbxc", "xy", 1), 1);
  EXPECT_EQ(find_last_of("axbxc", "xy", 2), 1);
  EXPECT_EQ(find_last_of("axbxc", "xy", 3), 3);
  EXPECT_EQ(find_last_of("axbxc", "xy", 4), 3);
  EXPECT_EQ(find_last_of("axbxc", "xy", IREE_STRING_VIEW_NPOS), 3);
  EXPECT_EQ(find_last_of("aybxc", "xy", 0), IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(find_last_of("aybxc", "xy", 1), 1);
  EXPECT_EQ(find_last_of("aybxc", "xy", 2), 1);
  EXPECT_EQ(find_last_of("aybxc", "xy", 3), 3);
  EXPECT_EQ(find_last_of("aybxc", "xy", 4), 3);
  EXPECT_EQ(find_last_of("aybxc", "xy", IREE_STRING_VIEW_NPOS), 3);
}

TEST(StringViewTest, StartsWith) {
  auto starts_with = [](const char* value, const char* prefix) -> bool {
    return iree_string_view_starts_with(iree_make_cstring_view(value),
                                        iree_make_cstring_view(prefix));
  };
  EXPECT_TRUE(starts_with("a", "a"));
  EXPECT_TRUE(starts_with("ab", "a"));
  EXPECT_TRUE(starts_with("ab", "ab"));
  EXPECT_TRUE(starts_with("abc", "ab"));
  EXPECT_TRUE(starts_with("abc", "abc"));
  EXPECT_FALSE(starts_with("abc", ""));
  EXPECT_FALSE(starts_with("", ""));
  EXPECT_FALSE(starts_with("", "a"));
  EXPECT_FALSE(starts_with("", "abc"));
  EXPECT_FALSE(starts_with("abc", "b"));
  EXPECT_FALSE(starts_with("abc", "bc"));
  EXPECT_FALSE(starts_with("a", "abc"));
}

TEST(StringViewTest, EndsWith) {
  auto ends_with = [](const char* value, const char* suffix) -> bool {
    return iree_string_view_ends_with(iree_make_cstring_view(value),
                                      iree_make_cstring_view(suffix));
  };
  EXPECT_TRUE(ends_with("a", "a"));
  EXPECT_TRUE(ends_with("ab", "b"));
  EXPECT_TRUE(ends_with("ab", "ab"));
  EXPECT_TRUE(ends_with("abc", "bc"));
  EXPECT_TRUE(ends_with("abc", "c"));
  EXPECT_FALSE(ends_with("abc", ""));
  EXPECT_FALSE(ends_with("", ""));
  EXPECT_FALSE(ends_with("", "a"));
  EXPECT_FALSE(ends_with("", "abc"));
  EXPECT_FALSE(ends_with("abc", "b"));
  EXPECT_FALSE(ends_with("abc", "ab"));
  EXPECT_FALSE(ends_with("a", "abc"));
}

TEST(StringViewTest, RemovePrefix) {
  auto remove_prefix = [](const char* value,
                          iree_host_size_t n) -> std::string {
    return ToString(
        iree_string_view_remove_prefix(iree_make_cstring_view(value), n));
  };
  EXPECT_EQ(remove_prefix("", 0), "");
  EXPECT_EQ(remove_prefix("", 1), "");
  EXPECT_EQ(remove_prefix("a", 10), "");
  EXPECT_EQ(remove_prefix("ab", 1), "b");
  EXPECT_EQ(remove_prefix("ab", 2), "");
  EXPECT_EQ(remove_prefix("abcdef", 2), "cdef");
}

TEST(StringViewTest, RemoveSuffix) {
  auto remove_suffix = [](const char* value,
                          iree_host_size_t n) -> std::string {
    return ToString(
        iree_string_view_remove_suffix(iree_make_cstring_view(value), n));
  };
  EXPECT_EQ(remove_suffix("", 0), "");
  EXPECT_EQ(remove_suffix("", 1), "");
  EXPECT_EQ(remove_suffix("a", 10), "");
  EXPECT_EQ(remove_suffix("ab", 1), "a");
  EXPECT_EQ(remove_suffix("ab", 2), "");
  EXPECT_EQ(remove_suffix("abcdef", 2), "abcd");
}

TEST(StringViewTest, StripPrefix) {
  auto strip_prefix = [](const char* value, const char* prefix) -> std::string {
    return ToString(iree_string_view_strip_prefix(
        iree_make_cstring_view(value), iree_make_cstring_view(prefix)));
  };
  EXPECT_EQ(strip_prefix("", ""), "");
  EXPECT_EQ(strip_prefix("", "a"), "");
  EXPECT_EQ(strip_prefix("a", ""), "a");
  EXPECT_EQ(strip_prefix("a", "a"), "");
  EXPECT_EQ(strip_prefix("ab", "a"), "b");
  EXPECT_EQ(strip_prefix("ab", "b"), "ab");
  EXPECT_EQ(strip_prefix("ab", "ab"), "");
  EXPECT_EQ(strip_prefix("ab", "abc"), "ab");
  EXPECT_EQ(strip_prefix("abcdef", "ab"), "cdef");
  EXPECT_EQ(strip_prefix("abcdef", "bc"), "abcdef");
}

TEST(StringViewTest, StripSuffix) {
  auto strip_suffix = [](const char* value, const char* suffix) -> std::string {
    return ToString(iree_string_view_strip_suffix(
        iree_make_cstring_view(value), iree_make_cstring_view(suffix)));
  };
  EXPECT_EQ(strip_suffix("", ""), "");
  EXPECT_EQ(strip_suffix("", "a"), "");
  EXPECT_EQ(strip_suffix("a", ""), "a");
  EXPECT_EQ(strip_suffix("a", "a"), "");
  EXPECT_EQ(strip_suffix("ab", "a"), "ab");
  EXPECT_EQ(strip_suffix("ab", "b"), "a");
  EXPECT_EQ(strip_suffix("ab", "ab"), "");
  EXPECT_EQ(strip_suffix("ab", "abc"), "ab");
  EXPECT_EQ(strip_suffix("abcdef", "ef"), "abcd");
  EXPECT_EQ(strip_suffix("abcdef", "de"), "abcdef");
}

TEST(StringViewTest, ConsumePrefix) {
  auto consume_prefix = [](const char* value,
                           const char* prefix) -> std::string {
    iree_string_view_t value_sv = iree_make_cstring_view(value);
    if (iree_string_view_consume_prefix(&value_sv,
                                        iree_make_cstring_view(prefix))) {
      return ToString(value_sv);
    } else {
      return "FAILED";
    }
  };
  EXPECT_EQ(consume_prefix("", ""), "FAILED");
  EXPECT_EQ(consume_prefix("", "a"), "FAILED");
  EXPECT_EQ(consume_prefix("a", ""), "FAILED");
  EXPECT_EQ(consume_prefix("a", "a"), "");
  EXPECT_EQ(consume_prefix("ab", "a"), "b");
  EXPECT_EQ(consume_prefix("ab", "b"), "FAILED");
  EXPECT_EQ(consume_prefix("ab", "ab"), "");
  EXPECT_EQ(consume_prefix("ab", "abc"), "FAILED");
  EXPECT_EQ(consume_prefix("abcdef", "ab"), "cdef");
  EXPECT_EQ(consume_prefix("abcdef", "bc"), "FAILED");
}

TEST(StringViewTest, ConsumeSuffix) {
  auto consume_suffix = [](const char* value,
                           const char* suffix) -> std::string {
    iree_string_view_t value_sv = iree_make_cstring_view(value);
    if (iree_string_view_consume_suffix(&value_sv,
                                        iree_make_cstring_view(suffix))) {
      return ToString(value_sv);
    } else {
      return "FAILED";
    }
  };
  EXPECT_EQ(consume_suffix("", ""), "FAILED");
  EXPECT_EQ(consume_suffix("", "a"), "FAILED");
  EXPECT_EQ(consume_suffix("a", ""), "FAILED");
  EXPECT_EQ(consume_suffix("a", "a"), "");
  EXPECT_EQ(consume_suffix("ab", "a"), "FAILED");
  EXPECT_EQ(consume_suffix("ab", "b"), "a");
  EXPECT_EQ(consume_suffix("ab", "ab"), "");
  EXPECT_EQ(consume_suffix("ab", "abc"), "FAILED");
  EXPECT_EQ(consume_suffix("abcdef", "ef"), "abcd");
  EXPECT_EQ(consume_suffix("abcdef", "de"), "FAILED");
}

TEST(StringViewTest, Trim) {
  auto trim = [](const char* value) -> std::string {
    return ToString(iree_string_view_trim(iree_make_cstring_view(value)));
  };
  EXPECT_EQ(trim(""), "");
  EXPECT_EQ(trim("a"), "a");
  EXPECT_EQ(trim(" a"), "a");
  EXPECT_EQ(trim("a "), "a");
  EXPECT_EQ(trim("a b"), "a b");
  EXPECT_EQ(trim(" a b "), "a b");
  EXPECT_EQ(trim("\t\t\na b\n \t "), "a b");
  EXPECT_EQ(trim("\n"), "");
  EXPECT_EQ(trim("\r\n"), "");
}

TEST(StringViewTest, Substr) {
  auto substr = [](const char* value, iree_host_size_t pos,
                   iree_host_size_t n) {
    return ToString(
        iree_string_view_substr(iree_make_cstring_view(value), pos, n));
  };
  EXPECT_EQ(substr("", 0, 0), "");
  EXPECT_EQ(substr("", 0, 1), "");
  EXPECT_EQ(substr("", 0, INTPTR_MAX), "");
  EXPECT_EQ(substr("", 1, 0), "");
  EXPECT_EQ(substr("", 1, 1), "");
  EXPECT_EQ(substr("", 1, INTPTR_MAX), "");

  EXPECT_EQ(substr("a", 0, 0), "");
  EXPECT_EQ(substr("a", 0, 1), "a");
  EXPECT_EQ(substr("a", 0, 2), "a");
  EXPECT_EQ(substr("a", 0, INTPTR_MAX), "a");
  EXPECT_EQ(substr("a", 1, 0), "");
  EXPECT_EQ(substr("a", 1, 1), "");
  EXPECT_EQ(substr("a", 1, INTPTR_MAX), "");

  EXPECT_EQ(substr("abc", 0, 1), "a");
  EXPECT_EQ(substr("abc", 1, 1), "b");
  EXPECT_EQ(substr("abc", 2, 1), "c");
  EXPECT_EQ(substr("abc", 0, 2), "ab");
  EXPECT_EQ(substr("abc", 1, 2), "bc");
  EXPECT_EQ(substr("abc", 1, INTPTR_MAX), "bc");
  EXPECT_EQ(substr("abc", 0, 3), "abc");
  EXPECT_EQ(substr("abc", 0, INTPTR_MAX), "abc");
}

TEST(StringViewTest, Split) {
  auto split =
      [](const char* value,
         char split_char) -> std::tuple<intptr_t, std::string, std::string> {
    iree_string_view_t lhs;
    iree_string_view_t rhs;
    intptr_t index = iree_string_view_split(iree_make_cstring_view(value),
                                            split_char, &lhs, &rhs);
    return std::make_tuple(index, ToString(lhs), ToString(rhs));
  };
  EXPECT_EQ(split("", 'x'), std::make_tuple(-1, "", ""));
  EXPECT_EQ(split(" ", 'x'), std::make_tuple(-1, " ", ""));
  EXPECT_EQ(split("x", 'x'), std::make_tuple(0, "", ""));
  EXPECT_EQ(split(" x ", 'x'), std::make_tuple(1, " ", " "));
  EXPECT_EQ(split("axb", 'x'), std::make_tuple(1, "a", "b"));
  EXPECT_EQ(split("axxxb", 'x'), std::make_tuple(1, "a", "xxb"));
  EXPECT_EQ(split("ax", 'x'), std::make_tuple(1, "a", ""));
  EXPECT_EQ(split("xb", 'x'), std::make_tuple(0, "", "b"));
  EXPECT_EQ(split("axbxc", 'x'), std::make_tuple(1, "a", "bxc"));
}

TEST(StringViewTest, ReplaceChar) {
  auto replace_char = [](const char* value, char old_char, char new_char) {
    std::string value_clone(value);
    iree_string_view_replace_char(
        iree_make_string_view(value_clone.data(), value_clone.size()), old_char,
        new_char);
    return value_clone;
  };
  EXPECT_EQ(replace_char("", 'x', 'y'), "");
  EXPECT_EQ(replace_char(" ", 'x', 'y'), " ");
  EXPECT_EQ(replace_char("a", 'x', 'y'), "a");
  EXPECT_EQ(replace_char("x", 'x', 'y'), "y");
  EXPECT_EQ(replace_char("xx", 'x', 'y'), "yy");
  EXPECT_EQ(replace_char("axbxc", 'x', 'y'), "aybyc");
}

}  // namespace
