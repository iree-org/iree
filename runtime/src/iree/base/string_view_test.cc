// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>
#include <string_view>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::StatusCode;
using iree::testing::status::IsOkAndHolds;
using iree::testing::status::StatusIs;
using testing::ElementsAre;
using testing::Eq;

static std::string ToString(iree_string_view_t value) {
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
  EXPECT_TRUE(equal("a_c", "a_c"));
}

TEST(StringViewTest, EqualCase) {
  auto equal_case = [](const char* lhs, const char* rhs) -> bool {
    return iree_string_view_equal_case(iree_make_cstring_view(lhs),
                                       iree_make_cstring_view(rhs));
  };
  EXPECT_TRUE(equal_case("", ""));
  EXPECT_FALSE(equal_case("a", ""));
  EXPECT_FALSE(equal_case("", "a"));
  EXPECT_TRUE(equal_case("a", "a"));
  EXPECT_TRUE(equal_case("A", "a"));
  EXPECT_TRUE(equal_case("a", "A"));
  EXPECT_FALSE(equal_case("a", "ab"));
  EXPECT_FALSE(equal_case("b", "ab"));
  EXPECT_TRUE(equal_case("abc", "abc"));
  EXPECT_TRUE(equal_case("abc", "aBc"));
  EXPECT_TRUE(equal_case("a_c", "a_C"));
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
    iree_string_view_t lhs = iree_string_view_empty();
    iree_string_view_t rhs = iree_string_view_empty();
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

// Tests that partial returns from iree_string_view_split (only LHS or RHS) work
// as expected even when no storage is provided.
TEST(StringViewTest, SplitLHSOnly) {
  auto split_lhs = [](const char* value,
                      char split_char) -> std::tuple<intptr_t, std::string> {
    iree_string_view_t lhs = iree_string_view_empty();
    intptr_t index = iree_string_view_split(iree_make_cstring_view(value),
                                            split_char, &lhs, nullptr);
    return std::make_tuple(index, ToString(lhs));
  };
  EXPECT_EQ(split_lhs("", 'x'), std::make_tuple(-1, ""));
  EXPECT_EQ(split_lhs(" ", 'x'), std::make_tuple(-1, " "));
  EXPECT_EQ(split_lhs("x", 'x'), std::make_tuple(0, ""));
  EXPECT_EQ(split_lhs(" x ", 'x'), std::make_tuple(1, " "));
  EXPECT_EQ(split_lhs("axb", 'x'), std::make_tuple(1, "a"));
  EXPECT_EQ(split_lhs("axxxb", 'x'), std::make_tuple(1, "a"));
  EXPECT_EQ(split_lhs("ax", 'x'), std::make_tuple(1, "a"));
  EXPECT_EQ(split_lhs("xb", 'x'), std::make_tuple(0, ""));
  EXPECT_EQ(split_lhs("axbxc", 'x'), std::make_tuple(1, "a"));
}
TEST(StringViewTest, SplitRHSOnly) {
  auto split_rhs = [](const char* value,
                      char split_char) -> std::tuple<intptr_t, std::string> {
    iree_string_view_t rhs = iree_string_view_empty();
    intptr_t index = iree_string_view_split(iree_make_cstring_view(value),
                                            split_char, nullptr, &rhs);
    return std::make_tuple(index, ToString(rhs));
  };
  EXPECT_EQ(split_rhs("", 'x'), std::make_tuple(-1, ""));
  EXPECT_EQ(split_rhs(" ", 'x'), std::make_tuple(-1, ""));
  EXPECT_EQ(split_rhs("x", 'x'), std::make_tuple(0, ""));
  EXPECT_EQ(split_rhs(" x ", 'x'), std::make_tuple(1, " "));
  EXPECT_EQ(split_rhs("axb", 'x'), std::make_tuple(1, "b"));
  EXPECT_EQ(split_rhs("axxxb", 'x'), std::make_tuple(1, "xxb"));
  EXPECT_EQ(split_rhs("ax", 'x'), std::make_tuple(1, ""));
  EXPECT_EQ(split_rhs("xb", 'x'), std::make_tuple(0, "b"));
  EXPECT_EQ(split_rhs("axbxc", 'x'), std::make_tuple(1, "bxc"));
}
TEST(StringViewTest, SplitReturnOnly) {
  // This is effectively a find but with extra steps.
  auto split_return = [](const char* value, char split_char) -> intptr_t {
    return iree_string_view_split(iree_make_cstring_view(value), split_char,
                                  nullptr, nullptr);
  };
  EXPECT_EQ(split_return("", 'x'), -1);
  EXPECT_EQ(split_return(" ", 'x'), -1);
  EXPECT_EQ(split_return("x", 'x'), 0);
  EXPECT_EQ(split_return(" x ", 'x'), 1);
  EXPECT_EQ(split_return("axb", 'x'), 1);
  EXPECT_EQ(split_return("axxxb", 'x'), 1);
  EXPECT_EQ(split_return("ax", 'x'), 1);
  EXPECT_EQ(split_return("xb", 'x'), 0);
  EXPECT_EQ(split_return("axbxc", 'x'), 1);
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

TEST(StringViewTest, ToCStringEmpty) {
  char buffer[3] = {0x7F, 0x7F, 0x7F};
  iree_string_view_to_cstring(IREE_SV(""), buffer, sizeof(buffer));
  EXPECT_EQ(buffer[0], 0);     // NUL
  EXPECT_EQ(buffer[1], 0x7F);  // unchanged
}

TEST(StringViewTest, ToCStringNoBuffer) {
  // Nothing to test but ASAN ensuring we don't null deref.
  iree_string_view_to_cstring(IREE_SV("abc"), NULL, 0);
}

TEST(StringViewTest, ToCString) {
  char buffer[6] = {0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F};
  iree_string_view_to_cstring(IREE_SV("abc"), buffer, sizeof(buffer));
  EXPECT_THAT(buffer, ElementsAre('a', 'b', 'c', 0, 0x7F, 0x7F));
}

TEST(StringViewTest, ToCStringTruncate) {
  char buffer[3] = {0x7F, 0x7F, 0x7F};
  iree_string_view_to_cstring(IREE_SV("abcdef"), buffer, sizeof(buffer));
  EXPECT_THAT(buffer, ElementsAre('a', 'b', 0));
}

TEST(StringViewTest, AppendToBuffer) {
  char buffer[6] = {0, 1, 2, 3, 4, 5};
  iree_string_view_t source = iree_make_cstring_view("test");
  iree_string_view_t target = {};
  const iree_host_size_t size =
      iree_string_view_append_to_buffer(source, &target, buffer);
  EXPECT_EQ(size, source.size);
  EXPECT_EQ(target.size, source.size);
  EXPECT_EQ(target.data, buffer);
  // Make sure we did not write past the source size.
  EXPECT_THAT(buffer, ElementsAre('t', 'e', 's', 't', 4, 5));
}

TEST(StringViewTest, AppendToBufferEmptySource) {
  char buffer[4] = {0, 1, 2, 3};
  iree_string_view_t source = iree_make_string_view(nullptr, 0);
  iree_string_view_t target = {};
  const iree_host_size_t size =
      iree_string_view_append_to_buffer(source, &target, buffer);
  EXPECT_EQ(size, 0u);
  EXPECT_EQ(target.size, 0u);
  EXPECT_EQ(target.data, buffer);
  // Make sure we did not write to the buffer.
  EXPECT_THAT(buffer, ElementsAre(0, 1, 2, 3));
}

TEST(StringViewTest, AppendToBufferEmptySourceAndBuffer) {
  iree_string_view_t source = iree_make_string_view(nullptr, 0);
  iree_string_view_t target = {};
  const iree_host_size_t size =
      iree_string_view_append_to_buffer(source, &target, nullptr);
  EXPECT_EQ(size, 0u);
  EXPECT_EQ(target.size, 0u);
  EXPECT_EQ(target.data, nullptr);
}

template <size_t N>
static iree::StatusOr<std::array<uint8_t, N>> ParseHex(const char* value) {
  std::array<uint8_t, N> buffer;
  if (!iree_string_view_parse_hex_bytes(iree_make_cstring_view(value),
                                        buffer.size(), buffer.data())) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  return {buffer};
}

TEST(StringViewTest, ParseHex0) {
  EXPECT_THAT(ParseHex<0>(""), IsOkAndHolds(ElementsAre()));
  EXPECT_THAT(ParseHex<0>(" "), IsOkAndHolds(ElementsAre()));
  EXPECT_THAT(ParseHex<0>("  "), IsOkAndHolds(ElementsAre()));
}

TEST(StringViewTest, ParseHex1) {
  EXPECT_THAT(ParseHex<1>("EF"), IsOkAndHolds(ElementsAre(0xEF)));
  EXPECT_THAT(ParseHex<1>("Ef"), IsOkAndHolds(ElementsAre(0xEF)));
  EXPECT_THAT(ParseHex<1>("00"), IsOkAndHolds(ElementsAre(0x00)));
  EXPECT_THAT(ParseHex<1>("FF"), IsOkAndHolds(ElementsAre(0xFF)));
  EXPECT_THAT(ParseHex<1>(" EF "), IsOkAndHolds(ElementsAre(0xEF)));
}

TEST(StringViewTest, ParseHex2) {
  // No separators:
  EXPECT_THAT(ParseHex<2>("ABEF"), IsOkAndHolds(ElementsAre(0xAB, 0xEF)));
  // Mixed-case with separators:
  EXPECT_THAT(ParseHex<2>("Ab-eF"), IsOkAndHolds(ElementsAre(0xAB, 0xEF)));
  // Upper-case with separators:
  EXPECT_THAT(ParseHex<2>("AB-EF"), IsOkAndHolds(ElementsAre(0xAB, 0xEF)));
  // Lower-case with separators:
  EXPECT_THAT(ParseHex<2>("ab-ef"), IsOkAndHolds(ElementsAre(0xAB, 0xEF)));
  // Min:
  EXPECT_THAT(ParseHex<2>("00-00"), IsOkAndHolds(ElementsAre(0x00, 0x00)));
  // Max:
  EXPECT_THAT(ParseHex<2>("FF-FF"), IsOkAndHolds(ElementsAre(0xFF, 0xFF)));
}

TEST(StringViewTest, ParseHex16) {
  // No separators:
  EXPECT_THAT(ParseHex<16>("000102030405060708090A0B0C0D0E0F"),
              IsOkAndHolds(ElementsAre(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                       0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                       0x0E, 0x0F)));
  // Mixed-case with separators:
  EXPECT_THAT(ParseHex<16>("00010203-0405-0607-0809-0a0B0c0D0e0F"),
              IsOkAndHolds(ElementsAre(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                       0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                       0x0E, 0x0F)));
  // Upper-case with separators:
  EXPECT_THAT(ParseHex<16>("00010203-0405-0607-0809-0A0B0C0D0E0F"),
              IsOkAndHolds(ElementsAre(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                       0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                       0x0E, 0x0F)));
  // Lower-case with separators:
  EXPECT_THAT(ParseHex<16>("00010203-0405-0607-0809-0a0b0c0d0e0f"),
              IsOkAndHolds(ElementsAre(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                       0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                       0x0E, 0x0F)));
  // Min:
  EXPECT_THAT(ParseHex<16>("00000000-0000-0000-0000-000000000000"),
              IsOkAndHolds(ElementsAre(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                       0x00, 0x00)));
  // Max:
  EXPECT_THAT(ParseHex<16>("FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"),
              IsOkAndHolds(ElementsAre(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                       0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                       0xFF, 0xFF)));
  // Morse code:
  EXPECT_THAT(ParseHex<16>("00-01-02-03-04-05-06-07-08-09-0a-0b-0c-0d-0e-0f"),
              IsOkAndHolds(ElementsAre(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                       0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                       0x0E, 0x0F)));
}

TEST(StringViewTest, InvalidParseHex) {
  // Leading/trailing -'s:
  EXPECT_THAT(ParseHex<1>("-"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseHex<1>("-ab"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseHex<1>("ab-"), StatusIs(StatusCode::kInvalidArgument));

  // Too little data:
  EXPECT_THAT(ParseHex<1>(""), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseHex<1>("a"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseHex<2>("abc"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseHex<2>("ab-a"), StatusIs(StatusCode::kInvalidArgument));

  // Too much data:
  EXPECT_THAT(ParseHex<0>("ab"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseHex<1>("abc"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseHex<1>("abcd"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseHex<1>("ab-"), StatusIs(StatusCode::kInvalidArgument));

  // Invalid characters:
  EXPECT_THAT(ParseHex<1>("ZF"), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseHex<2>("AB-C?"), StatusIs(StatusCode::kInvalidArgument));
}

iree::StatusOr<iree_device_size_t> ParseDeviceSize(std::string_view value) {
  iree_device_size_t size = 0;
  IREE_RETURN_IF_ERROR(iree::Status(iree_string_view_parse_device_size(
      iree_string_view_t{value.data(), value.size()}, &size)));
  return size;
}

TEST(StringViewTest, ParseDeviceSize) {
  EXPECT_THAT(ParseDeviceSize("0"), IsOkAndHolds(Eq(0u)));
  EXPECT_THAT(ParseDeviceSize("1"), IsOkAndHolds(Eq(1u)));
  EXPECT_THAT(ParseDeviceSize("10000"), IsOkAndHolds(Eq(10000u)));
  EXPECT_THAT(ParseDeviceSize("0b"), IsOkAndHolds(Eq(0u)));
  EXPECT_THAT(ParseDeviceSize("0kb"), IsOkAndHolds(Eq(0u)));
  EXPECT_THAT(ParseDeviceSize("0gib"), IsOkAndHolds(Eq(0u)));
  EXPECT_THAT(ParseDeviceSize("1b"), IsOkAndHolds(Eq(1)));
  EXPECT_THAT(ParseDeviceSize("1kb"), IsOkAndHolds(Eq(1 * 1000u)));
  EXPECT_THAT(ParseDeviceSize("1kib"), IsOkAndHolds(Eq(1 * 1024u)));
  EXPECT_THAT(ParseDeviceSize("1000b"), IsOkAndHolds(Eq(1000 * 1u)));
  EXPECT_THAT(ParseDeviceSize("1000kb"), IsOkAndHolds(Eq(1000 * 1000u)));
  EXPECT_THAT(ParseDeviceSize("1000kib"), IsOkAndHolds(Eq(1000 * 1024u)));

  // NOTE: we don't verify very hard here and accept random suffixes just like
  // atoi does (because under the covers we're using atoi).
  EXPECT_THAT(ParseDeviceSize("123fb"), IsOkAndHolds(Eq(123u)));
}

TEST(StringViewTest, ParseDeviceSizeInvalid) {
  EXPECT_THAT(ParseDeviceSize(""), StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseDeviceSize("abc"), StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
