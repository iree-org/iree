// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/json.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

#define EXPECT_SV_EQ(actual, expected) \
  EXPECT_TRUE(iree_string_view_equal(actual, expected))

//===----------------------------------------------------------------------===//
// Consume String Tests
//===----------------------------------------------------------------------===//

TEST(JsonConsumeStringTest, Simple) {
  iree_string_view_t str = IREE_SV("\"hello\"");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_string(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("hello"));
  EXPECT_EQ(str.size, 0);
}

TEST(JsonConsumeStringTest, Empty) {
  iree_string_view_t str = IREE_SV("\"\"");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_string(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV(""));
  EXPECT_EQ(str.size, 0);
}

TEST(JsonConsumeStringTest, WithEscapes) {
  iree_string_view_t str = IREE_SV("\"hello\\nworld\"");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_string(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("hello\\nworld"));  // Raw, not unescaped.
}

TEST(JsonConsumeStringTest, WithAllEscapes) {
  iree_string_view_t str = IREE_SV("\"\\\"\\\\/\\b\\f\\n\\r\\t\"");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_string(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("\\\"\\\\/\\b\\f\\n\\r\\t"));
}

TEST(JsonConsumeStringTest, WithUnicodeEscape) {
  iree_string_view_t str = IREE_SV("\"\\u0041\"");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_string(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("\\u0041"));  // Raw.
}

TEST(JsonConsumeStringTest, Unterminated) {
  iree_string_view_t str = IREE_SV("\"hello");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_string(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeStringTest, MissingPrefix) {
  iree_string_view_t str = IREE_SV("hello\"");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_string(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeStringTest, ControlCharacter) {
  // Control characters (0x00-0x1F) must be escaped per RFC 8259.
  iree_string_view_t str = IREE_SV("\"hello\tworld\"");  // Tab is 0x09.
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_string(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeStringTest, ControlCharacterNewline) {
  // Unescaped newline is invalid.
  iree_string_view_t str = IREE_SV("\"hello\nworld\"");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_string(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeStringTest, InvalidHexDigitInUnicode) {
  // \uNNNN requires valid hex digits.
  iree_string_view_t str = IREE_SV("\"\\u00GX\"");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_string(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeStringTest, TrailingContent) {
  iree_string_view_t str = IREE_SV("\"hello\", more");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_string(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("hello"));
  EXPECT_SV_EQ(str, IREE_SV(", more"));
}

//===----------------------------------------------------------------------===//
// Consume Number Tests
//===----------------------------------------------------------------------===//

TEST(JsonConsumeNumberTest, Integer) {
  iree_string_view_t str = IREE_SV("123");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_number(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("123"));
}

TEST(JsonConsumeNumberTest, Zero) {
  iree_string_view_t str = IREE_SV("0");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_number(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("0"));
}

TEST(JsonConsumeNumberTest, Negative) {
  iree_string_view_t str = IREE_SV("-456");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_number(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("-456"));
}

TEST(JsonConsumeNumberTest, Float) {
  iree_string_view_t str = IREE_SV("3.14");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_number(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("3.14"));
}

TEST(JsonConsumeNumberTest, NegativeFloat) {
  iree_string_view_t str = IREE_SV("-2.5");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_number(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("-2.5"));
}

TEST(JsonConsumeNumberTest, Scientific) {
  iree_string_view_t str = IREE_SV("1.5e10");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_number(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("1.5e10"));
}

TEST(JsonConsumeNumberTest, ScientificUpperE) {
  iree_string_view_t str = IREE_SV("1E-5");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_number(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("1E-5"));
}

TEST(JsonConsumeNumberTest, ScientificPositiveExp) {
  iree_string_view_t str = IREE_SV("1e+10");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_number(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("1e+10"));
}

TEST(JsonConsumeNumberTest, TrailingContent) {
  iree_string_view_t str = IREE_SV("123, more");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_number(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("123"));
  EXPECT_SV_EQ(str, IREE_SV(", more"));
}

TEST(JsonConsumeNumberTest, Empty) {
  iree_string_view_t str = IREE_SV("");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_number(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeNumberTest, JustMinus) {
  iree_string_view_t str = IREE_SV("-");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_number(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Consume Keyword Tests
//===----------------------------------------------------------------------===//

TEST(JsonConsumeKeywordTest, True) {
  iree_string_view_t str = IREE_SV("true");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_keyword(&str, IREE_SV("true"), &value));
  EXPECT_SV_EQ(value, IREE_SV("true"));
}

TEST(JsonConsumeKeywordTest, False) {
  iree_string_view_t str = IREE_SV("false");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_keyword(&str, IREE_SV("false"), &value));
  EXPECT_SV_EQ(value, IREE_SV("false"));
}

TEST(JsonConsumeKeywordTest, Null) {
  iree_string_view_t str = IREE_SV("null");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_keyword(&str, IREE_SV("null"), &value));
  EXPECT_SV_EQ(value, IREE_SV("null"));
}

TEST(JsonConsumeKeywordTest, WrongKeyword) {
  iree_string_view_t str = IREE_SV("false");
  iree_string_view_t value;
  iree_status_t status =
      iree_json_consume_keyword(&str, IREE_SV("true"), &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Consume Object Tests
//===----------------------------------------------------------------------===//

TEST(JsonConsumeObjectTest, Empty) {
  iree_string_view_t str = IREE_SV("{}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{}"));
}

TEST(JsonConsumeObjectTest, Simple) {
  iree_string_view_t str = IREE_SV("{\"key\": 123}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"key\": 123}"));
}

TEST(JsonConsumeObjectTest, Multiple) {
  iree_string_view_t str = IREE_SV("{\"a\": 1, \"b\": 2}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"a\": 1, \"b\": 2}"));
}

TEST(JsonConsumeObjectTest, Nested) {
  iree_string_view_t str = IREE_SV("{\"a\": {\"b\": 1}}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"a\": {\"b\": 1}}"));
}

TEST(JsonConsumeObjectTest, MissingBrace) {
  iree_string_view_t str = IREE_SV("{\"key\": 123");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_object(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeObjectTest, TrailingComma) {
  // JSONC allows trailing commas.
  iree_string_view_t str = IREE_SV("{\"key\": 123,}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_TRUE(iree_string_view_equal(value, IREE_SV("{\"key\": 123,}")));
}

TEST(JsonConsumeObjectTest, WhitespaceBeforeClosingBrace) {
  // Whitespace before closing brace should be accepted.
  iree_string_view_t str = IREE_SV("{\"key\": 123  }");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"key\": 123  }"));
}

TEST(JsonConsumeObjectTest, WhitespaceAfterComma) {
  // Whitespace after comma (before next key) should be accepted.
  iree_string_view_t str = IREE_SV("{\"a\": 1,  \"b\": 2}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"a\": 1,  \"b\": 2}"));
}

TEST(JsonConsumeObjectTest, SingleLineComment) {
  // JSONC single-line comments should be skipped.
  iree_string_view_t str = IREE_SV("{// comment\n\"key\": 123}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{// comment\n\"key\": 123}"));
}

TEST(JsonConsumeObjectTest, MultiLineComment) {
  // JSONC multi-line comments should be skipped.
  iree_string_view_t str = IREE_SV("{/* comment */\"key\": 123}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{/* comment */\"key\": 123}"));
}

TEST(JsonConsumeObjectTest, CommentAfterColon) {
  // Comments allowed after the colon before the value.
  iree_string_view_t str = IREE_SV("{\"key\":/* value */ 123}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"key\":/* value */ 123}"));
}

TEST(JsonConsumeObjectTest, UnterminatedComment) {
  // Unterminated multi-line comments should error.
  iree_string_view_t str = IREE_SV("{/* unterminated \"key\": 123}");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_object(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Consume Array Tests
//===----------------------------------------------------------------------===//

TEST(JsonConsumeArrayTest, Empty) {
  iree_string_view_t str = IREE_SV("[]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[]"));
}

TEST(JsonConsumeArrayTest, SingleElement) {
  iree_string_view_t str = IREE_SV("[123]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[123]"));
}

TEST(JsonConsumeArrayTest, Multiple) {
  iree_string_view_t str = IREE_SV("[1, 2, 3]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[1, 2, 3]"));
}

TEST(JsonConsumeArrayTest, Mixed) {
  iree_string_view_t str = IREE_SV("[1, \"two\", true, null]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[1, \"two\", true, null]"));
}

TEST(JsonConsumeArrayTest, Nested) {
  iree_string_view_t str = IREE_SV("[[1, 2], [3, 4]]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[[1, 2], [3, 4]]"));
}

TEST(JsonConsumeArrayTest, MissingBracket) {
  iree_string_view_t str = IREE_SV("[1, 2, 3");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_array(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeArrayTest, TrailingComma) {
  // JSONC allows trailing commas.
  iree_string_view_t str = IREE_SV("[1, 2,]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_TRUE(iree_string_view_equal(value, IREE_SV("[1, 2,]")));
}

TEST(JsonConsumeArrayTest, WhitespaceBeforeClosingBracket) {
  // Whitespace before closing bracket should be accepted.
  iree_string_view_t str = IREE_SV("[1, 2, 3  ]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[1, 2, 3  ]"));
}

TEST(JsonConsumeArrayTest, WhitespaceAfterComma) {
  // Whitespace after comma (before next element) should be accepted.
  iree_string_view_t str = IREE_SV("[1,  2,  3]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[1,  2,  3]"));
}

TEST(JsonConsumeArrayTest, SingleLineComment) {
  // JSONC single-line comments should be skipped.
  iree_string_view_t str = IREE_SV("[// comment\n1, 2]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[// comment\n1, 2]"));
}

TEST(JsonConsumeArrayTest, MultiLineComment) {
  // JSONC multi-line comments should be skipped.
  iree_string_view_t str = IREE_SV("[/* comment */1, 2]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[/* comment */1, 2]"));
}

TEST(JsonConsumeArrayTest, CommentBetweenElements) {
  // Comments allowed between elements.
  iree_string_view_t str = IREE_SV("[1,/* mid */2]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[1,/* mid */2]"));
}

//===----------------------------------------------------------------------===//
// Consume Value Tests
//===----------------------------------------------------------------------===//

TEST(JsonConsumeValueTest, String) {
  iree_string_view_t str = IREE_SV("\"hello\"");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("hello"));
}

TEST(JsonConsumeValueTest, Number) {
  iree_string_view_t str = IREE_SV("123");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("123"));
}

TEST(JsonConsumeValueTest, NegativeNumber) {
  iree_string_view_t str = IREE_SV("-456");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("-456"));
}

TEST(JsonConsumeValueTest, Object) {
  iree_string_view_t str = IREE_SV("{}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{}"));
}

TEST(JsonConsumeValueTest, Array) {
  iree_string_view_t str = IREE_SV("[]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[]"));
}

TEST(JsonConsumeValueTest, True) {
  iree_string_view_t str = IREE_SV("true");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("true"));
}

TEST(JsonConsumeValueTest, WithLeadingWhitespace) {
  iree_string_view_t str = IREE_SV("  \n\t123");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("123"));
}

TEST(JsonConsumeValueTest, CommentBeforeValue) {
  // JSONC comments before a value should be skipped.
  iree_string_view_t str = IREE_SV("/* comment */ 123");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("123"));
}

TEST(JsonConsumeValueTest, SingleLineCommentBeforeValue) {
  // JSONC single-line comments before a value should be skipped.
  iree_string_view_t str = IREE_SV("// comment\n123");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("123"));
}

TEST(JsonConsumeValueTest, Empty) {
  // Empty input should return an error.
  iree_string_view_t str = IREE_SV("");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_value(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeValueTest, WhitespaceOnly) {
  // Whitespace-only input should return an error (no actual value).
  iree_string_view_t str = IREE_SV("   \n\t  ");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_value(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Enumerate Object Tests
//===----------------------------------------------------------------------===//

struct ObjectEntry {
  std::string key;
  std::string value;
};

static iree_status_t CollectObjectEntries(void* user_data,
                                          iree_string_view_t key,
                                          iree_string_view_t value) {
  auto* entries = static_cast<std::vector<ObjectEntry>*>(user_data);
  entries->push_back(
      {std::string(key.data, key.size), std::string(value.data, value.size)});
  return iree_ok_status();
}

TEST(JsonEnumerateObjectTest, Empty) {
  std::vector<ObjectEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_object(IREE_SV("{}"), CollectObjectEntries,
                                            &entries));
  EXPECT_TRUE(entries.empty());
}

TEST(JsonEnumerateObjectTest, Single) {
  std::vector<ObjectEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_object(IREE_SV("{\"key\": 123}"),
                                            CollectObjectEntries, &entries));
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].key, "key");
  EXPECT_EQ(entries[0].value, "123");
}

TEST(JsonEnumerateObjectTest, Multiple) {
  std::vector<ObjectEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_object(
      IREE_SV("{\"a\": 1, \"b\": \"two\", \"c\": true}"), CollectObjectEntries,
      &entries));
  ASSERT_EQ(entries.size(), 3);
  EXPECT_EQ(entries[0].key, "a");
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].key, "b");
  EXPECT_EQ(entries[1].value, "two");
  EXPECT_EQ(entries[2].key, "c");
  EXPECT_EQ(entries[2].value, "true");
}

TEST(JsonEnumerateObjectTest, NestedValue) {
  std::vector<ObjectEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_object(IREE_SV("{\"obj\": {\"x\": 1}}"),
                                            CollectObjectEntries, &entries));
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].key, "obj");
  EXPECT_EQ(entries[0].value, "{\"x\": 1}");
}

static iree_status_t StopAfterTwo(void* user_data, iree_string_view_t key,
                                  iree_string_view_t value) {
  auto* count = static_cast<int*>(user_data);
  (*count)++;
  if (*count >= 2) {
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  return iree_ok_status();
}

TEST(JsonEnumerateObjectTest, EarlyTermination) {
  int count = 0;
  IREE_ASSERT_OK(iree_json_enumerate_object(
      IREE_SV("{\"a\": 1, \"b\": 2, \"c\": 3}"), StopAfterTwo, &count));
  EXPECT_EQ(count, 2);
}

TEST(JsonEnumerateObjectTest, TrailingComma) {
  // JSONC allows trailing commas.
  std::vector<ObjectEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_object(IREE_SV("{\"key\": 123,}"),
                                            CollectObjectEntries, &entries));
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].key, "key");
  EXPECT_EQ(entries[0].value, "123");
}

TEST(JsonEnumerateObjectTest, WhitespaceBeforeClose) {
  std::vector<ObjectEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_object(IREE_SV("{\"key\": 123  }"),
                                            CollectObjectEntries, &entries));
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].key, "key");
  EXPECT_EQ(entries[0].value, "123");
}

//===----------------------------------------------------------------------===//
// Lookup Object Value Tests
//===----------------------------------------------------------------------===//

TEST(JsonLookupObjectValueTest, Found) {
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_lookup_object_value(IREE_SV("{\"key\": 123}"),
                                               IREE_SV("key"), &value));
  EXPECT_SV_EQ(value, IREE_SV("123"));
}

TEST(JsonLookupObjectValueTest, NotFound) {
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_lookup_object_value(IREE_SV("{\"key\": 123}"),
                                               IREE_SV("missing"), &value));
  EXPECT_EQ(value.size, 0);
}

TEST(JsonLookupObjectValueTest, NestedObject) {
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      IREE_SV("{\"outer\": {\"inner\": 1}}"), IREE_SV("outer"), &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"inner\": 1}"));
}

TEST(JsonLookupObjectValueTest, StringValue) {
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_lookup_object_value(IREE_SV("{\"key\": \"value\"}"),
                                               IREE_SV("key"), &value));
  EXPECT_SV_EQ(value, IREE_SV("value"));
}

//===----------------------------------------------------------------------===//
// Enumerate Array Tests
//===----------------------------------------------------------------------===//

struct ArrayEntry {
  iree_host_size_t index;
  std::string value;
};

static iree_status_t CollectArrayEntries(void* user_data,
                                         iree_host_size_t index,
                                         iree_string_view_t value) {
  auto* entries = static_cast<std::vector<ArrayEntry>*>(user_data);
  entries->push_back({index, std::string(value.data, value.size)});
  return iree_ok_status();
}

TEST(JsonEnumerateArrayTest, Empty) {
  std::vector<ArrayEntry> entries;
  IREE_ASSERT_OK(
      iree_json_enumerate_array(IREE_SV("[]"), CollectArrayEntries, &entries));
  EXPECT_TRUE(entries.empty());
}

TEST(JsonEnumerateArrayTest, Single) {
  std::vector<ArrayEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_array(IREE_SV("[123]"),
                                           CollectArrayEntries, &entries));
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].index, 0);
  EXPECT_EQ(entries[0].value, "123");
}

TEST(JsonEnumerateArrayTest, Multiple) {
  std::vector<ArrayEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_array(IREE_SV("[1, \"two\", true]"),
                                           CollectArrayEntries, &entries));
  ASSERT_EQ(entries.size(), 3);
  EXPECT_EQ(entries[0].index, 0);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].index, 1);
  EXPECT_EQ(entries[1].value, "two");
  EXPECT_EQ(entries[2].index, 2);
  EXPECT_EQ(entries[2].value, "true");
}

static iree_status_t StopAfterTwoArray(void* user_data, iree_host_size_t index,
                                       iree_string_view_t value) {
  auto* count = static_cast<int*>(user_data);
  (*count)++;
  if (*count >= 2) {
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  return iree_ok_status();
}

TEST(JsonEnumerateArrayTest, EarlyTermination) {
  int count = 0;
  IREE_ASSERT_OK(iree_json_enumerate_array(IREE_SV("[1, 2, 3, 4]"),
                                           StopAfterTwoArray, &count));
  EXPECT_EQ(count, 2);
}

TEST(JsonEnumerateArrayTest, TrailingComma) {
  // JSONC allows trailing commas.
  std::vector<ArrayEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_array(IREE_SV("[1, 2,]"),
                                           CollectArrayEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].value, "2");
}

TEST(JsonEnumerateArrayTest, WhitespaceBeforeClose) {
  std::vector<ArrayEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_array(IREE_SV("[1, 2  ]"),
                                           CollectArrayEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].value, "2");
}

//===----------------------------------------------------------------------===//
// JSONL (JSON Lines) Tests
//===----------------------------------------------------------------------===//

struct LineEntry {
  iree_json_line_number_t line_number;
  iree_host_size_t index;
  std::string value;
};

static iree_status_t CollectLineEntries(void* user_data,
                                        iree_json_line_number_t line_number,
                                        iree_host_size_t index,
                                        iree_string_view_t value) {
  auto* entries = static_cast<std::vector<LineEntry>*>(user_data);
  entries->push_back({line_number, index, std::string(value.data, value.size)});
  return iree_ok_status();
}

static iree_status_t StopAfterTwoLines(void* user_data,
                                       iree_json_line_number_t line_number,
                                       iree_host_size_t index,
                                       iree_string_view_t value) {
  auto* count = static_cast<int*>(user_data);
  (*count)++;
  if (*count >= 2) {
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  return iree_ok_status();
}

TEST(JsonEnumerateLinesTest, Empty) {
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(
      iree_json_enumerate_lines(IREE_SV(""), CollectLineEntries, &entries));
  EXPECT_TRUE(entries.empty());
}

TEST(JsonEnumerateLinesTest, SingleLine) {
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_lines(IREE_SV("{\"key\": 123}"),
                                           CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].line_number, 1);  // 1-based.
  EXPECT_EQ(entries[0].index, 0);        // 0-based.
  EXPECT_EQ(entries[0].value, "{\"key\": 123}");
}

TEST(JsonEnumerateLinesTest, MultipleLines) {
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(
      iree_json_enumerate_lines(IREE_SV("{\"a\": 1}\n{\"b\": 2}\n{\"c\": 3}"),
                                CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 3);
  EXPECT_EQ(entries[0].line_number, 1);
  EXPECT_EQ(entries[0].index, 0);
  EXPECT_EQ(entries[0].value, "{\"a\": 1}");
  EXPECT_EQ(entries[1].line_number, 2);
  EXPECT_EQ(entries[1].index, 1);
  EXPECT_EQ(entries[1].value, "{\"b\": 2}");
  EXPECT_EQ(entries[2].line_number, 3);
  EXPECT_EQ(entries[2].index, 2);
  EXPECT_EQ(entries[2].value, "{\"c\": 3}");
}

TEST(JsonEnumerateLinesTest, MixedValueTypes) {
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(
      iree_json_enumerate_lines(IREE_SV("123\n\"hello\"\ntrue\nnull\n[1, 2]"),
                                CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 5);
  EXPECT_EQ(entries[0].value, "123");
  EXPECT_EQ(entries[1].value, "hello");
  EXPECT_EQ(entries[2].value, "true");
  EXPECT_EQ(entries[3].value, "null");
  EXPECT_EQ(entries[4].value, "[1, 2]");
}

TEST(JsonEnumerateLinesTest, EmptyLinesSkipped) {
  std::vector<LineEntry> entries;
  // Input: "1\n\n2\n\n\n3" has values on lines 1, 3, 6.
  IREE_ASSERT_OK(iree_json_enumerate_lines(IREE_SV("1\n\n2\n\n\n3"),
                                           CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 3);
  EXPECT_EQ(entries[0].line_number, 1);
  EXPECT_EQ(entries[0].index, 0);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].line_number, 3);
  EXPECT_EQ(entries[1].index, 1);
  EXPECT_EQ(entries[1].value, "2");
  EXPECT_EQ(entries[2].line_number, 6);
  EXPECT_EQ(entries[2].index, 2);
  EXPECT_EQ(entries[2].value, "3");
}

TEST(JsonEnumerateLinesTest, WhitespaceOnlyLinesSkipped) {
  std::vector<LineEntry> entries;
  // Input: "1\n   \n2\n\t\n3" has values on lines 1, 3, 5.
  IREE_ASSERT_OK(iree_json_enumerate_lines(IREE_SV("1\n   \n2\n\t\n3"),
                                           CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 3);
  EXPECT_EQ(entries[0].line_number, 1);
  EXPECT_EQ(entries[1].line_number, 3);
  EXPECT_EQ(entries[2].line_number, 5);
}

TEST(JsonEnumerateLinesTest, LeadingTrailingWhitespaceOnLines) {
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_lines(IREE_SV("  1  \n\t2\t\n  3"),
                                           CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 3);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].value, "2");
  EXPECT_EQ(entries[2].value, "3");
}

TEST(JsonEnumerateLinesTest, TrailingNewline) {
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_lines(IREE_SV("1\n2\n"),
                                           CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].value, "2");
}

TEST(JsonEnumerateLinesTest, CRLFLineEndings) {
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_lines(IREE_SV("1\r\n2\r\n3"),
                                           CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 3);
  EXPECT_EQ(entries[0].line_number, 1);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].line_number, 2);
  EXPECT_EQ(entries[1].value, "2");
  EXPECT_EQ(entries[2].line_number, 3);
  EXPECT_EQ(entries[2].value, "3");
}

TEST(JsonEnumerateLinesTest, MixedLineEndings) {
  std::vector<LineEntry> entries;
  // Mix of LF, CRLF, and LF again.
  IREE_ASSERT_OK(iree_json_enumerate_lines(IREE_SV("1\n2\r\n3\n4"),
                                           CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 4);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].value, "2");
  EXPECT_EQ(entries[2].value, "3");
  EXPECT_EQ(entries[3].value, "4");
}

TEST(JsonEnumerateLinesTest, TrailingContentOnLine) {
  std::vector<LineEntry> entries;
  // After parsing "123", there's " extra" remaining on the line.
  iree_status_t status = iree_json_enumerate_lines(
      IREE_SV("123 extra\n456"), CollectLineEntries, &entries);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonEnumerateLinesTest, EarlyTermination) {
  int count = 0;
  IREE_ASSERT_OK(iree_json_enumerate_lines(IREE_SV("1\n2\n3\n4"),
                                           StopAfterTwoLines, &count));
  EXPECT_EQ(count, 2);
}

TEST(JsonEnumerateLinesTest, InvalidJson) {
  std::vector<LineEntry> entries;
  iree_status_t status = iree_json_enumerate_lines(
      IREE_SV("{\"valid\": 1}\ninvalid json\n{\"also\": 2}"),
      CollectLineEntries, &entries);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonEnumerateLinesTest, CommentLinesSkipped) {
  // Lines that are only comments should be skipped like empty lines.
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_lines(
      IREE_SV("// comment line\n123\n/* another comment */\n456"),
      CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].line_number, 2);
  EXPECT_EQ(entries[0].value, "123");
  EXPECT_EQ(entries[1].line_number, 4);
  EXPECT_EQ(entries[1].value, "456");
}

TEST(JsonEnumerateLinesTest, TrailingComment) {
  // Comments after a value on the same line are allowed.
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_lines(
      IREE_SV("123 // trailing comment\n456"), CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].value, "123");
  EXPECT_EQ(entries[1].value, "456");
}

//===----------------------------------------------------------------------===//
// Unescape String Tests
//===----------------------------------------------------------------------===//

TEST(JsonUnescapeStringTest, NoEscapes) {
  char buf[64];
  iree_host_size_t len;
  IREE_ASSERT_OK(
      iree_json_unescape_string(IREE_SV("hello"), sizeof(buf), buf, &len));
  EXPECT_EQ(len, 5);
  EXPECT_EQ(std::string(buf, len), "hello");
}

TEST(JsonUnescapeStringTest, Empty) {
  char buf[64];
  iree_host_size_t len;
  IREE_ASSERT_OK(
      iree_json_unescape_string(IREE_SV(""), sizeof(buf), buf, &len));
  EXPECT_EQ(len, 0);
}

TEST(JsonUnescapeStringTest, SimpleEscapes) {
  char buf[64];
  iree_host_size_t len;
  IREE_ASSERT_OK(
      iree_json_unescape_string(IREE_SV("\\n\\t\\r"), sizeof(buf), buf, &len));
  EXPECT_EQ(len, 3);
  EXPECT_EQ(buf[0], '\n');
  EXPECT_EQ(buf[1], '\t');
  EXPECT_EQ(buf[2], '\r');
}

TEST(JsonUnescapeStringTest, AllSimpleEscapes) {
  char buf[64];
  iree_host_size_t len;
  IREE_ASSERT_OK(iree_json_unescape_string(
      IREE_SV("\\\"\\\\\\b\\f\\n\\r\\t\\/"), sizeof(buf), buf, &len));
  EXPECT_EQ(len, 8);
  EXPECT_EQ(buf[0], '"');
  EXPECT_EQ(buf[1], '\\');
  EXPECT_EQ(buf[2], '\b');
  EXPECT_EQ(buf[3], '\f');
  EXPECT_EQ(buf[4], '\n');
  EXPECT_EQ(buf[5], '\r');
  EXPECT_EQ(buf[6], '\t');
  EXPECT_EQ(buf[7], '/');
}

TEST(JsonUnescapeStringTest, UnicodeBasic) {
  char buf[64];
  iree_host_size_t len;
  // \u0041 = 'A' (ASCII 65)
  IREE_ASSERT_OK(
      iree_json_unescape_string(IREE_SV("\\u0041"), sizeof(buf), buf, &len));
  EXPECT_EQ(len, 1);
  EXPECT_EQ(buf[0], 'A');
}

TEST(JsonUnescapeStringTest, UnicodeTwoByte) {
  char buf[64];
  iree_host_size_t len;
  // \u00E9 = 'é' (U+00E9) -> UTF-8: 0xC3 0xA9
  IREE_ASSERT_OK(
      iree_json_unescape_string(IREE_SV("\\u00E9"), sizeof(buf), buf, &len));
  EXPECT_EQ(len, 2);
  EXPECT_EQ((unsigned char)buf[0], 0xC3);
  EXPECT_EQ((unsigned char)buf[1], 0xA9);
}

TEST(JsonUnescapeStringTest, UnicodeThreeByte) {
  char buf[64];
  iree_host_size_t len;
  // \u4E2D = '中' (U+4E2D) -> UTF-8: 0xE4 0xB8 0xAD
  IREE_ASSERT_OK(
      iree_json_unescape_string(IREE_SV("\\u4E2D"), sizeof(buf), buf, &len));
  EXPECT_EQ(len, 3);
  EXPECT_EQ((unsigned char)buf[0], 0xE4);
  EXPECT_EQ((unsigned char)buf[1], 0xB8);
  EXPECT_EQ((unsigned char)buf[2], 0xAD);
}

TEST(JsonUnescapeStringTest, UnicodeSurrogate) {
  char buf[64];
  iree_host_size_t len;
  // \uD83D\uDE00 = 😀 (U+1F600) -> UTF-8: 0xF0 0x9F 0x98 0x80
  IREE_ASSERT_OK(iree_json_unescape_string(IREE_SV("\\uD83D\\uDE00"),
                                           sizeof(buf), buf, &len));
  EXPECT_EQ(len, 4);
  EXPECT_EQ((unsigned char)buf[0], 0xF0);
  EXPECT_EQ((unsigned char)buf[1], 0x9F);
  EXPECT_EQ((unsigned char)buf[2], 0x98);
  EXPECT_EQ((unsigned char)buf[3], 0x80);
}

TEST(JsonUnescapeStringTest, MixedContent) {
  char buf[64];
  iree_host_size_t len;
  IREE_ASSERT_OK(iree_json_unescape_string(IREE_SV("hello\\nworld"),
                                           sizeof(buf), buf, &len));
  EXPECT_EQ(len, 11);
  EXPECT_EQ(std::string(buf, len), "hello\nworld");
}

TEST(JsonUnescapeStringTest, BufferTooSmall) {
  char buf[4];
  iree_host_size_t len;
  iree_status_t status =
      iree_json_unescape_string(IREE_SV("hello"), sizeof(buf), buf, &len);
  EXPECT_TRUE(iree_status_is_resource_exhausted(status));
  iree_status_ignore(status);
  EXPECT_EQ(len, 5);  // Required size.
}

TEST(JsonUnescapeStringTest, LengthOnly) {
  iree_host_size_t len;
  IREE_ASSERT_OK(
      iree_json_unescape_string(IREE_SV("hello\\nworld"), 0, NULL, &len));
  EXPECT_EQ(len, 11);
}

TEST(JsonUnescapeStringTest, InvalidEscape) {
  char buf[64];
  iree_host_size_t len;
  iree_status_t status =
      iree_json_unescape_string(IREE_SV("\\x"), sizeof(buf), buf, &len);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonUnescapeStringTest, TruncatedUnicode) {
  char buf[64];
  iree_host_size_t len;
  iree_status_t status =
      iree_json_unescape_string(IREE_SV("\\u00"), sizeof(buf), buf, &len);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonUnescapeStringTest, LoneSurrogate) {
  char buf[64];
  iree_host_size_t len;
  // High surrogate without low surrogate.
  iree_status_t status =
      iree_json_unescape_string(IREE_SV("\\uD83D"), sizeof(buf), buf, &len);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonUnescapeStringTest, InvalidHexDigit) {
  char buf[64];
  iree_host_size_t len;
  // 'G' is not a valid hex digit.
  iree_status_t status =
      iree_json_unescape_string(IREE_SV("\\u00GX"), sizeof(buf), buf, &len);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Parse Number Tests
//===----------------------------------------------------------------------===//

TEST(JsonParseInt64Test, Zero) {
  int64_t value;
  IREE_ASSERT_OK(iree_json_parse_int64(IREE_SV("0"), &value));
  EXPECT_EQ(value, 0);
}

TEST(JsonParseInt64Test, Positive) {
  int64_t value;
  IREE_ASSERT_OK(iree_json_parse_int64(IREE_SV("123"), &value));
  EXPECT_EQ(value, 123);
}

TEST(JsonParseInt64Test, Negative) {
  int64_t value;
  IREE_ASSERT_OK(iree_json_parse_int64(IREE_SV("-456"), &value));
  EXPECT_EQ(value, -456);
}

TEST(JsonParseInt64Test, Max) {
  int64_t value;
  IREE_ASSERT_OK(iree_json_parse_int64(IREE_SV("9223372036854775807"), &value));
  EXPECT_EQ(value, INT64_MAX);
}

TEST(JsonParseInt64Test, Min) {
  int64_t value;
  IREE_ASSERT_OK(
      iree_json_parse_int64(IREE_SV("-9223372036854775808"), &value));
  EXPECT_EQ(value, INT64_MIN);
}

TEST(JsonParseInt64Test, Overflow) {
  int64_t value;
  iree_status_t status =
      iree_json_parse_int64(IREE_SV("9223372036854775808"), &value);
  EXPECT_TRUE(iree_status_is_out_of_range(status));
  iree_status_ignore(status);
}

TEST(JsonParseInt64Test, Float) {
  int64_t value;
  iree_status_t status = iree_json_parse_int64(IREE_SV("3.14"), &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonParseUint64Test, Zero) {
  uint64_t value;
  IREE_ASSERT_OK(iree_json_parse_uint64(IREE_SV("0"), &value));
  EXPECT_EQ(value, 0u);
}

TEST(JsonParseUint64Test, Positive) {
  uint64_t value;
  IREE_ASSERT_OK(iree_json_parse_uint64(IREE_SV("123"), &value));
  EXPECT_EQ(value, 123u);
}

TEST(JsonParseUint64Test, Max) {
  uint64_t value;
  IREE_ASSERT_OK(
      iree_json_parse_uint64(IREE_SV("18446744073709551615"), &value));
  EXPECT_EQ(value, UINT64_MAX);
}

TEST(JsonParseUint64Test, Negative) {
  uint64_t value;
  iree_status_t status = iree_json_parse_uint64(IREE_SV("-1"), &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonParseDoubleTest, Integer) {
  double value;
  IREE_ASSERT_OK(iree_json_parse_double(IREE_SV("123"), &value));
  EXPECT_DOUBLE_EQ(value, 123.0);
}

TEST(JsonParseDoubleTest, Float) {
  double value;
  IREE_ASSERT_OK(iree_json_parse_double(IREE_SV("3.14"), &value));
  EXPECT_NEAR(value, 3.14, 0.001);
}

TEST(JsonParseDoubleTest, Negative) {
  double value;
  IREE_ASSERT_OK(iree_json_parse_double(IREE_SV("-2.5"), &value));
  EXPECT_DOUBLE_EQ(value, -2.5);
}

TEST(JsonParseDoubleTest, Scientific) {
  double value;
  IREE_ASSERT_OK(iree_json_parse_double(IREE_SV("1.5e10"), &value));
  EXPECT_DOUBLE_EQ(value, 1.5e10);
}

TEST(JsonParseDoubleTest, NegativeExponent) {
  double value;
  IREE_ASSERT_OK(iree_json_parse_double(IREE_SV("1e-5"), &value));
  EXPECT_DOUBLE_EQ(value, 1e-5);
}

//===----------------------------------------------------------------------===//
// UTF-8 BOM Tests
//===----------------------------------------------------------------------===//

TEST(JsonBOMTest, ConsumeValueWithBOM) {
  // UTF-8 BOM: 0xEF 0xBB 0xBF followed by JSON.
  const char bom_json[] = "\xEF\xBB\xBF{\"key\": 123}";
  iree_string_view_t str =
      iree_make_string_view(bom_json, sizeof(bom_json) - 1);
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"key\": 123}"));
}

TEST(JsonBOMTest, ConsumeObjectWithBOM) {
  const char bom_json[] = "\xEF\xBB\xBF{\"key\": 123}";
  iree_string_view_t str =
      iree_make_string_view(bom_json, sizeof(bom_json) - 1);
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"key\": 123}"));
}

TEST(JsonBOMTest, ConsumeArrayWithBOM) {
  const char bom_json[] = "\xEF\xBB\xBF[1, 2, 3]";
  iree_string_view_t str =
      iree_make_string_view(bom_json, sizeof(bom_json) - 1);
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[1, 2, 3]"));
}

TEST(JsonBOMTest, EnumerateLinesWithBOM) {
  const char bom_jsonl[] = "\xEF\xBB\xBF{\"a\": 1}\n{\"b\": 2}";
  iree_string_view_t input =
      iree_make_string_view(bom_jsonl, sizeof(bom_jsonl) - 1);
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(
      iree_json_enumerate_lines(input, CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].value, "{\"a\": 1}");
  EXPECT_EQ(entries[1].value, "{\"b\": 2}");
}

TEST(JsonBOMTest, BOMWithWhitespace) {
  // BOM followed by whitespace then JSON.
  const char bom_json[] = "\xEF\xBB\xBF  \n  {\"key\": 1}";
  iree_string_view_t str =
      iree_make_string_view(bom_json, sizeof(bom_json) - 1);
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"key\": 1}"));
}

//===----------------------------------------------------------------------===//
// Recursion Depth Limit Tests
//===----------------------------------------------------------------------===//

TEST(JsonDepthTest, DeeplyNestedArrays) {
  // Build a deeply nested array: [[[[...]]]]
  // Default depth limit is 128, so 130 levels should fail.
  std::string deep_json;
  for (int i = 0; i < 130; ++i) deep_json += "[";
  deep_json += "1";
  for (int i = 0; i < 130; ++i) deep_json += "]";

  iree_string_view_t str =
      iree_make_string_view(deep_json.data(), deep_json.size());
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_value(&str, &value);
  EXPECT_TRUE(iree_status_is_resource_exhausted(status));
  iree_status_ignore(status);
}

TEST(JsonDepthTest, DeeplyNestedObjects) {
  // Build deeply nested objects: {"a":{"a":{"a":...}}}
  std::string deep_json;
  for (int i = 0; i < 130; ++i) deep_json += "{\"a\":";
  deep_json += "1";
  for (int i = 0; i < 130; ++i) deep_json += "}";

  iree_string_view_t str =
      iree_make_string_view(deep_json.data(), deep_json.size());
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_value(&str, &value);
  EXPECT_TRUE(iree_status_is_resource_exhausted(status));
  iree_status_ignore(status);
}

TEST(JsonDepthTest, MixedNestedStructures) {
  // Mix of arrays and objects.
  std::string deep_json;
  for (int i = 0; i < 130; ++i) {
    deep_json += (i % 2 == 0) ? "[{\"a\":" : "{\"b\":[";
  }
  deep_json += "1";
  for (int i = 129; i >= 0; --i) {
    deep_json += (i % 2 == 0) ? "}]" : "]}";
  }

  iree_string_view_t str =
      iree_make_string_view(deep_json.data(), deep_json.size());
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_value(&str, &value);
  EXPECT_TRUE(iree_status_is_resource_exhausted(status));
  iree_status_ignore(status);
}

TEST(JsonDepthTest, AcceptableDepth) {
  // 50 levels of nesting should be fine (well under 128 limit).
  std::string json;
  for (int i = 0; i < 50; ++i) json += "[";
  json += "1";
  for (int i = 0; i < 50; ++i) json += "]";

  iree_string_view_t str = iree_make_string_view(json.data(), json.size());
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
}

//===----------------------------------------------------------------------===//
// Multi-line Block Comment Tests (JSONL)
//===----------------------------------------------------------------------===//

TEST(JsonEnumerateLinesTest, MultiLineBlockComment) {
  // Block comment spanning multiple lines.
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_lines(
      IREE_SV("/* This is a\nmulti-line\ncomment */\n123\n456"),
      CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  // After 3-line comment and newline, values are on lines 4 and 5.
  EXPECT_EQ(entries[0].line_number, 4);
  EXPECT_EQ(entries[0].value, "123");
  EXPECT_EQ(entries[1].line_number, 5);
  EXPECT_EQ(entries[1].value, "456");
}

TEST(JsonEnumerateLinesTest, MultiLineBlockCommentBetweenValues) {
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_lines(
      IREE_SV("123\n/* comment\nspanning\nlines */\n456"), CollectLineEntries,
      &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].line_number, 1);
  EXPECT_EQ(entries[0].value, "123");
  EXPECT_EQ(entries[1].line_number, 5);
  EXPECT_EQ(entries[1].value, "456");
}

TEST(JsonEnumerateLinesTest, MultiLineBlockCommentOnSameLine) {
  // Block comment that starts and ends on same line as value.
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(
      iree_json_enumerate_lines(IREE_SV("/* comment */ 123 /* another */\n456"),
                                CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].value, "123");
  EXPECT_EQ(entries[1].value, "456");
}

TEST(JsonEnumerateLinesTest, UnterminatedMultiLineComment) {
  std::vector<LineEntry> entries;
  iree_status_t status = iree_json_enumerate_lines(
      IREE_SV("/* unterminated\ncomment\n123"), CollectLineEntries, &entries);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonEnumerateLinesTest, MultipleValuesOnSameLine) {
  // JSONL requires one value per line - multiple values should fail.
  std::vector<LineEntry> entries;
  iree_status_t status = iree_json_enumerate_lines(
      IREE_SV("123 456\n789"), CollectLineEntries, &entries);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonEnumerateLinesTest, MultipleValuesWithCommentBetween) {
  // Even with a comment between, multiple values on same line should fail.
  std::vector<LineEntry> entries;
  iree_status_t status = iree_json_enumerate_lines(
      IREE_SV("123 /* comment */ 456"), CollectLineEntries, &entries);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonEnumerateLinesTest, CROnlyLineEndings) {
  // Old Mac-style CR-only line endings should work.
  std::vector<LineEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_lines(IREE_SV("1\r2\r3"),
                                           CollectLineEntries, &entries));
  ASSERT_EQ(entries.size(), 3);
  EXPECT_EQ(entries[0].line_number, 1);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].line_number, 2);
  EXPECT_EQ(entries[1].value, "2");
  EXPECT_EQ(entries[2].line_number, 3);
  EXPECT_EQ(entries[2].value, "3");
}

//===----------------------------------------------------------------------===//
// Empty Containers with Comments Tests
//===----------------------------------------------------------------------===//

TEST(JsonConsumeObjectTest, EmptyWithComment) {
  iree_string_view_t str = IREE_SV("{/* empty */}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{/* empty */}"));
}

TEST(JsonConsumeObjectTest, EmptyWithSingleLineComment) {
  iree_string_view_t str = IREE_SV("{// comment\n}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{// comment\n}"));
}

TEST(JsonConsumeArrayTest, EmptyWithComment) {
  iree_string_view_t str = IREE_SV("[/* empty */]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[/* empty */]"));
}

TEST(JsonConsumeArrayTest, EmptyWithSingleLineComment) {
  iree_string_view_t str = IREE_SV("[// comment\n]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("[// comment\n]"));
}

//===----------------------------------------------------------------------===//
// Comments in Various Positions Tests
//===----------------------------------------------------------------------===//

TEST(JsonConsumeObjectTest, CommentBeforeColon) {
  // Comment between key and colon.
  iree_string_view_t str = IREE_SV("{\"key\"/* comment */: 123}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("{\"key\"/* comment */: 123}"));
}

TEST(JsonConsumeObjectTest, CommentsEverywhere) {
  // Comments in all allowed positions.
  iree_string_view_t str =
      IREE_SV("{ /* 1 */ \"key\" /* 2 */ : /* 3 */ 123 /* 4 */ }");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
}

TEST(JsonConsumeArrayTest, CommentsEverywhere) {
  // Comments in all allowed positions.
  iree_string_view_t str = IREE_SV("[ /* 1 */ 1 /* 2 */ , /* 3 */ 2 /* 4 */ ]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
}

TEST(JsonConsumeObjectTest, NestedWithComments) {
  // Comments inside nested structures.
  iree_string_view_t str =
      IREE_SV("{\"outer\": {/* inner comment */\"inner\": [1,/* mid */2]}}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
}

TEST(JsonEnumerateObjectTest, CommentsInEntries) {
  std::vector<ObjectEntry> entries;
  IREE_ASSERT_OK(
      iree_json_enumerate_object(IREE_SV("{\"a\"/* c1 */: 1,/* c2 */\"b\": 2}"),
                                 CollectObjectEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].key, "a");
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].key, "b");
  EXPECT_EQ(entries[1].value, "2");
}

TEST(JsonEnumerateArrayTest, CommentsInElements) {
  std::vector<ArrayEntry> entries;
  IREE_ASSERT_OK(iree_json_enumerate_array(
      IREE_SV("[/* c1 */1,/* c2 */2/* c3 */]"), CollectArrayEntries, &entries));
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].value, "1");
  EXPECT_EQ(entries[1].value, "2");
}

//===----------------------------------------------------------------------===//
// Multiple Trailing Commas Tests
//===----------------------------------------------------------------------===//

TEST(JsonConsumeObjectTest, MultipleTrailingCommas) {
  // Multiple trailing commas - after first trailing comma we hit }, so ok.
  iree_string_view_t str = IREE_SV("{\"key\": 123,}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));
}

TEST(JsonConsumeArrayTest, MultipleTrailingCommas) {
  // Multiple trailing commas - similar behavior.
  iree_string_view_t str = IREE_SV("[1, 2,]");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_array(&str, &value));
}

//===----------------------------------------------------------------------===//
// Edge Cases for Whitespace and Comments
//===----------------------------------------------------------------------===//

TEST(JsonConsumeValueTest, MultipleCommentsBeforeValue) {
  iree_string_view_t str = IREE_SV("// c1\n/* c2 */\n// c3\n123");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_value(&str, &value));
  EXPECT_SV_EQ(value, IREE_SV("123"));
}

TEST(JsonConsumeValueTest, CommentOnly) {
  // Input with only comments should fail (no value).
  iree_string_view_t str = IREE_SV("// just a comment\n/* another */");
  iree_string_view_t value;
  iree_status_t status = iree_json_consume_value(&str, &value);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

TEST(JsonConsumeObjectTest, CommentInsideString) {
  // Comment syntax inside a string should be treated as literal characters.
  iree_string_view_t str = IREE_SV("{\"key\": \"/* not a comment */\"}");
  iree_string_view_t value;
  IREE_ASSERT_OK(iree_json_consume_object(&str, &value));

  // Verify the string value is preserved.
  iree_string_view_t str_value;
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(value, IREE_SV("key"), &str_value));
  EXPECT_SV_EQ(str_value, IREE_SV("/* not a comment */"));
}

}  // namespace
