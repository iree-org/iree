// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/transforms/transform.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

// Callback that accumulates segments into a vector.
struct EncodeContext {
  std::vector<std::string>* segments;
};

static iree_status_t AccumulateSegments(void* user_data,
                                        iree_string_view_list_t segments) {
  auto* context = static_cast<EncodeContext*>(user_data);
  for (size_t i = 0; i < segments.count; ++i) {
    context->segments->push_back(
        std::string(segments.values[i].data, segments.values[i].size));
  }
  return iree_ok_status();
}

class MetaspaceTransformTest : public ::testing::Test {
 protected:
  std::vector<std::string> Encode(
      const char* text, uint32_t replacement = 0x2581,
      iree_tokenizer_prepend_scheme_t prepend_scheme =
          IREE_TOKENIZER_PREPEND_ALWAYS,
      iree_tokenizer_metaspace_flags_t flags =
          IREE_TOKENIZER_METASPACE_FLAG_SPLIT) {
    iree_tokenizer_text_transform_t transform;
    iree_tokenizer_text_transform_initialize_metaspace(
        replacement, prepend_scheme, flags, &transform);

    std::vector<std::string> segments;
    EncodeContext context = {&segments};
    iree_status_t status = iree_tokenizer_text_transform_encode(
        NULL, &transform, IREE_SV(text), AccumulateSegments, &context);
    IREE_EXPECT_OK(status);

    iree_tokenizer_text_transform_deinitialize(&transform);
    return segments;
  }

  std::string Decode(const char* text, uint32_t replacement = 0x2581,
                     iree_tokenizer_prepend_scheme_t prepend_scheme =
                         IREE_TOKENIZER_PREPEND_ALWAYS) {
    iree_tokenizer_text_transform_t transform;
    iree_tokenizer_text_transform_initialize_metaspace(
        replacement, prepend_scheme, IREE_TOKENIZER_METASPACE_FLAG_SPLIT,
        &transform);

    char decoded[1024];
    iree_host_size_t decoded_size = 0;
    iree_status_t status = iree_tokenizer_text_transform_decode(
        &transform, IREE_SV(text), decoded, sizeof(decoded), &decoded_size);
    IREE_EXPECT_OK(status);

    iree_tokenizer_text_transform_deinitialize(&transform);
    return std::string(decoded, decoded_size);
  }
};

TEST_F(MetaspaceTransformTest, EmptyInput) {
  auto segments = Encode("");
  EXPECT_TRUE(segments.empty());
}

TEST_F(MetaspaceTransformTest, SingleWordPrependAlways) {
  auto segments = Encode("hello", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello");  // ▁hello
}

TEST_F(MetaspaceTransformTest, TwoWordsPrependAlways) {
  auto segments = Encode("hello world", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello");
  EXPECT_EQ(segments[1], "\xE2\x96\x81world");
}

TEST_F(MetaspaceTransformTest, PrependFirst) {
  auto segments = Encode("hello world", 0x2581, IREE_TOKENIZER_PREPEND_FIRST,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello");
  EXPECT_EQ(segments[1], "\xE2\x96\x81world");
}

TEST_F(MetaspaceTransformTest, PrependFirstWithLeadingSpace) {
  auto segments = Encode(" hello world", 0x2581, IREE_TOKENIZER_PREPEND_FIRST,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello");
  EXPECT_EQ(segments[1], "\xE2\x96\x81world");
}

TEST_F(MetaspaceTransformTest, PrependNever) {
  auto segments = Encode("hello world", 0x2581, IREE_TOKENIZER_PREPEND_NEVER,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "\xE2\x96\x81world");
}

TEST_F(MetaspaceTransformTest, NoSplit) {
  auto segments = Encode("hello world", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_DEFAULT);
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello\xE2\x96\x81world");  // ▁hello▁world
}

TEST_F(MetaspaceTransformTest, MultipleSpaces) {
  // HuggingFace: "hello   world" (3 spaces) -> 4 segments.
  // Consecutive spaces produce standalone ▁ segments.
  auto segments = Encode("hello   world", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  ASSERT_EQ(segments.size(), 4);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello");
  EXPECT_EQ(segments[1], "\xE2\x96\x81");  // First extra space.
  EXPECT_EQ(segments[2], "\xE2\x96\x81");  // Second extra space.
  EXPECT_EQ(segments[3], "\xE2\x96\x81world");
}

TEST_F(MetaspaceTransformTest, CustomReplacement) {
  auto segments = Encode("hello world", '_', IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "_hello");
  EXPECT_EQ(segments[1], "_world");
}

// Metaspace only replaces ASCII space (0x20) with the replacement character.
// Tabs, newlines, and other whitespace are preserved as-is.
// This matches HuggingFace tokenizers behavior.
TEST_F(MetaspaceTransformTest, NewlinesPreserved) {
  // Input with newline - newline should be preserved, not replaced.
  auto segments = Encode("hello\nworld", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  // No space means no split - entire string is one segment with newline intact.
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello\nworld");
}

TEST_F(MetaspaceTransformTest, TabsPreserved) {
  // Tabs should be preserved, not replaced with metaspace.
  auto segments = Encode("hello\tworld", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello\tworld");
}

TEST_F(MetaspaceTransformTest, MixedWhitespace) {
  // Only spaces split/replace; tabs and newlines are preserved.
  auto segments =
      Encode("hello\tworld foo\nbar", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
             IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  // Split only on the space between "foo" and the preceding text.
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello\tworld");  // Tab preserved.
  EXPECT_EQ(segments[1],
            "\xE2\x96\x81"
            "foo\nbar");  // Newline preserved.
}

TEST_F(MetaspaceTransformTest, CodeWithNewlines) {
  // Regression test for Mistral tokenization bug.
  // Newlines in code should be preserved, only spaces replaced.
  auto segments =
      Encode("def hello():\n    return 42", 0x2581,
             IREE_TOKENIZER_PREPEND_FIRST, IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  // Splits on spaces only. The newline stays with "hello():".
  // Input: "def hello():\n    return 42"
  //        "def " -> ▁def
  //        " hello():\n" -> ▁hello():\n
  //        "    " (4 spaces) -> ▁ | ▁ | ▁ | (4th space joins next word)
  //        " return" -> ▁return
  //        " 42" -> ▁42
  ASSERT_EQ(segments.size(), 7);
  EXPECT_EQ(segments[0],
            "\xE2\x96\x81"
            "def");
  EXPECT_EQ(segments[1], "\xE2\x96\x81hello():\n");  // Newline in segment!
  // 3 empty segments for consecutive spaces, 4th joins "return".
  EXPECT_EQ(segments[2], "\xE2\x96\x81");
  EXPECT_EQ(segments[3], "\xE2\x96\x81");
  EXPECT_EQ(segments[4], "\xE2\x96\x81");
  EXPECT_EQ(segments[5], "\xE2\x96\x81return");
  EXPECT_EQ(segments[6],
            "\xE2\x96\x81"
            "42");
}

TEST_F(MetaspaceTransformTest, NewlinesPreservedNoSplit) {
  // No-split mode: newlines should still be preserved.
  auto segments = Encode("hello\nworld", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_DEFAULT);
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello\nworld");
}

TEST_F(MetaspaceTransformTest, OnlyWhitespace) {
  auto segments = Encode("   ", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  SUCCEED();
}

TEST_F(MetaspaceTransformTest, UnicodeText) {
  auto segments = Encode("hello 你好", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_SPLIT);
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello");
  EXPECT_EQ(segments[1], "\xE2\x96\x81\xE4\xBD\xA0\xE5\xA5\xBD");  // ▁你好
}

//===----------------------------------------------------------------------===//
// Metaspace Decode Tests
//===----------------------------------------------------------------------===//

TEST_F(MetaspaceTransformTest, DecodeBasic) {
  // "▁hello▁world" -> "hello world"
  auto decoded = Decode("\xE2\x96\x81hello\xE2\x96\x81world", 0x2581,
                        IREE_TOKENIZER_PREPEND_ALWAYS);
  EXPECT_EQ(decoded, "hello world");
}

TEST_F(MetaspaceTransformTest, DecodeEmpty) {
  auto decoded = Decode("", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS);
  EXPECT_EQ(decoded, "");
}

TEST_F(MetaspaceTransformTest, DecodeNoReplacement) {
  auto decoded = Decode("hello", 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS);
  EXPECT_EQ(decoded, "hello");
}

TEST_F(MetaspaceTransformTest, DecodePrependNever) {
  // With PREPEND_NEVER, the leading ▁ is NOT stripped.
  auto decoded = Decode("\xE2\x96\x81hello\xE2\x96\x81world", 0x2581,
                        IREE_TOKENIZER_PREPEND_NEVER);
  EXPECT_EQ(decoded, " hello world");
}

TEST_F(MetaspaceTransformTest, RoundTripBasic) {
  const char* original = "hello world";
  auto segments = Encode(original, 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_DEFAULT);
  ASSERT_EQ(segments.size(), 1);
  auto decoded =
      Decode(segments[0].c_str(), 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS);
  EXPECT_EQ(decoded, original);
}

TEST_F(MetaspaceTransformTest, RoundTripUnicode) {
  const char* original = "hello 你好 world";
  auto segments = Encode(original, 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
                         IREE_TOKENIZER_METASPACE_FLAG_DEFAULT);
  ASSERT_EQ(segments.size(), 1);
  auto decoded =
      Decode(segments[0].c_str(), 0x2581, IREE_TOKENIZER_PREPEND_ALWAYS);
  EXPECT_EQ(decoded, original);
}

//===----------------------------------------------------------------------===//
// Edge Case and Stress Tests
//===----------------------------------------------------------------------===//

// Regression test for underflow bug when flushing after long whitespace runs.
// This test creates enough whitespace to trigger a buffer flush, then follows
// with a word to ensure pending_replacement is handled correctly.
TEST_F(MetaspaceTransformTest, LongWhitespaceRunThenWord) {
  // Create a string with many spaces followed by a word.
  // Each space becomes ▁ (3 bytes), so ~1400 spaces will exceed 4KB buffer.
  std::string input(1500, ' ');
  input += "hello";

  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &transform);

  std::vector<std::string> segments;
  EncodeContext context = {&segments};
  iree_status_t status = iree_tokenizer_text_transform_encode(
      NULL, &transform, iree_make_string_view(input.data(), input.size()),
      AccumulateSegments, &context);
  IREE_EXPECT_OK(status);

  // Should have at least one segment containing "hello" with leading ▁.
  ASSERT_GE(segments.size(), 1u);
  // The last segment should be the word.
  EXPECT_EQ(segments.back(), "\xE2\x96\x81hello");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(MetaspaceTransformTest, DecodeBufferTooSmall) {
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &transform);

  // "▁hello▁world" decodes to "hello world" (11 chars).
  const char* encoded = "\xE2\x96\x81hello\xE2\x96\x81world";
  char decoded[5];  // Too small.
  iree_host_size_t decoded_size = 0;
  iree_status_t status = iree_tokenizer_text_transform_decode(
      &transform, IREE_SV(encoded), decoded, sizeof(decoded), &decoded_size);

  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));
  iree_tokenizer_text_transform_deinitialize(&transform);
}

// Callback that returns an error to test error propagation.
static iree_status_t FailingCallback(void* user_data,
                                     iree_string_view_list_t segments) {
  (void)user_data;
  (void)segments;
  return iree_make_status(IREE_STATUS_CANCELLED, "intentional failure");
}

TEST_F(MetaspaceTransformTest, CallbackErrorPropagation) {
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &transform);

  iree_status_t status = iree_tokenizer_text_transform_encode(
      NULL, &transform, IREE_SVL("hello world"), FailingCallback, nullptr);

  EXPECT_THAT(Status(std::move(status)), StatusIs(StatusCode::kCancelled));
  iree_tokenizer_text_transform_deinitialize(&transform);
}

}  // namespace
