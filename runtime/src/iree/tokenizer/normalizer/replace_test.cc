// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/replace.h"

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer/normalizer_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedNormalizer;
using testing::ScopedNormalizerState;
using testing::TestLimitedOutputCapacity;
using testing::TestWithAllChunkSizes;

//===----------------------------------------------------------------------===//
// Test fixture for replace normalizer tests.
//===----------------------------------------------------------------------===//

class ReplaceNormalizerTest : public ::testing::Test {
 protected:
  // Creates a replace normalizer with the given pattern and content.
  ScopedNormalizer CreateReplace(iree_string_view_t pattern,
                                 iree_string_view_t content) {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_EXPECT_OK(iree_tokenizer_normalizer_replace_allocate(
        pattern, content, iree_allocator_system(), &raw_normalizer));
    return ScopedNormalizer(raw_normalizer);
  }

  // Convenience overload for single-byte pattern.
  ScopedNormalizer CreateSingleByteReplace(uint8_t target_byte,
                                           iree_string_view_t content) {
    char pattern_char = static_cast<char>(target_byte);
    return CreateReplace(iree_make_string_view(&pattern_char, 1), content);
  }
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, CreateAndDestroy) {
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(ReplaceNormalizerTest, EmptyPatternFails) {
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_replace_allocate(
      iree_string_view_empty(), IREE_SV("X"), iree_allocator_system(),
      &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw_normalizer, nullptr);
}

TEST_F(ReplaceNormalizerTest, EmptyContentAllowed) {
  // Empty content = deletion.
  auto normalizer = CreateReplace(IREE_SV("X"), iree_string_view_empty());
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(ReplaceNormalizerTest, PatternTooLong) {
  std::string long_pattern(33,
                           'X');  // 33 > IREE_TOKENIZER_REPLACE_MAX_PATTERN
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_replace_allocate(
      iree_make_string_view(long_pattern.data(), long_pattern.size()),
      IREE_SV("Y"), iree_allocator_system(), &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw_normalizer, nullptr);
}

TEST_F(ReplaceNormalizerTest, ContentTooLong) {
  std::string long_content(33,
                           'Y');  // 33 > IREE_TOKENIZER_REPLACE_MAX_CONTENT
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_replace_allocate(
      IREE_SV("X"),
      iree_make_string_view(long_content.data(), long_content.size()),
      iree_allocator_system(), &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw_normalizer, nullptr);
}

TEST_F(ReplaceNormalizerTest, StateSizeIsReasonable) {
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer.get());
  EXPECT_GT(state_size, 0u);
  EXPECT_LE(state_size, 128u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, EmptyInputProducesEmptyOutput) {
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  std::string result = ProcessAndFinalize(
      normalizer.get(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(ReplaceNormalizerTest, NoMatchPassesThrough) {
  // Input has no spaces, so nothing is replaced.
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "hello", "hello",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Metaspace Replacement (Space → ▁)
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, SingleSpaceReplaced) {
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), " ", "\xE2\x96\x81",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, SpaceInMiddle) {
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "hello world",
                        "hello\xE2\x96\x81world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, MultipleSpaces) {
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "a b c",
                        "a\xE2\x96\x81"
                        "b\xE2\x96\x81"
                        "c",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, ConsecutiveSpaces) {
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "a  b",
                        "a\xE2\x96\x81\xE2\x96\x81"
                        "b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, LeadingAndTrailingSpaces) {
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), " hello ",
                        "\xE2\x96\x81hello\xE2\x96\x81",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, AllSpaces) {
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "   ",
                        "\xE2\x96\x81\xE2\x96\x81\xE2\x96\x81",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Non-space Target Bytes
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, ReplaceNewline) {
  // Replace newlines with space.
  auto normalizer = CreateSingleByteReplace('\n', IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a\nb\nc", "a b c",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, ReplaceNullByte) {
  // Replace null bytes with a marker.
  auto normalizer = CreateSingleByteReplace(0x00, IREE_SV("NULL"));
  std::string input = std::string(
      "a\x00"
      "b",
      3);
  TestWithAllChunkSizes(normalizer.get(), input, "aNULLb",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, ReplaceSingleByteWithSingleByte) {
  // 1:1 replacement (no expansion).
  auto normalizer = CreateReplace(IREE_SV("X"), IREE_SV("Y"));
  TestWithAllChunkSizes(normalizer.get(), "aXbXc", "aYbYc",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, SingleByteDeletion) {
  // Delete all 'X' characters.
  auto normalizer = CreateReplace(IREE_SV("X"), iree_string_view_empty());
  TestWithAllChunkSizes(normalizer.get(), "aXbXc", "abc",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Partial Writes
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, LimitedOutputCapacity) {
  // The 1:3 expansion (space → ▁) exercises partial replacement writes.
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  TestLimitedOutputCapacity(normalizer.get(), "a b c",
                            "a\xE2\x96\x81"
                            "b\xE2\x96\x81"
                            "c");
}

TEST_F(ReplaceNormalizerTest, PartialReplacementResume) {
  // Output buffer of 1 byte with a 3-byte replacement: must resume correctly.
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  ScopedNormalizerState state(normalizer.get());

  std::string result;
  const char* input = " x";
  size_t input_position = 0;
  char output_byte;

  // Process byte-by-byte output.
  while (input_position < 2 ||
         iree_tokenizer_normalizer_state_has_pending(state.get())) {
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    iree_string_view_t remaining = {input + input_position, 2 - input_position};
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(), remaining, iree_make_mutable_string_view(&output_byte, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
    if (written > 0) {
      result += output_byte;
    }
    ASSERT_TRUE(consumed > 0 || written > 0) << "No progress";
    input_position += consumed;
  }

  // Finalize.
  iree_host_size_t finalize_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(&output_byte, 1),
      &finalize_written));
  EXPECT_EQ(finalize_written, 0u);

  // Should produce ▁x (0xE2 0x96 0x81 followed by 'x').
  EXPECT_EQ(result, "\xE2\x96\x81x");
}

//===----------------------------------------------------------------------===//
// Unicode Input Passthrough
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, Utf8InputPassthrough) {
  // Non-target bytes (including multi-byte UTF-8) pass through unchanged.
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  // café → café (no spaces, nothing replaced)
  TestWithAllChunkSizes(normalizer.get(), "caf\xC3\xA9", "caf\xC3\xA9",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, Utf8WithSpaces) {
  // Mixed UTF-8 and spaces.
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  // "日本 語" → "日本▁語"
  TestWithAllChunkSizes(normalizer.get(),
                        "\xE6\x97\xA5\xE6\x9C\xAC \xE8\xAA\x9E",
                        "\xE6\x97\xA5\xE6\x9C\xAC\xE2\x96\x81\xE8\xAA\x9E",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// State Reset
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, StateResetClearsPartialReplacement) {
  // If we reset mid-replacement, the partial state should be cleared.
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("\xE2\x96\x81"));
  ScopedNormalizerState state(normalizer.get());

  // Process a space with output capacity of 1 (starts partial replacement).
  char output_byte;
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV(" "), iree_make_mutable_string_view(&output_byte, 1),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 1u);
  EXPECT_EQ(written, 1u);
  EXPECT_TRUE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  // Reset and process "hello" — no leftover replacement bytes.
  state.Reset();
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  char output[64];
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("hello"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(std::string(output, written), "hello");
}

//===----------------------------------------------------------------------===//
// Long Replacement String
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, LongReplacement) {
  // 10-byte replacement for each space.
  auto normalizer = CreateSingleByteReplace(0x20, IREE_SV("XXXXXXXXXX"));
  TestWithAllChunkSizes(normalizer.get(), "a b", "aXXXXXXXXXXb",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Basic Pattern Matching
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, TwoBytePattern) {
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  TestWithAllChunkSizes(normalizer.get(), "aXYb", "aZb",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, ThreeBytePattern) {
  auto normalizer = CreateReplace(IREE_SV("XYZ"), IREE_SV("!"));
  TestWithAllChunkSizes(normalizer.get(), "aXYZb", "a!b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, PatternAtStart) {
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  TestWithAllChunkSizes(normalizer.get(), "XYabc", "Zabc",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, PatternAtEnd) {
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  TestWithAllChunkSizes(normalizer.get(), "abcXY", "abcZ",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, ConsecutivePatterns) {
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  TestWithAllChunkSizes(normalizer.get(), "XYXY", "ZZ",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, OverlappingText) {
  // "XYXYZ" contains "XYZ" starting at position 2, not "XY" at position 0.
  // Actually XY matches at 0, leaving XYZ. Then XYZ doesn't match XY.
  auto normalizer = CreateReplace(IREE_SV("XYZ"), IREE_SV("!"));
  TestWithAllChunkSizes(normalizer.get(), "XYXYZ", "XY!",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, NoMatchMultiByte) {
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  TestWithAllChunkSizes(normalizer.get(), "abcdef", "abcdef",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, PartialMatchAtEnd) {
  // Input ends with partial pattern - should output the non-matching prefix.
  auto normalizer = CreateReplace(IREE_SV("XYZ"), IREE_SV("!"));
  TestWithAllChunkSizes(normalizer.get(), "abcXY", "abcXY",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Content Variations
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, MultiByteEmptyContent) {
  // Deletion: "XY" → ""
  auto normalizer = CreateReplace(IREE_SV("XY"), iree_string_view_empty());
  TestWithAllChunkSizes(normalizer.get(), "aXYbXYc", "abc",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, LongerContent) {
  // Expansion: "X" → "YYY"
  auto normalizer = CreateReplace(IREE_SV("X"), IREE_SV("YYY"));
  TestWithAllChunkSizes(normalizer.get(), "aXb", "aYYYb",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, ShorterContent) {
  // Contraction: "XYZ" → "!"
  auto normalizer = CreateReplace(IREE_SV("XYZ"), IREE_SV("!"));
  TestWithAllChunkSizes(normalizer.get(), "aXYZb", "a!b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, SameLengthContent) {
  // Same length: "XY" → "AB"
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("AB"));
  TestWithAllChunkSizes(normalizer.get(), "aXYb", "aABb",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Unicode Patterns
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, UnicodePatternToSpace) {
  // Replace "▁" (U+2581, 3 bytes) with space.
  auto normalizer = CreateReplace(IREE_SV("\xE2\x96\x81"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "hello\xE2\x96\x81world",
                        "hello world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, SpaceToUnicodePattern) {
  // Replace space with "▁" using multi-byte path (pattern.size > 1).
  // Note: This actually uses single-byte path, but let's test multi-byte
  // content with a 2-byte pattern.
  auto normalizer = CreateReplace(IREE_SV("  "), IREE_SV("\xE2\x96\x81"));
  TestWithAllChunkSizes(normalizer.get(), "a  b",
                        "a\xE2\x96\x81"
                        "b",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// UTF-8 Character Boundary Preservation
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, TwoByteUtf8PreservedInOutput) {
  // Pattern "``" (2 ASCII bytes). For 2-byte UTF-8 input like é (C3 A9),
  // Phase 2 scans up to safe_end = in_end - 1, emitting C3. Without the fix,
  // Phase 3 would buffer A9, splitting the character.
  auto normalizer = CreateReplace(IREE_SV("``"), IREE_SV("\""));
  std::string result =
      ProcessAndFinalize(normalizer.get(), "caf\xC3\xA9",
                         /*expect_pending_after_process=*/false);
  EXPECT_EQ(result, "caf\xC3\xA9");
}

TEST_F(ReplaceNormalizerTest, ThreeByteUtf8PreservedInOutput) {
  // 3-byte UTF-8: ▁ (E2 96 81). With a 2-byte pattern, the last byte would
  // be split into overlap without the UTF-8 boundary fix.
  auto normalizer = CreateReplace(IREE_SV("``"), IREE_SV("\""));
  std::string result =
      ProcessAndFinalize(normalizer.get(),
                         "a\xE2\x96\x81"
                         "b",
                         /*expect_pending_after_process=*/false);
  EXPECT_EQ(result,
            "a\xE2\x96\x81"
            "b");
}

TEST_F(ReplaceNormalizerTest, FourByteUtf8PreservedInOutput) {
  // 4-byte emoji: 🌍 (F0 9F 8C 8D). With a 2-byte pattern, the last byte
  // would be split into overlap without the UTF-8 boundary fix.
  auto normalizer = CreateReplace(IREE_SV("``"), IREE_SV("\""));
  std::string result =
      ProcessAndFinalize(normalizer.get(), "\xF0\x9F\x8C\x8D",
                         /*expect_pending_after_process=*/false);
  EXPECT_EQ(result, "\xF0\x9F\x8C\x8D");
}

TEST_F(ReplaceNormalizerTest, EmojiSurroundedByAsciiPreserved) {
  // Emoji between ASCII text - tests that the UTF-8 boundary fix doesn't
  // affect the surrounding bytes.
  auto normalizer = CreateReplace(IREE_SV("``"), IREE_SV("\""));
  std::string result =
      ProcessAndFinalize(normalizer.get(),
                         "a\xF0\x9F\x8C\x8D"
                         "b",
                         /*expect_pending_after_process=*/false);
  EXPECT_EQ(result,
            "a\xF0\x9F\x8C\x8D"
            "b");
}

TEST_F(ReplaceNormalizerTest, MultipleEmojisPreserved) {
  // Multiple 4-byte emojis with a 2-byte pattern that doesn't match any of
  // them. Each emoji ends with a continuation byte that must not be buffered.
  auto normalizer = CreateReplace(IREE_SV("``"), IREE_SV("\""));
  std::string input = "Hello \xF0\x9F\x91\x8B World \xF0\x9F\x8C\x8D!";
  std::string result = ProcessAndFinalize(
      normalizer.get(), input, /*expect_pending_after_process=*/false);
  EXPECT_EQ(result, input);
}

TEST_F(ReplaceNormalizerTest, PatternMatchWithSurroundingUtf8) {
  // Pattern "``" matches while surrounding multi-byte UTF-8 is preserved.
  auto normalizer = CreateReplace(IREE_SV("``"), IREE_SV("\""));
  std::string result =
      ProcessAndFinalize(normalizer.get(),
                         "\xC3\xA9``\xF0\x9F\x8C\x8D",  // é``🌍
                         /*expect_pending_after_process=*/false);
  EXPECT_EQ(result,
            "\xC3\xA9\"\xF0\x9F\x8C\x8D");  // é"🌍
}

TEST_F(ReplaceNormalizerTest, ThreeBytePatternWithUtf8Passthrough) {
  // 3-byte pattern with surrounding multi-byte UTF-8. The tail for a 3-byte
  // pattern is 2 bytes, which can split a 3- or 4-byte character.
  auto normalizer = CreateReplace(IREE_SV("abc"), IREE_SV("!"));
  std::string result =
      ProcessAndFinalize(normalizer.get(),
                         "\xF0\x9F\x98\x80"
                         "abc\xF0\x9F\x8C\x8D",  // 😀abc🌍
                         /*expect_pending_after_process=*/false);
  EXPECT_EQ(result,
            "\xF0\x9F\x98\x80!\xF0\x9F\x8C\x8D");  // 😀!🌍
}

//===----------------------------------------------------------------------===//
// Cross-chunk Boundary Tests
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, SplitPattern_1_1) {
  // Pattern "XY" split: chunk1="aX", chunk2="Yb"
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  std::string result;

  // Process chunk 1: "aX"
  // The 'X' goes into the overlap buffer waiting for more input to see if
  // it's the start of "XY". has_pending() is false because we don't need
  // more OUTPUT space - we're waiting for more INPUT.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("aX"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  result.append(output, written);
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  // Process chunk 2: "Yb"
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("Yb"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  result.append(output, written);

  // Finalize.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));
  result.append(output, written);

  EXPECT_EQ(result, "aZb");
}

TEST_F(ReplaceNormalizerTest, SplitPattern_1_2) {
  // Pattern "XYZ" split: chunk1="aX", chunk2="YZb"
  auto normalizer = CreateReplace(IREE_SV("XYZ"), IREE_SV("!"));
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  std::string result;

  // Process chunk 1: "aX"
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("aX"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  result.append(output, written);

  // Process chunk 2: "YZb"
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("YZb"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  result.append(output, written);

  // Finalize.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));
  result.append(output, written);

  EXPECT_EQ(result, "a!b");
}

TEST_F(ReplaceNormalizerTest, SplitPattern_2_1) {
  // Pattern "XYZ" split: chunk1="aXY", chunk2="Zb"
  auto normalizer = CreateReplace(IREE_SV("XYZ"), IREE_SV("!"));
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  std::string result;

  // Process chunk 1: "aXY"
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("aXY"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  result.append(output, written);

  // Process chunk 2: "Zb"
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("Zb"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  result.append(output, written);

  // Finalize.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));
  result.append(output, written);

  EXPECT_EQ(result, "a!b");
}

TEST_F(ReplaceNormalizerTest, MultipleSplits) {
  // Pattern "XY" with single-byte chunks: "X", "Y", "X", "Y"
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  std::string result;

  const char* chunks[] = {"X", "Y", "X", "Y"};
  for (const char* chunk : chunks) {
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(), iree_make_cstring_view(chunk),
        iree_make_mutable_string_view(output, sizeof(output)),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
    result.append(output, written);
  }

  // Finalize.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));
  result.append(output, written);

  EXPECT_EQ(result, "ZZ");
}

//===----------------------------------------------------------------------===//
// Limited Output Capacity
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, MultiLimitedOutputCapacity) {
  // Multi-byte pattern with limited output.
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("ZZZ"));
  TestLimitedOutputCapacity(normalizer.get(), "aXYbXYc", "aZZZbZZZc");
}

TEST_F(ReplaceNormalizerTest, MultiOutputCapacity1) {
  // 1-byte output buffer with multi-byte pattern.
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("AB"));
  ScopedNormalizerState state(normalizer.get());

  std::string result;
  const char* input = "XY";
  size_t input_position = 0;
  char output_byte;

  // Process byte-by-byte output.
  while (input_position < 2 ||
         iree_tokenizer_normalizer_state_has_pending(state.get())) {
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    iree_string_view_t remaining = {input + input_position, 2 - input_position};
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(), remaining, iree_make_mutable_string_view(&output_byte, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
    if (written > 0) {
      result += output_byte;
    }
    ASSERT_TRUE(consumed > 0 || written > 0) << "No progress";
    input_position += consumed;
  }

  // Finalize.
  iree_host_size_t finalize_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(&output_byte, 1),
      &finalize_written));
  if (finalize_written > 0) {
    result += output_byte;
  }

  EXPECT_EQ(result, "AB");
}

TEST_F(ReplaceNormalizerTest, Phase1OverlapWithTinyChunksAndOutput) {
  // Exercises Phase 1 overlap accumulation with tiny input chunks and 1-byte
  // output. This stresses the cross-boundary handling when neither input nor
  // output provides much room to work with.
  //
  // Scenario: pattern "ABCD" (4 bytes), input split into single bytes.
  // The overlap buffer can hold up to 3 bytes (pattern_length - 1).
  // Each input byte may trigger overlap buffer adjustments that need to
  // emit bytes before there's room to accumulate more.
  auto normalizer = CreateReplace(IREE_SV("ABCD"), IREE_SV("!"));
  ScopedNormalizerState state(normalizer.get());

  std::string result;
  // Input includes non-matching bytes, partial prefix, and full match.
  const char* input = "xAByzABCDw";
  size_t input_length = strlen(input);
  size_t input_position = 0;
  char output_byte;

  // Process byte-by-byte input AND byte-by-byte output.
  while (input_position < input_length ||
         iree_tokenizer_normalizer_state_has_pending(state.get())) {
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    iree_string_view_t remaining = {input + input_position,
                                    input_length - input_position};
    // Feed only 1 byte of input at a time for maximum stress.
    if (remaining.size > 1) {
      remaining.size = 1;
    }
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(), remaining, iree_make_mutable_string_view(&output_byte, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
    if (written > 0) {
      result += output_byte;
    }
    ASSERT_TRUE(consumed > 0 || written > 0)
        << "No progress at position " << input_position;
    input_position += consumed;
  }

  // Finalize - may need multiple calls with 1-byte output.
  while (true) {
    iree_host_size_t finalize_written = 0;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
        state.get(), iree_make_mutable_string_view(&output_byte, 1),
        &finalize_written));
    if (finalize_written > 0) {
      result += output_byte;
    } else {
      break;
    }
  }

  // "xAByzABCDw" -> "xAByz!w" (only ABCD matches, AB/AByz don't)
  EXPECT_EQ(result, "xAByz!w");
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, MaxPatternLength) {
  // 32-byte pattern (maximum allowed).
  std::string pattern(32, 'X');
  std::string input = "a" + pattern + "b";
  auto normalizer = CreateReplace(
      iree_make_string_view(pattern.data(), pattern.size()), IREE_SV("!"));
  TestWithAllChunkSizes(normalizer.get(), input, "a!b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, MaxContentLength) {
  // 32-byte content (maximum allowed).
  std::string content(32, 'Y');
  std::string expected = "a" + content + "b";
  auto normalizer = CreateReplace(
      IREE_SV("X"), iree_make_string_view(content.data(), content.size()));
  TestWithAllChunkSizes(normalizer.get(), "aXb", expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, SingleByteViaNewAPI) {
  // Verify 1-byte pattern still uses optimized single-byte path.
  auto normalizer = CreateReplace(IREE_SV("X"), IREE_SV("YYY"));
  TestWithAllChunkSizes(normalizer.get(), "aXbXc", "aYYYbYYYc",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, FalseMatchPrefix) {
  // Input has first byte of pattern but not full match.
  auto normalizer = CreateReplace(IREE_SV("XYZ"), IREE_SV("!"));
  TestWithAllChunkSizes(normalizer.get(), "XYaXYbXY", "XYaXYbXY",
                        /*expect_pending_after_process=*/false);
}

TEST_F(ReplaceNormalizerTest, MultiStateReset) {
  // Reset mid-overlap clears state.
  auto normalizer = CreateReplace(IREE_SV("XYZ"), IREE_SV("!"));
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;

  // Process "XY" (partial pattern). The bytes go into the overlap buffer
  // waiting for more input. has_pending() is false because we don't need
  // more OUTPUT space - we're waiting for more INPUT.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("XY"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  // Reset and process new input.
  state.Reset();
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("abc"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  iree_host_size_t finalize_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(),
      iree_make_mutable_string_view(output + written, 64 - written),
      &finalize_written));

  EXPECT_EQ(std::string(output, written + finalize_written), "abc");
}

//===----------------------------------------------------------------------===//
// Performance-oriented Tests
//===----------------------------------------------------------------------===//

TEST_F(ReplaceNormalizerTest, LongInputNoMatch) {
  // Large input with no matches - exercises memchr skip.
  std::string input(1000, 'a');
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  std::string result = ProcessAndFinalize(
      normalizer.get(), input, /*expect_pending_after_process=*/false);
  EXPECT_EQ(result, input);
}

TEST_F(ReplaceNormalizerTest, LongInputAllMatch) {
  // Pattern repeated many times.
  std::string input;
  std::string expected;
  for (int i = 0; i < 100; ++i) {
    input += "XY";
    expected += "Z";
  }
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  std::string result = ProcessAndFinalize(
      normalizer.get(), input, /*expect_pending_after_process=*/false);
  EXPECT_EQ(result, expected);
}

TEST_F(ReplaceNormalizerTest, RareFirstByte) {
  // Pattern[0] is rare in input - memchr should skip efficiently.
  std::string input = std::string(500, 'a') + "XY" + std::string(500, 'b');
  auto normalizer = CreateReplace(IREE_SV("XY"), IREE_SV("Z"));
  std::string expected = std::string(500, 'a') + "Z" + std::string(500, 'b');
  std::string result = ProcessAndFinalize(
      normalizer.get(), input, /*expect_pending_after_process=*/false);
  EXPECT_EQ(result, expected);
}

}  // namespace
}  // namespace iree::tokenizer
