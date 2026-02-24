// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/regex_replace.h"

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

class RegexReplaceNormalizerTest : public ::testing::Test {
 protected:
  ScopedNormalizer CreateRegexReplace(iree_string_view_t pattern,
                                      iree_string_view_t content) {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_EXPECT_OK(iree_tokenizer_normalizer_regex_replace_allocate(
        pattern, content, iree_allocator_system(), &raw_normalizer));
    return ScopedNormalizer(raw_normalizer);
  }
};

// Lifecycle tests.

TEST_F(RegexReplaceNormalizerTest, CreateAndDestroy) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(RegexReplaceNormalizerTest, EmptyPatternFails) {
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_regex_replace_allocate(
      iree_string_view_empty(), IREE_SV(" "), iree_allocator_system(),
      &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw_normalizer, nullptr);
}

TEST_F(RegexReplaceNormalizerTest, EmptyContentAllowed) {
  auto normalizer =
      CreateRegexReplace(IREE_SV(" {2,}"), iree_string_view_empty());
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(RegexReplaceNormalizerTest, InvalidRegexFails) {
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_regex_replace_allocate(
      IREE_SV("[unclosed"), IREE_SV(" "), iree_allocator_system(),
      &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw_normalizer, nullptr);
}

TEST_F(RegexReplaceNormalizerTest, ContentTooLong) {
  std::string long_content(33, 'Y');
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_regex_replace_allocate(
      IREE_SV(" +"),
      iree_make_string_view(long_content.data(), long_content.size()),
      iree_allocator_system(), &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(raw_normalizer, nullptr);
}

// No-op tests.

TEST_F(RegexReplaceNormalizerTest, EmptyInputProducesEmptyOutput) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  std::string result = ProcessAndFinalize(
      normalizer.get(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(RegexReplaceNormalizerTest, NoMatchPassesThrough) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "hello", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, SingleSpaceNoMatch) {
  // Pattern " {2,}" requires 2+ spaces.
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "hello world", "hello world",
                        /*expect_pending_after_process=*/false);
}

// DeBERTa-v3 pattern: " {2,}" -> " " (collapse multiple spaces).
// Validated against HuggingFace tokenizers.

TEST_F(RegexReplaceNormalizerTest, DeBERTa_TwoSpaces) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a  b", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, DeBERTa_ThreeSpaces) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a   b", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, DeBERTa_ManySpaces) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a      b", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, DeBERTa_MultipleRuns) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a  b   c    d", "a b c d",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, DeBERTa_LeadingSpaces) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "  hello", " hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, DeBERTa_TrailingSpaces) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "hello  ", "hello ",
                        /*expect_pending_after_process=*/true);
}

TEST_F(RegexReplaceNormalizerTest, DeBERTa_OnlySpaces) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "    ", " ",
                        /*expect_pending_after_process=*/true);
}

// CLIP pattern: "\s+" -> " " (normalize all whitespace to single space).
// Validated against HuggingFace tokenizers.

TEST_F(RegexReplaceNormalizerTest, CLIP_SingleSpace) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a b", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, CLIP_MultipleSpaces) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a   b", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, CLIP_Tab) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a\tb", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, CLIP_Newline) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a\nb", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, CLIP_MixedWhitespace) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a \n\t b", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, CLIP_CRLF) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "a\r\nb", "a b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, CLIP_LeadingWhitespace) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), " \t hello", " hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, CLIP_TrailingWhitespace) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "hello \n ", "hello ",
                        /*expect_pending_after_process=*/true);
}

// Deletion tests (empty replacement).

TEST_F(RegexReplaceNormalizerTest, Deletion_Digits) {
  auto normalizer =
      CreateRegexReplace(IREE_SV("\\d+"), iree_string_view_empty());
  TestWithAllChunkSizes(normalizer.get(), "a123b456c", "abc",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, Deletion_Whitespace) {
  auto normalizer =
      CreateRegexReplace(IREE_SV("\\s+"), iree_string_view_empty());
  TestWithAllChunkSizes(normalizer.get(), "a b\tc\nd", "abcd",
                        /*expect_pending_after_process=*/false);
}

// Streaming tests.

TEST_F(RegexReplaceNormalizerTest, StreamingDeBERTa) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestLimitedOutputCapacity(normalizer.get(), "a  b  c", "a b c");
}

TEST_F(RegexReplaceNormalizerTest, StreamingCLIP) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestLimitedOutputCapacity(normalizer.get(), "a \t\n b", "a b");
}

// Edge cases.

TEST_F(RegexReplaceNormalizerTest, MatchAtStart) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "   hello", " hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, MatchAtEnd) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "hello   ", "hello ",
                        /*expect_pending_after_process=*/true);
}

TEST_F(RegexReplaceNormalizerTest, ConsecutiveMatches) {
  // Multiple adjacent matches - each gets replaced.
  auto normalizer = CreateRegexReplace(IREE_SV("[0-9]+"), IREE_SV("X"));
  TestWithAllChunkSizes(normalizer.get(), "a1b23c456d", "aXbXcXd",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, OverlappingPotential) {
  // Test leftmost-longest matching.
  // Pattern "a+" should match the longest run of 'a's.
  auto normalizer = CreateRegexReplace(IREE_SV("a+"), IREE_SV("X"));
  TestWithAllChunkSizes(normalizer.get(), "baaab", "bXb",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, Unicode_NonBreakingSpace) {
  // U+00A0 Non-Breaking Space is matched by \s.
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(),
                        "a\xC2\xA0"
                        "b",
                        "a b", /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, Unicode_Passthrough) {
  // Unicode text without matches passes through unchanged.
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(),
                        "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",  // 日本語
                        "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, Unicode_WithSpaces) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  // "日本  語" -> "日本 語"
  TestWithAllChunkSizes(normalizer.get(),
                        "\xE6\x97\xA5\xE6\x9C\xAC  \xE8\xAA\x9E",
                        "\xE6\x97\xA5\xE6\x9C\xAC \xE8\xAA\x9E",
                        /*expect_pending_after_process=*/false);
}

// Complex patterns.

TEST_F(RegexReplaceNormalizerTest, WordCharacter) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\w+"), IREE_SV("X"));
  TestWithAllChunkSizes(normalizer.get(), "hello world", "X X",
                        /*expect_pending_after_process=*/true);
}

TEST_F(RegexReplaceNormalizerTest, NonWordCharacter) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\W+"), IREE_SV("_"));
  TestWithAllChunkSizes(normalizer.get(), "hello, world!", "hello_world_",
                        /*expect_pending_after_process=*/true);
}

TEST_F(RegexReplaceNormalizerTest, AlternationPattern) {
  auto normalizer = CreateRegexReplace(IREE_SV("cat|dog"), IREE_SV("pet"));
  TestWithAllChunkSizes(normalizer.get(), "my cat and dog", "my pet and pet",
                        /*expect_pending_after_process=*/true);
}

TEST_F(RegexReplaceNormalizerTest, CharacterClass) {
  auto normalizer = CreateRegexReplace(IREE_SV("[aeiou]+"), IREE_SV("_"));
  TestWithAllChunkSizes(normalizer.get(), "hello", "h_ll_",
                        /*expect_pending_after_process=*/true);
}

// Longer content replacement.

TEST_F(RegexReplaceNormalizerTest, LongerReplacement) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s"), IREE_SV("___"));
  TestWithAllChunkSizes(normalizer.get(), "a b", "a___b",
                        /*expect_pending_after_process=*/false);
}

TEST_F(RegexReplaceNormalizerTest, LongerReplacementStreaming) {
  auto normalizer = CreateRegexReplace(IREE_SV("\\s"), IREE_SV("___"));
  TestLimitedOutputCapacity(normalizer.get(), "a b c", "a___b___c");
}

TEST_F(RegexReplaceNormalizerTest, TrailingPassthrough_SingleSpace) {
  // Pattern " {2,}" with input " " (single space): DFA enters partial match
  // (could be start of 2+ spaces), then finalize resolves as non-match.
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  std::string result =
      ProcessAndFinalize(normalizer.get(), " ",
                         /*expect_pending_after_process=*/true);
  EXPECT_EQ(result, " ");
}

TEST_F(RegexReplaceNormalizerTest, TrailingPassthrough_SingleSpace_Chunked) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), " ", " ",
                        /*expect_pending_after_process=*/true);
}

TEST_F(RegexReplaceNormalizerTest,
       TrailingPassthrough_SingleSpace_LimitedOutput) {
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestLimitedOutputCapacity(normalizer.get(), " ", " ");
}

TEST_F(RegexReplaceNormalizerTest, TrailingPassthrough_TrailingSpaceAfterText) {
  // "hello " — the trailing space enters partial match during process().
  // Finalize resolves as non-match and emits the space.
  // has_pending=true because the DFA holds the trailing space in partial match
  // (output_position=5 < bytes_processed=6).
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "hello ", "hello ",
                        /*expect_pending_after_process=*/true);
}

TEST_F(RegexReplaceNormalizerTest, TrailingPassthrough_DoubleSpaceIsMatch) {
  // "  " (double space) — DFA matches " {2,}" and emits replacement " ".
  // has_pending=true because the DFA holds both spaces in partial match during
  // process() (could continue matching more spaces), resolved at finalize.
  auto normalizer = CreateRegexReplace(IREE_SV(" {2,}"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), "  ", " ",
                        /*expect_pending_after_process=*/true);
}

TEST_F(RegexReplaceNormalizerTest, TrailingPassthrough_SingleTab) {
  // Pattern "\s+" with input "\t" — single whitespace enters partial match.
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  std::string result =
      ProcessAndFinalize(normalizer.get(), "\t",
                         /*expect_pending_after_process=*/true);
  EXPECT_EQ(result, " ");
}

TEST_F(RegexReplaceNormalizerTest, TrailingPassthrough_WhitespaceOnly_Match) {
  // Pattern "\s+" with input " \t\n" — all whitespace forms one match.
  // has_pending=true because the DFA holds all bytes in partial match during
  // process() (could continue matching more whitespace), resolved at finalize.
  auto normalizer = CreateRegexReplace(IREE_SV("\\s+"), IREE_SV(" "));
  TestWithAllChunkSizes(normalizer.get(), " \t\n", " ",
                        /*expect_pending_after_process=*/true);
}

TEST_F(RegexReplaceNormalizerTest, TrailingPassthrough_WordCharsAtEnd) {
  // Pattern "\w+" with input "abc" — entire input is partial match, resolves
  // at finalize to a full match → replaced with "X".
  auto normalizer = CreateRegexReplace(IREE_SV("\\w+"), IREE_SV("X"));
  std::string result =
      ProcessAndFinalize(normalizer.get(), "abc",
                         /*expect_pending_after_process=*/true);
  EXPECT_EQ(result, "X");
}

}  // namespace
}  // namespace iree::tokenizer
