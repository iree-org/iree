// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/transforms/transform.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

// Callback that collects segments into a vector.
struct CollectContext {
  std::vector<std::string>* segments;
};

static iree_status_t CollectSegments(void* user_data,
                                     iree_string_view_list_t segments) {
  auto* context = static_cast<CollectContext*>(user_data);
  for (size_t i = 0; i < segments.count; ++i) {
    context->segments->emplace_back(segments.values[i].data,
                                    segments.values[i].size);
  }
  return iree_ok_status();
}

class SplitTransformTest : public ::testing::Test {
 protected:
  std::vector<std::string> Split(
      const char* pattern, const char* text,
      iree_tokenizer_regex_split_behavior_t behavior =
          IREE_TOKENIZER_REGEX_SPLIT_REMOVED,
      bool invert = false) {
    return SplitWithNormalizer(pattern, text, nullptr, behavior, invert);
  }

  std::vector<std::string> SplitWithNormalizer(
      const char* pattern, const char* text,
      const iree_tokenizer_normalizer_t* normalizer,
      iree_tokenizer_regex_split_behavior_t behavior =
          IREE_TOKENIZER_REGEX_SPLIT_REMOVED,
      bool invert = false) {
    iree_tokenizer_text_transform_t transform;
    iree_status_t status = iree_tokenizer_text_transform_initialize_split(
        IREE_SV(pattern), behavior, invert, iree_allocator_system(),
        &transform);
    IREE_EXPECT_OK(status);

    std::vector<std::string> segments;
    CollectContext context = {&segments};
    status = iree_tokenizer_text_transform_encode(
        normalizer, &transform, IREE_SV(text), CollectSegments, &context);
    IREE_EXPECT_OK(status);

    iree_tokenizer_text_transform_deinitialize(&transform);
    return segments;
  }
};

//===----------------------------------------------------------------------===//
// Basic Split Tests
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, EmptyInput) {
  auto segments = Split("\\s+", "");
  EXPECT_EQ(segments.size(), 0u);
}

TEST_F(SplitTransformTest, NoMatches) {
  // Pattern doesn't match anything, entire text is emitted.
  auto segments = Split("X", "hello");
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "hello");
}

TEST_F(SplitTransformTest, WhitespaceSplitRemoved) {
  // Behavior 0 = REMOVED: discard the delimiter.
  auto segments =
      Split("\\s+", "hello world", IREE_TOKENIZER_REGEX_SPLIT_REMOVED);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
}

TEST_F(SplitTransformTest, WhitespaceSplitIsolated) {
  // Behavior 1 = ISOLATED: emit delimiter as separate segment.
  auto segments =
      Split("\\s+", "hello world", IREE_TOKENIZER_REGEX_SPLIT_ISOLATED);
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], " ");
  EXPECT_EQ(segments[2], "world");
}

TEST_F(SplitTransformTest, MultipleSpaces) {
  auto segments =
      Split("\\s+", "a  b   c", IREE_TOKENIZER_REGEX_SPLIT_ISOLATED);
  ASSERT_EQ(segments.size(), 5u);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], "  ");
  EXPECT_EQ(segments[2], "b");
  EXPECT_EQ(segments[3], "   ");
  EXPECT_EQ(segments[4], "c");
}

//===----------------------------------------------------------------------===//
// Behavior Tests
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, MergedWithPrevious) {
  // Behavior 2 = MERGED_WITH_PREVIOUS: offset-tracking only, NOT BPE
  // boundaries. Emits single segment because BPE needs to see full text for
  // cross-segment merges (e.g., " " + "world" -> " world" in Gemma 3).
  auto segments =
      Split(",", "a,b,c", IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_PREVIOUS);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "a,b,c");
}

TEST_F(SplitTransformTest, MergedWithNext) {
  // Behavior 3 = MERGED_WITH_NEXT: offset-tracking only, NOT BPE boundaries.
  // Emits single segment because BPE needs to see full text for cross-segment
  // merges.
  auto segments =
      Split(",", "a,b,c", IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "a,b,c");
}

TEST_F(SplitTransformTest, Contiguous) {
  // Behavior 4 = CONTIGUOUS: merge consecutive delimiters.
  auto segments = Split(",", "a,,b,,,c", IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS);
  ASSERT_EQ(segments.size(), 5u);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], ",,");
  EXPECT_EQ(segments[2], "b");
  EXPECT_EQ(segments[3], ",,,");
  EXPECT_EQ(segments[4], "c");
}

//===----------------------------------------------------------------------===//
// Invert Mode Tests
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, InvertMode) {
  // Invert = true: emit only matches, discard everything else.
  auto segments =
      Split("\\w+", "hello world", IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
}

TEST_F(SplitTransformTest, InvertModeWithPunctuation) {
  // Matches are words, gaps are punctuation/whitespace.
  auto segments = Split("[a-zA-Z]+", "hello, world!",
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
}

//===----------------------------------------------------------------------===//
// Unicode Tests
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, UnicodeLetters) {
  // Split on Unicode letters using \p{L}.
  auto segments =
      Split("\\s+", "hello 世界 test", IREE_TOKENIZER_REGEX_SPLIT_ISOLATED);
  ASSERT_EQ(segments.size(), 5u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], " ");
  EXPECT_EQ(segments[2], "世界");
  EXPECT_EQ(segments[3], " ");
  EXPECT_EQ(segments[4], "test");
}

TEST_F(SplitTransformTest, ExactUnicodeCodepointRange_CJK) {
  // Test exact codepoint range matching (not category approximation).
  // This would have crashed before the fix that added ranges/num_ranges to
  // split config, because the DFA range transitions weren't being propagated.
  //
  // Pattern [一-龥] matches CJK Unified Ideographs U+4E00-U+9FA5.
  // In invert mode, we emit the CJK matches.
  auto segments = Split("[一-龥]+", "hello世界test漢字ok",
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "世界");  // U+4E16 U+754C - both in range.
  EXPECT_EQ(segments[1], "漢字");  // U+6F22 U+5B57 - both in range.
}

TEST_F(SplitTransformTest, ExactUnicodeCodepointRange_CJK_Isolated) {
  // Test exact range matching with ISOLATED behavior (delimiter is separate).
  // Pattern splits on CJK characters, emitting them separately.
  auto segments = Split("[一-龥]+", "hello世界test",
                        IREE_TOKENIZER_REGEX_SPLIT_ISOLATED);  // ISOLATED.
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "世界");  // CJK delimiter isolated.
  EXPECT_EQ(segments[2], "test");
}

TEST_F(SplitTransformTest, ExactUnicodeCodepointRange_Hiragana) {
  // Test exact range matching for Hiragana U+3040-U+309F.
  // Hiragana "あいう" = U+3042 U+3044 U+3046.
  auto segments = Split("[ぁ-ゟ]+", "testあいうend",
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "あいう");
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, DelimiterAtStart) {
  auto segments = Split(",", ",a,b", IREE_TOKENIZER_REGEX_SPLIT_ISOLATED);
  ASSERT_EQ(segments.size(), 4u);
  // Empty segment before first comma is skipped (empty segments not emitted).
  EXPECT_EQ(segments[0], ",");
  EXPECT_EQ(segments[1], "a");
  EXPECT_EQ(segments[2], ",");
  EXPECT_EQ(segments[3], "b");
}

TEST_F(SplitTransformTest, DelimiterAtEnd) {
  auto segments = Split(",", "a,b,", IREE_TOKENIZER_REGEX_SPLIT_ISOLATED);
  ASSERT_EQ(segments.size(), 4u);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], "b");
  EXPECT_EQ(segments[3], ",");
}

TEST_F(SplitTransformTest, OnlyDelimiters) {
  auto segments = Split(",", ",,", IREE_TOKENIZER_REGEX_SPLIT_ISOLATED);
  // Just the delimiters, no gaps.
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], ",");
  EXPECT_EQ(segments[1], ",");
}

//===----------------------------------------------------------------------===//
// Behavior Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, MergedWithPreviousAtStart) {
  // MERGED_WITH_PREVIOUS: offset-tracking only, emits single segment.
  auto segments =
      Split(",", ",a,b", IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_PREVIOUS);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], ",a,b");
}

TEST_F(SplitTransformTest, MergedWithNextAtEnd) {
  // MERGED_WITH_NEXT: offset-tracking only, emits single segment.
  auto segments =
      Split(",", "a,b,", IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "a,b,");
}

TEST_F(SplitTransformTest, MergedWithNextNoMatches) {
  // MERGED_WITH_NEXT when pattern doesn't match anything.
  auto segments =
      Split("X", "hello", IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "hello");
}

TEST_F(SplitTransformTest, ContiguousWithLeadingDelimiters) {
  // CONTIGUOUS with delimiters at the start.
  auto segments = Split(",", ",,a,b", IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS);
  ASSERT_EQ(segments.size(), 4u);
  EXPECT_EQ(segments[0], ",,");  // Leading contiguous delimiters.
  EXPECT_EQ(segments[1], "a");
  EXPECT_EQ(segments[2], ",");
  EXPECT_EQ(segments[3], "b");
}

TEST_F(SplitTransformTest, ContiguousWithTrailingDelimiters) {
  // CONTIGUOUS with delimiters at the end.
  auto segments = Split(",", "a,b,,", IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS);
  ASSERT_EQ(segments.size(), 4u);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], "b");
  EXPECT_EQ(segments[3], ",,");  // Trailing contiguous delimiters.
}

TEST_F(SplitTransformTest, ContiguousOnlyDelimiters) {
  // CONTIGUOUS when input is all delimiters.
  auto segments = Split(",", ",,,", IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], ",,,");  // All merged into one.
}

TEST_F(SplitTransformTest, InvertModeNoMatches) {
  // Invert mode when pattern doesn't match anything.
  // No matches = no segments emitted.
  auto segments =
      Split("X", "hello world", IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  EXPECT_EQ(segments.size(), 0u);
}

TEST_F(SplitTransformTest, InputIsSingleFullMatch) {
  // Entire input is one match (invert mode).
  auto segments =
      Split("[a-z]+", "hello", IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "hello");
}

TEST_F(SplitTransformTest, InputIsEntirelyDelimiter) {
  // Entire input matches the delimiter pattern (REMOVED behavior).
  auto segments = Split("\\s+", "   ", IREE_TOKENIZER_REGEX_SPLIT_REMOVED);
  // All is delimiter, gaps are empty, nothing emitted.
  EXPECT_EQ(segments.size(), 0u);
}

TEST_F(SplitTransformTest, SplitBySingleCharDelimiter) {
  // Split on a specific single character (pipe).
  auto segments =
      Split("\\|", "hello|world|test", IREE_TOKENIZER_REGEX_SPLIT_ISOLATED);
  ASSERT_EQ(segments.size(), 5u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "|");
  EXPECT_EQ(segments[2], "world");
  EXPECT_EQ(segments[3], "|");
  EXPECT_EQ(segments[4], "test");
}

TEST_F(SplitTransformTest, UnicodePropertyLetters) {
  // Use \p{L} to match Unicode letters (what real tokenizers do).
  // Invert mode: emit matches (the letters), discard gaps (spaces/punctuation).
  auto segments = Split("\\p{L}+", "hello 世界 café!",
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "世界");
  EXPECT_EQ(segments[2], "café");
}

TEST_F(SplitTransformTest, UnicodePropertyNumbers) {
  // Use \p{N} to match Unicode numbers.
  auto segments = Split("\\p{N}+", "abc123def456",
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "123");
  EXPECT_EQ(segments[1], "456");
}

TEST_F(SplitTransformTest, GPT2StylePattern) {
  // Simplified GPT-2 pattern: words, numbers, or punctuation.
  // Real GPT-2 uses: 's|'t|'re|...|\\p{L}+|\\p{N}+|...
  auto segments = Split("\\p{L}+|\\p{N}+|[!.,?]", "Hello, world! 123",
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 5u);
  EXPECT_EQ(segments[0], "Hello");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], "world");
  EXPECT_EQ(segments[3], "!");
  EXPECT_EQ(segments[4], "123");
}

// The real GPT-2 pattern includes optional space before letters/numbers.
// This is critical: " world" should be captured as ONE segment (space + word).
TEST_F(SplitTransformTest, GPT2FullPatternHelloWorld) {
  // The exact GPT-2 pattern from transform_json.c.
  const char* pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  auto segments =
      Split(pattern, "hello world", IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], " world");  // Space MUST be included with "world".
}

TEST_F(SplitTransformTest, GPT2FullPatternMixedContent) {
  const char* pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  auto segments = Split(pattern, "Hello, world! 123",
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  // Expected: ["Hello", ",", " world", "!", " 123"]
  ASSERT_EQ(segments.size(), 5u);
  EXPECT_EQ(segments[0], "Hello");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], " world");  // Space included.
  EXPECT_EQ(segments[3], "!");
  EXPECT_EQ(segments[4], " 123");  // Space included.
}

TEST_F(SplitTransformTest, GPT2FullPatternContractions) {
  const char* pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  auto segments =
      Split(pattern, "I'm going", IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  // Expected: ["I", "'m", " going"]
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[0], "I");
  EXPECT_EQ(segments[1], "'m");
  EXPECT_EQ(segments[2], " going");
}

// GPT-2 pattern with ISOLATED behavior, invert=false.
// ISOLATED emits gaps then matches. Since the GPT-2 pattern matches everything
// (letters, numbers, punctuation, contractions), there are no gaps, so we get
// just the matches: ["hello", " world"].
TEST_F(SplitTransformTest, GPT2PatternIsolatedBehavior) {
  const char* pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  auto segments =
      Split(pattern, "hello world", IREE_TOKENIZER_REGEX_SPLIT_ISOLATED, false);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], " world");
}

//===----------------------------------------------------------------------===//
// Streaming / Long Input Tests
//===----------------------------------------------------------------------===//

// The streaming implementation processes matches one at a time, enabling
// unlimited input length. These tests verify this works correctly.

TEST_F(SplitTransformTest, ManyMatchesStreaming) {
  // 17 commas - would have exceeded old 16-match capacity.
  auto segments = Split(",", "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r",
                        IREE_TOKENIZER_REGEX_SPLIT_ISOLATED);
  // 18 letters + 17 commas = 35 segments.
  ASSERT_EQ(segments.size(), 35u);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[34], "r");
}

TEST_F(SplitTransformTest, LongInputStreaming) {
  // Create input with 100 words - far exceeds any fixed buffer.
  std::string input;
  for (int i = 0; i < 100; ++i) {
    if (i > 0) input += " ";
    input += "word";
  }
  auto segments = Split("\\s+", input.c_str(),
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED);  // REMOVED.
  ASSERT_EQ(segments.size(), 100u);
  for (const auto& seg : segments) {
    EXPECT_EQ(seg, "word");
  }
}

TEST_F(SplitTransformTest, LongInputContiguous) {
  // Test CONTIGUOUS behavior with many matches.
  auto segments =
      Split(",", "a,,b,,,c,,,,d", IREE_TOKENIZER_REGEX_SPLIT_CONTIGUOUS);
  ASSERT_EQ(segments.size(), 7u);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], ",,");
  EXPECT_EQ(segments[2], "b");
  EXPECT_EQ(segments[3], ",,,");
  EXPECT_EQ(segments[4], "c");
  EXPECT_EQ(segments[5], ",,,,");
  EXPECT_EQ(segments[6], "d");
}

TEST_F(SplitTransformTest, LongInputMergedWithNext) {
  // MERGED_WITH_NEXT: offset-tracking only, emits single segment.
  std::string input;
  for (int i = 0; i < 50; ++i) {
    if (i > 0) input += ",";
    input += "x";
  }
  auto segments =
      Split(",", input.c_str(),
            IREE_TOKENIZER_REGEX_SPLIT_MERGED_WITH_NEXT);  // MERGED_WITH_NEXT.
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], input);
}

//===----------------------------------------------------------------------===//
// Decode (Passthrough) Tests
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, DecodeIsPassthrough) {
  iree_tokenizer_text_transform_t transform;
  iree_status_t status = iree_tokenizer_text_transform_initialize_split(
      IREE_SVL("\\s+"), IREE_TOKENIZER_REGEX_SPLIT_REMOVED, false,
      iree_allocator_system(), &transform);
  IREE_EXPECT_OK(status);

  const char* input = "hello world";
  char output[256];
  iree_host_size_t output_size = 0;
  status = iree_tokenizer_text_transform_decode(
      &transform, IREE_SV(input), output, sizeof(output), &output_size);
  IREE_EXPECT_OK(status);
  EXPECT_EQ(output_size, strlen(input));
  EXPECT_EQ(std::string(output, output_size), input);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// Error Handling Tests
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, InvalidPattern) {
  iree_tokenizer_text_transform_t transform;
  // Unmatched parenthesis is invalid.
  iree_status_t status = iree_tokenizer_text_transform_initialize_split(
      IREE_SVL("(abc"), IREE_TOKENIZER_REGEX_SPLIT_REMOVED, false,
      iree_allocator_system(), &transform);
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);
}

TEST_F(SplitTransformTest, DecodeBufferTooSmall) {
  iree_tokenizer_text_transform_t transform;
  iree_status_t status = iree_tokenizer_text_transform_initialize_split(
      IREE_SVL("\\s+"), IREE_TOKENIZER_REGEX_SPLIT_REMOVED, false,
      iree_allocator_system(), &transform);
  IREE_EXPECT_OK(status);

  const char* input = "hello world";
  char output[5];  // Too small for 11 bytes.
  iree_host_size_t output_size = 0;
  status = iree_tokenizer_text_transform_decode(
      &transform, IREE_SV(input), output, sizeof(output), &output_size);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));

  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// GPT-2 Style Pattern Tests
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, SimpleWordPattern) {
  // A simplified GPT-2 style pattern: match word characters.
  // Using invert=true to emit the matches as segments.
  auto segments = Split("[a-zA-Z]+|[0-9]+", "hello 123 world",
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "123");
  EXPECT_EQ(segments[2], "world");
}

TEST_F(SplitTransformTest, WordsAndPunctuation) {
  // Match words, numbers, and punctuation separately.
  // This is closer to GPT-2 behavior.
  auto segments = Split("[a-zA-Z]+|[0-9]+|[!.,?]", "Hello, world! 123",
                        IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 5u);
  EXPECT_EQ(segments[0], "Hello");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], "world");
  EXPECT_EQ(segments[3], "!");
  EXPECT_EQ(segments[4], "123");
}

//===----------------------------------------------------------------------===//
// NFC Normalization Tests
//===----------------------------------------------------------------------===//

TEST_F(SplitTransformTest, NfcNormalizerComposesDecomposedInput) {
  // Test that decomposed Unicode (e + combining acute) is normalized to
  // composed form (é) before splitting. This is critical for tokenizers like
  // Qwen that use NFC normalization.
  iree_tokenizer_normalizer_t normalizer;
  iree_tokenizer_normalizer_initialize_nfc(&normalizer);

  // Decomposed: "cafe" + combining acute accent (U+0301).
  const char decomposed[] = "cafe\xcc\x81";
  auto segments = SplitWithNormalizer("\\s+", decomposed, &normalizer);
  ASSERT_EQ(segments.size(), 1u);
  // After NFC: should be "café" with composed é (U+00E9 = C3 A9).
  EXPECT_EQ(segments[0], "caf\xc3\xa9");
}

TEST_F(SplitTransformTest, NfcNormalizerMultipleDecomposed) {
  iree_tokenizer_normalizer_t normalizer;
  iree_tokenizer_normalizer_initialize_nfc(&normalizer);

  // Multiple decomposed characters separated by space.
  // "e\xcc\x81" = e + combining acute → é
  // "a\xcc\x80" = a + combining grave → à
  const char decomposed[] = "e\xcc\x81 a\xcc\x80";
  auto segments = SplitWithNormalizer("\\s+", decomposed, &normalizer,
                                      IREE_TOKENIZER_REGEX_SPLIT_REMOVED);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "\xc3\xa9");  // é (U+00E9).
  EXPECT_EQ(segments[1], "\xc3\xa0");  // à (U+00C0).
}

TEST_F(SplitTransformTest, NfcNormalizerAsciiPassthrough) {
  iree_tokenizer_normalizer_t normalizer;
  iree_tokenizer_normalizer_initialize_nfc(&normalizer);

  // ASCII input should pass through unchanged (fast path).
  auto segments = SplitWithNormalizer("\\s+", "hello world", &normalizer,
                                      IREE_TOKENIZER_REGEX_SPLIT_REMOVED);
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
}

TEST_F(SplitTransformTest, NfcNormalizerAlreadyComposed) {
  iree_tokenizer_normalizer_t normalizer;
  iree_tokenizer_normalizer_initialize_nfc(&normalizer);

  // Already composed café should remain unchanged.
  auto segments = SplitWithNormalizer("\\s+", "caf\xc3\xa9", &normalizer,
                                      IREE_TOKENIZER_REGEX_SPLIT_REMOVED);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "caf\xc3\xa9");
}

TEST_F(SplitTransformTest, NfcNormalizerWithInvertMode) {
  iree_tokenizer_normalizer_t normalizer;
  iree_tokenizer_normalizer_initialize_nfc(&normalizer);

  // Test NFC with invert mode (like GPT-2 style pattern matching).
  // Pattern matches words, input has decomposed accents.
  const char decomposed[] = "hello cafe\xcc\x81 world";
  auto segments = SplitWithNormalizer("\\p{L}+", decomposed, &normalizer,
                                      IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true);
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "caf\xc3\xa9");  // Decomposed → composed.
  EXPECT_EQ(segments[2], "world");
}

}  // namespace
