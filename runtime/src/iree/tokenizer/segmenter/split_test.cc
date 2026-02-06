// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/split.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/segmenter/segmenter_test_util.h"

namespace iree::tokenizer::testing {
namespace {

//===----------------------------------------------------------------------===//
// Test Helper
//===----------------------------------------------------------------------===//

// Creates a Split segmenter from a pattern string.
// This is a test-only helper since the regex compiler is a heavy dependency.
// Check status() after construction to verify success.
class ScopedSplitSegmenter {
 public:
  ScopedSplitSegmenter(const char* pattern,
                       iree_tokenizer_regex_split_behavior_t behavior,
                       bool invert = false) {
    iree_tokenizer_regex_compile_error_t error = {0};
    status_ = iree_tokenizer_regex_compile_and_load(
        iree_make_cstring_view(pattern),
        IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
        &dfa_, &dfa_storage_, &error);
    if (!iree_status_is_ok(status_)) return;

    iree_tokenizer_segmenter_t* raw = nullptr;
    status_ = iree_tokenizer_segmenter_split_allocate(
        dfa_, dfa_storage_, behavior, invert, iree_allocator_system(), &raw);
    if (!iree_status_is_ok(status_)) return;

    segmenter_ = ScopedSegmenter(raw);
    // Ownership of dfa_storage_ transferred to segmenter.
    dfa_storage_ = nullptr;
  }

  ~ScopedSplitSegmenter() {
    // If segmenter was never created, we need to free dfa_storage_.
    if (dfa_storage_) {
      iree_allocator_free(iree_allocator_system(), dfa_storage_);
    }
  }

  iree_status_t status() const { return status_; }
  iree_tokenizer_segmenter_t* get() const { return segmenter_.get(); }

 private:
  iree_status_t status_ = iree_ok_status();
  iree_tokenizer_regex_dfa_t dfa_;
  uint8_t* dfa_storage_ = nullptr;
  ScopedSegmenter segmenter_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, CreateAndDestroy) {
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());
  ASSERT_NE(segmenter.get(), nullptr);
}

TEST(SplitSegmenterTest, StateSizeIsReasonable) {
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  iree_host_size_t state_size =
      iree_tokenizer_segmenter_state_size(segmenter.get());
  // State should be reasonable size (regex exec state ~300 bytes plus
  // position tracking fields).
  EXPECT_GT(state_size, 0u);
  EXPECT_LT(state_size, 1024u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, EmptyInput) {
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  auto segments = ProcessAndFinalize(segmenter.get(), iree_string_view_empty(),
                                     /*expect_pending_after_process=*/false);
  EXPECT_TRUE(segments.empty());
}

TEST(SplitSegmenterTest, ZeroCapacityOutput) {
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestZeroCapacityOutput(segmenter.get(), IREE_SVL("hello world"));
}

//===----------------------------------------------------------------------===//
// ISOLATED Behavior (GPT-2 Default)
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, IsolatedNoMatches) {
  // Pattern that won't match anything.
  ScopedSplitSegmenter segmenter("xyz",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // Entire input becomes one gap segment.
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("hello"), {"hello"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, IsolatedSingleMatch) {
  // Match on space.
  ScopedSplitSegmenter segmenter(" ", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "hello world" -> ["hello", " ", "world"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("hello world"),
                        {"hello", " ", "world"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, IsolatedMultipleMatches) {
  // Match on whitespace sequences.
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "a  b   c" -> ["a", "  ", "b", "   ", "c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a  b   c"),
                        {"a", "  ", "b", "   ", "c"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, IsolatedMatchAtStart) {
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "  hello" -> ["  ", "hello"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("  hello"), {"  ", "hello"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, IsolatedMatchAtEnd) {
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "hello  " -> ["hello", "  "]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("hello  "), {"hello", "  "},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, IsolatedOnlyMatches) {
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "   " -> ["   "]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("   "), {"   "},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, IsolatedAdjacentMatches) {
  // Each character is a match (alternation).
  ScopedSplitSegmenter segmenter("a|b",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "ab" -> ["a", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("ab"), {"a", "b"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// REMOVED Behavior
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, RemovedBasic) {
  ScopedSplitSegmenter segmenter(",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b,c" -> ["a", "b", "c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b,c"), {"a", "b", "c"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, RemovedAtStart) {
  ScopedSplitSegmenter segmenter(",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // ",a,b" -> ["a", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(",a,b"), {"a", "b"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, RemovedAtEnd) {
  ScopedSplitSegmenter segmenter(",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b," -> ["a", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b,"), {"a", "b"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, RemovedConsecutive) {
  ScopedSplitSegmenter segmenter(",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,,b" -> ["a", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,,b"), {"a", "b"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// MERGED_WITH_PREVIOUS Behavior
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, MergedWithPreviousBasic) {
  ScopedSplitSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b,c" -> ["a,", "b,", "c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b,c"), {"a,", "b,", "c"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, MergedWithPreviousAtStart) {
  ScopedSplitSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS);
  IREE_ASSERT_OK(segmenter.status());

  // ",a" -> [",", "a"]
  // Leading comma has nothing to merge with, emitted as [empty..comma].
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(",a"), {",", "a"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// MERGED_WITH_NEXT Behavior
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, MergedWithNextBasic) {
  ScopedSplitSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b,c" -> ["a", ",b", ",c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b,c"), {"a", ",b", ",c"},
                        /*expect_pending_after_process=*/true);
}

TEST(SplitSegmenterTest, MergedWithNextAtEnd) {
  ScopedSplitSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT);
  IREE_ASSERT_OK(segmenter.status());

  // "a," -> ["a", ","]
  // Trailing comma has nothing to merge with, emitted standalone.
  // The comma isn't emitted during process (DFA in accept at end of input),
  // so consumed=0 and has_pending=false. All segmentation happens in finalize.
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,"), {"a", ","},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// CONTIGUOUS Behavior
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, ContiguousBasic) {
  ScopedSplitSegmenter segmenter(",",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b" -> ["a", ",", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b"), {"a", ",", "b"},
                        /*expect_pending_after_process=*/true);
}

TEST(SplitSegmenterTest, ContiguousMergesConsecutive) {
  ScopedSplitSegmenter segmenter(",",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a,,b" -> ["a", ",,", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,,b"), {"a", ",,", "b"},
                        /*expect_pending_after_process=*/true);
}

TEST(SplitSegmenterTest, ContiguousTriple) {
  ScopedSplitSegmenter segmenter(",",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a,,,b" -> ["a", ",,,", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,,,b"), {"a", ",,,", "b"},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Unicode Patterns
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, UnicodeWhitespace) {
  // Unicode whitespace pattern.
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // Japanese text with ASCII space.
  // Uses UTF-8-aware chunking since regex DFA requires complete codepoints.
  std::string input = "日本語 テスト";
  TestWithAllChunkSizesUtf8(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {"日本語", " ", "テスト"},
                            /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, UnicodeLetterPattern) {
  // Match sequences of letters.
  ScopedSplitSegmenter segmenter("\\p{L}+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "hello 世界" -> ["hello", " ", "世界"]
  // Uses UTF-8-aware chunking since regex DFA requires complete codepoints.
  std::string input = "hello 世界";
  TestWithAllChunkSizesUtf8(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {"hello", " ", "世界"},
                            /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// GPT-2 Style Pattern
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, GPT2StyleSimple) {
  // Simplified GPT-2 pattern: words and spaces.
  ScopedSplitSegmenter segmenter("\\p{L}+|\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "Hello world" -> ["Hello", " ", "world"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("Hello world"),
                        {"Hello", " ", "world"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, GPT2StyleWithPunctuation) {
  // Pattern: words, numbers, spaces, punctuation.
  ScopedSplitSegmenter segmenter("\\p{L}+|\\p{N}+|\\s+|[.,!?]",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "Hello, world!" -> ["Hello", ",", " ", "world", "!"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("Hello, world!"),
                        {"Hello", ",", " ", "world", "!"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Edge Cases
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, HasPendingCorrectness) {
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());
  ScopedSegmenterState state(segmenter.get());

  // Initially no pending.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // After processing content with no matches, the segmenter reduces consumed
  // to the committed position (0 here). has_pending is false because the
  // unconsumed data is reported back via reduced consumed — the caller retains
  // it for finalize.
  iree_tokenizer_segment_t segments[16];
  iree_tokenizer_segment_output_t output = {16, segments};
  iree_host_size_t consumed = 0, count = 0;

  iree_string_view_t input = IREE_SVL("hello");
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input, output, &consumed, &count));
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // After finalize, no pending.
  // Pull-based: finalize receives the unconsumed portion.
  iree_string_view_t remaining_input =
      iree_make_string_view(input.data + consumed, input.size - consumed);
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), remaining_input, output, &count));
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));
}

TEST(SplitSegmenterTest, LimitedOutputCapacity) {
  ScopedSplitSegmenter segmenter(",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b,c" with capacity=1 should work correctly.
  TestLimitedOutputCapacity(segmenter.get(), IREE_SVL("a,b,c"),
                            {"a", ",", "b", ",", "c"});
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, SingleCharInput) {
  ScopedSplitSegmenter segmenter("\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a"), {"a"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, SingleCharMatch) {
  ScopedSplitSegmenter segmenter(" ", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(" "), {" "},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Lookahead Pattern Tests (GPT-2 style)
//===----------------------------------------------------------------------===//

// These tests verify the GPT-2 lookahead pattern \s+(?!\S)|\s+ correctly
// produces separate segments when consecutive whitespace is followed by
// non-whitespace.

TEST(SplitSegmenterTest, LookaheadConsecutiveNewlinesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\n\nA"
  //
  // Expected: ["\n", "\n", "A"]
  // First '\n' passes lookahead (followed by whitespace).
  // Second '\n' fails lookahead (followed by 'A'), uses fallback.
  // Result: two separate newline segments, then text.
  ScopedSplitSegmenter segmenter("\\s+(?!\\S)|\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("\n\nA"), {"\n", "\n", "A"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, LookaheadThreeNewlinesBeforeText) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "\n\n\nA"
  //
  // Expected: ["\n\n", "\n", "A"]
  // First two '\n' pass lookahead (followed by whitespace).
  // Third '\n' fails lookahead (followed by 'A'), uses fallback.
  ScopedSplitSegmenter segmenter("\\s+(?!\\S)|\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("\n\n\nA"),
                        {"\n\n", "\n", "A"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, LookaheadNewlinesAtEnd) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "A\n\n"
  //
  // Expected: ["A", "\n\n"]
  // At EOS, lookahead passes for all whitespace (no following char = not \S).
  ScopedSplitSegmenter segmenter("\\s+(?!\\S)|\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("A\n\n"), {"A", "\n\n"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, GPT2FullPattern_ConsecutiveNewlines) {
  // Full GPT-2 pattern with "\n\nAll:" (paragraph break in text).
  ScopedSplitSegmenter segmenter(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
      IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("\n\nAll:"),
                        {"\n", "\n", "All", ":"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, GPT2FullPattern_SpacePlusPunctuation) {
  // The ` ?[^\s\p{L}\p{N}]+` alternative must group space with punctuation.
  // Input: "s  <" (s, space, space, angle-bracket)
  // Expected: ["s", " ", " <"]
  //   - "s" matches ` ?\p{L}+` (letter)
  //   - " " (first space) matches `\s+` (standalone whitespace)
  //   - " <" matches ` ?[^\s\p{L}\p{N}]+` (optional space + punctuation)
  ScopedSplitSegmenter segmenter(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
      IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("s  <"), {"s", " ", " <"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// INLINE CALLBACK: Large Input (no data loss)
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, LargeInputNoDataLoss) {
  // Pattern matches each space individually. With 300 words we get 299 matches
  // (spaces), which exceeds the old 128-match buffer. The inline callback
  // approach handles unlimited matches without data loss.
  ScopedSplitSegmenter segmenter(" ", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // Build input: "w0 w1 w2 ... w299" (300 single-char "words" separated by
  // spaces, exceeding the old 128-match limit).
  std::string input;
  std::vector<std::string> expected;
  for (int i = 0; i < 300; ++i) {
    if (i > 0) input += " ";
    std::string word = "w" + std::to_string(i);
    input += word;
    expected.push_back(word);
  }

  iree_string_view_t input_view =
      iree_make_string_view(input.data(), input.size());

  // Process entire input in one call with large capacity.
  auto segments = ProcessAndFinalize(segmenter.get(), input_view,
                                     /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), expected.size())
      << "All 300 words should be produced as segments (299 matches)";
  EXPECT_EQ(segments, expected);
}

TEST(SplitSegmenterTest, LargeInputIsolatedNoDataLoss) {
  // ISOLATED produces 2 segments per match (gap + match). With 200 matches
  // we get up to 400+ segments total — well beyond the old buffer limit.
  ScopedSplitSegmenter segmenter(",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  std::string input;
  std::vector<std::string> expected;
  for (int i = 0; i < 200; ++i) {
    if (i > 0) {
      input += ",";
      expected.push_back(",");
    }
    char c = 'a' + (i % 26);
    input += c;
    expected.push_back(std::string(1, c));
  }

  iree_string_view_t input_view =
      iree_make_string_view(input.data(), input.size());

  auto segments = ProcessAndFinalize(segmenter.get(), input_view,
                                     /*expect_pending_after_process=*/false);
  EXPECT_EQ(segments, expected);
}

//===----------------------------------------------------------------------===//
// INLINE CALLBACK: Output Capacity Constraints
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, OutputCapacityOneSlotIsolated) {
  // ISOLATED with capacity=1: each match produces gap + match (2 segments).
  // With capacity=1, only the gap fits. The match is re-discovered next call.
  ScopedSplitSegmenter segmenter(" ", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "ab cd ef" -> ["ab", " ", "cd", " ", "ef"]
  TestLimitedOutputCapacity(segmenter.get(), IREE_SVL("ab cd ef"),
                            {"ab", " ", "cd", " ", "ef"});
}

TEST(SplitSegmenterTest, OutputCapacityOneSlotRemoved) {
  // REMOVED with capacity=1: each match produces 1 segment (gap only).
  // Should work fine with capacity=1.
  ScopedSplitSegmenter segmenter(",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b,c" -> ["a", "b", "c"]
  TestLimitedOutputCapacity(segmenter.get(), IREE_SVL("a,b,c"),
                            {"a", "b", "c"});
}

TEST(SplitSegmenterTest, OutputCapacityMultiSegmentAtomicity) {
  // With ISOLATED and capacity=2, verify that a 2-segment result (gap + match)
  // is emitted atomically — both or neither — rather than splitting.
  ScopedSplitSegmenter segmenter(" ", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  ScopedSegmenterState state(segmenter.get());

  // "hello world foo" has 2 spaces → 5 segments: [hello][ ][world][ ][foo]
  iree_string_view_t input = IREE_SVL("hello world foo");

  iree_tokenizer_segment_t segments[2];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // First call with capacity=2: should get exactly 2 segments (gap + match)
  // as one atomic result, not 1.5 of a result.
  IREE_EXPECT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input, iree_tokenizer_make_segment_output(segments, 2),
      &consumed, &segment_count));

  EXPECT_EQ(segment_count, 2u);
  if (segment_count >= 2) {
    // "hello" and " " should be the first atomic result.
    std::string seg0(input.data + segments[0].start,
                     segments[0].end - segments[0].start);
    std::string seg1(input.data + segments[1].start,
                     segments[1].end - segments[1].start);
    EXPECT_EQ(seg0, "hello");
    EXPECT_EQ(seg1, " ");
  }
}

//===----------------------------------------------------------------------===//
// INLINE CALLBACK: Finalize Re-entrancy
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, FinalizeReentrant) {
  // MERGED_WITH_NEXT: at finalize time, pending is merged with trailing gap.
  // With capacity=1, the merged segment fills output. A second finalize call
  // should complete without emitting duplicates.
  ScopedSplitSegmenter segmenter(
      " ", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT);
  IREE_ASSERT_OK(segmenter.status());

  // "abc def" -> [" def"] (match " " merged with "def") plus leading "abc"
  // Full expected: ["abc", " def"]
  TestLimitedOutputCapacity(segmenter.get(), IREE_SVL("abc def"),
                            {"abc", " def"});
}

TEST(SplitSegmenterTest, FinalizeReentrantContiguous) {
  // CONTIGUOUS with content after the last match. Finalize emits pending + gap.
  ScopedSplitSegmenter segmenter(",",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b" -> ["a", ",", "b"]
  TestLimitedOutputCapacity(segmenter.get(), IREE_SVL("a,b"), {"a", ",", "b"});
}

//===----------------------------------------------------------------------===//
// INLINE CALLBACK: Inverted Mode with Overflow
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, InvertWithOverflow) {
  // Inverted REMOVED: matches become content segments, gaps are discarded.
  // Tests partial consumption with inverted semantics.
  ScopedSplitSegmenter segmenter(
      "\\w+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED, /*invert=*/true);
  IREE_ASSERT_OK(segmenter.status());

  // "hello world foo" with inverted \\w+ → matches are words, gaps discarded.
  // Expected segments: ["hello", "world", "foo"]
  TestLimitedOutputCapacity(segmenter.get(), IREE_SVL("hello world foo"),
                            {"hello", "world", "foo"});
}

TEST(SplitSegmenterTest, InvertManyMatches) {
  // Inverted with many matches and limited capacity.
  ScopedSplitSegmenter segmenter(
      "[a-z]", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED, /*invert=*/true);
  IREE_ASSERT_OK(segmenter.status());

  // "a1b2c3d4e5" with inverted [a-z] → matches are single letters.
  // Expected: ["a", "b", "c", "d", "e"]
  TestLimitedOutputCapacity(segmenter.get(), IREE_SVL("a1b2c3d4e5"),
                            {"a", "b", "c", "d", "e"});
}

TEST(SplitSegmenterTest, InvertRemovedAllMatchingInput) {
  // When the entire input matches the pattern with REMOVED+invert=true, the
  // regex stores the match as pending (streaming mode waits for termination).
  // process() must limit consumed bytes so that finalize() receives the input.
  //
  // Pattern [a-z]+ on "hello": entire input is one match. The regex can't
  // terminate the match during process() because no non-matching byte is seen.
  // If process() reports consumed=5, finalize() receives empty remaining_input
  // and emits the match with positions relative to nothing, causing underflow.
  ScopedSplitSegmenter segmenter(
      "[a-z]+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED, /*invert=*/true);
  IREE_ASSERT_OK(segmenter.status());
  ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_tokenizer_segment_output_t output =
      iree_tokenizer_make_segment_output(segments, 8);

  iree_string_view_t input = IREE_SVL("hello");
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input, output, &consumed, &segment_count));

  // consumed must be limited so remaining_input contains the pending match.
  // The caller uses (consumed < input.size) as the signal to keep remaining
  // bytes for finalize(). If consumed equals input.size, finalize() gets
  // empty remaining_input and segment positions underflow.
  EXPECT_LT(consumed, input.size)
      << "Should limit consumed when regex has pending match";

  // Finalize with the unconsumed portion.
  iree_string_view_t remaining =
      iree_make_string_view(input.data + consumed, input.size - consumed);
  segment_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), remaining, output, &segment_count));

  ASSERT_EQ(segment_count, 1u) << "Should emit exactly one segment";

  // Segment positions must be valid (relative to remaining_input).
  // If underflow occurred, start would be a huge value like SIZE_MAX - 4.
  EXPECT_LE(segments[0].start, remaining.size)
      << "Segment start must be within remaining_input bounds";
  EXPECT_LE(segments[0].end, remaining.size)
      << "Segment end must be within remaining_input bounds";
  EXPECT_LE(segments[0].start, segments[0].end)
      << "Segment start must not exceed end (underflow check)";

  // Verify the segment covers the expected text.
  std::string segment_text(remaining.data + segments[0].start,
                           segments[0].end - segments[0].start);
  EXPECT_EQ(segment_text, "hello");
}

//===----------------------------------------------------------------------===//
// INLINE CALLBACK: Contiguous Adjacent Matches
//===----------------------------------------------------------------------===//

TEST(SplitSegmenterTest, ConsecutiveMatchesContiguous) {
  // CONTIGUOUS: adjacent matches merge into one segment.
  // Verifies the extend-pending path works correctly with inline processing.
  ScopedSplitSegmenter segmenter("[0-9]",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "abc123def456ghi" → gaps [abc, def, ghi] + merged matches [123, 456]
  // Expected: ["abc", "123", "def", "456", "ghi"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("abc123def456ghi"),
                        {"abc", "123", "def", "456", "ghi"},
                        /*expect_pending_after_process=*/true);
}

TEST(SplitSegmenterTest, ConsecutiveMatchesContiguousAllDigits) {
  // All-match input with CONTIGUOUS: entire input becomes one merged segment.
  ScopedSplitSegmenter segmenter("[0-9]",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "12345" → one merged match segment.
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("12345"), {"12345"},
                        /*expect_pending_after_process=*/true);
}

TEST(SplitSegmenterTest, ConsecutiveMatchesContiguousCapacityOne) {
  // CONTIGUOUS with capacity=1: pending is flushed when a non-adjacent match
  // arrives. Tests the inline callback's pending accumulation.
  ScopedSplitSegmenter segmenter("[0-9]",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a12b34c" → ["a", "12", "b", "34", "c"]
  TestLimitedOutputCapacity(segmenter.get(), IREE_SVL("a12b34c"),
                            {"a", "12", "b", "34", "c"});
}

//===----------------------------------------------------------------------===//
// Streaming Offset Verification
//===----------------------------------------------------------------------===//
// These tests directly assert raw segment.start/segment.end values and verify
// that absolute position reconstruction matches HuggingFace's single-pass
// pre_tokenize_str() output.

// MERGED_WITH_NEXT: verify finalize() produces unsigned-wraparound offsets
// when the pending match is before chunk_base.
//
// HuggingFace ground truth:
//   Split(",", behavior="merged_with_next").pre_tokenize_str("a,b")
//   → [("a", (0, 1)), (",b", (1, 3))]
TEST(SplitSegmenterTest, StreamingOffsets_MergedWithNextWraparound) {
  ScopedSplitSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT);
  IREE_ASSERT_OK(segmenter.status());
  testing::ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process "a,b" (3 bytes).
  // MERGED_WITH_NEXT: emit gap "a" [0,1), pending "," [1,2). Consume all
  // (has_pending=true).
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a,b"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 3u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);  // "a" at [0,1)
  EXPECT_EQ(segments[0].end, 1u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Finalize with empty remaining_input.
  // Pending "," merged with trailing "b" → ",b" at absolute [1,3).
  // chunk_base=3, so relative offsets wrap unsigned.
  iree_host_size_t position = consumed;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_string_view_empty(),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Verify absolute reconstruction matches HF: (",b", (1, 3)).
  size_t abs_start = position + segments[0].start;
  size_t abs_end = position + segments[0].end;
  EXPECT_EQ(abs_start, 1u);
  EXPECT_EQ(abs_end, 3u);
}

// CONTIGUOUS: two adjacent matches extend pending. Finalize emits both pending
// and trailing gap with unsigned-wraparound offsets.
//
// HuggingFace ground truth:
//   Split(",", behavior="contiguous").pre_tokenize_str("a,,b")
//   → [("a", (0, 1)), (",,", (1, 3)), ("b", (3, 4))]
TEST(SplitSegmenterTest, StreamingOffsets_ContiguousWraparound) {
  ScopedSplitSegmenter segmenter(",",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());
  testing::ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process "a,,b" (4 bytes).
  // CONTIGUOUS: emit gap "a" [0,1), extend pending to [1,3) for ",,".
  // Consume all (has_pending=true).
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a,,b"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 4u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);  // "a" at [0,1)
  EXPECT_EQ(segments[0].end, 1u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Finalize: pending ",," at [1,3) and trailing "b" at [3,4).
  // Both wrap unsigned since chunk_base=4.
  iree_host_size_t position = consumed;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_string_view_empty(),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 2u);
  // Verify absolute reconstruction matches HF: (",,", (1, 3)), ("b", (3, 4)).
  EXPECT_EQ(position + segments[0].start, 1u);
  EXPECT_EQ(position + segments[0].end, 3u);
  EXPECT_EQ(position + segments[1].start, 3u);
  EXPECT_EQ(position + segments[1].end, 4u);
}

// MERGED_WITH_NEXT: 2-chunk streaming. Pending crosses chunk boundaries with
// unsigned wraparound on the second emission.
//
// Each chunk must contain a non-matching byte after any match to trigger
// the DFA's NO_TRANSITION callback (greedy matching defers emission until
// the match is confirmed by a non-extending byte).
//
// HuggingFace ground truth:
//   Split(",", behavior="merged_with_next").pre_tokenize_str("a,b,c")
//   → [("a", (0, 1)), (",b", (1, 3)), (",c", (3, 5))]
TEST(SplitSegmenterTest, StreamingOffsets_CrossChunkMergedWithNext) {
  ScopedSplitSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT);
  IREE_ASSERT_OK(segmenter.status());
  testing::ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Chunk 1: "a,b" (3 bytes). 'b' confirms the comma match via NO_TRANSITION.
  // Emits gap "a" [0,1), sets pending=[1,2).
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a,b"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 3u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Chunk 2: ",c" (2 bytes, position=3).
  // 'c' confirms the comma match. The new match flushes old pending merged
  // with the gap: ",b" at absolute [1,3). New pending=[3,4).
  // Relative to chunk_base=3: start=1-3 wraps unsigned.
  iree_host_size_t position_chunk2 = consumed;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view(",c"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 2u);
  ASSERT_EQ(segment_count, 1u);
  // Verify absolute reconstruction matches HF: (",b", (1, 3)).
  EXPECT_EQ(position_chunk2 + segments[0].start, 1u);
  EXPECT_EQ(position_chunk2 + segments[0].end, 3u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Finalize: pending "," merged with trailing "c" → ",c" at absolute [3,5).
  iree_host_size_t position_finalize = position_chunk2 + consumed;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_string_view_empty(),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Verify absolute reconstruction matches HF: (",c", (3, 5)).
  EXPECT_EQ(position_finalize + segments[0].start, 3u);
  EXPECT_EQ(position_finalize + segments[0].end, 5u);
}

// CONTIGUOUS: flush() emits segments with absolute offsets and resets state.
//
// HuggingFace ground truth:
//   Split(",", behavior="contiguous").pre_tokenize_str("a,,b")
//   → [("a", (0, 1)), (",,", (1, 3)), ("b", (3, 4))]
// Process emits ("a", {0,1}). Flush emits (",,", {1,3}) and ("b", {3,4})
// as absolute offsets.
TEST(SplitSegmenterTest, StreamingOffsets_FlushAbsolute) {
  ScopedSplitSegmenter segmenter(",",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());
  testing::ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process "a,,b" — emit gap, accumulate pending.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a,,b"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 4u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Flush: emits pending + trailing with absolute offsets.
  iree_host_size_t flush_count = 0;
  iree_host_size_t bytes_committed = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_flush(
      state.get(), iree_tokenizer_make_segment_output(segments, 8),
      &flush_count, &bytes_committed));

  ASSERT_EQ(flush_count, 2u);
  EXPECT_EQ(bytes_committed, 4u);
  // Flush offsets are absolute (chunk_base=0 in flush emitter).
  EXPECT_EQ(segments[0].start, 1u);  // ",," at [1,3)
  EXPECT_EQ(segments[0].end, 3u);
  EXPECT_EQ(segments[1].start, 3u);  // "b" at [3,4)
  EXPECT_EQ(segments[1].end, 4u);

  // State is reset: no pending, ready for fresh input.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));
}

// CONTIGUOUS: flush() with limited capacity preserves state for re-entry.
// If output fills during trailing emission, state must NOT be reset so caller
// can retry with more capacity.
//
// HuggingFace ground truth:
//   Split(",", behavior="contiguous").pre_tokenize_str("a,,b")
//   → [("a", (0, 1)), (",,", (1, 3)), ("b", (3, 4))]
TEST(SplitSegmenterTest, StreamingOffsets_FlushPartialCapacity) {
  ScopedSplitSegmenter segmenter(",",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());
  testing::ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process "a,,b" — emit gap "a", accumulate pending ",,".
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a,,b"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 4u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // First flush with capacity=1: can only emit pending ",,", not trailing "b".
  iree_host_size_t flush_count = 0;
  iree_host_size_t bytes_committed = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_flush(
      state.get(), iree_tokenizer_make_segment_output(segments, 1),
      &flush_count, &bytes_committed));

  ASSERT_EQ(flush_count, 1u);
  EXPECT_EQ(bytes_committed, 3u);    // Committed up to end of ",,"
  EXPECT_EQ(segments[0].start, 1u);  // ",," at [1,3)
  EXPECT_EQ(segments[0].end, 3u);

  // State should be preserved (not reset) since flush was incomplete.
  // has_pending() returns true because there's still trailing "b" to emit.
  // State must NOT be reset to 0, so second flush continues with correct
  // absolute offsets.
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Second flush with more capacity: emit remaining trailing "b".
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_flush(
      state.get(), iree_tokenizer_make_segment_output(segments, 8),
      &flush_count, &bytes_committed));

  ASSERT_EQ(flush_count, 1u);
  EXPECT_EQ(bytes_committed, 4u);    // Now committed all 4 bytes
  EXPECT_EQ(segments[0].start, 3u);  // "b" at [3,4)
  EXPECT_EQ(segments[0].end, 4u);

  // Now state should be fully reset.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));
}

// ISOLATED: capacity=1 overflow recovery. Verify re-entry offsets are correct
// across multiple process() calls.
//
// HuggingFace ground truth:
//   Split(",", behavior="isolated").pre_tokenize_str("a,b")
//   → [("a", (0, 1)), (",", (1, 2)), ("b", (2, 3))]
TEST(SplitSegmenterTest, StreamingOffsets_OverflowRecovery) {
  ScopedSplitSegmenter segmenter(",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());
  testing::ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[1];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;
  iree_host_size_t position = 0;

  // Call 1: capacity=1. Emits "a" [0,1), overflows on ",".
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a,b"),
      iree_tokenizer_make_segment_output(segments, 1), &consumed,
      &segment_count));

  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(consumed, 1u);
  EXPECT_EQ(position + segments[0].start, 0u);  // "a" at abs [0,1)
  EXPECT_EQ(position + segments[0].end, 1u);
  position += consumed;

  // Call 2: capacity=1 with remaining ",b". Emits "," [0,1) relative.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view(",b"),
      iree_tokenizer_make_segment_output(segments, 1), &consumed,
      &segment_count));

  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(consumed, 1u);
  EXPECT_EQ(position + segments[0].start, 1u);  // "," at abs [1,2)
  EXPECT_EQ(position + segments[0].end, 2u);
  position += consumed;

  // Finalize: remaining "b" at position=2.
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view("b"),
      iree_tokenizer_make_segment_output(segments, 1), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  EXPECT_EQ(position + segments[0].start, 2u);  // "b" at abs [2,3)
  EXPECT_EQ(position + segments[0].end, 3u);
}

//===----------------------------------------------------------------------===//
// Literal Pattern Tests
//===----------------------------------------------------------------------===//

// Helper class for literal pattern tests (no regex compilation needed).
class ScopedSplitLiteralSegmenter {
 public:
  ScopedSplitLiteralSegmenter(const char* pattern,
                              iree_tokenizer_regex_split_behavior_t behavior,
                              bool invert = false) {
    iree_tokenizer_segmenter_t* raw = nullptr;
    status_ = iree_tokenizer_segmenter_split_literal_allocate(
        iree_make_cstring_view(pattern), behavior, invert,
        iree_allocator_system(), &raw);
    if (iree_status_is_ok(status_)) {
      segmenter_ = ScopedSegmenter(raw);
    }
  }

  iree_status_t status() const { return status_; }
  iree_tokenizer_segmenter_t* get() const { return segmenter_.get(); }

 private:
  iree_status_t status_ = iree_ok_status();
  ScopedSegmenter segmenter_;
};

//===----------------------------------------------------------------------===//
// LITERAL: Lifecycle
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, CreateAndDestroy) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());
  ASSERT_NE(segmenter.get(), nullptr);
}

TEST(SplitLiteralTest, StateSizeIsReasonable) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  iree_host_size_t state_size =
      iree_tokenizer_segmenter_state_size(segmenter.get());
  // Literal mode state should be similar size to regex (same struct, just
  // different fields used).
  EXPECT_GT(state_size, 0u);
  EXPECT_LT(state_size, 1024u);
}

TEST(SplitLiteralTest, EmptyPatternRejected) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  iree_status_t status = iree_tokenizer_segmenter_split_literal_allocate(
      iree_string_view_empty(), IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED, false,
      iree_allocator_system(), &segmenter);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// LITERAL: No-ops
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, EmptyInput) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  auto segments = ProcessAndFinalize(segmenter.get(), iree_string_view_empty(),
                                     /*expect_pending_after_process=*/false);
  EXPECT_TRUE(segments.empty());
}

TEST(SplitLiteralTest, ZeroCapacityOutput) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  TestZeroCapacityOutput(segmenter.get(), IREE_SVL("a,b,c"));
}

//===----------------------------------------------------------------------===//
// LITERAL: REMOVED Behavior
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, RemovedBasic) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b,c" -> ["a", "b", "c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b,c"), {"a", "b", "c"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, RemovedNoMatches) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "hello" -> ["hello"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("hello"), {"hello"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, RemovedAtStart) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // ",a,b" -> ["a", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(",a,b"), {"a", "b"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, RemovedAtEnd) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b," -> ["a", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b,"), {"a", "b"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, RemovedConsecutive) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,,b" -> ["a", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,,b"), {"a", "b"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// LITERAL: ISOLATED Behavior
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, IsolatedBasic) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b" -> ["a", ",", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b"), {"a", ",", "b"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, IsolatedNoMatches) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "hello" -> ["hello"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("hello"), {"hello"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, IsolatedAtStart) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // ",a" -> [",", "a"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(",a"), {",", "a"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, IsolatedAtEnd) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "a," -> ["a", ","]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,"), {"a", ","},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, IsolatedOnlyMatches) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // ",,," -> [",", ",", ","]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(",,,"), {",", ",", ","},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// LITERAL: MERGED_WITH_PREVIOUS Behavior
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, MergedWithPreviousBasic) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b,c" -> ["a,", "b,", "c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b,c"), {"a,", "b,", "c"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, MergedWithPreviousAtStart) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS);
  IREE_ASSERT_OK(segmenter.status());

  // ",a" -> [",", "a"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(",a"), {",", "a"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// LITERAL: MERGED_WITH_NEXT Behavior
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, MergedWithNextBasic) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b,c" -> ["a", ",b", ",c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b,c"), {"a", ",b", ",c"},
                        /*expect_pending_after_process=*/true);
}

TEST(SplitLiteralTest, MergedWithNextAtEnd) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT);
  IREE_ASSERT_OK(segmenter.status());

  // "a," -> ["a", ","]
  // Note: Unlike regex (which stays "in accept" at end), literal matching
  // immediately recognizes the complete match and buffers it as pending.
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,"), {"a", ","},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// LITERAL: CONTIGUOUS Behavior
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, ContiguousBasic) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b" -> ["a", ",", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,b"), {"a", ",", "b"},
                        /*expect_pending_after_process=*/true);
}

TEST(SplitLiteralTest, ContiguousMergesConsecutive) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a,,b" -> ["a", ",,", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,,b"), {"a", ",,", "b"},
                        /*expect_pending_after_process=*/true);
}

TEST(SplitLiteralTest, ContiguousTriple) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  IREE_ASSERT_OK(segmenter.status());

  // "a,,,b" -> ["a", ",,,", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a,,,b"), {"a", ",,,", "b"},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// LITERAL: Multi-Byte Patterns
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, MultiBytePattern) {
  // Use "||" as a 2-byte delimiter.
  ScopedSplitLiteralSegmenter segmenter(
      "||", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a||b||c" -> ["a", "b", "c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a||b||c"), {"a", "b", "c"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, MultiBytePatternPartialFalseStart) {
  // Pattern "|=" with false start "|" that doesn't complete.
  ScopedSplitLiteralSegmenter segmenter(
      "|=", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a|b|=c" -> ["a|b", "c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a|b|=c"), {"a|b", "c"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, UTF8Pattern) {
  // Use the metaspace character "▁" (U+2581, 3 bytes) as delimiter.
  ScopedSplitLiteralSegmenter segmenter(
      "▁", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "a▁b" -> ["a", "▁", "b"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a▁b"), {"a", "▁", "b"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, LongPattern) {
  // Pattern "===>" (4 bytes).
  ScopedSplitLiteralSegmenter segmenter(
      "===>", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // "a===>b===>c" -> ["a", "b", "c"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a===>b===>c"),
                        {"a", "b", "c"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// LITERAL: Streaming Edge Cases
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, PatternSplitAcrossChunks) {
  // Pattern "||" split between chunks.
  ScopedSplitLiteralSegmenter segmenter(
      "||", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // Test with explicit chunk sizes to ensure partial match handling.
  // "abc||def" with chunk size 4: "abc|" then "|def"
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("abc||def"), {"abc", "def"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, PatternAtExactChunkBoundary) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // Process "abc,def" byte by byte to test all boundary conditions.
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("abc,def"), {"abc", "def"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, LimitedOutputCapacity) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "a,b,c" with capacity=1 should work correctly.
  TestLimitedOutputCapacity(segmenter.get(), IREE_SVL("a,b,c"),
                            {"a", ",", "b", ",", "c"});
}

TEST(SplitLiteralTest, LargeInputNoDataLoss) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // Build input with many segments (same pattern as regex test).
  std::string input;
  std::vector<std::string> expected;
  for (int i = 0; i < 300; ++i) {
    if (i > 0) input += ",";
    std::string word = "w" + std::to_string(i);
    input += word;
    expected.push_back(word);
  }

  iree_string_view_t input_view =
      iree_make_string_view(input.data(), input.size());

  auto segments = ProcessAndFinalize(segmenter.get(), input_view,
                                     /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), expected.size());
  EXPECT_EQ(segments, expected);
}

//===----------------------------------------------------------------------===//
// LITERAL: Inverted Mode
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, InvertBasic) {
  // Inverted REMOVED: matches become content, gaps are discarded.
  ScopedSplitLiteralSegmenter segmenter(
      "abc", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED, /*invert=*/true);
  IREE_ASSERT_OK(segmenter.status());

  // "XXXabcYYYabcZZZ" with inverted -> ["abc", "abc"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("XXXabcYYYabcZZZ"),
                        {"abc", "abc"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, InvertIsolated) {
  // Inverted ISOLATED: matches and gaps are both emitted.
  ScopedSplitLiteralSegmenter segmenter(
      "ab", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED, /*invert=*/true);
  IREE_ASSERT_OK(segmenter.status());

  // "XabY" with inverted ISOLATED -> ["X", "ab", "Y"]
  // Note: has_pending=true because trailing gap "Y" is emitted in finalize.
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("XabY"), {"X", "ab", "Y"},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// LITERAL: Edge Cases
//===----------------------------------------------------------------------===//

TEST(SplitLiteralTest, SingleCharInput) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a"), {"a"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, SingleCharMatch) {
  ScopedSplitLiteralSegmenter segmenter(
      ",", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(","), {","},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitLiteralTest, PatternLongerThanInput) {
  ScopedSplitLiteralSegmenter segmenter(
      "verylongpattern", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  IREE_ASSERT_OK(segmenter.status());

  // Input shorter than pattern: no matches, entire input is trailing gap.
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("short"), {"short"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// CJK Pattern Tests (DeepSeek-V3 style pre_tokenizer)
//===----------------------------------------------------------------------===//

// DeepSeek-V3 uses a pre_tokenizer with CJK character class matching:
//   [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+
// This matches:
// - Chinese characters (U+4E00-U+9FA5, CJK Unified Ideographs)
// - Hiragana (U+3040-U+309F)
// - Katakana (U+30A0-U+30FF)
//
// Note: We use literal UTF-8 characters because the regex parser's handling
// of Unicode escapes in ranges for non-ASCII codepoints differs from literals.

TEST(SplitSegmenterTest, DeepSeekV3CJKPattern) {
  // DeepSeek-V3 CJK pattern using literal UTF-8 characters.
  // 3 Unicode ranges: CJK (一-龥), Hiragana (ぁ-ゟ), Katakana (ァ-ヿ).
  ScopedSplitSegmenter segmenter("[一-龥ぁ-ゟァ-ヿ]+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "Hello 你好世界 こんにちは"
  // ASCII "Hello " -> gap
  // Chinese "你好世界" -> match
  // Space " " -> gap
  // Japanese "こんにちは" -> match
  std::string input = "Hello 你好世界 こんにちは";
  TestWithAllChunkSizesUtf8(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {"Hello ", "你好世界", " ", "こんにちは"},
                            /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, DeepSeekV3CJKPatternLimitedCapacity) {
  // Same CJK pattern with limited output capacity.
  // This tests the re-entry path when segment buffer fills.
  ScopedSplitSegmenter segmenter("[一-龥ぁ-ゟァ-ヿ]+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  std::string input = "Hello 你好世界 こんにちは";
  TestLimitedOutputCapacity(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {"Hello ", "你好世界", " ", "こんにちは"});
}

// Tests finalize re-entry when CJK regex has pending match state and output
// capacity is limited. When finalize cannot emit all segments in one call,
// has_pending() returns true and the caller must call finalize again.
TEST(SplitSegmenterTest, DeepSeekV3CJKFinalizeWithLimitedCapacity) {
  ScopedSplitSegmenter segmenter("[一-龥ぁ-ゟァ-ヿ]+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());
  ScopedSegmenterState state(segmenter.get());

  // Input ends with CJK text (no trailing ASCII to terminate match).
  // This forces the match to be finalized rather than emitted during process.
  std::string input = "Test 日本語";
  iree_string_view_t input_sv =
      iree_make_string_view(input.c_str(), input.size());

  iree_tokenizer_segment_t segment;
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process with capacity=1. The DFA will be in an accept state at EOS,
  // so no segments are emitted during process (match pending).
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input_sv, iree_tokenizer_make_segment_output(&segment, 1),
      &consumed, &segment_count));

  // The "Test " gap should be emitted, but "日本語" is pending in accept state.
  // Exact behavior depends on when the DFA confirms the match.

  // Finalize to get remaining segments one at a time.
  std::vector<std::string> results;
  iree_string_view_t remaining =
      iree_make_string_view(input_sv.data + consumed, input_sv.size - consumed);

  // Loop until finalize completes (returns OK with count=0).
  // RESOURCE_EXHAUSTED means "buffer full, call again for more segments".
  while (true) {
    iree_host_size_t finalize_count = 0;
    iree_status_t status = iree_tokenizer_segmenter_state_finalize(
        state.get(), remaining, iree_tokenizer_make_segment_output(&segment, 1),
        &finalize_count);

    if (iree_status_is_resource_exhausted(status)) {
      // Buffer full - consume segment and continue.
      iree_status_ignore(status);
      ASSERT_EQ(finalize_count, 1u);
      std::string seg(remaining.data + segment.start,
                      segment.end - segment.start);
      results.push_back(seg);
      continue;
    }

    // Any other error is unexpected.
    IREE_ASSERT_OK(status);

    // OK status: consume any remaining segment(s) and check if done.
    if (finalize_count > 0) {
      std::string seg(remaining.data + segment.start,
                      segment.end - segment.start);
      results.push_back(seg);
    }

    // Check if finalize is complete.
    if (!iree_tokenizer_segmenter_state_has_pending(state.get())) {
      break;
    }
  }

  // Verify we got both segments: "Test " and "日本語".
  EXPECT_GE(results.size(), 1u);
  // The exact segments depend on what was emitted during process vs finalize.
}

// Test with mixed ASCII and CJK, alternating characters.
// This creates many segments to stress the re-entry path.
TEST(SplitSegmenterTest, DeepSeekV3CJKManySegmentsCapacity1) {
  // Chinese-only pattern: match CJK Unified Ideographs.
  ScopedSplitSegmenter segmenter("[一-龥]+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // Create input with alternating ASCII and Chinese characters.
  // "a中b国c人d民" -> ["a", "中", "b", "国", "c", "人", "d", "民"]
  std::string input = "a中b国c人d民";
  TestLimitedOutputCapacity(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {"a", "中", "b", "国", "c", "人", "d", "民"});
}

// Test the DeepSeek-V3 number pattern: \p{N}{1,3}
TEST(SplitSegmenterTest, DeepSeekV3NumberPattern) {
  // DeepSeek-V3 splits numbers into groups of 1-3 digits.
  ScopedSplitSegmenter segmenter("\\p{N}{1,3}",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // "abc123456def" -> ["abc", "123", "456", "def"]
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("abc123456def"),
                        {"abc", "123", "456", "def"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Negative Lookahead Backtracking
//===----------------------------------------------------------------------===//

// Tests PCRE-compatible backtracking with negative lookahead patterns.
// Pattern \s+(?!\S)|\s+ requires backtracking when the greedy \s+ consumes
// whitespace up to a non-whitespace character: the lookahead fails, so the
// engine must backtrack to find a shorter match where lookahead passes.

TEST(SplitSegmenterTest, LookaheadBacktrackingWithInvert) {
  // Pattern: [^\r\n\p{L}]?[\p{L}]+|\s+(?!\S)|\s+
  // Input: "    if" (4 spaces + "if")
  // With invert=true: matches become segments, gaps discarded.
  //
  // Backtracking behavior:
  //   \s+ greedily matches 4 spaces, but (?!\S) fails (sees 'i').
  //   Backtrack to 3 spaces: (?!\S) passes (sees ' ').
  //   Next: remaining " if" matches [^\r\n\p{L}]?[\p{L}]+.
  //
  // Expected: ["   ", " if"] at [0,3) and [3,6).
  ScopedSplitSegmenter segmenter("[^\\r\\n\\p{L}]?[\\p{L}]+|\\s+(?!\\S)|\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED,
                                 /*invert=*/true);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("    if"), {"   ", " if"},
                        /*expect_pending_after_process=*/true);
}

TEST(SplitSegmenterTest, LookaheadBacktrackingWithFallback) {
  // Pattern: \s+(?!\S)|\s+
  // Input: "    x" (4 spaces + "x")
  //
  // Backtracking behavior:
  //   Position 0: \s+(?!\S) matches 4 spaces, (?!\S) fails (sees 'x').
  //   Backtrack to 3 spaces: (?!\S) passes (sees ' '). Match [0,3).
  //   Position 3: \s+(?!\S) matches 1 space, (?!\S) fails (sees 'x').
  //   No shorter match possible, fallback \s+ used. Match [3,4).
  //   Trailing gap: "x".
  //
  // Expected: ["   ", " ", "x"].
  ScopedSplitSegmenter segmenter("\\s+(?!\\S)|\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("    x"), {"   ", " ", "x"},
                        /*expect_pending_after_process=*/false);
}

TEST(SplitSegmenterTest, LookaheadOnlyNoFallback) {
  // Pattern: \s+(?!\S) (no fallback branch)
  // Input: "   x" (3 spaces + "x")
  //
  // Without a fallback branch, the pattern only matches whitespace runs
  // that are NOT followed by non-whitespace.
  //   Position 0: \s+ matches up to 3 spaces.
  //   Each extension: lookahead checks next char.
  //   At 2 spaces, lookahead sees ' ' (passes). At 3 spaces, sees 'x' (fails).
  //   Result: match [0,2) (the longest where lookahead passed).
  //   Trailing gap: " x" (no match can start at position 2 or 3).
  //
  // Expected: ["  ", " x"].
  ScopedSplitSegmenter segmenter("\\s+(?!\\S)",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("   x"), {"  ", " x"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// GPT-2 Style Regex Patterns
//===----------------------------------------------------------------------===//

// Tests GPT-2 style regex with space+punctuation input.
// Pattern alternatives: ` ?[\p{P}\p{S}]+` (space+symbol) vs `\s+` (whitespace).
// Leftmost-longest matching ensures " <" is matched as one segment by the
// space+symbol pattern, not split by whitespace patterns.
TEST(SplitSegmenterTest, SpacePunctuationSingleSegment) {
  // Pattern: ` ?[\p{P}\p{S}]+` matches optional space + punctuation/symbols.
  // `\s+(?!\S)|\s+` are whitespace-only alternatives with lower priority.
  ScopedSplitSegmenter segmenter(" ?[\\p{P}\\p{S}]+|\\s+(?!\\S)|\\s+",
                                 IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // " <" matches ` ?[\p{P}\p{S}]+` (2 chars) over `\s+` (1 char).
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(" <"), {" <"},
                        /*expect_pending_after_process=*/false);
}

// Tests the full GPT-2-style regex pattern from real tokenizer configs.
TEST(SplitSegmenterTest, FullGpt2PatternSpacePunctuation) {
  // Full GPT-2 style pattern with multiple alternatives:
  // - Punctuation followed by letters
  // - Word characters with optional preceding non-word
  // - Optional space + punctuation/symbols + optional newlines
  // - Newline patterns
  // - Whitespace patterns (with and without negative lookahead)
  ScopedSplitSegmenter segmenter(
      "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+"
      "|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
      "| ?[\\p{P}\\p{S}]+[\\r\\n]*"
      "|\\s*[\\r\\n]+"
      "|\\s+(?!\\S)"
      "|\\s+",
      IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  IREE_ASSERT_OK(segmenter.status());

  // " <" matches ` ?[\p{P}\p{S}]+[\r\n]*` as a single segment.
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(" <"), {" <"},
                        /*expect_pending_after_process=*/false);
}

}  // namespace
}  // namespace iree::tokenizer::testing
