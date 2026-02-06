// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/whitespace.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/segmenter/segmenter_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ScopedSegmenter;
using testing::ScopedSegmenterState;

//===----------------------------------------------------------------------===//
// Test fixture providing a whitespace segmenter for each test.
//===----------------------------------------------------------------------===//

class WhitespaceSegmenterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_whitespace_allocate(
        iree_allocator_system(), &raw_segmenter));
    segmenter_ = ScopedSegmenter(raw_segmenter);
  }

  iree_tokenizer_segmenter_t* segmenter() { return segmenter_.get(); }

 private:
  ScopedSegmenter segmenter_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(WhitespaceSegmenterTest, CreateAndDestroy) {
  // Segmenter was created in SetUp, will be destroyed in TearDown.
  // This test verifies the allocation/free path doesn't crash.
  EXPECT_NE(segmenter(), nullptr);
}

TEST_F(WhitespaceSegmenterTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_segmenter_state_size(segmenter());
  // State should exist and be small (whitespace segmenter tracks minimal info).
  EXPECT_GT(state_size, 0u);
  EXPECT_LE(state_size, 64u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(WhitespaceSegmenterTest, EmptyInput) {
  // Empty input should produce no segments and no pending data.
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view(""),
                                 /*expected_segments=*/{},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(WhitespaceSegmenterTest, ZeroCapacityOutput) {
  // Zero capacity output should consume nothing and produce nothing.
  testing::TestZeroCapacityOutput(segmenter(),
                                  iree_make_cstring_view("hello world"));
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(WhitespaceSegmenterTest, OnlyWhitespace) {
  // Input with only whitespace should produce no segments.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("   \t\n   "),
                                 /*expected_segments=*/{},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(WhitespaceSegmenterTest, SingleChar) {
  // Single character input has no terminating whitespace, so it's pending.
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("a"),
                                 /*expected_segments=*/{"a"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, SingleCharWithTrailingWhitespace) {
  // Single character followed by whitespace should emit immediately.
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("a "),
                                 /*expected_segments=*/{"a"},
                                 /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

TEST_F(WhitespaceSegmenterTest, SingleWord) {
  // Single word without trailing whitespace - pending until finalize.
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("hello"),
                                 /*expected_segments=*/{"hello"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, SingleWordWithTrailingWhitespace) {
  // Single word with trailing whitespace - emitted immediately, nothing
  // pending.
  testing::TestWithAllChunkSizes(segmenter(), iree_make_cstring_view("hello "),
                                 /*expected_segments=*/{"hello"},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(WhitespaceSegmenterTest, TwoWords) {
  // Two words: first emitted on whitespace, second pending until finalize.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("hello world"),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, TwoWordsWithTrailingWhitespace) {
  // Two words with trailing whitespace - both emitted, nothing pending.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("hello world "),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(WhitespaceSegmenterTest, ThreeWords) {
  // Three words: first two emitted on whitespace, third pending.
  testing::TestWithAllChunkSizes(
      segmenter(), iree_make_cstring_view("hello beautiful world"),
      /*expected_segments=*/{"hello", "beautiful", "world"},
      /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, ManyWords) {
  // Many words to stress test the segmenter.
  testing::TestWithAllChunkSizes(
      segmenter(),
      iree_make_cstring_view("the quick brown fox jumps over the lazy dog"),
      /*expected_segments=*/
      {"the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"},
      /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
// Note: has_pending() and finalize() are internally verified by test utils.
// These tests cover specific streaming scenarios beyond what
// TestWithAllChunkSizes provides.
//===----------------------------------------------------------------------===//

TEST_F(WhitespaceSegmenterTest, StreamingChunkedAtWordBoundary) {
  // Chunking exactly at word boundaries should work correctly.
  // "hello world" chunked as "hello " + "world"
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("hello world"),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, StreamingChunkedMidWord) {
  // Chunking in the middle of words should work correctly.
  // This is tested by TestWithAllChunkSizes with chunk_size=3:
  // "hello world" as "hel" + "lo " + "wor" + "ld"
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("hello world"),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Output Buffer Handling
//===----------------------------------------------------------------------===//

TEST_F(WhitespaceSegmenterTest, LimitedOutputCapacity) {
  // With capacity=1, should still produce all segments across multiple calls.
  testing::TestLimitedOutputCapacity(
      segmenter(), iree_make_cstring_view("one two three four "),
      /*expected_segments=*/{"one", "two", "three", "four"});
}

TEST_F(WhitespaceSegmenterTest, LimitedOutputCapacityWithPending) {
  // With capacity=1 and no trailing whitespace, last word comes from finalize.
  testing::TestLimitedOutputCapacity(
      segmenter(), iree_make_cstring_view("one two three"),
      /*expected_segments=*/{"one", "two", "three"});
}

//===----------------------------------------------------------------------===//
// Whitespace Variations
//===----------------------------------------------------------------------===//

TEST_F(WhitespaceSegmenterTest, TabSeparator) {
  // Tab should work as word separator.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("hello\tworld"),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, NewlineSeparator) {
  // Newline should work as word separator.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("hello\nworld\n"),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(WhitespaceSegmenterTest, CarriageReturnSeparator) {
  // Carriage return should work as word separator.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("hello\rworld"),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, MixedWhitespace) {
  // Mixed whitespace characters should all work as separators.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("a\t\n\r b"),
                                 /*expected_segments=*/{"a", "b"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, LeadingWhitespace) {
  // Leading whitespace should be consumed without producing empty segments.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("  hello world"),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, TrailingWhitespace) {
  // Trailing whitespace completes the last word, nothing pending.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("hello world  "),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(WhitespaceSegmenterTest, LeadingAndTrailingWhitespace) {
  // Both leading and trailing whitespace.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("  hello world  "),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/false);
}

TEST_F(WhitespaceSegmenterTest, MultipleConsecutiveWhitespace) {
  // Multiple consecutive whitespace characters should not produce empty
  // segments.
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("hello    world"),
                                 /*expected_segments=*/{"hello", "world"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, UnicodeContent) {
  // Non-ASCII content should be preserved as segments (whitespace is ASCII).
  testing::TestWithAllChunkSizes(segmenter(),
                                 iree_make_cstring_view("こんにちは 世界"),
                                 /*expected_segments=*/{"こんにちは", "世界"},
                                 /*expect_pending_after_process=*/true);
}

TEST_F(WhitespaceSegmenterTest, MixedAsciiAndUnicode) {
  // Mixed ASCII and Unicode with various whitespace.
  testing::TestWithAllChunkSizes(
      segmenter(), iree_make_cstring_view("hello 世界 world"),
      /*expected_segments=*/{"hello", "世界", "world"},
      /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Streaming Offsets: Verify Absolute Offset Reconstruction Matches HuggingFace
//===----------------------------------------------------------------------===//

// Single process + finalize. Process emits "ab" immediately (whitespace
// terminates it), finalize emits the trailing "cd".
//
// HuggingFace ground truth:
//   WhitespaceSplit().pre_tokenize_str("ab cd")
//   → [("ab", (0, 2)), ("cd", (3, 5))]
TEST_F(WhitespaceSegmenterTest, StreamingOffsets_ProcessAndFinalize) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process full input. "ab" is emitted (terminated by space), "cd" is pending.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("ab cd"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 3u);
  ASSERT_EQ(segment_count, 1u);
  // Verify absolute: "ab" at [0, 2).
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 2u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Finalize with remaining "cd".
  iree_host_size_t position = consumed;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view("cd"),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Verify absolute reconstruction matches HF: ("cd", (3, 5)).
  EXPECT_EQ(position + segments[0].start, 3u);
  EXPECT_EQ(position + segments[0].end, 5u);
}

// Multi-chunk streaming. The unconsumed tail from each process() is re-fed
// concatenated with new data. Verifies offsets stay correct across re-entry.
//
// HuggingFace ground truth:
//   WhitespaceSplit().pre_tokenize_str("ab cd ef")
//   → [("ab", (0, 2)), ("cd", (3, 5)), ("ef", (6, 8))]
TEST_F(WhitespaceSegmenterTest, StreamingOffsets_MultiChunkPending) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Feed 1: "ab cd" (first 5 bytes). Emits "ab", "cd" starts pending.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("ab cd"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 3u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 2u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Feed 2: remaining "cd" + new " ef". The segmenter sees this as continuation
  // since bytes_processed tracks the absolute position.
  iree_host_size_t position_feed2 = consumed;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("cd ef"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 3u);
  ASSERT_EQ(segment_count, 1u);
  // Verify absolute reconstruction matches HF: ("cd", (3, 5)).
  EXPECT_EQ(position_feed2 + segments[0].start, 3u);
  EXPECT_EQ(position_feed2 + segments[0].end, 5u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Finalize with remaining "ef".
  iree_host_size_t position_finalize = position_feed2 + consumed;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view("ef"),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Verify absolute reconstruction matches HF: ("ef", (6, 8)).
  EXPECT_EQ(position_finalize + segments[0].start, 6u);
  EXPECT_EQ(position_finalize + segments[0].end, 8u);
}

// Overflow recovery with capacity=1. Each re-entry must produce correct
// absolute offsets despite the output buffer forcing incremental emission.
//
// HuggingFace ground truth:
//   WhitespaceSplit().pre_tokenize_str("a b c")
//   → [("a", (0, 1)), ("b", (2, 3)), ("c", (4, 5))]
TEST_F(WhitespaceSegmenterTest, StreamingOffsets_OverflowRecovery) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[1];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Feed 1: capacity=1. Emits "a", can't emit "b" → consumed stops before "b".
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a b c"),
      iree_tokenizer_make_segment_output(segments, 1), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 2u);
  ASSERT_EQ(segment_count, 1u);
  // Verify absolute: "a" at [0, 1).
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);

  // Feed 2: re-feed from unconsumed position. "b c" with capacity=1.
  iree_host_size_t position_feed2 = consumed;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("b c"),
      iree_tokenizer_make_segment_output(segments, 1), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 2u);
  ASSERT_EQ(segment_count, 1u);
  // Verify absolute reconstruction matches HF: ("b", (2, 3)).
  EXPECT_EQ(position_feed2 + segments[0].start, 2u);
  EXPECT_EQ(position_feed2 + segments[0].end, 3u);

  // Feed 3: finalize with remaining "c".
  iree_host_size_t position_finalize = position_feed2 + consumed;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view("c"),
      iree_tokenizer_make_segment_output(segments, 1), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Verify absolute reconstruction matches HF: ("c", (4, 5)).
  EXPECT_EQ(position_finalize + segments[0].start, 4u);
  EXPECT_EQ(position_finalize + segments[0].end, 5u);
}

}  // namespace
}  // namespace iree::tokenizer
