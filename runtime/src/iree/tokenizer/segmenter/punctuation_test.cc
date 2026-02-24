// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/punctuation.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/segmenter/segmenter_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ScopedSegmenter;

//===----------------------------------------------------------------------===//
// Test fixture with parameterized behavior.
//===----------------------------------------------------------------------===//

class PunctuationSegmenterTest : public ::testing::Test {
 protected:
  void SetUpWithBehavior(iree_tokenizer_regex_split_behavior_t behavior) {
    iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_punctuation_allocate(
        behavior, iree_allocator_system(), &raw_segmenter));
    segmenter_ = ScopedSegmenter(raw_segmenter);
  }

  iree_tokenizer_segmenter_t* segmenter() { return segmenter_.get(); }

 private:
  ScopedSegmenter segmenter_;
};

//===----------------------------------------------------------------------===//
// ISOLATED (default behavior)
// All expectations verified against:
//   tokenizers.pre_tokenizers.Punctuation(behavior='isolated')
//===----------------------------------------------------------------------===//

class PunctuationIsolatedTest : public PunctuationSegmenterTest {
 protected:
  void SetUp() override {
    SetUpWithBehavior(IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  }
};

TEST_F(PunctuationIsolatedTest, CreateAndDestroy) {
  EXPECT_NE(segmenter(), nullptr);
}

TEST_F(PunctuationIsolatedTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_segmenter_state_size(segmenter());
  EXPECT_GT(state_size, 0u);
  EXPECT_LE(state_size, 128u);
}

TEST_F(PunctuationIsolatedTest, EmptyInput) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view(""),
                                     /*expected_segments=*/{},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, ZeroCapacityOutput) {
  testing::TestZeroCapacityOutput(segmenter(),
                                  iree_make_cstring_view("hello world!"));
}

TEST_F(PunctuationIsolatedTest, HF_BasicSentence) {
  // From HuggingFace punctuation.rs test.
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("Hey friend!     How are you?!?"),
      /*expected_segments=*/
      {"Hey friend", "!", "     How are you", "?", "!", "?"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_NoWhitespaceRemoval) {
  // Whitespace is NOT removed (unlike BertPreTokenizer).
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("hello world"),
                                     /*expected_segments=*/{"hello world"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_DotSeparated) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a.b.c"),
                                     /*expected_segments=*/
                                     {"a", ".", "b", ".", "c"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_LeadingPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("!hello!"),
                                     /*expected_segments=*/{"!", "hello", "!"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_OnlyPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!@#"),
                                     /*expected_segments=*/{"!", "@", "#"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_NoPunctuation) {
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("no punctuation here"),
      /*expected_segments=*/{"no punctuation here"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_ConsecutiveDots) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a...b"),
                                     /*expected_segments=*/
                                     {"a", ".", ".", ".", "b"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_SinglePunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!"),
                                     /*expected_segments=*/{"!"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_SingleWord) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("test"),
                                     /*expected_segments=*/{"test"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_FullwidthParens) {
  // Unicode punctuation (Ps/Pe categories).
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("hello\xef\xbc\x88world\xef\xbc\x89"),
      /*expected_segments=*/{"hello", "\xef\xbc\x88", "world", "\xef\xbc\x89"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_EmDash) {
  // U+2014 (em dash) is Unicode punctuation (Pd category).
  testing::TestWithAllChunkSizesUtf8(
      segmenter(),
      iree_make_cstring_view("test\xe2\x80\x94"
                             "dash"),
      /*expected_segments=*/{"test", "\xe2\x80\x94", "dash"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_MiddleDot) {
  // U+00B7 (middle dot) is Unicode punctuation (Po category).
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("x\xc2\xb7y"),
      /*expected_segments=*/{"x", "\xc2\xb7", "y"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_AccentedNotSplit) {
  // Accented characters are NOT punctuation.
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("caf\xc3\xa9 r\xc3\xa9sum\xc3\xa9"),
      /*expected_segments=*/{"caf\xc3\xa9 r\xc3\xa9sum\xc3\xa9"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_CJKNotSplit) {
  // CJK characters are NOT punctuation.
  testing::TestWithAllChunkSizesUtf8(
      segmenter(),
      iree_make_cstring_view(
          "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf"),
      /*expected_segments=*/
      {"\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, HF_NonBreakingSpaceNotSplit) {
  // U+00A0 (non-breaking space) is NOT punctuation.
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a\xc2\xa0"
                                                            "b"),
                                     /*expected_segments=*/
                                     {"a\xc2\xa0"
                                      "b"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationIsolatedTest, LimitedOutputCapacity) {
  testing::TestLimitedOutputCapacity(
      segmenter(), iree_make_cstring_view("a.b.c"),
      /*expected_segments=*/{"a", ".", "b", ".", "c"});
}

TEST_F(PunctuationIsolatedTest, LimitedOutputCapacityLeadingPunct) {
  testing::TestLimitedOutputCapacity(segmenter(),
                                     iree_make_cstring_view("!hello!"),
                                     /*expected_segments=*/{"!", "hello", "!"});
}

//===----------------------------------------------------------------------===//
// REMOVED behavior
// All expectations verified against:
//   tokenizers.pre_tokenizers.Punctuation(behavior='removed')
//===----------------------------------------------------------------------===//

class PunctuationRemovedTest : public PunctuationSegmenterTest {
 protected:
  void SetUp() override {
    SetUpWithBehavior(IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  }
};

TEST_F(PunctuationRemovedTest, HF_BasicSentence) {
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("Hey friend!     How are you?!?"),
      /*expected_segments=*/{"Hey friend", "     How are you"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationRemovedTest, HF_NoWhitespaceRemoval) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("hello world"),
                                     /*expected_segments=*/{"hello world"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationRemovedTest, HF_DotSeparated) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a.b.c"),
                                     /*expected_segments=*/{"a", "b", "c"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationRemovedTest, HF_LeadingPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("!hello!"),
                                     /*expected_segments=*/{"hello"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationRemovedTest, HF_OnlyPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!@#"),
                                     /*expected_segments=*/{},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationRemovedTest, HF_NoPunctuation) {
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("no punctuation here"),
      /*expected_segments=*/{"no punctuation here"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationRemovedTest, HF_ConsecutiveDots) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a...b"),
                                     /*expected_segments=*/{"a", "b"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationRemovedTest, HF_SinglePunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!"),
                                     /*expected_segments=*/{},
                                     /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// MERGED_WITH_PREVIOUS behavior
// All expectations verified against:
//   tokenizers.pre_tokenizers.Punctuation(behavior='merged_with_previous')
//===----------------------------------------------------------------------===//

class PunctuationMergedWithPreviousTest : public PunctuationSegmenterTest {
 protected:
  void SetUp() override {
    SetUpWithBehavior(IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS);
  }
};

TEST_F(PunctuationMergedWithPreviousTest, HF_BasicSentence) {
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("Hey friend!     How are you?!?"),
      /*expected_segments=*/
      {"Hey friend!", "     How are you?", "!", "?"},
      /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationMergedWithPreviousTest, HF_NoWhitespaceRemoval) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("hello world"),
                                     /*expected_segments=*/{"hello world"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationMergedWithPreviousTest, HF_DotSeparated) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a.b.c"),
                                     /*expected_segments=*/{"a.", "b.", "c"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationMergedWithPreviousTest, HF_LeadingPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("!hello!"),
                                     /*expected_segments=*/{"!", "hello!"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationMergedWithPreviousTest, HF_OnlyPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!@#"),
                                     /*expected_segments=*/{"!", "@", "#"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationMergedWithPreviousTest, HF_ConsecutiveDots) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a...b"),
                                     /*expected_segments=*/
                                     {"a.", ".", ".", "b"},
                                     /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// MERGED_WITH_NEXT behavior
// All expectations verified against:
//   tokenizers.pre_tokenizers.Punctuation(behavior='merged_with_next')
//===----------------------------------------------------------------------===//

class PunctuationMergedWithNextTest : public PunctuationSegmenterTest {
 protected:
  void SetUp() override {
    SetUpWithBehavior(IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT);
  }
};

TEST_F(PunctuationMergedWithNextTest, HF_BasicSentence) {
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("Hey friend!     How are you?!?"),
      /*expected_segments=*/
      {"Hey friend", "!     How are you", "?", "!", "?"},
      /*expect_pending_after_process=*/true);
}

TEST_F(PunctuationMergedWithNextTest, HF_NoWhitespaceRemoval) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("hello world"),
                                     /*expected_segments=*/{"hello world"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationMergedWithNextTest, HF_DotSeparated) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a.b.c"),
                                     /*expected_segments=*/
                                     {"a", ".b", ".c"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(PunctuationMergedWithNextTest, HF_LeadingPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("!hello!"),
                                     /*expected_segments=*/{"!hello", "!"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(PunctuationMergedWithNextTest, HF_OnlyPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!@#"),
                                     /*expected_segments=*/{"!", "@", "#"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(PunctuationMergedWithNextTest, HF_ConsecutiveDots) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a...b"),
                                     /*expected_segments=*/
                                     {"a", ".", ".", ".b"},
                                     /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// CONTIGUOUS behavior
// All expectations verified against:
//   tokenizers.pre_tokenizers.Punctuation(behavior='contiguous')
//===----------------------------------------------------------------------===//

class PunctuationContiguousTest : public PunctuationSegmenterTest {
 protected:
  void SetUp() override {
    SetUpWithBehavior(IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS);
  }
};

TEST_F(PunctuationContiguousTest, HF_BasicSentence) {
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("Hey friend!     How are you?!?"),
      /*expected_segments=*/
      {"Hey friend", "!", "     How are you", "?!?"},
      /*expect_pending_after_process=*/true);
}

TEST_F(PunctuationContiguousTest, HF_NoWhitespaceRemoval) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("hello world"),
                                     /*expected_segments=*/{"hello world"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(PunctuationContiguousTest, HF_DotSeparated) {
  // Non-consecutive dots are separate segments.
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a.b.c"),
                                     /*expected_segments=*/
                                     {"a", ".", "b", ".", "c"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(PunctuationContiguousTest, HF_LeadingPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("!hello!"),
                                     /*expected_segments=*/{"!", "hello", "!"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(PunctuationContiguousTest, HF_OnlyPunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!@#"),
                                     /*expected_segments=*/{"!@#"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(PunctuationContiguousTest, HF_ConsecutiveDots) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a...b"),
                                     /*expected_segments=*/
                                     {"a", "...", "b"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(PunctuationContiguousTest, HF_SinglePunctuation) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!"),
                                     /*expected_segments=*/{"!"},
                                     /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Output buffer handling
//===----------------------------------------------------------------------===//

TEST_F(PunctuationIsolatedTest, LimitedOutputCapacitySentence) {
  testing::TestLimitedOutputCapacity(
      segmenter(), iree_make_cstring_view("Hey friend!     How are you?!?"),
      /*expected_segments=*/
      {"Hey friend", "!", "     How are you", "?", "!", "?"});
}

//===----------------------------------------------------------------------===//
// Streaming Offset Verification
// These tests directly assert on raw segment.start/segment.end values to
// verify correct offset arithmetic, particularly the unsigned wraparound
// used by MERGED_WITH_NEXT and CONTIGUOUS when pending state spans across
// process()/finalize() boundaries.
//===----------------------------------------------------------------------===//

// Isolated: verify process() emits correct relative offsets and finalize()
// handles the trailing gap at offset 0.
TEST_F(PunctuationIsolatedTest, StreamingOffsets_ProcessAndFinalize) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process "a.b.c" (5 bytes).
  // Expected: 4 segments emitted, consumed=4 (trailing "c" left for finalize).
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a.b.c"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 4u);
  ASSERT_EQ(segment_count, 4u);
  // "a" at [0,1), relative to chunk_base=0.
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);
  // "." at [1,2).
  EXPECT_EQ(segments[1].start, 1u);
  EXPECT_EQ(segments[1].end, 2u);
  // "b" at [2,3).
  EXPECT_EQ(segments[2].start, 2u);
  EXPECT_EQ(segments[2].end, 3u);
  // "." at [3,4).
  EXPECT_EQ(segments[3].start, 3u);
  EXPECT_EQ(segments[3].end, 4u);

  // Finalize with remaining "c" (1 byte).
  // remaining_input starts at position=4 in the original.
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view("c"),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // "c" at [0,1) relative to finalize's chunk_base=4.
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);
  // Reconstruct absolute: position=4 + start=0 = 4, position=4 + end=1 = 5.
}

// MergedWithNext: verify finalize() receives trailing text to merge with
// pending punctuation from process().
TEST_F(PunctuationMergedWithNextTest, StreamingOffsets_FinalizeWraparound) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process "a.b" (3 bytes).
  // MergedWithNext: emit "a" [0,1), pending=[1,2) ".".
  // consumed=1 (only "a" emitted), leaving ".b" for finalize.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a.b"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 1u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);

  // Finalize with ".b" as remaining (pending plus trailing text).
  // chunk_base=1, remaining=".b" (2 bytes), input_end=3.
  // MergedWithNext flush: emit [pending_start=1, input_end=3) = ".b".
  // Relative to chunk_base=1: start=0, end=2.
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view(".b"),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Segment is [0,2) relative to remaining_input ".b".
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 2u);
  // Verify absolute position reconstruction.
  size_t position = 1;  // bytes consumed before finalize.
  size_t abs_start = position + segments[0].start;  // 1 + 0 = 1.
  size_t abs_end = position + segments[0].end;      // 1 + 2 = 3.
  EXPECT_EQ(abs_start, 1u);
  EXPECT_EQ(abs_end, 3u);
}

// MergedWithNext: verify offsets when pending state crosses chunk boundaries.
// HuggingFace expected output for "a.b.c": ["a", ".b", ".c"]
// This test simulates proper streaming: unconsumed bytes from previous call
// are prepended to new data for the next call.
TEST_F(PunctuationMergedWithNextTest, StreamingOffsets_CrossChunkPending) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Full input is "a.b.c" (5 bytes). We process in chunks, simulating proper
  // streaming where unconsumed bytes from previous call are prepended to new
  // data for the next call.

  // Chunk 1: "a." (2 bytes). chunk_base=0.
  // pos=0: 'a' not punct. pos=1: '.' punct.
  // MergedWithNext: emit [0,1)="a". pending=[1,2).
  // consumed=1 (only "a" emitted), remaining=".".
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a."),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));
  EXPECT_EQ(consumed, 1u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);

  // Chunk 2: Remaining from chunk 1 + new data = ".b." (3 bytes).
  // chunk_base now at position 1 (where unconsumed "." starts).
  // pos=0: '.' punct (abs 1), emit pending merged with gap → ".b" [1,3).
  // Actually let's trace: pending=[1,2) from chunk1.
  // Input ".b." at position 1: first char '.' at abs_pos=1.
  // It's punct! handle_match: gap=[1,1) empty, match=[1,2).
  // Since gap_end=1 <= segment_start=1 (pending_start), no segment emitted.
  // New pending=[1,2). Continue.
  // pos=1: 'b' at abs_pos=2, not punct.
  // pos=2: '.' at abs_pos=3, punct!
  // gap=[last_emit_end=1, 3)=[1,3). segment_start=pending_start=1.
  // Emit [1,3)=".b". New pending=[3,4).
  // consumed=2 (up to pending_start=3, relative to chunk_base=1 → 3-1=2).
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view(".b."),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));
  EXPECT_EQ(consumed, 2u);
  ASSERT_EQ(segment_count, 1u);
  // Segment is [0,2) relative to this chunk's input ".b.".
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 2u);
  // Absolute: position=1 + start=0 = 1, position=1 + end=2 = 3.
  size_t chunk2_position = 1;
  size_t abs_start = chunk2_position + segments[0].start;
  size_t abs_end = chunk2_position + segments[0].end;
  EXPECT_EQ(abs_start, 1u);
  EXPECT_EQ(abs_end, 3u);

  // Chunk 3: Remaining from chunk 2 + new data = ".c" (2 bytes).
  // chunk_base at position 3. pending=[3,4).
  // Input starts with '.' at pos 0 (abs 3) - this IS the pending punct.
  // 'c' at pos 1 (abs 4) - not punct.
  // No more punct, so no emit during process.
  // consumed=0 (pending_start=3, relative to chunk_base=3 → 3-3=0).
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view(".c"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));
  EXPECT_EQ(consumed, 0u);
  EXPECT_EQ(segment_count, 0u);

  // Finalize with ".c" as remaining. pending=[3,4), input_end=5.
  // MergedWithNext: emit [pending_start=3, input_end=5) = ".c" as [0,2).
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view(".c"),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));
  ASSERT_EQ(finalize_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 2u);
  size_t finalize_position = 3;
  abs_start = finalize_position + segments[0].start;  // 3 + 0 = 3.
  abs_end = finalize_position + segments[0].end;      // 3 + 2 = 5.
  EXPECT_EQ(abs_start, 3u);
  EXPECT_EQ(abs_end, 5u);
}

// Contiguous: verify pending flush and trailing gap offsets in finalize.
// HuggingFace expected output for "a..b": ["a", "..", "b"]
TEST_F(PunctuationContiguousTest,
       StreamingOffsets_PendingFlushWithTrailingGap) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process "a..b" (4 bytes).
  // pos=0: 'a' not punct.
  // pos=1: '.' punct. Contiguous: no pending, emit gap [0,1)="a". pending=[1,2)
  // pos=2: '.' punct. pending_end=2==match_start=2. Extend to [1,3).
  // pos=3: 'b' not punct.
  // consumed=1 (only "a" emitted), leaving "..b" for finalize.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a..b"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));
  EXPECT_EQ(consumed, 1u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);

  // Finalize with "..b" as remaining. chunk_base=1, pending=[1,3), input_end=4.
  // Contiguous flush: emit pending [1,3)=".." as [0,2), then trailing gap "b"
  // as [2,3).
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view("..b"),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));
  ASSERT_EQ(finalize_count, 2u);

  size_t position = 1;
  // Pending ".." at absolute [1,3), relative [0,2).
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 2u);
  size_t abs_start0 = position + segments[0].start;
  size_t abs_end0 = position + segments[0].end;
  EXPECT_EQ(abs_start0, 1u);
  EXPECT_EQ(abs_end0, 3u);
  // Trailing gap "b" at absolute [3,4), relative [2,3).
  EXPECT_EQ(segments[1].start, 2u);
  EXPECT_EQ(segments[1].end, 3u);
  size_t abs_start1 = position + segments[1].start;
  size_t abs_end1 = position + segments[1].end;
  EXPECT_EQ(abs_start1, 3u);
  EXPECT_EQ(abs_end1, 4u);
}

// Isolated: verify overflow recovery produces correct offsets on re-entry.
TEST_F(PunctuationIsolatedTest, StreamingOffsets_OverflowRecovery) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segment;
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process "a.b" with capacity=1.
  // First match: emit "a" [0,1). Second emit "." [1,2) overflows.
  // consumed = last_emitted_end - chunk_base = 1.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a.b"),
      iree_tokenizer_make_segment_output(&segment, 1), &consumed,
      &segment_count));
  EXPECT_EQ(consumed, 1u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segment.start, 0u);
  EXPECT_EQ(segment.end, 1u);

  // Re-entry with ".b" (chunk_base=1). Emits "." [0,1).
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view(".b"),
      iree_tokenizer_make_segment_output(&segment, 1), &consumed,
      &segment_count));
  EXPECT_EQ(consumed, 1u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segment.start, 0u);
  EXPECT_EQ(segment.end, 1u);

  // Re-entry with "b" (chunk_base=2). No punct, consumed=0.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("b"),
      iree_tokenizer_make_segment_output(&segment, 1), &consumed,
      &segment_count));
  EXPECT_EQ(consumed, 0u);
  EXPECT_EQ(segment_count, 0u);

  // Finalize with "b" (chunk_base=2). Emit trailing gap [0,1).
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view("b"),
      iree_tokenizer_make_segment_output(&segment, 1), &finalize_count));
  ASSERT_EQ(finalize_count, 1u);
  EXPECT_EQ(segment.start, 0u);
  EXPECT_EQ(segment.end, 1u);
}

}  // namespace
}  // namespace iree::tokenizer
