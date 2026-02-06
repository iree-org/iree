// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/bert.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/segmenter/segmenter_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ScopedSegmenter;
using testing::ScopedSegmenterState;

//===----------------------------------------------------------------------===//
// Test fixture providing a BERT segmenter for each test.
//===----------------------------------------------------------------------===//

class BertSegmenterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_bert_allocate(
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

TEST_F(BertSegmenterTest, CreateAndDestroy) { EXPECT_NE(segmenter(), nullptr); }

TEST_F(BertSegmenterTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_segmenter_state_size(segmenter());
  EXPECT_GT(state_size, 0u);
  EXPECT_LE(state_size, 128u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(BertSegmenterTest, EmptyInput) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view(""),
                                     /*expected_segments=*/{},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(BertSegmenterTest, ZeroCapacityOutput) {
  testing::TestZeroCapacityOutput(segmenter(),
                                  iree_make_cstring_view("hello world"));
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(BertSegmenterTest, OnlyWhitespace) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("   \t\n   "),
                                     /*expected_segments=*/{},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(BertSegmenterTest, SingleChar) {
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("a"),
                                     /*expected_segments=*/{"a"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, SinglePunctuation) {
  // A single punctuation character is emitted immediately (not pending).
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!"),
                                     /*expected_segments=*/{"!"},
                                     /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// HuggingFace-Validated Behavior
// All expectations verified against:
//   tokenizers.pre_tokenizers.BertPreTokenizer().pre_tokenize_str(input)
//===----------------------------------------------------------------------===//

TEST_F(BertSegmenterTest, HF_BasicSentence) {
  // HuggingFace test from bert.rs: "Hey friend!     How are you?!?"
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("Hey friend!     How are you?!?"),
      /*expected_segments=*/
      {"Hey", "friend", "!", "How", "are", "you", "?", "!", "?"},
      /*expect_pending_after_process=*/false);
}

TEST_F(BertSegmenterTest, HF_Contractions) {
  // Apostrophe is ASCII punctuation (range 33-47).
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("don't stop"),
      /*expected_segments=*/{"don", "'", "t", "stop"},
      /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_SimpleWords) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("hello world"),
                                     /*expected_segments=*/{"hello", "world"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_CommaAndExclamation) {
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("hello, world!"),
      /*expected_segments=*/{"hello", ",", "world", "!"},
      /*expect_pending_after_process=*/false);
}

TEST_F(BertSegmenterTest, HF_LeadingSpaces) {
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("  leading spaces"),
      /*expected_segments=*/{"leading", "spaces"},
      /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_TrailingSpaces) {
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("trailing spaces  "),
      /*expected_segments=*/{"trailing", "spaces"},
      /*expect_pending_after_process=*/false);
}

TEST_F(BertSegmenterTest, HF_DotSeparated) {
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a.b.c"),
                                     /*expected_segments=*/
                                     {"a", ".", "b", ".", "c"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_MixedPunctuation) {
  // $ is ASCII punctuation (range 33-47), . is too.
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("price: $100.00"),
      /*expected_segments=*/{"price", ":", "$", "100", ".", "00"},
      /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_Underscore) {
  // _ is ASCII punctuation (range 91-96).
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("foo_bar"),
                                     /*expected_segments=*/
                                     {"foo", "_", "bar"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_TabAndNewline) {
  // Tab and newline are Unicode whitespace.
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("hello\tworld\nnewline"),
      /*expected_segments=*/{"hello", "world", "newline"},
      /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_AccentedLatin) {
  // Non-ASCII letters are word characters.
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("caf\xc3\xa9 r\xc3\xa9sum\xc3\xa9"),
      /*expected_segments=*/{"caf\xc3\xa9", "r\xc3\xa9sum\xc3\xa9"},
      /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_CJKNotSplit) {
  // CJK characters are NOT split by BertPreTokenizer.
  // CJK isolation is done by the BertNormalizer (handle_chinese_chars=true).
  testing::TestWithAllChunkSizesUtf8(
      segmenter(),
      iree_make_cstring_view(
          "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf"
          "\xe4\xb8\x96\xe7\x95\x8c"),
      /*expected_segments=*/
      {"\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf"
       "\xe4\xb8\x96\xe7\x95\x8c"},
      /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_FullwidthParens) {
  // Fullwidth parentheses are Unicode punctuation (Ps/Pe categories).
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("hello\xef\xbc\x88world\xef\xbc\x89"),
      /*expected_segments=*/{"hello", "\xef\xbc\x88", "world", "\xef\xbc\x89"},
      /*expect_pending_after_process=*/false);
}

TEST_F(BertSegmenterTest, HF_NonBreakingSpace) {
  // U+00A0 (non-breaking space) is Unicode whitespace.
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a\xc2\xa0"
                                                            "b"),
                                     /*expected_segments=*/{"a", "b"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_EmDash) {
  // U+2014 (em dash) is Unicode punctuation (Pd category).
  testing::TestWithAllChunkSizesUtf8(
      segmenter(),
      iree_make_cstring_view("test\xe2\x80\x94"
                             "dash"),
      /*expected_segments=*/{"test", "\xe2\x80\x94", "dash"},
      /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, HF_MiddleDot) {
  // U+00B7 (middle dot) is Unicode punctuation (Po category).
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("x\xc2\xb7y"),
      /*expected_segments=*/{"x", "\xc2\xb7", "y"},
      /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Output Buffer Handling
//===----------------------------------------------------------------------===//

TEST_F(BertSegmenterTest, LimitedOutputCapacity) {
  testing::TestLimitedOutputCapacity(
      segmenter(), iree_make_cstring_view("hello, world! "),
      /*expected_segments=*/{"hello", ",", "world", "!"});
}

TEST_F(BertSegmenterTest, LimitedOutputCapacityWithPending) {
  testing::TestLimitedOutputCapacity(
      segmenter(), iree_make_cstring_view("a.b.c"),
      /*expected_segments=*/{"a", ".", "b", ".", "c"});
}

//===----------------------------------------------------------------------===//
// Additional Punctuation Coverage
//===----------------------------------------------------------------------===//

TEST_F(BertSegmenterTest, AllASCIIPunctuationRanges) {
  // Verify all four ASCII punctuation ranges are handled.
  // Range 33-47: !
  // Range 58-64: @
  // Range 91-96: ^
  // Range 123-126: ~
  testing::TestWithAllChunkSizesUtf8(
      segmenter(), iree_make_cstring_view("a!b@c^d~e"),
      /*expected_segments=*/
      {"a", "!", "b", "@", "c", "^", "d", "~", "e"},
      /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, ConsecutivePunctuation) {
  // Each punctuation character is its own segment, even when adjacent.
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view("a...b"),
                                     /*expected_segments=*/
                                     {"a", ".", ".", ".", "b"},
                                     /*expect_pending_after_process=*/true);
}

TEST_F(BertSegmenterTest, OnlyPunctuation) {
  // All punctuation, no word characters.
  testing::TestWithAllChunkSizesUtf8(segmenter(), iree_make_cstring_view("!@#"),
                                     /*expected_segments=*/
                                     {"!", "@", "#"},
                                     /*expect_pending_after_process=*/false);
}

TEST_F(BertSegmenterTest, PunctuationWithWhitespace) {
  // Whitespace around punctuation is consumed.
  testing::TestWithAllChunkSizesUtf8(segmenter(),
                                     iree_make_cstring_view(" ! @ "),
                                     /*expected_segments=*/{"!", "@"},
                                     /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Offsets: Verify Absolute Offset Reconstruction Matches HuggingFace
//===----------------------------------------------------------------------===//

// Single process + finalize. Process emits "ab" (word) and "!" (punctuation),
// while "cd" (trailing word) is left for finalize.
//
// HuggingFace ground truth:
//   BertPreTokenizer().pre_tokenize_str("ab!cd")
//   → [("ab", (0, 2)), ("!", (2, 3)), ("cd", (3, 5))]
TEST_F(BertSegmenterTest, StreamingOffsets_ProcessAndFinalize) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process "ab!cd". Punctuation terminates "ab" and is emitted immediately.
  // Trailing "cd" is pending.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("ab!cd"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 3u);
  ASSERT_EQ(segment_count, 2u);
  // "ab" at [0, 2), "!" at [2, 3).
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 2u);
  EXPECT_EQ(segments[1].start, 2u);
  EXPECT_EQ(segments[1].end, 3u);
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

// Multi-byte UTF-8 character at chunk boundary forces rewind (consumed=0).
// "aé!" is 4 bytes: 'a'(1) + 'é'(2: 0xC3 0xA9) + '!'(1).
// Feeding 2 bytes splits 'é' across chunks → INCOMPLETE → rewind.
//
// HuggingFace ground truth:
//   BertPreTokenizer().pre_tokenize_str("aé!")
//   → [("aé", (0, 3)), ("!", (3, 4))]  (byte offsets)
TEST_F(BertSegmenterTest, StreamingOffsets_MultiByteCharBoundary) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Feed "a\xC3" (2 bytes) — splits the 2-byte 'é' sequence.
  // The classifier sees 0xC3 needs 2 bytes but only 1 remains → INCOMPLETE.
  // Rewind: consumed=0.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_string_view("a\xC3", 2),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 0u);
  EXPECT_EQ(segment_count, 0u);

  // Finalize with full "aé!" (4 bytes) since nothing was consumed.
  iree_host_size_t position = consumed;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_string_view("a\xC3\xA9!", 4),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 2u);
  // Verify absolute reconstruction matches HF: ("aé", (0, 3)), ("!", (3, 4)).
  EXPECT_EQ(position + segments[0].start, 0u);
  EXPECT_EQ(position + segments[0].end, 3u);
  EXPECT_EQ(position + segments[1].start, 3u);
  EXPECT_EQ(position + segments[1].end, 4u);
}

// Overflow recovery with capacity=1. Each re-entry produces correct absolute
// offsets despite the output buffer forcing incremental emission.
//
// HuggingFace ground truth:
//   BertPreTokenizer().pre_tokenize_str("a.b")
//   → [("a", (0, 1)), (".", (1, 2)), ("b", (2, 3))]
TEST_F(BertSegmenterTest, StreamingOffsets_OverflowRecovery) {
  testing::ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[1];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Feed 1: capacity=1. Emits "a", can't emit "." → consumed stops at ".".
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("a.b"),
      iree_tokenizer_make_segment_output(segments, 1), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 1u);
  ASSERT_EQ(segment_count, 1u);
  // Verify absolute: "a" at [0, 1).
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);

  // Feed 2: re-feed from unconsumed ".b" with capacity=1.
  iree_host_size_t position_feed2 = consumed;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view(".b"),
      iree_tokenizer_make_segment_output(segments, 1), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 1u);
  ASSERT_EQ(segment_count, 1u);
  // Verify absolute reconstruction matches HF: (".", (1, 2)).
  EXPECT_EQ(position_feed2 + segments[0].start, 1u);
  EXPECT_EQ(position_feed2 + segments[0].end, 2u);

  // Feed 3: finalize with remaining "b".
  iree_host_size_t position_finalize = position_feed2 + consumed;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_cstring_view("b"),
      iree_tokenizer_make_segment_output(segments, 1), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Verify absolute reconstruction matches HF: ("b", (2, 3)).
  EXPECT_EQ(position_finalize + segments[0].start, 2u);
  EXPECT_EQ(position_finalize + segments[0].end, 3u);
}

}  // namespace
}  // namespace iree::tokenizer
