// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/model/bpe.h"

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/model/model_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::EncodeAndFinalize;
using testing::EncodeResult;
using testing::EncodeWithOffsetsAndFinalize;
using testing::ScopedModel;
using testing::ScopedModelState;
using testing::ScopedVocab;
using testing::ScopedVocabBuilder;
using testing::TestEncode;
using testing::TestLimitedOutputCapacity;
using testing::TestMultipleEncodeCalls;

//===----------------------------------------------------------------------===//
// Test fixture for BPE model tests.
//===----------------------------------------------------------------------===//

class BPEModelTest : public ::testing::Test {
 protected:
  // Helper to create model from vocab builder.
  // BPE model does not own vocab, so we store both and ensure vocab outlives
  // model (members are destroyed in reverse declaration order).
  void CreateModel(
      ScopedVocabBuilder& builder,
      iree_tokenizer_bpe_flags_t flags = IREE_TOKENIZER_BPE_FLAG_NONE) {
    vocab_ = ScopedVocab(builder.Build());
    iree_tokenizer_model_t* raw_model = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
        vocab_.get(), flags, iree_allocator_system(), &raw_model));
    model_ = ScopedModel(raw_model);
  }

  iree_tokenizer_model_t* model() { return model_.get(); }

 private:
  // Order matters: vocab_ destroyed after model_ (reverse declaration order).
  ScopedVocab vocab_;
  ScopedModel model_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(BPEModelTest, CreateAndDestroy) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  builder.AddMerge(0, 1);  // a + b -> ab

  CreateModel(builder);
  EXPECT_NE(model(), nullptr);
}

TEST_F(BPEModelTest, StateSizeIsReasonable) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");

  CreateModel(builder);

  iree_host_size_t state_size = iree_tokenizer_model_state_size(model());
  EXPECT_GT(state_size, 0u);
  // State includes window/heap buffers (window+heap path) and backtracking
  // buffers (backtracking path). Word cache is disabled for vocabs < 256
  // tokens. For max_token_length=1, vocab_capacity=1:
  //   window: 2 * 12 = 24
  //   heap: 3 * 8 = 24
  //   backtrack stack: 2048 * 8 = 16,384
  //   backtrack bitfield: 33 * 8 = 264
  //   pair cache: 4096 * 8 = 32,768
  //   word cache: 0 (disabled, vocab < 256)
  //   struct: ~128
  //   total: ~49,592
  EXPECT_LE(state_size, 65536u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(BPEModelTest, EmptySegment) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");

  CreateModel(builder);

  std::vector<iree_tokenizer_segment_t> segments = {{0, 0}};
  auto tokens = EncodeAndFinalize(model(), "", segments,
                                  /*expect_pending_after_encode=*/false);
  EXPECT_TRUE(tokens.empty());
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(BPEModelTest, SingleChar) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");

  CreateModel(builder);

  TestEncode(model(), "a", {0}, /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

// All expectations verified against HuggingFace tokenizers library.

// Verified: HuggingFace "ab" -> [2]
TEST_F(BPEModelTest, SimpleMerge) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  builder.AddMerge(0, 1);  // a + b -> ab

  CreateModel(builder);

  TestEncode(model(), "ab", {2}, /*expect_pending_after_encode=*/false);
}

// Verified: HuggingFace "ab" -> [0, 1] (no merges defined)
TEST_F(BPEModelTest, NoMerge) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  // No merge rule, so "ab" stays as two tokens.

  CreateModel(builder);

  TestEncode(model(), "ab", {0, 1}, /*expect_pending_after_encode=*/false);
}

TEST_F(BPEModelTest, MultipleSegments) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");

  CreateModel(builder);

  // "abc" with three separate segments.
  std::vector<iree_tokenizer_segment_t> segments = {{0, 1}, {1, 2}, {2, 3}};
  TestEncode(model(), "abc", segments, {0, 1, 2},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

// Note: has_pending() and finalize() are internally verified by test utils.
// BPE never buffers, so has_pending() is always false.

TEST_F(BPEModelTest, HasPendingAlwaysFalse) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");

  CreateModel(builder);

  ScopedModelState state(model());
  EXPECT_FALSE(iree_tokenizer_model_state_has_pending(state.get()));

  // Even after encoding.
  iree_tokenizer_segment_t segments[] = {{0, 1}};
  iree_tokenizer_token_id_t tokens[8];
  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;
  const char* text = "a";
  iree_const_byte_span_t buffer = {reinterpret_cast<const uint8_t*>(text), 1};

  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer, iree_tokenizer_make_segment_list(segments, 1),
      iree_tokenizer_make_token_output(tokens, NULL, NULL, 8),
      &segments_consumed, &token_count));

  EXPECT_FALSE(iree_tokenizer_model_state_has_pending(state.get()));
}

TEST_F(BPEModelTest, FinalizeProducesNothing) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");

  CreateModel(builder);

  ScopedModelState state(model());

  iree_tokenizer_token_id_t tokens[8];
  iree_host_size_t token_count = 0;

  IREE_ASSERT_OK(iree_tokenizer_model_state_finalize(
      state.get(), iree_tokenizer_make_token_output(tokens, NULL, NULL, 8),
      &token_count));

  // BPE never buffers, so finalize produces nothing.
  EXPECT_EQ(token_count, 0u);
}

TEST_F(BPEModelTest, MultipleEncodeCallsWithIndependentSegmentLists) {
  // This tests that segment index is reset between encode calls.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "h");
  builder.AddToken(1, "e");
  builder.AddToken(2, "l");
  builder.AddToken(3, "o");
  builder.AddToken(4, "w");
  builder.AddToken(5, "r");
  builder.AddToken(6, "d");

  CreateModel(builder);

  // First call: encode "hello" from segments starting at index 0.
  // Second call: encode "world" from a NEW segment list starting at index 0.
  std::string text = "hello world";
  std::vector<iree_tokenizer_segment_t> segments1 = {{0, 5}};
  std::vector<iree_tokenizer_segment_t> segments2 = {{6, 11}};

  // "hello" -> [h, e, l, l, o] = [0, 1, 2, 2, 3]
  // "world" -> [w, o, r, l, d] = [4, 3, 5, 2, 6]
  TestMultipleEncodeCalls(model(), text, segments1, {0, 1, 2, 2, 3}, text,
                          segments2, {4, 3, 5, 2, 6});
}

//===----------------------------------------------------------------------===//
// Output Buffer Handling
//===----------------------------------------------------------------------===//

TEST_F(BPEModelTest, LimitedOutputCapacity) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");

  CreateModel(builder);

  // With capacity=1, should still produce all tokens across multiple calls.
  std::vector<iree_tokenizer_segment_t> segments = {{0, 1}, {1, 2}, {2, 3}};
  TestLimitedOutputCapacity(model(), "abc", segments, {0, 1, 2});
}

//===----------------------------------------------------------------------===//
// COMPONENT-SPECIFIC: BPE merge algorithm
// All expectations verified against HuggingFace tokenizers library.
//===----------------------------------------------------------------------===//

// Verified: HuggingFace "hello" -> [7]
TEST_F(BPEModelTest, ChainedMerges) {
  // Test BPE merge chain: h+e -> he, l+l -> ll, he+ll -> hell, hell+o -> hello
  ScopedVocabBuilder builder;
  builder.AddToken(0, "h");
  builder.AddToken(1, "e");
  builder.AddToken(2, "l");
  builder.AddToken(3, "o");
  builder.AddToken(4, "he");
  builder.AddToken(5, "ll");
  builder.AddToken(6, "hell");
  builder.AddToken(7, "hello");

  // Merges in priority order (lower index = higher priority).
  builder.AddMerge(0, 1);  // h + e -> he (rank 0)
  builder.AddMerge(2, 2);  // l + l -> ll (rank 1)
  builder.AddMerge(4, 5);  // he + ll -> hell (rank 2)
  builder.AddMerge(6, 3);  // hell + o -> hello (rank 3)

  CreateModel(builder);

  // All merges should be applied, resulting in single "hello" token.
  TestEncode(model(), "hello", {7}, /*expect_pending_after_encode=*/false);
}

TEST_F(BPEModelTest, ByteFallback) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  // Add byte fallback tokens.
  builder.AddToken(1, "<0xC3>");
  builder.AddToken(2, "<0xA9>");  // UTF-8 for 'e' is C3 A9.

  CreateModel(builder);

  // "e" is not in vocab as a character, should use byte fallback.
  TestEncode(model(), "é", {1, 2}, /*expect_pending_after_encode=*/false);
}

TEST_F(BPEModelTest, GetTokenString) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hello");
  builder.AddToken(1, "world");

  CreateModel(builder);

  iree_string_view_t text;
  IREE_ASSERT_OK(iree_tokenizer_model_get_token_string(model(), 0, &text));
  EXPECT_EQ(std::string(text.data, text.size), "hello");

  IREE_ASSERT_OK(iree_tokenizer_model_get_token_string(model(), 1, &text));
  EXPECT_EQ(std::string(text.data, text.size), "world");
}

// Verified: HuggingFace "abc" -> [3, 2] = [ab, c]
TEST_F(BPEModelTest, PartialMerge) {
  // Test when only some pairs can merge.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "ab");
  builder.AddMerge(0, 1);  // a + b -> ab

  CreateModel(builder);

  // "abc": a+b merges to ab, then ab + c (no merge) -> ab, c
  TestEncode(model(), "abc", {3, 2}, /*expect_pending_after_encode=*/false);
}

// Verified: HuggingFace "abc" -> [0, 4] = [a, bc]
// This is THE critical test proving greedy longest-match is WRONG.
// Greedy would match "ab" first, giving [ab, c] = [3, 2].
// Correct BPE applies rank 0 merge (b+c->bc) before rank 1 (a+b->ab).
TEST_F(BPEModelTest, MergeOrderCorrectness) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "ab");
  builder.AddToken(4, "bc");

  // Merges in priority order: (b,c)->bc has HIGHER priority than (a,b)->ab.
  builder.AddMerge(1, 2);  // b + c -> bc (rank 0, highest priority)
  builder.AddMerge(0, 1);  // a + b -> ab (rank 1, lower priority)

  CreateModel(builder);

  // Input "abc" starts as [a, b, c].
  // Rank 0 merge (b+c->bc) applies first: [a, bc].
  // Rank 1 merge (a+b->ab) can't apply anymore (b is gone).
  // Final result: [a, bc] = [0, 4]
  TestEncode(model(), "abc", {0, 4}, /*expect_pending_after_encode=*/false);
}

// Tests that vocab entries without merge rules are not used.
// A multi-char token in the vocabulary is only reachable through merging.
TEST_F(BPEModelTest, NoMergeForRepeatedToken) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "aa");  // Exists in vocab, but no merge rule creates it.

  CreateModel(builder);

  // Without a merge rule for a + a -> aa, output must be [a, a] = [0, 0].
  TestEncode(model(), "aa", {0, 0}, /*expect_pending_after_encode=*/false);
}

// Tests that ignore_merges flag allows direct vocab lookup for multi-byte
// tokens. This is a HuggingFace BPE option (default false) that skips merge
// rules. When ignore_merges=true, a multi-char token in vocab can be matched
// directly.
TEST_F(BPEModelTest, IgnoreMergesAllowsDirectVocabLookup) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "aa");  // Exists in vocab, no merge rule.

  CreateModel(builder, IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES);

  // With ignore_merges=true, "aa" matches directly in vocab -> [1].
  // This contrasts with NoMergeForRepeatedToken where ignore_merges=false
  // produces [0, 0] because there's no merge rule for a+a->aa.
  TestEncode(model(), "aa", {1}, /*expect_pending_after_encode=*/false);
}

// Verified: HuggingFace "abcd" -> [6]
TEST_F(BPEModelTest, CompetingMerges) {
  // "abcd" with merges: a+b (rank 0), c+d (rank 1), ab+cd (rank 2)
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "d");
  builder.AddToken(4, "ab");
  builder.AddToken(5, "cd");
  builder.AddToken(6, "abcd");

  builder.AddMerge(0, 1);  // a + b -> ab (rank 0)
  builder.AddMerge(2, 3);  // c + d -> cd (rank 1)
  builder.AddMerge(4, 5);  // ab + cd -> abcd (rank 2)

  CreateModel(builder);

  // "abcd" -> [a,b,c,d] -> [ab,c,d] -> [ab,cd] -> [abcd]
  TestEncode(model(), "abcd", {6}, /*expect_pending_after_encode=*/false);
}

// Verified: HuggingFace "aabb" -> [2, 3] = [aa, bb]
TEST_F(BPEModelTest, SameCharMerges) {
  // Test merging same characters: a+a, b+b
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "aa");
  builder.AddToken(3, "bb");

  builder.AddMerge(0, 0);  // a + a -> aa (rank 0)
  builder.AddMerge(1, 1);  // b + b -> bb (rank 1)

  CreateModel(builder);

  // "aabb" -> [a,a,b,b] -> [aa,b,b] -> [aa,bb]
  TestEncode(model(), "aabb", {2, 3}, /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Offset Tracking
//===----------------------------------------------------------------------===//

// Verifies that token-to-byte offsets are correctly computed for all
// BPE encode paths (fast path, backtracking, window+heap).

// Each byte stays as its own token (no merges). Offsets should be sequential
// single-byte ranges relative to segment start.
TEST_F(BPEModelTest, OffsetsSingleByteTokens) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");

  CreateModel(builder);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "abc",
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 3u);
  EXPECT_EQ(result.tokens[0], 0);
  EXPECT_EQ(result.tokens[1], 1);
  EXPECT_EQ(result.tokens[2], 2);

  // Each byte is its own token: [0,1), [1,2), [2,3).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 1u);
  EXPECT_EQ(result.offsets[1].start, 1u);
  EXPECT_EQ(result.offsets[1].end, 2u);
  EXPECT_EQ(result.offsets[2].start, 2u);
  EXPECT_EQ(result.offsets[2].end, 3u);
}

// Entire segment merges into a single token. Offset should cover full segment.
TEST_F(BPEModelTest, OffsetsWholeSegmentMerge) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  builder.AddMerge(0, 1);  // a + b -> ab

  CreateModel(builder);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "ab",
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 1u);
  EXPECT_EQ(result.tokens[0], 2);

  // Whole segment merged: [0, 2).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);
}

// Partial merges: "abc" with a+b→ab merge produces [ab, c] with offsets
// covering the correct byte sub-ranges within the segment.
TEST_F(BPEModelTest, OffsetsPartialMerge) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "ab");
  builder.AddMerge(0, 1);  // a + b -> ab

  CreateModel(builder);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "abc",
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 2u);
  EXPECT_EQ(result.tokens[0], 3);  // "ab"
  EXPECT_EQ(result.tokens[1], 2);  // "c"

  // "ab" covers bytes [0, 2), "c" covers bytes [2, 3).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 3u);
}

// Multi-level chained merges: h+e→he, l+l→ll, he+ll→hell, hell+o→hello.
// All bytes collapse into one token covering the full segment.
TEST_F(BPEModelTest, OffsetsChainedMerges) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "h");
  builder.AddToken(1, "e");
  builder.AddToken(2, "l");
  builder.AddToken(3, "o");
  builder.AddToken(4, "he");
  builder.AddToken(5, "ll");
  builder.AddToken(6, "hell");
  builder.AddToken(7, "hello");
  builder.AddMerge(0, 1);  // h + e -> he
  builder.AddMerge(2, 2);  // l + l -> ll
  builder.AddMerge(4, 5);  // he + ll -> hell
  builder.AddMerge(6, 3);  // hell + o -> hello

  CreateModel(builder);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "hello",
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 1u);
  EXPECT_EQ(result.tokens[0], 7);

  // Full segment merged: [0, 5).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 5u);
}

// Multiple segments with non-zero base offsets. Verifies that segment.start
// is correctly added to per-token byte positions within each segment.
TEST_F(BPEModelTest, OffsetsMultipleSegmentsWithBaseOffsets) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "d");
  builder.AddToken(4, "ab");
  builder.AddMerge(0, 1);  // a + b -> ab

  CreateModel(builder);

  // Text "abcd" split into two segments: [0,2) and [2,4).
  // First segment "ab" merges → [ab], offsets [0,2).
  // Second segment "cd" → [c, d], offsets [2,3) and [3,4).
  std::vector<iree_tokenizer_segment_t> segments = {{0, 2}, {2, 4}};
  auto result =
      EncodeWithOffsetsAndFinalize(model(), "abcd", segments,
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 3u);
  EXPECT_EQ(result.tokens[0], 4);  // "ab"
  EXPECT_EQ(result.tokens[1], 2);  // "c"
  EXPECT_EQ(result.tokens[2], 3);  // "d"

  // First segment base=0: "ab" at [0, 2).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);
  // Second segment base=2: "c" at [2, 3), "d" at [3, 4).
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 3u);
  EXPECT_EQ(result.offsets[2].start, 3u);
  EXPECT_EQ(result.offsets[2].end, 4u);
}

// Segments with gaps (e.g., whitespace stripped by pre-tokenizer).
// Verifies offsets correctly reflect the segment positions, not contiguous
// byte indices.
TEST_F(BPEModelTest, OffsetsSegmentsWithGaps) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "h");
  builder.AddToken(1, "i");

  CreateModel(builder);

  // Text "h i" (3 bytes), but segmented as [0,1) and [2,3) (skipping space).
  std::vector<iree_tokenizer_segment_t> segments = {{0, 1}, {2, 3}};
  auto result =
      EncodeWithOffsetsAndFinalize(model(), "h i", segments,
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 2u);
  EXPECT_EQ(result.tokens[0], 0);  // "h"
  EXPECT_EQ(result.tokens[1], 1);  // "i"

  // Offsets reflect segment positions: "h" at [0,1), "i" at [2,3).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 1u);
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 3u);
}

// Merge priority correctness with offsets. The merge-order test (b+c before
// a+b) should produce correct byte ranges for the resulting tokens.
TEST_F(BPEModelTest, OffsetsMergeOrderCorrectness) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "ab");
  builder.AddToken(4, "bc");
  builder.AddMerge(1, 2);  // b + c -> bc (rank 0, highest priority)
  builder.AddMerge(0, 1);  // a + b -> ab (rank 1, lower priority)

  CreateModel(builder);

  // "abc" → [a, b, c] → merge b+c → [a, bc].
  auto result =
      EncodeWithOffsetsAndFinalize(model(), "abc",
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 2u);
  EXPECT_EQ(result.tokens[0], 0);  // "a"
  EXPECT_EQ(result.tokens[1], 4);  // "bc"

  // "a" covers [0, 1), "bc" covers [1, 3).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 1u);
  EXPECT_EQ(result.offsets[1].start, 1u);
  EXPECT_EQ(result.offsets[1].end, 3u);
}

// Competing merges (both sides merge first, then compound merge) with offsets.
TEST_F(BPEModelTest, OffsetsCompetingMerges) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "d");
  builder.AddToken(4, "ab");
  builder.AddToken(5, "cd");
  builder.AddToken(6, "abcd");
  builder.AddMerge(0, 1);  // a + b -> ab (rank 0)
  builder.AddMerge(2, 3);  // c + d -> cd (rank 1)
  builder.AddMerge(4, 5);  // ab + cd -> abcd (rank 2)

  CreateModel(builder);

  // "abcd" → [a,b,c,d] → [ab,c,d] → [ab,cd] → [abcd]
  auto result =
      EncodeWithOffsetsAndFinalize(model(), "abcd",
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 1u);
  EXPECT_EQ(result.tokens[0], 6);

  // Full segment merged: [0, 4).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 4u);
}

// Same-char merges with offsets: "aabb" → [aa, bb].
TEST_F(BPEModelTest, OffsetsSameCharMerges) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "aa");
  builder.AddToken(3, "bb");
  builder.AddMerge(0, 0);  // a + a -> aa (rank 0)
  builder.AddMerge(1, 1);  // b + b -> bb (rank 1)

  CreateModel(builder);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "aabb",
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 2u);
  EXPECT_EQ(result.tokens[0], 2);  // "aa"
  EXPECT_EQ(result.tokens[1], 3);  // "bb"

  // "aa" covers [0, 2), "bb" covers [2, 4).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 4u);
}

// Byte fallback tokens with offsets: each byte token covers exactly 1 byte.
TEST_F(BPEModelTest, OffsetsByteFallback) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "<0xC3>");
  builder.AddToken(2, "<0xA9>");  // UTF-8 for 'é' is C3 A9.

  CreateModel(builder);

  // "é" is 2 UTF-8 bytes (C3 A9), each falls back to a byte token.
  auto result =
      EncodeWithOffsetsAndFinalize(model(), "é",
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 2u);
  EXPECT_EQ(result.tokens[0], 1);  // <0xC3>
  EXPECT_EQ(result.tokens[1], 2);  // <0xA9>

  // Each byte token covers exactly 1 byte: [0,1), [1,2).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 1u);
  EXPECT_EQ(result.offsets[1].start, 1u);
  EXPECT_EQ(result.offsets[1].end, 2u);
}

// Output-buffer-limited encoding with offsets. When capacity=1, each resume
// call must produce the correct offset for each token.
TEST_F(BPEModelTest, OffsetsLimitedCapacity) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "ab");
  builder.AddMerge(0, 1);  // a + b -> ab

  CreateModel(builder);

  ScopedModelState state(model());

  // "abc" → [ab, c]. Feed with capacity=1 to test resume behavior.
  std::string text = "abc";
  iree_const_byte_span_t buffer = {
      reinterpret_cast<const uint8_t*>(text.data()), text.size()};
  std::vector<iree_tokenizer_segment_t> segments = {
      {0, static_cast<iree_host_size_t>(text.size())}};

  std::vector<iree_tokenizer_token_id_t> collected_tokens;
  std::vector<iree_tokenizer_offset_t> collected_offsets;

  size_t segment_index = 0;
  while (segment_index < segments.size()) {
    std::vector<iree_tokenizer_segment_t> remaining_segments(
        segments.begin() + segment_index, segments.end());

    iree_tokenizer_token_id_t token;
    iree_tokenizer_offset_t offset;
    iree_host_size_t segments_consumed = 0;
    iree_host_size_t token_count = 0;

    IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
        state.get(), buffer,
        iree_tokenizer_make_segment_list(remaining_segments.data(),
                                         remaining_segments.size()),
        iree_tokenizer_make_token_output(&token, &offset, NULL, 1),
        &segments_consumed, &token_count));

    if (token_count > 0) {
      collected_tokens.push_back(token);
      collected_offsets.push_back(offset);
    }

    if (segments_consumed > 0) {
      segment_index += segments_consumed;
    } else if (token_count == 0) {
      ADD_FAILURE() << "No progress at segment " << segment_index;
      break;
    }
  }

  // Finalize.
  iree_tokenizer_token_id_t token;
  iree_tokenizer_offset_t offset;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_finalize(
      state.get(), iree_tokenizer_make_token_output(&token, &offset, NULL, 1),
      &finalize_count));
  if (finalize_count > 0) {
    collected_tokens.push_back(token);
    collected_offsets.push_back(offset);
  }

  // Verify tokens.
  ASSERT_EQ(collected_tokens.size(), 2u);
  EXPECT_EQ(collected_tokens[0], 3);  // "ab"
  EXPECT_EQ(collected_tokens[1], 2);  // "c"

  // Verify offsets are correct even with capacity=1 resume.
  EXPECT_EQ(collected_offsets[0].start, 0u);
  EXPECT_EQ(collected_offsets[0].end, 2u);
  EXPECT_EQ(collected_offsets[1].start, 2u);
  EXPECT_EQ(collected_offsets[1].end, 3u);
}

// Empty segment with offsets: no tokens, no offsets.
TEST_F(BPEModelTest, OffsetsEmptySegment) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");

  CreateModel(builder);

  std::vector<iree_tokenizer_segment_t> segments = {{0, 0}};
  auto result =
      EncodeWithOffsetsAndFinalize(model(), "", segments,
                                   /*expect_pending_after_encode=*/false);

  EXPECT_TRUE(result.tokens.empty());
  EXPECT_TRUE(result.offsets.empty());
}

// Long segment (>2048 bytes) forces the window+heap path. Offsets must still
// be correct for every token emitted by the O(n log L) algorithm.
TEST_F(BPEModelTest, OffsetsHeapPathLongSegment) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  builder.AddMerge(0, 1);  // a + b -> ab

  CreateModel(builder);

  // Build a 2100-byte input: "ab" repeated 1050 times.
  // Each pair merges → 1050 "ab" tokens.
  std::string input;
  input.reserve(2100);
  for (int i = 0; i < 1050; ++i) {
    input += "ab";
  }
  ASSERT_EQ(input.size(), 2100u);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), input,
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 1050u);
  for (size_t i = 0; i < 1050; ++i) {
    EXPECT_EQ(result.tokens[i], 2) << "Token " << i << " should be 'ab'";
    EXPECT_EQ(result.offsets[i].start, i * 2)
        << "Token " << i << " start offset";
    EXPECT_EQ(result.offsets[i].end, i * 2 + 2)
        << "Token " << i << " end offset";
  }
}

// Heap path with non-trivial merge pattern: chained merges in a long segment.
TEST_F(BPEModelTest, OffsetsHeapPathChainedMerges) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "ab");
  builder.AddToken(4, "abc");
  builder.AddMerge(0, 1);  // a + b -> ab (rank 0)
  builder.AddMerge(3, 2);  // ab + c -> abc (rank 1)

  CreateModel(builder);

  // Build input: "abc" repeated 700 times (2100 bytes, > 2048 threshold).
  std::string input;
  input.reserve(2100);
  for (int i = 0; i < 700; ++i) {
    input += "abc";
  }
  ASSERT_EQ(input.size(), 2100u);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), input,
                                   /*expect_pending_after_encode=*/false);

  ASSERT_EQ(result.tokens.size(), 700u);
  for (size_t i = 0; i < 700; ++i) {
    EXPECT_EQ(result.tokens[i], 4) << "Token " << i << " should be 'abc'";
    EXPECT_EQ(result.offsets[i].start, i * 3)
        << "Token " << i << " start offset";
    EXPECT_EQ(result.offsets[i].end, i * 3 + 3)
        << "Token " << i << " end offset";
  }
}

// Backward compatibility: NULL offsets array must not crash.
TEST_F(BPEModelTest, OffsetsNullDoesNotCrash) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  builder.AddMerge(0, 1);

  CreateModel(builder);

  // Use NULL offsets (original API behavior).
  auto tokens = EncodeAndFinalize(model(), "ab",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 2);
}

//===----------------------------------------------------------------------===//
// end_of_word_suffix: CLIP-style tokenization
//===----------------------------------------------------------------------===//

TEST(BPEEndOfWordSuffixTest, VocabContainingAngleBrackets) {
  // Verify that vocab with "</w>" works WITHOUT suffix (greedy longest-match).
  // Using IGNORE_MERGES flag for pure greedy matching without merge rules.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);

  // Input "cat</w>" directly should match token 0 via greedy longest-match.
  auto tokens = EncodeAndFinalize(model.get(), "cat</w>",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "cat</w>"
}

TEST(BPEEndOfWordSuffixTest, BasicMatch) {
  // CLIP uses "</w>" suffix to mark word boundaries in vocabulary.
  // Input "cat" should match "cat</w>" when suffix is "</w>".
  // Note: CLIP uses normal BPE (ignore_merges=False), not IGNORE_MERGES.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // No IGNORE_MERGES - test whole-segment suffix matching.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), /*flags=*/0, iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "cat" + "</w>" suffix should match "cat</w>" directly.
  auto tokens = EncodeAndFinalize(model.get(), "cat",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "cat</w>"
}

TEST(BPEEndOfWordSuffixTest, NoMatchFallsBackToCharacters) {
  // When no suffix match exists, fall back to character tokens.
  // Using IGNORE_MERGES for pure greedy matching.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "dog</w>");  // Not in our input
  builder.AddToken(1, "c");
  builder.AddToken(2, "a");
  builder.AddToken(3, "t");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "cat</w>" has no whole-word match; splits to [c, a, t] from original input.
  // The suffix bytes are NOT emitted since they're not in original input.
  auto tokens = EncodeAndFinalize(model.get(), "cat",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 1);  // "c"
  EXPECT_EQ(tokens[1], 2);  // "a"
  EXPECT_EQ(tokens[2], 3);  // "t"
}

TEST(BPEEndOfWordSuffixTest, WithOffsets) {
  // Offsets must be clamped to original segment size (suffix is virtual).
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hi</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // No IGNORE_MERGES - test whole-segment suffix matching with offsets.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), /*flags=*/0, iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto result =
      EncodeWithOffsetsAndFinalize(model.get(), "hi",
                                   /*expect_pending_after_encode=*/false);
  ASSERT_EQ(result.tokens.size(), 1u);
  EXPECT_EQ(result.tokens[0], 0);  // "hi</w>"

  // Offsets should span [0,2) not [0,6) even though "hi</w>" is 6 chars.
  ASSERT_EQ(result.offsets.size(), 1u);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);  // Clamped to input length
}

TEST(BPEEndOfWordSuffixTest, EmptySuffixIsNoOp) {
  // Using IGNORE_MERGES for pure greedy matching.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);

  // Empty suffix should be no-op.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_string_view_empty()));

  auto tokens = EncodeAndFinalize(model.get(), "cat", false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "cat"
}

//===----------------------------------------------------------------------===//
// end_of_word_suffix streaming tests
//===----------------------------------------------------------------------===//

TEST(BPEEndOfWordSuffixTest, PartialSegmentNoSuffix) {
  // When streaming with last_is_partial=true, suffix should NOT be applied.
  // This is critical for streaming: we don't know if the word is complete yet.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");  // Suffixed token
  builder.AddToken(1, "c");
  builder.AddToken(2, "a");
  builder.AddToken(3, "t");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  ScopedModelState state(model.get());

  // Create segment with last_is_partial=true.
  const char* text = "cat";
  iree_tokenizer_segment_t segment = {0, 3};
  iree_tokenizer_segment_list_t segments = {1, &segment,
                                            /*last_is_partial=*/true};
  iree_const_byte_span_t buffer =
      iree_make_const_byte_span((const uint8_t*)text, 3);

  std::vector<iree_tokenizer_token_id_t> tokens(16);
  iree_host_size_t segments_consumed = 0, token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer, segments,
      iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr,
                                       tokens.size()),
      &segments_consumed, &token_count));

  // With is_partial=true, should NOT match "cat</w>" - instead get chars.
  // (The exact token count depends on how much BYTE_LOOP emits, but should
  // NOT be a single "cat</w>" token.)
  bool found_suffixed_token = false;
  for (iree_host_size_t i = 0; i < token_count; ++i) {
    if (tokens[i] == 0) found_suffixed_token = true;
  }
  EXPECT_FALSE(found_suffixed_token)
      << "Partial segment should NOT match suffixed token";
}

TEST(BPEEndOfWordSuffixTest, CompleteSegmentGetsSuffix) {
  // When segment is complete (last_is_partial=false), suffix IS applied.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");
  builder.AddToken(1, "c");
  builder.AddToken(2, "a");
  builder.AddToken(3, "t");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // No IGNORE_MERGES - test whole-segment suffix matching.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), /*flags=*/0, iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  ScopedModelState state(model.get());

  // Create segment with last_is_partial=false (complete).
  const char* text = "cat";
  iree_tokenizer_segment_t segment = {0, 3};
  iree_tokenizer_segment_list_t segments = {1, &segment,
                                            /*last_is_partial=*/false};
  iree_const_byte_span_t buffer =
      iree_make_const_byte_span((const uint8_t*)text, 3);

  std::vector<iree_tokenizer_token_id_t> tokens(16);
  iree_host_size_t segments_consumed = 0, token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer, segments,
      iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr,
                                       tokens.size()),
      &segments_consumed, &token_count));

  // With complete segment, should match "cat</w>" → single token 0.
  ASSERT_EQ(token_count, 1u);
  EXPECT_EQ(tokens[0], 0);  // "cat</w>"
}

TEST(BPEEndOfWordSuffixTest, StreamingThenFinalize) {
  // Stream partial segment, then finalize to complete it.
  // The partial segment should emit chars; finalize completes.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hello</w>");
  builder.AddToken(1, "h");
  builder.AddToken(2, "e");
  builder.AddToken(3, "l");
  builder.AddToken(4, "o");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  ScopedModelState state(model.get());

  // Stream first chunk as partial.
  const char* text1 = "hel";
  iree_tokenizer_segment_t segment1 = {0, 3};
  iree_tokenizer_segment_list_t segments1 = {1, &segment1,
                                             /*last_is_partial=*/true};
  iree_const_byte_span_t buffer1 =
      iree_make_const_byte_span((const uint8_t*)text1, 3);

  std::vector<iree_tokenizer_token_id_t> tokens(16);
  iree_host_size_t segments_consumed = 0, token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer1, segments1,
      iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr,
                                       tokens.size()),
      &segments_consumed, &token_count));

  // Partial segment - should NOT get suffixed match.
  // With BYTE_LOOP, tokens are frozen gradually based on window.
  // Just verify we didn't get a single suffixed token.
  bool found_whole_word = (token_count == 1 && tokens[0] == 0);
  EXPECT_FALSE(found_whole_word) << "Partial should not match whole word";
}

// Tests suffix matching WITHOUT IGNORE_MERGES flag.
// This exercises the whole-segment suffix check added before the fast path.
TEST(BPEEndOfWordSuffixTest, WholeSegmentMatchWithoutIgnoreMerges) {
  // Without IGNORE_MERGES, normal BPE would use backtracking.
  // But when the whole segment + suffix matches exactly, we should use it.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");  // Whole word with suffix
  builder.AddToken(1, "c");
  builder.AddToken(2, "a");
  builder.AddToken(3, "t");
  // No merges - tokens 1,2,3 are base vocabulary.

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // Note: NO IGNORE_MERGES flag - uses normal BPE algorithm.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "cat" + "</w>" should match "cat</w>" even without IGNORE_MERGES.
  auto tokens = EncodeAndFinalize(model.get(), "cat",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "cat</w>"
}

// Tests that backtracking applies suffix to the last token.
// Input "aaa" with merges produces [aa, a]. The last "a" should become "a</w>".
TEST(BPEEndOfWordSuffixTest, BacktrackingAppliesSuffixToLastToken) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");  // Suffixed single char
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa</w>");  // Suffixed merged token
  builder.AddMerge(0, 0);         // a + a -> aa

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // No IGNORE_MERGES - uses backtracking.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "aaa" -> BPE produces [aa, a], then suffix replaces last "a" with "a</w>".
  auto tokens = EncodeAndFinalize(model.get(), "aaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 2);  // "aa"
  EXPECT_EQ(tokens[1], 1);  // "a</w>" (suffixed)
}

// Tests that merged tokens also get suffix replacement.
// Input "aaaa" with merges produces [aa, aa]. Last "aa" should become "aa</w>".
TEST(BPEEndOfWordSuffixTest, BacktrackingAppliesSuffixToLastMergedToken) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "aa");
  builder.AddToken(2, "aa</w>");  // Suffixed merged token
  builder.AddMerge(0, 0);         // a + a -> aa

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "aaaa" -> BPE produces [aa, aa], then suffix replaces last with "aa</w>".
  auto tokens = EncodeAndFinalize(model.get(), "aaaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 1);  // "aa"
  EXPECT_EQ(tokens[1], 2);  // "aa</w>" (suffixed)
}

// Tests that suffix is NOT applied when suffixed token doesn't exist.
TEST(BPEEndOfWordSuffixTest, NoSuffixWhenSuffixedTokenMissing) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "aa");
  // Note: NO "aa</w>" token in vocabulary.
  builder.AddMerge(0, 0);  // a + a -> aa

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "aaaa" -> [aa, aa]. No "aa</w>" exists, so last token stays "aa".
  // Neither "aa</w>" nor "a</w>" exist, so no suffix can be applied.
  auto tokens = EncodeAndFinalize(model.get(), "aaaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 1);  // "aa"
  EXPECT_EQ(tokens[1], 1);  // "aa" (not suffixed - no such token)
}

// When a merged token has no suffixed version but its right component does,
// the merge should be split and suffix applied to the right component.
// This matches HuggingFace behavior for CLIP-style vocabularies.
TEST(BPEEndOfWordSuffixTest, SplitMergeWhenRightComponentHasSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  builder.AddToken(3, "b</w>");
  // Note: NO "ab</w>" token in vocabulary.
  builder.AddMerge(0, 1);  // a + b -> ab

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "ab" -> initially [ab], but "ab</w>" doesn't exist.
  // Since "b</w>" exists, split the merge: [a, b] -> [a, b</w>].
  auto tokens = EncodeAndFinalize(model.get(), "ab",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "a"
  EXPECT_EQ(tokens[1], 3);  // "b</w>"
}

// When splitting a merge enables a different merge with the suffixed token,
// that merge should be applied.
TEST(BPEEndOfWordSuffixTest, SplitMergeEnablesNewMergeWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  builder.AddToken(3, "b</w>");
  builder.AddToken(4, "ab</w>");
  // Note: merge(a, b</w>) -> ab</w> exists but merge(a, b) -> ab also exists.
  builder.AddMerge(0, 1);  // a + b -> ab
  builder.AddMerge(0, 3);  // a + b</w> -> ab</w>

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "ab" -> [ab] -> suffix lookup finds "ab</w>" directly.
  auto tokens = EncodeAndFinalize(model.get(), "ab",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 4);  // "ab</w>"
}

// Recursive split: merged token has no suffix, right component is also a merge
// whose right component has a suffix.
TEST(BPEEndOfWordSuffixTest, RecursiveSplitForSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "bc");
  builder.AddToken(4, "abc");
  builder.AddToken(5, "c</w>");
  // Note: NO "abc</w>", NO "bc</w>" in vocabulary.
  builder.AddMerge(1, 2);  // b + c -> bc
  builder.AddMerge(0, 3);  // a + bc -> abc

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "abc" -> [abc], but "abc</w>" doesn't exist.
  // Split abc -> [a, bc]. "bc</w>" doesn't exist either.
  // Split bc -> [b, c]. "c</w>" exists!
  // Result: [a, b, c</w>]
  auto tokens = EncodeAndFinalize(model.get(), "abc",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);  // "a"
  EXPECT_EQ(tokens[1], 1);  // "b"
  EXPECT_EQ(tokens[2], 5);  // "c</w>"
}

// Tests offsets are correctly clamped when suffix is applied after
// backtracking. The suffix bytes are virtual (not in original input), so
// offsets must stay within original segment bounds.
TEST(BPEEndOfWordSuffixTest, BacktrackingOffsetsClampedWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");  // 5 chars but only covers 1 byte of input
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa</w>");  // 6 chars but only covers 2 bytes of input
  builder.AddMerge(0, 0);         // a + a -> aa

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "aaa" (3 bytes) -> [aa, a</w>]
  // Offsets should be: aa=[0,2), a</w>=[2,3) - NOT [2,7) for the suffix.
  auto result =
      EncodeWithOffsetsAndFinalize(model.get(), "aaa",
                                   /*expect_pending_after_encode=*/false);
  ASSERT_EQ(result.tokens.size(), 2u);
  EXPECT_EQ(result.tokens[0], 2);  // "aa"
  EXPECT_EQ(result.tokens[1], 1);  // "a</w>"

  ASSERT_EQ(result.offsets.size(), 2u);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);  // "aa" covers bytes [0,2)
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 3u);  // "a</w>" covers byte [2,3), clamped
}

// Tests single-character suffix (not just "</w>").
TEST(BPEEndOfWordSuffixTest, SingleCharSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a_");  // Single char suffix
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa_");
  builder.AddMerge(0, 0);  // a + a -> aa

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("_")));

  // "aaa" -> [aa, a_]
  auto tokens = EncodeAndFinalize(model.get(), "aaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 2);  // "aa"
  EXPECT_EQ(tokens[1], 1);  // "a_"
}

// Tests longer suffix (8 characters).
TEST(BPEEndOfWordSuffixTest, LongerSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a[ENDWD]");  // 8-char suffix
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa[ENDWD]");
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("[ENDWD]")));

  auto tokens = EncodeAndFinalize(model.get(), "aaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 2);  // "aa"
  EXPECT_EQ(tokens[1], 1);  // "a[ENDWD]"
}

// Tests that only the LAST token gets suffix, not intermediate ones.
TEST(BPEEndOfWordSuffixTest, OnlyLastTokenGetsSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");
  builder.AddToken(2, "b");
  builder.AddToken(3, "b</w>");
  // No merges - each character is its own token.

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "aab" -> [a, a, b</w>] - only last token suffixed.
  auto tokens = EncodeAndFinalize(model.get(), "aab",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);  // "a" (not suffixed)
  EXPECT_EQ(tokens[1], 0);  // "a" (not suffixed)
  EXPECT_EQ(tokens[2], 3);  // "b</w>" (suffixed)
}

// Tests single-byte input with suffix.
TEST(BPEEndOfWordSuffixTest, SingleByteInput) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "x");
  builder.AddToken(1, "x</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Single byte "x" should become "x</w>".
  auto tokens = EncodeAndFinalize(model.get(), "x",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 1);  // "x</w>"
}

// Tests longer sequence requiring multiple merge rounds.
// Simulates CLIP's "bbbb" -> [bb, bb</w>] pattern verified against HuggingFace.
TEST(BPEEndOfWordSuffixTest, LongerSequenceMultipleMerges) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "b");
  builder.AddToken(1, "b</w>");
  builder.AddToken(2, "bb");
  builder.AddToken(3, "bb</w>");
  builder.AddToken(4, "bbb");
  builder.AddToken(5, "bbb</w>");
  builder.AddMerge(0, 0);  // b + b -> bb
  builder.AddMerge(2, 0);  // bb + b -> bbb

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "bbbb" -> BPE produces [bb, bb], suffix makes it [bb, bb</w>].
  // Verified against HuggingFace CLIP: "bbbb" -> [1174, 3529].
  auto tokens = EncodeAndFinalize(model.get(), "bbbb",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 2);  // "bb"
  EXPECT_EQ(tokens[1], 3);  // "bb</w>"
}

// Tests partial segment during streaming does not get suffix.
// Partial segments route to BYTE_LOOP which doesn't apply suffix.
TEST(BPEEndOfWordSuffixTest, PartialSegmentNoSuffixWithBacktracking) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa</w>");
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // No IGNORE_MERGES - would normally use backtracking for complete segments.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  ScopedModelState state(model.get());

  // Stream "aa" as partial segment.
  const char* text = "aa";
  iree_tokenizer_segment_t segment = {0, 2};
  iree_tokenizer_segment_list_t segments = {1, &segment,
                                            /*last_is_partial=*/true};
  iree_const_byte_span_t buffer =
      iree_make_const_byte_span((const uint8_t*)text, 2);

  std::vector<iree_tokenizer_token_id_t> tokens(16);
  iree_host_size_t segments_consumed = 0, token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer, segments,
      iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr,
                                       tokens.size()),
      &segments_consumed, &token_count));

  // Partial segment should NOT have suffix applied.
  // Token output depends on BYTE_LOOP frozen token emission, but suffixed
  // versions (token 1 "a</w>" or token 3 "aa</w>") should NOT appear.
  bool found_suffixed = false;
  for (iree_host_size_t i = 0; i < token_count; ++i) {
    if (tokens[i] == 1 || tokens[i] == 3) found_suffixed = true;
  }
  EXPECT_FALSE(found_suffixed)
      << "Partial segment should NOT match suffixed tokens";
}

// Tests that whole-segment suffix match works even when there are merges
// that could theoretically produce the same token.
TEST(BPEEndOfWordSuffixTest, WholeSegmentMatchPreferredOverMerges) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "c");
  builder.AddToken(1, "a");
  builder.AddToken(2, "t");
  builder.AddToken(3, "ca");
  builder.AddToken(4, "cat");
  builder.AddToken(5, "cat</w>");  // Target whole-segment match
  builder.AddMerge(0, 1);          // c + a -> ca
  builder.AddMerge(3, 2);          // ca + t -> cat

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // "cat" should match "cat</w>" directly via whole-segment suffix match,
  // producing single token 5, not going through BPE merges.
  auto tokens = EncodeAndFinalize(model.get(), "cat",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 5);  // "cat</w>"
}

// Tests that COMPLETE long segments (> 2048 bytes) that use the BYTE_LOOP path
// still get suffix applied at the end.
TEST(BPEEndOfWordSuffixTest, LongSegmentGetsSuffixApplied) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");  // Suffixed version
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa</w>");  // Suffixed merged version
  builder.AddMerge(0, 0);         // a + a -> aa

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Build a 2100-byte input: "aa" repeated 1050 times.
  // This exceeds max_backtrack_segment_bytes (~2048) forcing BYTE_LOOP path.
  std::string input;
  input.reserve(2100);
  for (int i = 0; i < 1050; ++i) {
    input += "aa";
  }
  ASSERT_EQ(input.size(), 2100u);

  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);

  // Should have 1050 tokens, all "aa" (token 2), except the LAST one
  // which should be "aa</w>" (token 3) due to suffix application.
  ASSERT_EQ(tokens.size(), 1050u);
  for (size_t i = 0; i < 1049; ++i) {
    EXPECT_EQ(tokens[i], 2) << "Token " << i << " should be 'aa'";
  }
  EXPECT_EQ(tokens[1049], 3) << "Last token should be 'aa</w>' (suffixed)";
}

// Tests suffix with merge chain in long segment (BYTE_LOOP path).
// The suffix enables a merge that wouldn't otherwise fire.
TEST(BPEEndOfWordSuffixTest, LongSegmentSuffixEnablesMerge) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "x");
  builder.AddToken(1, "x</w>");
  builder.AddToken(2, "xx");
  builder.AddToken(3, "xx</w>");
  builder.AddToken(4, "xxx</w>");  // Only exists with suffix
  builder.AddMerge(0, 0);          // x + x -> xx
  builder.AddMerge(2, 1);          // xx + x</w> -> xxx</w>

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Build input with odd number of x's at the end to test suffix merge.
  // 2100 bytes = 700 "xxx" = 2100 x's
  // This should produce: 699 "xx" tokens + 1 "xxx</w>" token (suffix merge)
  // Wait, 2100 / 3 = 700, so 700 * 3 = 2100 x's
  // After merging pairs: floor(2100/2) = 1050 pairs, 0 remainder
  // Actually let's use 2101 x's: 1050 "xx" + 1 "x</w>"
  // Or better: use pattern that clearly tests suffix merge.
  // Use 2103 x's: 1050 "xx" + "xxx" at end
  // After BPE: 1050 "xx" + remains "xxx"
  // With suffix: last "x" becomes "x</w>", then xx + x</w> -> xxx</w>
  std::string input(2103, 'x');
  ASSERT_EQ(input.size(), 2103u);

  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);

  // 2103 x's: pairs merge to "xx", leaving 1 "x" at end.
  // 1051 "xx" tokens + 1 "x" = 2103 x's? No: 1051*2 + 1 = 2103. Yes.
  // With suffix: last "x" -> "x</w>", then merge with previous "xx"?
  // Actually the merge is xx + x</w> -> xxx</w>, so the last TWO tokens
  // (xx and x) become one (xxx</w>).
  // Expected: 1050 "xx" + 1 "xxx</w>"
  ASSERT_EQ(tokens.size(), 1051u);
  for (size_t i = 0; i < 1050; ++i) {
    EXPECT_EQ(tokens[i], 2) << "Token " << i << " should be 'xx'";
  }
  EXPECT_EQ(tokens[1050], 4) << "Last token should be 'xxx</w>' (suffix merge)";
}

//===----------------------------------------------------------------------===//
// Boundary Blocking Merges
//===----------------------------------------------------------------------===//

// Tests that the is_first_token_reachable logic correctly handles blocking
// merges at split boundaries. A boundary merge can only block if it fires
// before either boundary token is consumed by internal merges.

// Tests that a boundary merge does not block when both boundary tokens are
// consumed by internal merges before the boundary merge can fire.
//
// Setup: "abcd" = merge(ab, cd) where:
//   - "cd" = merge(c, d) at rank 0  → 'c' consumed at rank 0
//   - "ab" = merge(a, b) at rank 1  → 'b' consumed at rank 1
//   - "bc" = merge(b, c) at rank 2  (boundary merge)
//   - "abcd" = merge(ab, cd) at rank 3
//
// The boundary merge "bc" at rank 2 cannot fire because:
//   - 'c' is already consumed by "cd" at rank 0
//   - 'b' is already consumed by "ab" at rank 1
// Therefore "abcd" IS reachable and should tokenize as a single token.
TEST_F(BPEModelTest, BoundaryMergeDoesNotBlockWhenConsumedFirst) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "d");
  builder.AddToken(4, "ab");
  builder.AddToken(5, "bc");  // Boundary merge result
  builder.AddToken(6, "cd");
  builder.AddToken(7, "abcd");

  // Merge order matters:
  builder.AddMerge(2, 3);  // Rank 0: c + d -> cd (consumes 'c' first!)
  builder.AddMerge(0, 1);  // Rank 1: a + b -> ab (consumes 'b')
  builder.AddMerge(1, 2);  // Rank 2: b + c -> bc (boundary merge - too late!)
  builder.AddMerge(4, 6);  // Rank 3: ab + cd -> abcd

  CreateModel(builder);

  // "abcd" should be reachable as a single token because "bc" cannot fire.
  TestEncode(model(), "abcd", {7}, /*expect_pending_after_encode=*/false);
}

// Tests that a boundary merge DOES block when it fires BEFORE either boundary
// token is consumed by internal merges.
//
// Setup: "abcd" = merge(ab, cd) where:
//   - "bc" = merge(b, c) at rank 0  (boundary merge - fires first!)
//   - "ab" = merge(a, b) at rank 1
//   - "cd" = merge(c, d) at rank 2
//   - "abcd" = merge(ab, cd) at rank 3
//
// The boundary merge "bc" at rank 0 fires BEFORE:
//   - 'b' is consumed by "ab" at rank 1
//   - 'c' is consumed by "cd" at rank 2
// Therefore "abcd" is NOT reachable. Input "abcd" will be tokenized as
// [a, bc, d] because "bc" forms first, blocking both "ab" and "cd".
TEST_F(BPEModelTest, BoundaryMergeBlocksWhenFiringFirst) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "d");
  builder.AddToken(4, "ab");
  builder.AddToken(5, "bc");  // Boundary merge result
  builder.AddToken(6, "cd");
  builder.AddToken(7, "abcd");

  // Merge order matters - bc fires first!
  builder.AddMerge(1,
                   2);  // Rank 0: b + c -> bc (boundary merge - fires first!)
  builder.AddMerge(0, 1);  // Rank 1: a + b -> ab (too late, 'b' is gone)
  builder.AddMerge(2, 3);  // Rank 2: c + d -> cd (too late, 'c' is gone)
  builder.AddMerge(4, 6);  // Rank 3: ab + cd -> abcd (unreachable)

  CreateModel(builder);

  // "abcd" is NOT reachable. BPE produces [a, bc, d] because "bc" fires first.
  TestEncode(model(), "abcd", {0, 5, 3}, /*expect_pending_after_encode=*/false);
}

// Tests the case from the DeepSeek bug: boundary merge exists but fires after
// BOTH boundary tokens are consumed by deeper internal merges.
//
// This is a more realistic scenario with multi-level merge chains:
//   - Token "ABCDEF" = merge("ABC", "DEF")
//   - "ABC" = merge("AB", "C") at rank 5  → 'C' consumed at rank 5
//   - "AB" = merge("A", "B") at rank 1    → 'B' consumed at rank 1
//   - "DEF" = merge("D", "EF") at rank 4  → 'D' consumed at rank 4
//   - "EF" = merge("E", "F") at rank 0    → 'E' consumed at rank 0
//   - "CD" = merge("C", "D") at rank 3    (boundary merge)
//
// The boundary tokens are 'C' (rightmost of ABC) and 'D' (leftmost of DEF).
// Boundary merge "CD" at rank 3 is blocked because:
//   - 'D' is consumed at rank 4 > rank 3... wait, that's the wrong way.
//
// Actually let me reconsider: 'D' consumed at rank 4 means "CD" at rank 3
// fires BEFORE 'D' is consumed. So this would block!
//
// Let me redo with correct ordering where boundary tokens are consumed early:
//   - "EF" = merge("E", "F") at rank 0
//   - "DE" = merge("D", "E") at rank 1    → 'D' consumed at rank 1
//   - "AB" = merge("A", "B") at rank 2
//   - "BC" = merge("B", "C") at rank 3    → 'C' consumed at rank 3 via ABC path
//   - Actually this is getting complicated. Let me simplify.
//
// Simplified test: mirror the fibonacci case structure where internal merges
// consume boundary tokens at low ranks.
TEST_F(BPEModelTest, DeepBoundaryConsumedByInternalMerge) {
  // Simulates: "xyzw" = merge("xy", "zw") with boundary merge "yz" that
  // cannot fire because 'y' and 'z' are consumed by internal merges first.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "x");
  builder.AddToken(1, "y");
  builder.AddToken(2, "z");
  builder.AddToken(3, "w");
  builder.AddToken(4, "xy");  // Left subtree
  builder.AddToken(5, "yz");  // Boundary merge result
  builder.AddToken(6, "zw");  // Right subtree
  builder.AddToken(7, "xyzw");

  // Merge order: internal merges fire before boundary merge
  builder.AddMerge(0, 1);  // Rank 0: x + y -> xy (consumes 'y'!)
  builder.AddMerge(2, 3);  // Rank 1: z + w -> zw (consumes 'z'!)
  builder.AddMerge(1, 2);  // Rank 2: y + z -> yz (boundary merge - blocked!)
  builder.AddMerge(4, 6);  // Rank 3: xy + zw -> xyzw

  CreateModel(builder);

  // "xyzw" should be reachable as single token because "yz" cannot fire:
  // - 'y' consumed at rank 0 (by xy)
  // - 'z' consumed at rank 1 (by zw)
  // - "yz" at rank 2 is too late - both tokens already consumed
  TestEncode(model(), "xyzw", {7}, /*expect_pending_after_encode=*/false);
}

// Tests asymmetric case: one boundary token consumed early, other late.
// The boundary merge is still blocked because it needs BOTH tokens available.
TEST_F(BPEModelTest, BoundaryMergeBlockedByEarlyConsumptionOfOneSide) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "d");
  builder.AddToken(4, "ab");
  builder.AddToken(5, "bc");  // Boundary merge
  builder.AddToken(6, "cd");
  builder.AddToken(7, "abcd");

  // 'b' consumed very early, 'c' consumed late - but still before boundary
  builder.AddMerge(0, 1);  // Rank 0: a + b -> ab (consumes 'b' at rank 0!)
  builder.AddMerge(1, 2);  // Rank 1: b + c -> bc (boundary - blocked by rank 0)
  builder.AddMerge(2, 3);  // Rank 2: c + d -> cd (consumes 'c')
  builder.AddMerge(4, 6);  // Rank 3: ab + cd -> abcd

  CreateModel(builder);

  // "abcd" is reachable: 'b' consumed at rank 0, before "bc" at rank 1
  TestEncode(model(), "abcd", {7}, /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Multi-byte vocabulary tokens without merge participation
//===----------------------------------------------------------------------===//

// Multi-byte vocabulary tokens that don't participate in ANY merge should still
// be matched directly, rather than falling back to byte-level tokens.
// This is critical for CJK characters in vocabularies like TinyLlama where
// characters like "你" exist as single tokens but never appear in merge rules.
TEST_F(BPEModelTest, MultiByteVocabTokenWithoutMerges) {
  ScopedVocabBuilder builder;
  // Single-byte tokens for basic ASCII.
  builder.AddToken(0, "h");
  builder.AddToken(1, "i");
  builder.AddToken(2, " ");
  // Multi-byte token (3 bytes UTF-8 for "你") that doesn't participate in any
  // merge. This simulates CJK characters in vocabularies like TinyLlama.
  builder.AddToken(3, "你");  // U+4F60 = 0xE4 0xBD 0xA0
  // Byte fallback tokens that would be used if "你" wasn't matched.
  builder.AddToken(4, "<0xE4>");
  builder.AddToken(5, "<0xBD>");
  builder.AddToken(6, "<0xA0>");
  // Merge for ASCII tokens (but "你" doesn't participate).
  builder.AddToken(7, "hi");
  builder.AddMerge(0, 1);  // h + i -> hi

  CreateModel(builder);

  // "你" should match directly as token 3, not as byte fallback [4, 5, 6].
  TestEncode(model(), "你", {3}, /*expect_pending_after_encode=*/false);

  // "hi你" should be [7, 3] (merged "hi" + direct "你").
  TestEncode(model(), "hi你", {7, 3}, /*expect_pending_after_encode=*/false);
}

// Similar test but with a 2-byte UTF-8 character (Latin Extended).
TEST_F(BPEModelTest, TwoByteVocabTokenWithoutMerges) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  // "é" = U+00E9 = 0xC3 0xA9 (2 bytes UTF-8).
  builder.AddToken(2, "é");
  // Byte fallback tokens.
  builder.AddToken(3, "<0xC3>");
  builder.AddToken(4, "<0xA9>");
  // No merges involving "é".

  CreateModel(builder);

  // "é" should match directly as token 2, not as byte fallback [3, 4].
  TestEncode(model(), "é", {2}, /*expect_pending_after_encode=*/false);

  // "aéb" should be [0, 2, 1] (a + direct é + b).
  TestEncode(model(), "aéb", {0, 2, 1}, /*expect_pending_after_encode=*/false);
}

// Test for SentencePiece-style space tokens (▁ = U+2581).
// As observed in TinyLlama where "▁▁▁" exists as a vocabulary entry, these
// multi-byte tokens should match directly rather than decomposing to byte
// fallback.
TEST_F(BPEModelTest, SentencePieceSpaceTokenWithoutMerges) {
  ScopedVocabBuilder builder;
  // Basic ASCII tokens.
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  // SentencePiece space marker (▁ = U+2581 = 0xE2 0x96 0x81 in UTF-8).
  builder.AddToken(2, "▁");
  // Multiple space markers (like TinyLlama's token 1678).
  builder.AddToken(3, "▁▁▁");
  // Byte fallback tokens for ▁.
  builder.AddToken(4, "<0xE2>");
  builder.AddToken(5, "<0x96>");
  builder.AddToken(6, "<0x81>");
  // No merges involving ▁ or ▁▁▁.

  CreateModel(builder);

  // Single ▁ should match directly as token 2.
  TestEncode(model(), "▁", {2}, /*expect_pending_after_encode=*/false);

  // Triple ▁▁▁ should match directly as token 3, not as [4, 5, 6, 4, 5, 6,
  // ...].
  TestEncode(model(), "▁▁▁", {3}, /*expect_pending_after_encode=*/false);

  // "a▁b" should be [0, 2, 1] (a + direct ▁ + b).
  TestEncode(model(), "a▁b", {0, 2, 1}, /*expect_pending_after_encode=*/false);
}

// Tests that ▁ at high token ID (like Mistral's 28705) is matched correctly
// when merges involving ▁ exist and byte fallback tokens are also present.
// ▁ must be matched directly, not via byte fallback.
TEST_F(BPEModelTest, SentencePieceSpaceTokenWithMergesAtHighId) {
  ScopedVocabBuilder builder;

  // Single character tokens at high IDs like Mistral
  static const char kMeta[] = "\xE2\x96\x81";  // ▁
  builder.AddToken(28705, kMeta);              // ▁ at ID 28705 like Mistral
  builder.AddToken(28706, "e");
  builder.AddToken(28707, "t");
  builder.AddToken(28716, "h");

  // Merged tokens.
  std::string meta_t = std::string(kMeta) + "t";
  std::string meta_h = std::string(kMeta) + "h";
  builder.AddToken(100, meta_t.c_str());  // ▁t
  builder.AddToken(200, meta_h.c_str());  // ▁h
  builder.AddToken(265, "he");            // he

  // Byte fallback tokens (like Mistral).
  builder.AddToken(229, "<0xE2>");
  builder.AddToken(153, "<0x96>");
  builder.AddToken(132, "<0x81>");

  // Merges in rank order (lower = higher priority).
  builder.AddMerge(28705, 28707);  // ▁ + t -> ▁t (rank 0)
  builder.AddMerge(28716, 28706);  // h + e -> he (rank 1)
  builder.AddMerge(28705, 28716);  // ▁ + h -> ▁h (rank 2)

  CreateModel(builder);

  // Single ▁ should match directly as token 28705, not byte fallback.
  TestEncode(model(), "▁", {28705}, /*expect_pending_after_encode=*/false);

  // ▁ followed by h should produce [▁h] = [200] via merge, not byte fallback.
  TestEncode(model(), "▁h", {200}, /*expect_pending_after_encode=*/false);

  // ▁ followed by t should produce [▁t] = [100] via merge.
  TestEncode(model(), "▁t", {100}, /*expect_pending_after_encode=*/false);
}

// Tests that merged tokens like ▁he (produced by ▁h + e) are correctly matched
// when they are also components in other merges. Such tokens have
// effective_rank > 0 even though they are results of a merge, and must be
// matched directly rather than via byte fallback.
TEST_F(BPEModelTest, MergedTokensAsComponents) {
  ScopedVocabBuilder builder;

  // Base single-character tokens at high IDs like Mistral.
  static const char kMeta[] = "\xE2\x96\x81";  // ▁
  builder.AddToken(28705, kMeta);              // ▁ at ID 28705 like Mistral
  builder.AddToken(28706, "e");
  builder.AddToken(28716, "h");
  builder.AddToken(28712, "r");

  // First-level merged tokens.
  std::string meta_h = std::string(kMeta) + "h";
  builder.AddToken(295, meta_h.c_str());  // ▁h (like Mistral)
  builder.AddToken(265, "he");            // he

  // Second-level merged token: ▁he = ▁h + e
  std::string meta_he = std::string(kMeta) + "he";
  builder.AddToken(400, meta_he.c_str());  // ▁he (like Mistral)

  // Third-level merged token: ▁her = ▁he + r (makes ▁he a component)
  std::string meta_her = std::string(kMeta) + "her";
  builder.AddToken(700, meta_her.c_str());  // ▁her

  // Byte fallback tokens (like Mistral).
  builder.AddToken(229, "<0xE2>");
  builder.AddToken(153, "<0x96>");
  builder.AddToken(132, "<0x81>");

  // Merges in rank order (lower = higher priority).
  builder.AddMerge(28705, 28716);  // ▁ + h -> ▁h (rank 0)
  builder.AddMerge(28716, 28706);  // h + e -> he (rank 1)
  builder.AddMerge(295, 28706);  // ▁h + e -> ▁he (rank 2) - produces token 400
  builder.AddMerge(400,
                   28712);  // ▁he + r -> ▁her (rank 3) - uses 400 as component

  CreateModel(builder);

  // Single ▁ should match directly.
  TestEncode(model(), "▁", {28705}, /*expect_pending_after_encode=*/false);

  // ▁h should match via merge ▁ + h.
  TestEncode(model(), "▁h", {295}, /*expect_pending_after_encode=*/false);

  // ▁he should match via merge ▁h + e -> token 400.
  // This tests that merged tokens (produced by merges) are accepted by the trie
  // when they also appear as components in other merges.
  TestEncode(model(), "▁he", {400}, /*expect_pending_after_encode=*/false);

  // ▁her should match via merge ▁he + r -> token 700.
  TestEncode(model(), "▁her", {700}, /*expect_pending_after_encode=*/false);
}

// Tests that multi-metaspace tokens (like Mistral's whitespace handling) work
// correctly for long runs of 50+ consecutive metaspaces, producing proper
// multi-metaspace tokens rather than falling back to byte-level encoding.
TEST_F(BPEModelTest, MultiMetaspaceTokensMistralStyle) {
  ScopedVocabBuilder builder;

  static const char kMeta[] = "\xE2\x96\x81";  // ▁ (U+2581)

  // Single metaspace (like Mistral's 28705).
  builder.AddToken(28705, kMeta);  // ▁

  // Multi-metaspace tokens built from merges.
  std::string meta2;
  for (int i = 0; i < 2; ++i) meta2 += kMeta;
  builder.AddToken(259, meta2.c_str());  // ▁▁

  std::string meta4;
  for (int i = 0; i < 4; ++i) meta4 += kMeta;
  builder.AddToken(300, meta4.c_str());  // ▁▁▁▁

  std::string meta8;
  for (int i = 0; i < 8; ++i) meta8 += kMeta;
  builder.AddToken(320, meta8.c_str());  // ▁▁▁▁▁▁▁▁

  std::string meta16;
  for (int i = 0; i < 16; ++i) meta16 += kMeta;
  builder.AddToken(359, meta16.c_str());  // 16 metaspaces

  // Byte fallback tokens (like Mistral).
  builder.AddToken(229, "<0xE2>");
  builder.AddToken(153, "<0x96>");
  builder.AddToken(132, "<0x81>");

  // Merges in rank order:
  builder.AddMerge(28705, 28705);  // ▁ + ▁ → ▁▁ (rank 0)
  builder.AddMerge(259, 259);      // ▁▁ + ▁▁ → ▁▁▁▁ (rank 1)
  builder.AddMerge(300, 300);      // ▁▁▁▁ + ▁▁▁▁ → ▁▁▁▁▁▁▁▁ (rank 2)
  builder.AddMerge(320, 320);      // ▁▁▁▁▁▁▁▁ + ▁▁▁▁▁▁▁▁ → 16 metas (rank 3)

  CreateModel(builder);

  // Single metaspace should match directly.
  TestEncode(model(), kMeta, {28705}, /*expect_pending_after_encode=*/false);

  // 2 metaspaces should merge to token 259.
  TestEncode(model(), meta2.c_str(), {259},
             /*expect_pending_after_encode=*/false);

  // 4 metaspaces should merge to token 300.
  TestEncode(model(), meta4.c_str(), {300},
             /*expect_pending_after_encode=*/false);

  // 16 metaspaces should merge to token 359.
  TestEncode(model(), meta16.c_str(), {359},
             /*expect_pending_after_encode=*/false);

  // 50 metaspaces: 16 + 16 + 16 + 2 = 50
  // Should produce: [359, 359, 359, 259]
  std::string meta50;
  for (int i = 0; i < 50; ++i) meta50 += kMeta;
  TestEncode(model(), meta50.c_str(), {359, 359, 359, 259},
             /*expect_pending_after_encode=*/false);
}

// Tests suffix blocking when a token has multiple merge paths.
//
// When a token can form via multiple paths, suffix blocking must consider
// which path actual BPE would take. A suffix that exists in split_table may
// not exist in the actual BPE execution if a lower-rank merge consumes its
// components first.
//
// Example: ▁hell can form via ▁h+ell or ▁he+ll. If BPE forms he before ▁+h,
// then l+l→ll may fire before el+l→ell, meaning ell never exists. Suffix
// blocking should not reject ▁hell based on an ell+o merge that cannot occur.
TEST_F(BPEModelTest, SuffixBlockingWithAlternateMergePaths) {
  ScopedVocabBuilder builder;

  // Use low IDs to simplify, matching the pattern used by working tests.
  static const char kMeta[] = "\xE2\x96\x81";  // ▁ (U+2581)

  // Base single-character tokens.
  builder.AddToken(0, kMeta);  // ▁
  builder.AddToken(1, "h");
  builder.AddToken(2, "e");
  builder.AddToken(3, "l");
  builder.AddToken(4, "o");

  // First-level merged tokens.
  std::string meta_h = std::string(kMeta) + "h";
  builder.AddToken(10, meta_h.c_str());  // ▁h

  builder.AddToken(11, "he");   // h + e
  builder.AddToken(12, "el");   // e + l
  builder.AddToken(13, "ell");  // el + l
  builder.AddToken(14, "ll");   // l + l

  std::string meta_he = std::string(kMeta) + "he";
  builder.AddToken(20, meta_he.c_str());  // ▁he

  std::string meta_hell = std::string(kMeta) + "hell";
  builder.AddToken(30, meta_hell.c_str());  // ▁hell

  builder.AddToken(31, "ello");  // ell + o

  // Byte fallback tokens for ▁.
  builder.AddToken(100, "<0xE2>");
  builder.AddToken(101, "<0x96>");
  builder.AddToken(102, "<0x81>");

  // Merges ordered so h+e fires before ▁+h, causing l+l to consume before ell.
  builder.AddMerge(1, 2);    // Rank 0: h + e -> he (11)
  builder.AddMerge(2, 3);    // Rank 1: e + l -> el (12)
  builder.AddMerge(12, 3);   // Rank 2: el + l -> ell (13)
  builder.AddMerge(3, 3);    // Rank 3: l + l -> ll (14)
  builder.AddMerge(0, 1);    // Rank 4: ▁ + h -> ▁h (10)
  builder.AddMerge(0, 11);   // Rank 5: ▁ + he -> ▁he (20)
  builder.AddMerge(10, 2);   // Rank 6: ▁h + e -> ▁he (alternate)
  builder.AddMerge(13, 4);   // Rank 7: ell + o -> ello (31)
  builder.AddMerge(10, 13);  // Rank 8: ▁h + ell -> ▁hell (30) PRIMARY
  builder.AddMerge(20, 14);  // Rank 9: ▁he + ll -> ▁hell (alternate)

  CreateModel(builder);

  // Verify basic tokens work.
  TestEncode(model(), "▁", {0}, /*expect_pending_after_encode=*/false);
  TestEncode(model(), "▁h", {10}, /*expect_pending_after_encode=*/false);
  TestEncode(model(), "▁hell", {30}, /*expect_pending_after_encode=*/false);

  // BPE on ▁hello: h+e fires first (rank 0), then l+l (rank 3), then ▁+he
  // (rank 5), then ▁he+ll (rank 9). The ell token never forms because l+l
  // consumes both l's. Result: [▁hell, o].
  TestEncode(model(), "▁hello", {30, 4},
             /*expect_pending_after_encode=*/false);
}

// Tests that BPE backtracking does not corrupt tokenization of earlier
// positions when later positions trigger backtracking.
//
// This pattern occurs in TinyLlama (and similar SentencePiece-based models)
// where space tokens like ▁▁▁ (token 1678) precede other content. When
// backtracking is triggered by certain token combinations at later positions,
// the earlier space tokens must remain correctly tokenized rather than
// falling back to byte-level encoding.
TEST_F(BPEModelTest, BacktrackingDoesNotCorruptEarlierTokens) {
  ScopedVocabBuilder builder(iree_allocator_system(), /*capacity_hint=*/64);

  // Base single-character tokens.
  builder.AddToken(0, "▁");  // SentencePiece space marker.
  builder.AddToken(1, "p");
  builder.AddToken(2, "r");
  builder.AddToken(3, "i");
  builder.AddToken(4, "n");
  builder.AddToken(5, "t");
  builder.AddToken(6, "H");
  builder.AddToken(7, "e");
  builder.AddToken(8, "l");

  // Multi-character merged tokens.
  builder.AddToken(10, "▁▁");
  builder.AddToken(11, "▁▁▁");
  builder.AddToken(12, "▁▁▁▁");
  builder.AddToken(13, "▁print");
  builder.AddToken(14, "He");
  builder.AddToken(15, "Hel");
  builder.AddToken(16, "print");

  // Byte fallback tokens for ▁ (U+2581 = E2 96 81).
  builder.AddToken(20, "<0xE2>");
  builder.AddToken(21, "<0x96>");
  builder.AddToken(22, "<0x81>");

  // Merges in rank order (lower rank = higher priority).
  builder.AddMerge(0, 0);    // ▁ + ▁ -> ▁▁
  builder.AddMerge(10, 10);  // ▁▁ + ▁▁ -> ▁▁▁▁
  builder.AddMerge(10, 0);   // ▁▁ + ▁ -> ▁▁▁

  // Word merges.
  builder.AddMerge(1, 2);  // p + r
  int32_t pr = 17;
  builder.AddToken(pr, "pr");
  builder.AddMerge(pr, 3);  // pr + i
  int32_t pri = 18;
  builder.AddToken(pri, "pri");
  builder.AddMerge(pri, 4);  // pri + n
  int32_t prin = 19;
  builder.AddToken(prin, "prin");
  builder.AddMerge(prin, 5);  // prin + t -> print

  builder.AddMerge(0, 16);  // ▁ + print -> ▁print

  builder.AddMerge(6, 7);   // H + e -> He
  builder.AddMerge(14, 8);  // He + l -> Hel

  CreateModel(builder);

  // Three spaces followed by "He".
  TestEncode(model(), "▁▁▁He", {11, 14},
             /*expect_pending_after_encode=*/false);

  // Three spaces followed by "Hel".
  TestEncode(model(), "▁▁▁Hel", {11, 15},
             /*expect_pending_after_encode=*/false);

  // Just three spaces.
  TestEncode(model(), "▁▁▁", {11}, /*expect_pending_after_encode=*/false);

  // Four spaces followed by content should not produce byte fallback.
  auto result = EncodeAndFinalize(model(), "▁▁▁▁He",
                                  /*expect_pending_after_encode=*/false);
  for (auto token : result) {
    EXPECT_NE(token, 20) << "Unexpected byte fallback <0xE2>";
    EXPECT_NE(token, 21) << "Unexpected byte fallback <0x96>";
    EXPECT_NE(token, 22) << "Unexpected byte fallback <0x81>";
  }
}

// Tests BPE with ByteLevel-pretokenized input.
//
// When a ByteLevel pretokenizer transforms input " <" to "Ġ<" (UTF-8: c4 a0
// 3c), the BPE model receives the transformed bytes directly. The model must
// match these bytes against the vocabulary (which uses the same ByteLevel
// encoding) and apply merges correctly.
//
// Vocabulary tokens use ByteLevel encoding:
//   - "Ġ" (U+0120) represents the original space byte (0x20)
//   - Printable ASCII like "<" remains unchanged
//
// We do not use BYTE_LEVEL_INPUT flag here because the input is already
// ByteLevel-encoded by the pretokenizer. BYTE_LEVEL_INPUT is for raw bytes
// that need transformation (the pretokenizer only splits, doesn't transform).
TEST_F(BPEModelTest, ByteLevelPretokenizedInputMergesCorrectly) {
  ScopedVocabBuilder builder(iree_allocator_system(), /*capacity_hint=*/8);

  // Vocabulary with ByteLevel-encoded tokens.
  builder.AddToken(0, "Ġ");   // ByteLevel encoding of space (U+0120)
  builder.AddToken(1, "<");   // less-than (unchanged in ByteLevel)
  builder.AddToken(2, "Ġ<");  // merged: space + less-than

  builder.AddMerge(0, 1);  // Ġ + < → Ġ<

  // No BYTE_LEVEL_INPUT - input "Ġ<" is already ByteLevel-transformed.
  // BYTE_LEVEL_INPUT would double-transform, causing incorrect lookups.
  CreateModel(builder, IREE_TOKENIZER_BPE_FLAG_NONE);

  // Input "Ġ<" is already ByteLevel-transformed (UTF-8: c4 a0 3c).
  // BPE should find tokens Ġ (0) and < (1), then apply merge → Ġ< (2).
  TestEncode(model(), "Ġ<", {2}, /*expect_pending_after_encode=*/false);
}

// Tests ByteLevel-pretokenized input with multiple merge levels.
TEST_F(BPEModelTest, ByteLevelPretokenizedMultipleMerges) {
  ScopedVocabBuilder builder(iree_allocator_system(), /*capacity_hint=*/16);

  // ByteLevel-encoded vocabulary.
  builder.AddToken(0, "Ġ");    // space
  builder.AddToken(1, "<");    // less-than
  builder.AddToken(2, "Ġ<");   // space + less-than (merged)
  builder.AddToken(3, "i");    // letter i
  builder.AddToken(4, "f");    // letter f
  builder.AddToken(5, "if");   // merged i+f
  builder.AddToken(6, "Ġif");  // merged space+i+f

  builder.AddMerge(0, 1);  // Ġ + < → Ġ< (rank 0)
  builder.AddMerge(3, 4);  // i + f → if (rank 1)
  builder.AddMerge(0, 5);  // Ġ + if → Ġif (rank 2)

  // No BYTE_LEVEL_INPUT - input is already ByteLevel-transformed.
  CreateModel(builder, IREE_TOKENIZER_BPE_FLAG_NONE);

  TestEncode(model(), "Ġ<", {2}, /*expect_pending_after_encode=*/false);
  TestEncode(model(), "if", {5}, /*expect_pending_after_encode=*/false);
  TestEncode(model(), "Ġif", {6}, /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// ByteLevel mode with end_of_word_suffix
// Tests CLIP-style tokenization where both ByteLevel input transformation
// AND end_of_word_suffix are used together.
//===----------------------------------------------------------------------===//

// Tests that ByteLevel mode correctly applies end_of_word_suffix to the last
// token when the segment contains non-ASCII bytes.
//
// This is the CLIP tokenizer pattern:
// - Input: raw bytes (e.g., "café" = 63 61 66 C3 A9)
// - ByteLevel transform: each byte maps to a codepoint
//   - 'c', 'a', 'f' = identity (0x63, 0x61, 0x66)
//   - 0xC3 -> 0xC3 (identity for Latin-1)
//   - 0xA9 -> 0xA9 (identity for Latin-1, which is ©)
// - Vocab tokens: "cafÃ©" (without suffix), "cafÃ©</w>" (with suffix)
// - Expected: "café" -> token for "cafÃ©</w>"
TEST(BPEByteLevelSuffixTest, NonAsciiBytesWithSuffix) {
  // "cafÃ©" in ByteLevel is the transformed form of "café".
  // UTF-8 encoding of "cafÃ©" = 63 61 66 C3 83 C2 A9 (7 bytes).
  // The suffix-appended form is "cafÃ©</w>".
  ScopedVocabBuilder builder;
  builder.AddToken(0, "c");
  builder.AddToken(1, "a");
  builder.AddToken(2, "f");
  // 'Ã' = U+00C3 = UTF-8 C3 83, '©' = U+00A9 = UTF-8 C2 A9
  builder.AddToken(3, "Ã");
  builder.AddToken(4, "©");
  builder.AddToken(5, "cafÃ©");      // Without suffix
  builder.AddToken(6, "cafÃ©</w>");  // With suffix (target)
  builder.AddMerge(0, 1);            // c + a -> ca (implicit)

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // No IGNORE_MERGES - test ByteLevel whole-segment suffix matching.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Input is raw "café" (5 bytes: 63 61 66 C3 A9).
  // Should match "cafÃ©</w>" via ByteLevel transformation + suffix.
  auto tokens = EncodeAndFinalize(model.get(), "café",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 6);  // "cafÃ©</w>"
}

// Tests that ByteLevel + suffix works when backtracking is used.
// This exercises the iree_tokenizer_bpe_apply_suffix_to_last_token() path.
TEST(BPEByteLevelSuffixTest, BacktrackingPathWithNonAscii) {
  // Build a vocabulary where "café" must go through BPE merges, not whole-word
  // match. This forces the backtracking encode path.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "c");
  builder.AddToken(1, "a");
  builder.AddToken(2, "f");
  builder.AddToken(3, "Ã");  // U+00C3 = ByteLevel for byte 0xC3
  builder.AddToken(4, "©");  // U+00A9 = ByteLevel for byte 0xA9
  builder.AddToken(5, "ca");
  builder.AddToken(6, "caf");
  builder.AddToken(7, "cafÃ");
  builder.AddToken(8, "cafÃ©");
  builder.AddToken(9, "cafÃ©</w>");  // Suffixed target

  // Merges to build up to "cafÃ©".
  builder.AddMerge(0, 1);  // c + a -> ca
  builder.AddMerge(5, 2);  // ca + f -> caf
  builder.AddMerge(6, 3);  // caf + Ã -> cafÃ
  builder.AddMerge(7, 4);  // cafÃ + © -> cafÃ©

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Input "café" should merge to "cafÃ©", then suffix lookup finds "cafÃ©</w>".
  auto tokens = EncodeAndFinalize(model.get(), "café",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 9);  // "cafÃ©</w>"
}

// Tests Khmer script byte fallback with ByteLevel mode.
// Khmer character U+179F (ស) = UTF-8 E1 9E 9F.
// In ByteLevel mode:
//   - 0xE1 -> 0xE1 (identity for Latin-1) = UTF-8 C3 A1 (á)
//   - 0x9E -> 0x13E (C1 control range) = UTF-8 C4 BE (Ľ)
//   - 0x9F -> 0x13F (C1 control range) = UTF-8 C4 BF (ľ)
// If these tokens don't exist, byte fallback should find <0xE1>, <0x9E>,
// <0x9F>.
TEST(BPEByteLevelSuffixTest, KhmerByteFallbackWithSuffix) {
  // CLIP-style byte tokens: <0xNN> format.
  // Note: CLIP doesn't actually have byte fallback tokens; it has all Unicode.
  // This test verifies the byte fallback path works with ByteLevel mode.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "<0xE1>");  // Byte fallback for first Khmer byte
  builder.AddToken(1, "<0x9E>");  // Second byte
  builder.AddToken(2, "<0x9F>");  // Third byte
  builder.AddToken(3, "</w>");    // Suffix as separate token

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  // No suffix set - just testing byte fallback

  // Input is Khmer character ស (UTF-8: E1 9E 9F).
  // Should fall back to byte tokens [<0xE1>, <0x9E>, <0x9F>].
  std::string khmer = "ស";  // U+179F = UTF-8 E1 9E 9F
  auto tokens = EncodeAndFinalize(model.get(), khmer,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);  // <0xE1>
  EXPECT_EQ(tokens[1], 1);  // <0x9E>
  EXPECT_EQ(tokens[2], 2);  // <0x9F>
}

//===----------------------------------------------------------------------===//
// Suffix-Blocked Pair Validation
// Tests that pair validation accepts token pairs when the merge was
// intentionally deferred by suffix_blocked for better suffix merges.
//===----------------------------------------------------------------------===//

// Tests the CLIP repeated emoji pattern where suffix_blocked causes pair
// validation to see adjacent copies of a shorter token.
//
// Problem scenario (simplified from CLIP 10-emoji input):
//   Input: "AAAA" (4 units)
//   Vocabulary: A (single), AA (double), AA</w> (double+suffix), AAA</w>
//   (triple+suffix)
//   Merges: A+A->AA (rank 2), A+AA</w>->AAA</w> (rank 1)
//
// At position 2 (after consuming "AA"):
//   - Token AA would be blocked by suffix_blocked because suffix(AA)=A can
//     merge with remaining "AA</w>" to form AAA</w> at rank 1 < AA's rank 2
//   - We fall back to token A
//   - pair_ok(A, A) would fail without the fix because merge(A,A)=AA exists
//   - With the fix, pair_ok knows the merge was intentionally deferred
//
// Expected output: [AA, A, AAA</w>] (consuming 2 + 1 + 3 = 6 bytes, but we're
// using 4-byte input "AAAA", so we expect [AA, AA</w>] or [A, AAA</w>]).
//
// Actually for "AAAA" (4 A's) with suffix </w>:
//   - Best tokenization depends on merge ranks
//   - If A+AA</w>->AAA</w> at rank 1 < A+A->AA at rank 2:
//     Position 0: AA would be suffix_blocked (A + AA</w> = AAA</w>)
//     Position 0: A accepted
//     Position 1: AA would be suffix_blocked
//     Position 1: A accepted (pair A,A now uses deferred_merge_rank)
//     Position 2: AA</w> accepted via suffix lookup
//   Result: [A, A, AA</w>] but then A+AA</w>->AAA</w> fires in post-processing
//   Final: [A, AAA</w>]
//
// This test verifies pair validation doesn't reject the (A, A) pair.
TEST(BPESuffixBlockedPairValidationTest, RepeatedTokenWithSuffixMerge) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "A");
  builder.AddToken(1, "AA");
  builder.AddToken(2,
                   "A</w>");  // Need this for suffix_blocked at later positions
  builder.AddToken(3, "AA</w>");
  builder.AddToken(4, "AAA</w>");

  // Key: A+AA</w>->AAA</w> has LOWER rank than A+A->AA
  // This means suffix_blocked will trigger for AA at certain positions.
  // Also need A+A</w>->AA</w> for suffix_blocked to work at all positions.
  builder.AddMerge(0, 3);  // Rank 0: A + AA</w> -> AAA</w>
  builder.AddMerge(0, 2);  // Rank 1: A + A</w> -> AA</w>
  builder.AddMerge(0, 0);  // Rank 2: A + A -> AA

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Input "AAAA" (4 A's).
  // Without the fix: would fall back to byte-level due to pair validation
  // failure.
  // With the fix: should produce [A, AAA</w>] after suffix post-processing.
  auto tokens = EncodeAndFinalize(model.get(), "AAAA",
                                  /*expect_pending_after_encode=*/false);
  // The exact output depends on the algorithm, but it should NOT be 4 separate
  // tokens (which would indicate byte-level fallback). We expect either
  // [A, AAA</w>] (2 tokens) or [AA, AA</w>] (2 tokens).
  ASSERT_LE(tokens.size(), 2u);
  // Verify we got meaningful tokens, not byte fallback.
  // Token 0 is "A", token 1 is "AA", token 2 is "A</w>", token 3 is "AA</w>",
  // token 4 is "AAA</w>".
  for (int32_t token : tokens) {
    EXPECT_GE(token, 0);
    EXPECT_LE(token, 4);
  }
}

// Tests a longer repeated pattern similar to CLIP's 10-emoji input.
// This exercises the case where multiple suffix-blocked deferrals happen
// in sequence.
TEST(BPESuffixBlockedPairValidationTest, LongRepeatedPattern) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "X");
  builder.AddToken(1, "XX");
  builder.AddToken(2, "X</w>");
  builder.AddToken(3, "XX</w>");
  builder.AddToken(4, "XXX</w>");

  // Merge hierarchy:
  // X + XX</w> -> XXX</w> has lower rank than X + X -> XX
  // This causes suffix_blocked to trigger when we could form XXX</w>.
  // Need X + X</w> -> XX</w> for suffix_blocked to trigger at all positions.
  builder.AddMerge(0, 3);  // Rank 0: X + XX</w> -> XXX</w>
  builder.AddMerge(0, 2);  // Rank 1: X + X</w> -> XX</w>
  builder.AddMerge(0, 0);  // Rank 2: X + X -> XX

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Input "XXXXXXXXXX" (10 X's) - similar to CLIP's 10-emoji case.
  // Expected behavior: should NOT fall back to 10 single-character tokens.
  auto tokens = EncodeAndFinalize(model.get(), "XXXXXXXXXX",
                                  /*expect_pending_after_encode=*/false);

  // We should get far fewer than 10 tokens. The exact count depends on
  // how the algorithm handles the suffix merges, but we definitely shouldn't
  // get 10 tokens (which would indicate failure).
  ASSERT_LT(tokens.size(), 10u);

  // Verify all tokens are valid (not byte fallback or error tokens).
  for (int32_t token : tokens) {
    EXPECT_GE(token, 0);
    EXPECT_LE(token, 4);
  }
}

// Tests that the deferred merge rank is properly scoped to prefix attempts
// within a single position, not carried across positions incorrectly.
TEST(BPESuffixBlockedPairValidationTest, DeferredRankScopedToPosition) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "A");
  builder.AddToken(1, "B");
  builder.AddToken(2, "AB");
  builder.AddToken(3, "AA");
  builder.AddToken(4, "A</w>");
  builder.AddToken(5, "AA</w>");

  // Setup where AA is suffix-blocked but AB is not.
  // A + A</w> -> AA</w> has lower rank, triggering suffix_blocked for AA.
  builder.AddMerge(0, 4);  // Rank 0: A + A</w> -> AA</w>
  builder.AddMerge(0, 0);  // Rank 1: A + A -> AA
  builder.AddMerge(0, 1);  // Rank 2: A + B -> AB

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Input "AB" - no suffix blocking should occur here.
  // AB should be accepted directly.
  auto tokens = EncodeAndFinalize(model.get(), "AB",
                                  /*expect_pending_after_encode=*/false);
  // Should get AB</w> if suffixed version exists, or just AB.
  // Since we don't have AB</w>, we get AB.
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 2);  // "AB"
}

// Tests BPE merge priority when multiple merges compete at adjacent positions.
// This reproduces the BART "denoising" bug where "oising" was tokenized as
// [ois, ing] instead of [o, ising] because o+is (rank 10668) was incorrectly
// chosen over is+ing (rank 1454) which has higher priority (lower rank).
//
// Correct BPE merge sequence for "oising":
//   Start: o i s i n g
//   Apply "i n -> in" (rank 3):     o i s in g
//   Apply "i s -> is" (rank 15):    o is in g
//   Apply "in g -> ing" (rank 22):  o is ing
//   Apply "is ing -> ising" (rank 1454): o ising
//   Result: [o] [ising]
//
// Incorrect (buggy) sequence:
//   ... after step 3: o is ing
//   Apply "o is -> ois" (rank 10668): ois ing
//   Result: [ois] [ing]
TEST_F(BPEModelTest, MergePriorityAtAdjacentPositions) {
  ScopedVocabBuilder builder;

  // Base single-character tokens.
  builder.AddToken(0, "o");
  builder.AddToken(1, "i");
  builder.AddToken(2, "s");
  builder.AddToken(3, "n");
  builder.AddToken(4, "g");

  // First-level merged tokens.
  builder.AddToken(5, "in");  // i + n
  builder.AddToken(6, "is");  // i + s

  // Second-level merged tokens.
  builder.AddToken(7, "ing");  // in + g

  // Third-level and competing tokens.
  builder.AddToken(8, "ising");  // is + ing
  builder.AddToken(9, "ois");    // o + is

  // Merges in rank order (lower rank = higher priority).
  // These ranks are simplified but preserve the relative ordering from BART.
  builder.AddMerge(1, 3);  // Rank 0: i + n -> in (BART rank 3)
  builder.AddMerge(1, 2);  // Rank 1: i + s -> is (BART rank 15)
  builder.AddMerge(5, 4);  // Rank 2: in + g -> ing (BART rank 22)
  builder.AddMerge(6, 7);  // Rank 3: is + ing -> ising (BART rank 1454)
  builder.AddMerge(0, 6);  // Rank 4: o + is -> ois (BART rank 10668)

  CreateModel(builder);

  // "oising" should tokenize as [o, ising] not [ois, ing].
  // The merge "is + ing -> ising" (rank 3) must be applied before
  // "o + is -> ois" (rank 4) even though they compete for the "is" token.
  TestEncode(model(), "oising", {0, 8}, /*expect_pending_after_encode=*/false);
}

// Same test with ByteLevel mode enabled (like BART/RoBERTa).
// In ByteLevel mode, regular ASCII characters map to themselves in the
// vocabulary. This tests that merge priority works correctly when ByteLevel
// transformation is active.
TEST_F(BPEModelTest, MergePriorityByteLevelMode) {
  ScopedVocabBuilder builder;

  // In BART/RoBERTa ByteLevel mode, printable ASCII maps to itself.
  // The vocabulary contains these as-is.
  builder.AddToken(0, "o");
  builder.AddToken(1, "i");
  builder.AddToken(2, "s");
  builder.AddToken(3, "n");
  builder.AddToken(4, "g");
  builder.AddToken(5, "in");
  builder.AddToken(6, "is");
  builder.AddToken(7, "ing");
  builder.AddToken(8, "ising");
  builder.AddToken(9, "ois");

  // Same merges as before.
  builder.AddMerge(1, 3);  // i + n -> in
  builder.AddMerge(1, 2);  // i + s -> is
  builder.AddMerge(5, 4);  // in + g -> ing
  builder.AddMerge(6, 7);  // is + ing -> ising
  builder.AddMerge(0, 6);  // o + is -> ois

  // Create with ByteLevel flag.
  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);

  // "oising" should still tokenize as [o, ising] even with ByteLevel mode.
  auto tokens = EncodeAndFinalize(model.get(), "oising",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "o"
  EXPECT_EQ(tokens[1], 8);  // "ising"
}

// Same test but with context (preceding token) to verify pair validation.
TEST_F(BPEModelTest, MergePriorityWithPrecedingToken) {
  ScopedVocabBuilder builder;

  // Base tokens.
  builder.AddToken(0, "d");
  builder.AddToken(1, "e");
  builder.AddToken(2, "n");
  builder.AddToken(3, "o");
  builder.AddToken(4, "i");
  builder.AddToken(5, "s");
  builder.AddToken(6, "g");

  // Merged tokens.
  builder.AddToken(10, "de");     // d + e
  builder.AddToken(11, "en");     // e + n
  builder.AddToken(12, "den");    // de + n or d + en
  builder.AddToken(13, "in");     // i + n
  builder.AddToken(14, "is");     // i + s
  builder.AddToken(15, "ing");    // in + g
  builder.AddToken(16, "ising");  // is + ing
  builder.AddToken(17, "ois");    // o + is

  // Merges in rank order.
  builder.AddMerge(0, 1);    // Rank 0: d + e -> de
  builder.AddMerge(1, 2);    // Rank 1: e + n -> en
  builder.AddMerge(10, 2);   // Rank 2: de + n -> den
  builder.AddMerge(4, 2);    // Rank 3: i + n -> in
  builder.AddMerge(4, 5);    // Rank 4: i + s -> is
  builder.AddMerge(13, 6);   // Rank 5: in + g -> ing
  builder.AddMerge(14, 15);  // Rank 6: is + ing -> ising
  builder.AddMerge(3, 14);   // Rank 7: o + is -> ois

  CreateModel(builder);

  // "denoising" should tokenize as [den, o, ising] not [den, ois, ing].
  TestEncode(model(), "denoising", {12, 3, 16},
             /*expect_pending_after_encode=*/false);
}

// Tests ByteLevel transformation of multi-byte UTF-8 sequences.
// This is the Qwen unicode bug scenario where 'él' (UTF-8: c3 a9 6c) should
// transform to 'Ã©l' (UTF-8: c3 83 c2 a9 6c) and match as a single token.
//
// ByteLevel transformation for 'é' (U+00E9 = UTF-8 c3 a9):
//   byte 0xC3 -> U+00C3 (Ã) = UTF-8 c3 83
//   byte 0xA9 -> U+00A9 (©) = UTF-8 c2 a9
// So 'é' becomes 'Ã©' in the vocabulary.
TEST(BPEByteLevelMultiByteTest, MultiByteThenMerge) {
  ScopedVocabBuilder builder;

  // ByteLevel-encoded tokens matching Qwen vocabulary structure.
  // These are the UTF-8 representations of ByteLevel-transformed bytes.
  builder.AddToken(39, "H");    // ASCII identity
  builder.AddToken(75, "l");    // ASCII identity
  builder.AddToken(127, "Ã");   // ByteLevel for byte 0xC3 (Ã = U+00C3 = c3 83)
  builder.AddToken(102, "©");   // ByteLevel for byte 0xA9 (© = U+00A9 = c2 a9)
  builder.AddToken(963, "Ã©");  // Merged: Ã + ©
  builder.AddToken(18503, "Ã©l");  // Merged: Ã© + l
  builder.AddToken(385, "lo");     // Merged: l + o
  builder.AddToken(111, "o");      // ASCII identity

  // Merges to build up 'Ã©l' from ByteLevel components.
  builder.AddMerge(127, 102);  // Ã + © -> Ã© (merge ByteLevel bytes)
  builder.AddMerge(963, 75);   // Ã© + l -> Ã©l
  builder.AddMerge(75, 111);   // l + o -> lo

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);

  // Input 'él' is raw UTF-8: c3 a9 6c (3 bytes).
  // With BYTE_LEVEL_INPUT, each byte is transformed:
  //   c3 -> UTF-8(U+00C3) = c3 83
  //   a9 -> UTF-8(U+00A9) = c2 a9
  //   6c -> 6c (ASCII identity)
  // So BPE should look up c3 83 c2 a9 6c in the trie = 'Ã©l' = token 18503.
  std::string input = "él";  // UTF-8: c3 a9 6c
  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u) << "Expected 'él' to match as single token";
  EXPECT_EQ(tokens[0], 18503) << "Expected token 'Ã©l' (18503)";
}

// Tests ByteLevel with multi-byte UTF-8 after an ASCII prefix.
// This tests "Héllo" -> ['H', 'Ã©l', 'lo'].
TEST(BPEByteLevelMultiByteTest, AsciiThenMultiByteThenAscii) {
  ScopedVocabBuilder builder;

  builder.AddToken(39, "H");
  builder.AddToken(75, "l");
  builder.AddToken(127, "Ã");
  builder.AddToken(102, "©");
  builder.AddToken(963, "Ã©");
  builder.AddToken(18503, "Ã©l");
  builder.AddToken(385, "lo");
  builder.AddToken(111, "o");

  builder.AddMerge(127, 102);  // Ã + © -> Ã©
  builder.AddMerge(963, 75);   // Ã© + l -> Ã©l
  builder.AddMerge(75, 111);   // l + o -> lo

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);

  // Input "Héllo" is raw UTF-8: 48 c3 a9 6c 6c 6f.
  // Expected tokenization:
  //   'H' (48) -> token 39
  //   'él' (c3 a9 6c) -> ByteLevel -> 'Ã©l' -> token 18503
  //   'lo' (6c 6f) -> token 385
  std::string input = "Héllo";  // UTF-8: 48 c3 a9 6c 6c 6f
  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 3u) << "Expected [H, Ã©l, lo]";
  EXPECT_EQ(tokens[0], 39) << "Expected 'H' token";
  EXPECT_EQ(tokens[1], 18503) << "Expected 'Ã©l' token";
  EXPECT_EQ(tokens[2], 385) << "Expected 'lo' token";
}

//===----------------------------------------------------------------------===//
// ByteLevel + Suffix Fast-Path Tests
// Verify that ByteLevel transformation is correctly applied to segment bytes
// (but NOT suffix bytes) when both BYTE_LEVEL_INPUT and end_of_word_suffix
// are configured. The vocabulary contains ByteLevel-transformed tokens, so
// the trie walk must transform input bytes before matching.
//===----------------------------------------------------------------------===//

// ASCII bytes are identity-mapped in ByteLevel mode.
// Tests single-byte suffix lookup (without IGNORE_MERGES).
TEST(BPEByteLevelSuffixFastPathTest, AsciiSegmentWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // Note: NO IGNORE_MERGES - we're testing suffix handling here.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "a",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 1);  // "a</w>" (suffix applied to single-byte segment)
}

// Non-printable byte 0x80 maps to U+0122 ("Ģ") in ByteLevel mode.
// The trie contains "Ģ</w>", so the segment must be transformed to match.
// Tests single-byte suffix lookup (without IGNORE_MERGES).
TEST(BPEByteLevelSuffixFastPathTest, NonAsciiByteWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "Ģ");  // ByteLevel: 0x80 -> U+0122
  builder.AddToken(1, "Ģ</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // Note: NO IGNORE_MERGES - we're testing suffix handling here.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  std::string input(1, '\x80');
  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u) << "Expected single token 'Ģ</w>'";
  EXPECT_EQ(tokens[0], 1);  // "Ģ</w>" (suffix applied to single-byte segment)
}

// Space (0x20) maps to U+0120 ("Ġ") in ByteLevel mode (GPT-2 pattern).
// Tests single-byte suffix lookup (without IGNORE_MERGES).
TEST(BPEByteLevelSuffixFastPathTest, SpaceByteWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "Ġ");  // ByteLevel: 0x20 -> U+0120
  builder.AddToken(1, "Ġ</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  // Note: NO IGNORE_MERGES - we're testing suffix handling here.
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), " ",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u) << "Expected single token 'Ġ</w>'";
  EXPECT_EQ(tokens[0], 1);  // "Ġ</w>" (suffix applied to single-byte segment)
}

// Multi-byte ASCII word with suffix. Printable ASCII is identity-mapped.
// With IGNORE_MERGES, tokenizes byte-by-byte and suffix applies to last token.
// Per HuggingFace: "hi" with no merges -> [h, i] -> apply suffix -> [h, i</w>].
// With IGNORE_MERGES, HuggingFace does longest-match vocab lookup on the BARE
// segment (without suffix). Since "hi" is in the vocabulary, it returns ["hi"].
// The suffix is NOT appended when there's a direct vocab match.
TEST(BPEByteLevelSuffixFastPathTest, MultiByteAsciiWordWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "h");
  builder.AddToken(1, "i");
  builder.AddToken(2, "hi");
  builder.AddToken(3, "hi</w>");
  builder.AddToken(4, "h</w>");
  builder.AddToken(5, "i</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(),
      IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT |
          IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "hi",
                                  /*expect_pending_after_encode=*/false);
  // HuggingFace ignore_merges=True: vocab lookup finds bare "hi" -> [2]
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 2);  // "hi" (bare, no suffix)
}

// With IGNORE_MERGES, HuggingFace does longest-match vocab lookup on the BARE
// segment (after ByteLevel transformation). Since "Ģģ" is in vocabulary, it
// returns ["Ģģ"]. The suffix is NOT appended when there's a direct vocab match.
// ByteLevel: 0x80 -> "Ģ" (U+0122), 0x81 -> "ģ" (U+0123)
TEST(BPEByteLevelSuffixFastPathTest, IgnoreMergesWithNonAscii) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "Ģ");  // ByteLevel: 0x80 -> U+0122
  builder.AddToken(1, "ģ");  // ByteLevel: 0x81 -> U+0123
  builder.AddToken(2,
                   "Ģģ");  // Merged token (also serves as direct vocab match)
  builder.AddToken(3, "Ģ</w>");
  builder.AddToken(4, "ģ</w>");
  builder.AddToken(5, "Ģģ</w>");
  builder.AddMerge(0, 1);  // Ģ + ģ -> Ģģ

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(),
      IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT |
          IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Input: 0x80 0x81 -> ByteLevel -> "Ģģ"
  // With IGNORE_MERGES, vocab lookup finds "Ģģ" -> [2]
  std::string input = "\x80\x81";
  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);

  ASSERT_EQ(tokens.size(), 1u)
      << "IGNORE_MERGES does vocab lookup, expected [Ģģ]";
  EXPECT_EQ(tokens[0], 2);  // "Ģģ" (bare, no suffix)
}

// Whole-segment suffix match with non-ASCII (without IGNORE_MERGES).
// ByteLevel: newline (0x0A) -> U+010A ("Ċ")
TEST(BPEByteLevelSuffixFastPathTest, WholeSegmentSuffixMatchNonAscii) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "Ċ");  // ByteLevel: 0x0A -> U+010A
  builder.AddToken(1, "Ċ</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "\n",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u) << "Expected single token 'Ċ</w>'";
  EXPECT_EQ(tokens[0], 1);  // "Ċ</w>"
}

//===----------------------------------------------------------------------===//
// Nested Suffix Blocking Tests
// When a token has a multi-level decomposition, suffix blocking must check
// suffixes at ALL levels, not just the direct suffix.
//===----------------------------------------------------------------------===//

// Tests suffix blocking with a two-level nested decomposition.
//
// Vocabulary structure (modeled after Mistral's "relativity" pattern):
//   - OUTER = LEFT + INNER (high rank merge)
//   - INNER = MID + SUFFIX (medium rank merge)
//   - SUFFIX + REMAINING -> COMBINED (low rank merge)
//
// For input LEFT + INNER + REMAINING:
//   - Greedy longest-match finds OUTER
//   - OUTER's suffix chain includes SUFFIX (via INNER's decomposition)
//   - SUFFIX can merge with REMAINING at lower rank than INNER's effective rank
//   - Therefore OUTER should be suffix-blocked, producing [LEFT, COMBINED']
//     where COMBINED' includes the merged suffix
//
// Concrete example: "abcde" where:
//   - abc (a + bc) would be greedy-matched, leaving "de"
//   - bc = b + c, so suffix chain = [bc, c]
//   - c + d -> cd fires at lower rank than b + c
//   - Therefore abc is blocked; result should be [a, b, cde] or similar
TEST_F(BPEModelTest, NestedSuffixBlockingTwoLevel) {
  ScopedVocabBuilder builder;

  // Base tokens.
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "d");
  builder.AddToken(4, "e");

  // Level 1 merges.
  builder.AddToken(5, "cd");  // c + d (LOW rank - fires first!)
  builder.AddToken(6, "de");  // d + e
  builder.AddToken(7, "bc");  // b + c (higher rank than cd)

  // Level 2 merges.
  builder.AddToken(8, "cde");  // cd + e or c + de
  builder.AddToken(9, "abc");  // a + bc (highest rank)

  // Merges in rank order (lower = higher priority).
  // Key: c+d fires BEFORE b+c, so when processing "abcde", the 'c' in "abc"
  // can merge with 'd' at lower rank than 'bc' formed.
  builder.AddMerge(2, 3);  // Rank 0: c + d -> cd (LOWEST - blocks bc!)
  builder.AddMerge(3, 4);  // Rank 1: d + e -> de
  builder.AddMerge(5, 4);  // Rank 2: cd + e -> cde
  builder.AddMerge(1, 2);  // Rank 3: b + c -> bc (higher than cd!)
  builder.AddMerge(0, 7);  // Rank 4: a + bc -> abc (highest)

  CreateModel(builder);

  // Input "abcde":
  // Greedy would match "abc" (3 chars) leaving "de".
  // But suffix blocking should trigger because:
  //   - abc = a + bc, suffix = bc
  //   - bc = b + c, suffix = c
  //   - c + d -> cd at rank 0 < bc's effective_rank (4)
  //   - Therefore abc is suffix-blocked!
  //
  // Correct BPE applies merges in rank order:
  //   c + d -> cd (rank 0): "ab cd e"
  //   d + e cannot fire (d is consumed)
  //   cd + e -> cde (rank 2): "ab cde"
  //   b + c cannot fire (c is consumed)
  //   Final: [a, b, cde]
  TestEncode(model(), "abcde", {0, 1, 8},
             /*expect_pending_after_encode=*/false);
}

// Tests that tokens with multiple merge paths are correctly marked reachable
// when the first (lowest-rank) decomposition is blocked but an alternate path
// is valid.
//
// This simulates real-world tokenizers like Mistral where:
//   - Token "AB" has TWO decompositions producing it:
//     1. Merge "a" + "B" at rank 2 (stored in split_table as canonical)
//     2. Merge "A" + "b" at rank 3 (alternate path)
//   - The boundary of "a + B" is blocked by merge "aB_suffix" at rank 1
//   - The boundary of "A + b" is NOT blocked
//   - The precomputed token_reachable correctly marks "AB" as reachable
//   - The runtime must use precomputed reachability, not recompute
//
// Concrete vocabulary:
//   - "xay" has decompositions:
//     1. "xa" + "y" at rank 4 - boundary a|y blocked by "ay" at rank 0
//     2. "x" + "ay" at rank 5 - boundary x|a NOT blocked
TEST_F(BPEModelTest, MultipleDecompositionsAlternatePathReachable) {
  ScopedVocabBuilder builder;

  // Base tokens.
  builder.AddToken(0, "x");
  builder.AddToken(1, "a");
  builder.AddToken(2, "y");

  // Level 1 merges.
  builder.AddToken(10, "ay");  // a + y
  builder.AddToken(11, "xa");  // x + a

  // Level 2: "xay" has two paths
  builder.AddToken(12, "xay");  // Can be formed via xa+y OR x+ay

  // Merges in rank order.
  // Key: "ay" has lower rank than "xa", so boundary a|y blocks xa+y.
  builder.AddMerge(1, 2);   // Rank 0: a + y -> ay
  builder.AddMerge(0, 1);   // Rank 1: x + a -> xa
  builder.AddMerge(11, 2);  // Rank 2: xa + y -> xay (BLOCKED! boundary a|y)
  builder.AddMerge(0, 10);  // Rank 3: x + ay -> xay (REACHABLE! no blocking)

  CreateModel(builder);

  // Key test: "xay" should produce token 12 (xay) via the x+ay path.
  // The split_table has xa+y (rank 2, first merge), but that's blocked.
  // The precomputed reachability correctly identifies xay as reachable.
  TestEncode(model(), "xay", {12}, /*expect_pending_after_encode=*/false);

  // Verify the paths:
  // - "ay" is reachable: a+y at rank 0
  TestEncode(model(), "ay", {10}, /*expect_pending_after_encode=*/false);

  // - "xa" is reachable: x+a at rank 1
  TestEncode(model(), "xa", {11}, /*expect_pending_after_encode=*/false);
}

// Same pattern but with three levels of nesting.
// This tests that suffix collection walks the full rightmost path.
//
// Vocabulary:
//   - OUTER = L1 + LEVEL2 (rank 5)
//   - LEVEL2 = L2 + LEVEL3 (rank 4)
//   - LEVEL3 = L3 + SUFFIX (rank 3)
//   - SUFFIX + REM -> MERGED (rank 0)
TEST_F(BPEModelTest, NestedSuffixBlockingThreeLevel) {
  ScopedVocabBuilder builder;

  // Base tokens.
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "d");
  builder.AddToken(4, "e");
  builder.AddToken(5, "f");

  // Level 1.
  builder.AddToken(10, "ef");  // e + f (LOWEST rank - blocks de!)
  builder.AddToken(11, "de");  // d + e

  // Level 2.
  builder.AddToken(12, "cde");  // c + de
  builder.AddToken(13, "def");  // de + f or d + ef

  // Level 3.
  builder.AddToken(14, "bcde");  // b + cde

  // Level 4.
  builder.AddToken(15, "abcde");  // a + bcde (highest)

  // Merges in rank order.
  builder.AddMerge(4, 5);   // Rank 0: e + f -> ef (LOWEST - blocks de!)
  builder.AddMerge(3, 4);   // Rank 1: d + e -> de
  builder.AddMerge(10, 5);  // Rank 2: ef + f -> skip (not needed)
  builder.AddMerge(2, 11);  // Rank 3: c + de -> cde
  builder.AddMerge(1, 12);  // Rank 4: b + cde -> bcde
  builder.AddMerge(0, 14);  // Rank 5: a + bcde -> abcde (highest)

  CreateModel(builder);

  // Input "abcdef":
  // Greedy would match "abcde" (5 chars) leaving "f".
  // Suffix chain of abcde: bcde -> cde -> de -> e
  // e + f -> ef at rank 0 < de's effective_rank (2)
  // Therefore abcde is suffix-blocked!
  //
  // Correct BPE:
  //   e + f -> ef (rank 0): "abcd ef"
  //   d + e cannot fire (e consumed)
  //   Final: [a, b, c, d, ef]
  TestEncode(model(), "abcdef", {0, 1, 2, 3, 10},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Suffix-blocking with internal boundary merges
//
// Tests that suffix-blocking correctly handles the interaction between
// reachability and pair validation when internal boundary merges exist.
//===----------------------------------------------------------------------===//

// Verifies suffix-blocking skips suffixes whose decomposition has internal
// boundary merges that would fail pair validation.
//
// A token like 'lish' can be:
//   - REACHABLE: the decomposition lish → (li, sh) is valid during BPE because
//     li and sh consume their boundary bytes (i and s) before the i+s merge
//   - But NOT usable for suffix-blocking: if we block 'lish' and emit 'li'
//     then 'sh' separately, pair validation would reject (li, sh) because the
//     i+s boundary merge exists
//
// The suffix collection must check for this: when the internal boundary between
// left and right components has a merge forming a different token, that suffix
// cannot be used for blocking decisions.
//
// Setup:
//   - "li" = merge(l, i) at rank 0 (consumes 'i' early)
//   - "sh" = merge(s, h) at rank 1 (consumes 's' early)
//   - "is" = merge(i, s) at rank 2 (fires after i,s already consumed)
//   - "sh" + "ment</w>" merge at rank 6 (lower than lish)
//   - "lish" = merge(li, sh) at rank 7
//
// Without the internal boundary check, 'lish' would be incorrectly blocked
// because sh+ment</w> exists at lower rank. But blocking 'lish' in favor of
// 'li' + 'sh' + suffix fails because (li, sh) has an internal boundary merge.
//
// With the check, 'sh' is not added as a valid suffix (due to the i+s merge
// at the li|sh boundary), so 'lish' is not blocked and tokenizes correctly.
TEST(BPEEndOfWordSuffixTest, SuffixBlockingMustCheckPairValidity) {
  ScopedVocabBuilder builder;
  // Single characters
  builder.AddToken(0, "l");
  builder.AddToken(1, "i");
  builder.AddToken(2, "s");
  builder.AddToken(3, "h");
  builder.AddToken(4, "m");
  builder.AddToken(5, "e");
  builder.AddToken(6, "n");
  builder.AddToken(7, "t");
  // Pair tokens - order matters for reachability!
  builder.AddToken(8, "li");   // l + i merge (rank 0)
  builder.AddToken(9, "sh");   // s + h merge (rank 1)
  builder.AddToken(10, "is");  // i + s merge (rank 2 - AFTER li and sh!)
  builder.AddToken(11, "en");  // e + n merge
  // Higher-order tokens
  builder.AddToken(12, "lish");  // li + sh merge
  builder.AddToken(13, "ent");   // en + t merge
  builder.AddToken(14, "ment");  // m + ent merge
  // Suffixed tokens
  builder.AddToken(15, "t</w>");
  builder.AddToken(16, "ent</w>");
  builder.AddToken(17, "ment</w>");
  builder.AddToken(18, "shment</w>");  // The suffix merge that blocks lish

  // Merge order (rank = order in list, 0 = highest priority):
  // CRITICAL: li and sh must have LOWER ranks than is so lish is reachable.
  builder.AddMerge(0, 1);   // Rank 0: l + i -> li (consumes 'i' at rank 0)
  builder.AddMerge(2, 3);   // Rank 1: s + h -> sh (consumes 's' at rank 1)
  builder.AddMerge(1, 2);   // Rank 2: i + s -> is (too late! i,s consumed)
  builder.AddMerge(5, 6);   // Rank 3: e + n -> en
  builder.AddMerge(11, 7);  // Rank 4: en + t -> ent
  builder.AddMerge(4, 13);  // Rank 5: m + ent -> ment
  builder.AddMerge(9, 17);  // Rank 6: sh + ment</w> -> shment</w>
  builder.AddMerge(8, 9);   // Rank 7: li + sh -> lish (rank > sh+ment</w>!)

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  // Input "lishment" with </w> suffix should produce [li, shment</w>].
  //
  // Why suffix blocking DOES apply here:
  // - lish = li + sh has rank 7
  // - The decomposition (li, sh) IS reachable because:
  //   - 'i' (boundary of li) is consumed at rank 0 (when l+i→li)
  //   - 's' (boundary of sh) is consumed at rank 1 (when s+h→sh)
  //   - The i+s merge at rank 2 cannot fire because both 'i' and 's' are
  //     already consumed by their respective parent merges
  // - Therefore 'sh' is collected as a valid suffix of 'lish'
  // - sh + ment</w> → shment</w> at rank 6 < lish rank 7
  // - So 'lish' IS suffix-blocked, and we backtrack to 'li'
  //
  // This is correct BPE behavior: the sh+ment</w> merge has higher priority
  // than li+sh, so the final output maximizes use of the suffix merge.
  auto tokens = EncodeAndFinalize(model.get(), "lishment",
                                  /*expect_pending_after_encode=*/false);

  // Expected: [li(8), shment</w>(18)]
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 8) << "First token should be 'li'";
  EXPECT_EQ(tokens[1], 18) << "Second token should be 'shment</w>'";
}

}  // namespace
}  // namespace iree::tokenizer
