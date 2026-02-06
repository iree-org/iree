// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/model/unigram.h"

#include <cmath>
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

//===----------------------------------------------------------------------===//
// Test fixture for Unigram model tests.
//===----------------------------------------------------------------------===//

class UnigramModelTest : public ::testing::Test {
 protected:
  // Helper to create model from vocab builder with default settings.
  void CreateModel(
      ScopedVocabBuilder& builder, iree_tokenizer_token_id_t unk_token_id,
      float unk_score = -10.0f,
      iree_tokenizer_unigram_flags_t flags = IREE_TOKENIZER_UNIGRAM_FLAG_NONE) {
    vocab_ = ScopedVocab(builder.Build());
    iree_tokenizer_model_t* raw_model = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
        vocab_.get(), unk_token_id, unk_score, flags, iree_allocator_system(),
        &raw_model));
    model_ = ScopedModel(raw_model);
  }

  // Helper for models without UNK (byte fallback only).
  void CreateModelNoUnk(
      ScopedVocabBuilder& builder,
      iree_tokenizer_unigram_flags_t flags = IREE_TOKENIZER_UNIGRAM_FLAG_NONE) {
    vocab_ = ScopedVocab(builder.Build());
    iree_tokenizer_model_t* raw_model = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
        vocab_.get(), IREE_TOKENIZER_TOKEN_ID_INVALID, -10.0f, flags,
        iree_allocator_system(), &raw_model));
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

TEST_F(UnigramModelTest, CreateAndDestroy) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);
  EXPECT_NE(model(), nullptr);
}

TEST_F(UnigramModelTest, StateSizeIsReasonable) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);

  iree_host_size_t state_size = iree_tokenizer_model_state_size(model());
  EXPECT_GT(state_size, 0u);
  // State = struct + DP tables (3 arrays * (chunk_size+1) entries)
  //       + pending buffer (chunk_size entries).
  // chunk_size = max(max_token_length, 128). For this vocab: chunk_size=128.
  // DP: 129*(4+4+2) = 1290 bytes, pending: 128*12 = 1536 bytes, struct ~48.
  // Total ~3KB.
  EXPECT_LE(state_size, 8192u);
}

TEST_F(UnigramModelTest, RequiresUnkWhenNoByteFallback) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  // No UNK token, but byte_fallback disabled.
  auto* vocab = builder.Build();
  iree_tokenizer_model_t* raw_model = nullptr;
  iree_status_t status = iree_tokenizer_unigram_model_allocate(
      vocab, IREE_TOKENIZER_TOKEN_ID_INVALID, -10.0f,
      IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK, iree_allocator_system(),
      &raw_model);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_tokenizer_vocab_free(vocab);
}

TEST_F(UnigramModelTest, AllowsNoUnkWithByteFallback) {
  // No UNK token is allowed when byte_fallback is enabled (default).
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  // Add byte fallback tokens.
  for (int i = 0; i < 256; ++i) {
    char text[8];
    snprintf(text, sizeof(text), "<0x%02X>", i);
    builder.AddTokenWithScore(1 + i, text, -5.0f);
  }

  CreateModelNoUnk(builder);
  EXPECT_NE(model(), nullptr);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, EmptySegment) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);

  std::vector<iree_tokenizer_segment_t> segments = {{0, 0}};
  auto tokens = EncodeAndFinalize(model(), "", segments,
                                  /*expect_pending_after_encode=*/false);
  EXPECT_TRUE(tokens.empty());
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

// Unigram uses Viterbi DP to find the segmentation with highest total score.
// Unlike WordPiece (greedy longest-match), Unigram considers all valid paths.

TEST_F(UnigramModelTest, SingleCharToken) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);

  TestEncode(model(), "a", {0}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, WholeWordMatch) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "hello", -2.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);

  TestEncode(model(), "hello", {0}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, MultipleTokens) {
  // "hello" can be split as "hel" + "lo" if those tokens exist.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "hel", -1.5f);
  builder.AddTokenWithScore(1, "lo", -1.5f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  TestEncode(model(), "hello", {0, 1}, /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Viterbi Optimal Selection
//===----------------------------------------------------------------------===//

// Key test: Unigram picks the highest-scoring path, not just greedy.

TEST_F(UnigramModelTest, ViterbiPicksHighestScorePath) {
  // Given "abc", test that Viterbi picks the path with highest total score.
  // Option 1: "abc" (single token, score -3.0)
  // Option 2: "a" + "bc" (scores -1.0 + -1.5 = -2.5) <-- better!
  // Option 3: "ab" + "c" (scores -2.0 + -1.0 = -3.0)
  // Viterbi should pick option 2 with total score -2.5.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "abc", -3.0f);  // Whole word
  builder.AddTokenWithScore(1, "a", -1.0f);    // Single char
  builder.AddTokenWithScore(2, "bc", -1.5f);   // Two chars
  builder.AddTokenWithScore(3, "ab", -2.0f);   // Two chars
  builder.AddTokenWithScore(4, "c", -1.0f);    // Single char
  builder.AddTokenWithScore(5, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 5);

  CreateModel(builder, /*unk_token_id=*/5);

  // Should pick "a" + "bc" (IDs 1, 2) with total -2.5, not "abc" (ID 0, -3.0).
  TestEncode(model(), "abc", {1, 2}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, ViterbiPrefersWholeWordWhenBetter) {
  // When whole word has better score, pick it.
  // Option 1: "abc" (single token, score -1.0) <-- better!
  // Option 2: "a" + "bc" (scores -2.0 + -2.0 = -4.0)
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "abc", -1.0f);  // Whole word - better score
  builder.AddTokenWithScore(1, "a", -2.0f);
  builder.AddTokenWithScore(2, "bc", -2.0f);
  builder.AddTokenWithScore(3, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 3);

  CreateModel(builder, /*unk_token_id=*/3);

  // Should pick "abc" (ID 0) with -1.0, not "a"+"bc" (IDs 1,2, -4.0).
  TestEncode(model(), "abc", {0}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, ViterbiWithLongerPath) {
  // "abcd": multiple paths exist.
  // Option 1: "abcd" (score -5.0)
  // Option 2: "ab" + "cd" (scores -1.0 + -1.0 = -2.0) <-- best!
  // Option 3: "a" + "b" + "cd" (scores -1.0 + -1.0 + -1.0 = -3.0)
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "abcd", -5.0f);
  builder.AddTokenWithScore(1, "ab", -1.0f);
  builder.AddTokenWithScore(2, "cd", -1.0f);
  builder.AddTokenWithScore(3, "a", -1.0f);
  builder.AddTokenWithScore(4, "b", -1.0f);
  builder.AddTokenWithScore(5, "c", -1.0f);
  builder.AddTokenWithScore(6, "d", -1.0f);
  builder.AddTokenWithScore(7, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 7);

  CreateModel(builder, /*unk_token_id=*/7);

  // Should pick "ab" + "cd" (IDs 1, 2) with total -2.0.
  TestEncode(model(), "abcd", {1, 2}, /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// UNK Handling
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, UnknownCharacter) {
  // Character not in vocab -> UNK (when byte fallback not available).
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  TestEncode(model(), "xyz", {1}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, UnkFusionDefault) {
  // By default, consecutive UNK-producing characters are fused into one UNK.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  // "xyz" -> single [UNK], not three [UNK]s.
  TestEncode(model(), "xyz", {1}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, NoUnkFusionFlag) {
  // With NO_FUSE_UNK flag, each UNK-producing character emits separate UNK.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK |
                  IREE_TOKENIZER_UNIGRAM_FLAG_NO_FUSE_UNK);

  // "xyz" -> three [UNK]s.
  TestEncode(model(), "xyz", {1, 1, 1}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, PartialUnknown) {
  // Mix of known and unknown characters.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "b", -1.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  // "axb" -> "a", [UNK], "b"
  TestEncode(model(), "axb", {0, 2, 1}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, SeparatedUnreachable_FuseUnk_ProducesSeparateUnks) {
  // UNK fusion only merges CONSECUTIVE unreachable characters. When a reachable
  // token separates two unreachable regions, each region produces its own UNK.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "b", -1.0f);
  builder.AddTokenWithScore(2, "c", -1.0f);
  builder.AddTokenWithScore(3, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 3);

  CreateModel(builder, /*unk_token_id=*/3, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  // "axbyc" - 'x' and 'y' are unreachable, separated by 'b'.
  // Expected: [a, UNK, b, UNK, c] — the two UNKs are not fused.
  TestEncode(model(), "axbyc", {0, 3, 1, 3, 2},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest,
       SeparatedEmojiUnreachable_FuseUnk_ProducesSeparateUnks) {
  // Same principle as SeparatedUnreachable above, but with 4-byte emoji
  // characters. Each emoji is a single UTF-8 character (4 bytes) that produces
  // one UNK, and the reachable 'b' between them prevents fusion.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "b", -1.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  // "a😀b🌍" — each emoji is 4 bytes, unreachable, separated by 'b'.
  // Expected: [a, UNK, b, UNK]
  TestEncode(model(),
             "a\xF0\x9F\x98\x80"
             "b\xF0\x9F\x8C\x8D",
             {0, 2, 1, 2},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest,
       MultipleSegments_SeparatedUnreachable_ProduceSeparateUnks) {
  // When multiple segments alternate between reachable and unreachable content,
  // each unreachable segment independently produces a UNK token. Segments are
  // tokenized independently, so no cross-segment UNK fusion occurs.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "hello", -2.0f);
  builder.AddTokenWithScore(1, "world", -2.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  // 4 segments: "hello", 😀, "world", 🌍
  std::string text = "hello\xF0\x9F\x98\x80world\xF0\x9F\x8C\x8D";
  std::vector<iree_tokenizer_segment_t> segments = {
      {0, 5},    // "hello"
      {5, 9},    // 😀 (4 bytes, unreachable)
      {9, 14},   // "world"
      {14, 18},  // 🌍 (4 bytes, unreachable)
  };
  auto tokens = EncodeAndFinalize(model(), text, segments,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 2);  // UNK for 😀
  EXPECT_EQ(tokens[2], 1);  // "world"
  EXPECT_EQ(tokens[3], 2);  // UNK for 🌍
}

TEST_F(UnigramModelTest, SegmentWithPrefixAndUnreachableEmoji) {
  // A segment containing a reachable prefix (▁), an unreachable emoji, and a
  // reachable suffix (!). The Viterbi DP should produce three tokens: the
  // prefix, a UNK for the emoji, and the suffix.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "\xE2\x96\x81", -2.0f);  // ▁
  builder.AddTokenWithScore(1, "!", -1.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  // "▁🌍!" — ▁ (3 bytes) is reachable, 🌍 (4 bytes) is unreachable, ! is
  // reachable. Expected: [▁, UNK, !]
  TestEncode(model(), "\xE2\x96\x81\xF0\x9F\x8C\x8D!", {0, 2, 1},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Byte Fallback
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, ByteFallbackForUnknown) {
  // With byte fallback (default), unknown chars become <0xXX> tokens.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  // Add byte token for 'x' (ASCII 0x78).
  builder.AddTokenWithScore(1, "<0x78>", -5.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  // "ax" -> "a" + <0x78>
  TestEncode(model(), "ax", {0, 1}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, ByteFallbackMultibyte) {
  // Multi-byte UTF-8 character falls back to byte tokens.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  // Add byte tokens for UTF-8 encoding of 'é' (0xC3 0xA9).
  builder.AddTokenWithScore(1, "<0xC3>", -5.0f);
  builder.AddTokenWithScore(2, "<0xA9>", -5.0f);
  builder.AddTokenWithScore(3, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 3);

  CreateModel(builder, /*unk_token_id=*/3);

  // "aé" -> "a" + <0xC3> + <0xA9>
  TestEncode(model(), "a\xc3\xa9", {0, 1, 2},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, NoByteFallbackFlag) {
  // With NO_BYTE_FALLBACK, unknown chars produce UNK even if byte tokens exist.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "<0x78>", -5.0f);  // Byte token exists.
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  // "ax" -> "a" + [UNK] (not <0x78>).
  TestEncode(model(), "ax", {0, 2}, /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// UTF-8 Handling
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, MultibyteCharacterToken) {
  // Token containing multi-byte UTF-8 character.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "\xc3\xa9", -1.0f);  // é
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);

  TestEncode(model(), "\xc3\xa9", {0}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, MixedAsciiAndMultibyte) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "caf", -1.5f);
  builder.AddTokenWithScore(1, "\xc3\xa9", -1.0f);  // é
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  // "café" = "caf" + "é"
  TestEncode(model(), "caf\xc3\xa9", {0, 1},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, FourByteUtf8Character) {
  // 4-byte UTF-8 character (emoji).
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "\xf0\x9f\x98\x80", -1.0f);  // Grinning face
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);

  TestEncode(model(), "\xf0\x9f\x98\x80", {0},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Multi-segment Encoding
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, MultipleSegments) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "hello", -2.0f);
  builder.AddTokenWithScore(1, "world", -2.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  std::string text = "helloworld";
  std::vector<iree_tokenizer_segment_t> segments = {{0, 5}, {5, 10}};
  auto tokens = EncodeAndFinalize(model(), text, segments,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "world"
}

TEST_F(UnigramModelTest, MultipleSegmentsWithUnknown) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "hello", -2.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  std::string text = "helloxyz";
  std::vector<iree_tokenizer_segment_t> segments = {{0, 5}, {5, 8}};
  auto tokens = EncodeAndFinalize(model(), text, segments,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "xyz" -> [UNK]
}

//===----------------------------------------------------------------------===//
// Offset Tracking
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, OffsetsTrackBytePositions) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "un", -1.0f);
  builder.AddTokenWithScore(1, "aff", -1.0f);
  builder.AddTokenWithScore(2, "able", -1.0f);
  builder.AddTokenWithScore(3, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 3);

  CreateModel(builder, /*unk_token_id=*/3);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "unaffable",
                                   /*expect_pending_after_encode=*/false);
  ASSERT_EQ(result.tokens.size(), 3u);
  EXPECT_EQ(result.tokens[0], 0);  // "un"
  EXPECT_EQ(result.tokens[1], 1);  // "aff"
  EXPECT_EQ(result.tokens[2], 2);  // "able"

  // Offsets are byte positions in the input.
  ASSERT_EQ(result.offsets.size(), 3u);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);  // "un"
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 5u);  // "aff"
  EXPECT_EQ(result.offsets[2].start, 5u);
  EXPECT_EQ(result.offsets[2].end, 9u);  // "able"
}

TEST_F(UnigramModelTest, OffsetsForUnknownWord) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);

  CreateModel(builder, /*unk_token_id=*/0, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "xyz",
                                   /*expect_pending_after_encode=*/false);
  ASSERT_EQ(result.tokens.size(), 1u);
  EXPECT_EQ(result.tokens[0], 0);  // [UNK]

  ASSERT_EQ(result.offsets.size(), 1u);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 3u);  // Covers entire word.
}

TEST_F(UnigramModelTest, OffsetsForByteFallback) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "<0x78>", -5.0f);  // 'x'
  builder.AddTokenWithScore(2, "<0x79>", -5.0f);  // 'y'
  builder.AddTokenWithScore(3, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 3);

  CreateModel(builder, /*unk_token_id=*/3);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "axy",
                                   /*expect_pending_after_encode=*/false);
  ASSERT_EQ(result.tokens.size(), 3u);
  EXPECT_EQ(result.tokens[0], 0);  // "a"
  EXPECT_EQ(result.tokens[1], 1);  // <0x78>
  EXPECT_EQ(result.tokens[2], 2);  // <0x79>

  ASSERT_EQ(result.offsets.size(), 3u);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 1u);
  EXPECT_EQ(result.offsets[1].start, 1u);
  EXPECT_EQ(result.offsets[1].end, 2u);
  EXPECT_EQ(result.offsets[2].start, 2u);
  EXPECT_EQ(result.offsets[2].end, 3u);
}

//===----------------------------------------------------------------------===//
// Output Buffer Handling
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, LimitedOutputCapacity) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "h", -1.0f);
  builder.AddTokenWithScore(1, "e", -1.0f);
  builder.AddTokenWithScore(2, "l", -1.0f);
  builder.AddTokenWithScore(3, "o", -1.0f);
  builder.AddTokenWithScore(4, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 4);

  CreateModel(builder, /*unk_token_id=*/4);

  std::vector<iree_tokenizer_segment_t> segments = {{0, 5}};
  TestLimitedOutputCapacity(model(), "hello", segments, {0, 1, 2, 2, 3});
}

TEST_F(UnigramModelTest, LimitedOutputCapacityMultiSegment) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "b", -1.0f);
  builder.AddTokenWithScore(2, "c", -1.0f);
  builder.AddTokenWithScore(3, "d", -1.0f);
  builder.AddTokenWithScore(4, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 4);

  CreateModel(builder, /*unk_token_id=*/4);

  std::vector<iree_tokenizer_segment_t> segments = {{0, 2}, {2, 4}};
  TestLimitedOutputCapacity(model(), "abcd", segments, {0, 1, 2, 3});
}

//===----------------------------------------------------------------------===//
// Large Segments (Chunked Viterbi)
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, LargeSegmentChunkedCorrectly) {
  // Segments larger than max_token_length are processed in chunks.
  // With single-char tokens, every chunk boundary is a valid split point.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  // Chunked Viterbi handles arbitrary segment sizes.
  TestEncode(model(), "aaaa", {0, 0, 0, 0},
             /*expect_pending_after_encode=*/false);
  TestEncode(model(), "aaa", {0, 0, 0},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Untokenizable Segments with UNK Fallback
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, UntokenizableSegmentEmitsUnk) {
  // When byte fallback is incomplete (missing some <0xXX> tokens) and a segment
  // cannot be tokenized, it should emit UNK rather than being silently dropped.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  // Only add byte fallback for 'a' (0x61), not for 'x' (0x78).
  builder.AddTokenWithScore(1, "<0x61>", -5.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  // "ax" - 'a' tokenizes normally, 'x' has no byte fallback -> UNK
  // Note: This tests the encode path, not tokenize_segment directly.
  // The 'x' character has no <0x78> token, so the forward pass cannot reach
  // segment end, but UNK should be emitted for the unreachable portion.
  TestEncode(model(), "x", {2}, /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, MixedTokenizableAndUntokenizable) {
  // Test that a tokenizable segment followed by an untokenizable one both work.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "hello", -2.0f);
  // No byte fallback tokens, no coverage for "xyz".
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  std::string text = "helloxyz";
  std::vector<iree_tokenizer_segment_t> segments = {{0, 5}, {5, 8}};
  auto tokens = EncodeAndFinalize(model(), text, segments,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "xyz" -> [UNK] (not silently dropped)
}

//===----------------------------------------------------------------------===//
// STREAMING: Partial segment handling
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, PartialSegmentDeferred) {
  // When last_is_partial=true, the last segment should be skipped.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "hello", -2.0f);
  builder.AddTokenWithScore(1, "world", -2.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  ScopedModelState state(model());
  std::string text = "helloworld";
  iree_const_byte_span_t buffer = {
      reinterpret_cast<const uint8_t*>(text.data()), text.size()};

  // Two segments: "hello" and "world", with "world" marked as partial.
  iree_tokenizer_segment_t segments[] = {{0, 5}, {5, 10}};
  iree_tokenizer_segment_list_t segment_list = {
      .count = 2,
      .values = segments,
      .last_is_partial = true,
  };

  iree_tokenizer_token_id_t tokens[8];
  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;

  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer, segment_list,
      iree_tokenizer_make_token_output(tokens, NULL, NULL, 8),
      &segments_consumed, &token_count));

  // Only "hello" should be tokenized; "world" is deferred.
  EXPECT_EQ(segments_consumed, 1u);
  EXPECT_EQ(token_count, 1u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
}

TEST_F(UnigramModelTest, StreamingMultipleChunks) {
  // Simulate streaming: first chunk has partial segment, second completes it.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "hello", -2.0f);
  builder.AddTokenWithScore(1, "world", -2.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  ScopedModelState state(model());

  // Chunk 1: "hello" complete, "wor" partial.
  std::string chunk1 = "hellowor";
  iree_const_byte_span_t buffer1 = {
      reinterpret_cast<const uint8_t*>(chunk1.data()), chunk1.size()};
  iree_tokenizer_segment_t segments1[] = {{0, 5}, {5, 8}};
  iree_tokenizer_segment_list_t list1 = {
      .count = 2, .values = segments1, .last_is_partial = true};

  iree_tokenizer_token_id_t tokens[8];
  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;

  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer1, list1,
      iree_tokenizer_make_token_output(tokens, NULL, NULL, 8),
      &segments_consumed, &token_count));

  EXPECT_EQ(segments_consumed, 1u);  // Only "hello" consumed.
  EXPECT_EQ(token_count, 1u);
  EXPECT_EQ(tokens[0], 0);  // "hello"

  // Chunk 2: "world" complete (caller moved partial data to front).
  std::string chunk2 = "world";
  iree_const_byte_span_t buffer2 = {
      reinterpret_cast<const uint8_t*>(chunk2.data()), chunk2.size()};
  iree_tokenizer_segment_t segments2[] = {{0, 5}};
  iree_tokenizer_segment_list_t list2 = {
      .count = 1, .values = segments2, .last_is_partial = false};

  state.Reset();  // Start fresh for the new chunk.
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer2, list2,
      iree_tokenizer_make_token_output(tokens, NULL, NULL, 8),
      &segments_consumed, &token_count));

  EXPECT_EQ(segments_consumed, 1u);
  EXPECT_EQ(token_count, 1u);
  EXPECT_EQ(tokens[0], 1);  // "world"
}

TEST_F(UnigramModelTest, AllSegmentsPartial) {
  // Edge case: single segment marked partial - nothing should be processed.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "hello", -2.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);

  ScopedModelState state(model());
  std::string text = "hel";  // Partial "hello".
  iree_const_byte_span_t buffer = {
      reinterpret_cast<const uint8_t*>(text.data()), text.size()};
  iree_tokenizer_segment_t segments[] = {{0, 3}};
  iree_tokenizer_segment_list_t list = {
      .count = 1, .values = segments, .last_is_partial = true};

  iree_tokenizer_token_id_t tokens[8];
  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;

  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer, list,
      iree_tokenizer_make_token_output(tokens, NULL, NULL, 8),
      &segments_consumed, &token_count));

  // Nothing should be processed.
  EXPECT_EQ(segments_consumed, 0u);
  EXPECT_EQ(token_count, 0u);
}

//===----------------------------------------------------------------------===//
// VALIDATION: Segment bounds checking
//===----------------------------------------------------------------------===//

TEST_F(UnigramModelTest, SegmentBoundsEndExceedsBuffer) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);

  ScopedModelState state(model());
  std::string text = "abc";
  iree_const_byte_span_t buffer = {
      reinterpret_cast<const uint8_t*>(text.data()), text.size()};

  // Segment end (10) exceeds buffer length (3).
  iree_tokenizer_segment_t segments[] = {{0, 10}};
  iree_tokenizer_segment_list_t list = {
      .count = 1, .values = segments, .last_is_partial = false};

  iree_tokenizer_token_id_t tokens[8];
  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;

  iree_status_t status = iree_tokenizer_model_state_encode(
      state.get(), buffer, list,
      iree_tokenizer_make_token_output(tokens, NULL, NULL, 8),
      &segments_consumed, &token_count);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(UnigramModelTest, SegmentBoundsStartExceedsEnd) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1);

  ScopedModelState state(model());
  std::string text = "abc";
  iree_const_byte_span_t buffer = {
      reinterpret_cast<const uint8_t*>(text.data()), text.size()};

  // Segment start (2) > end (1) - invalid.
  iree_tokenizer_segment_t segments[] = {{2, 1}};
  iree_tokenizer_segment_list_t list = {
      .count = 1, .values = segments, .last_is_partial = false};

  iree_tokenizer_token_id_t tokens[8];
  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;

  iree_status_t status = iree_tokenizer_model_state_encode(
      state.get(), buffer, list,
      iree_tokenizer_make_token_output(tokens, NULL, NULL, 8),
      &segments_consumed, &token_count);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

//===----------------------------------------------------------------------===//
// Consecutive Whitespace/Metaspace Tokens
//===----------------------------------------------------------------------===//

// These tests verify that after normalization, consecutive metaspace characters
// (▁) produce the expected tokenization - either merged tokens or individual
// tokens depending on vocabulary coverage.

TEST_F(UnigramModelTest, ConsecutiveMetaspace_PrefersMergedToken) {
  // When vocabulary contains both single and double metaspace tokens,
  // Viterbi should prefer the merged token if its score is better.
  // This models the behavior expected from SentencePiece/ALBERT tokenizers.
  //
  // U+2581 LOWER ONE EIGHTH BLOCK: UTF-8 = E2 96 81 (3 bytes).
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "\xE2\x96\x81", -2.0f);  // Single ▁
  builder.AddTokenWithScore(1, "\xE2\x96\x81\xE2\x96\x81",
                            -2.5f);  // Double ▁▁ (better)
  builder.AddTokenWithScore(2, "word", -1.0f);
  builder.AddTokenWithScore(3, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 3);

  CreateModel(builder, /*unk_token_id=*/3);

  // Input: "▁▁word" (two metaspaces + word)
  // Score option 1: ▁ + ▁ + word = -2.0 + -2.0 + -1.0 = -5.0
  // Score option 2: ▁▁ + word = -2.5 + -1.0 = -3.5 (better!)
  // Viterbi should pick option 2: [1, 2]
  TestEncode(model(), "\xE2\x96\x81\xE2\x96\x81word", {1, 2},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, ConsecutiveMetaspace_FallsBackToIndividual) {
  // When merged token doesn't exist, individual metaspaces are produced.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "\xE2\x96\x81", -2.0f);  // Single ▁ only
  builder.AddTokenWithScore(1, "word", -1.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  // Input: "▁▁word" - no ▁▁ token exists, so must use ▁ + ▁
  TestEncode(model(), "\xE2\x96\x81\xE2\x96\x81word", {0, 0, 1},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, TripleMetaspace_OptimalCoverage) {
  // Tests that Viterbi finds optimal coverage for three metaspaces.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "\xE2\x96\x81", -2.0f);  // ▁
  builder.AddTokenWithScore(1, "\xE2\x96\x81\xE2\x96\x81",
                            -3.0f);  // ▁▁ (worse than 2x single)
  builder.AddTokenWithScore(2, "\xE2\x96\x81\xE2\x96\x81\xE2\x96\x81",
                            -2.0f);  // ▁▁▁ (best!)
  builder.AddTokenWithScore(3, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 3);

  CreateModel(builder, /*unk_token_id=*/3);

  // Input: "▁▁▁" (three metaspaces)
  // Score option 1: ▁ + ▁ + ▁ = -6.0
  // Score option 2: ▁▁ + ▁ = -5.0
  // Score option 3: ▁▁▁ = -2.0 (best!)
  TestEncode(model(), "\xE2\x96\x81\xE2\x96\x81\xE2\x96\x81", {2},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Multiple UNK Tokens (NO_FUSE_UNK Behavior)
//===----------------------------------------------------------------------===//

// When FUSE_UNK is disabled, each unreachable character should produce
// a separate UNK token. This is critical for models that track character
// positions via UNK counts.

TEST_F(UnigramModelTest, MultipleUnreachable_NoFuseUnk_ProducesMultipleUnk) {
  // Test that multiple consecutive unreachable characters each produce [UNK]
  // when the NO_FUSE_UNK flag is set. This is important for models that need
  // to preserve character-level alignment through UNK tokens.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK |
                  IREE_TOKENIZER_UNIGRAM_FLAG_NO_FUSE_UNK);

  // Input: "axyza" - 'x', 'y', 'z' are unreachable.
  // With NO_FUSE_UNK: [a, UNK, UNK, UNK, a] - each char produces its own UNK.
  TestEncode(model(), "axyza", {0, 1, 1, 1, 0},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, MultipleUnreachable_FuseUnk_ProducesSingleUnk) {
  // Contrast test: with default (fuse UNK) behavior, consecutive unreachable
  // characters produce a single UNK.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);

  // Input: "axyza" - 'x', 'y', 'z' are unreachable.
  // With FUSE_UNK (default): [a, UNK, a] - consecutive unknowns merge.
  TestEncode(model(), "axyza", {0, 1, 0},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, MultiByteFallback_NoFuseUnk_EachCharSeparate) {
  // Verify that multi-byte UTF-8 characters that are unreachable each produce
  // separate UNK tokens when NO_FUSE_UNK is set. UTF-8 character boundaries
  // should be respected (each character = one UNK, not each byte).
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK |
                  IREE_TOKENIZER_UNIGRAM_FLAG_NO_FUSE_UNK);

  // Input: "a\xC3\xA9\xC3\xA9a" = "aééa" (each é is 2 bytes in UTF-8).
  // With NO_FUSE_UNK: [a, UNK, UNK, a] - each é character produces one UNK.
  TestEncode(model(),
             "a\xC3\xA9\xC3\xA9"
             "a",
             {0, 1, 1, 0},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, EmojiUnreachable_NoFuseUnk_EachEmojiSeparateUnk) {
  // Test that each emoji (4-byte UTF-8 sequence) produces a separate UNK
  // when the emoji is not in vocabulary and NO_FUSE_UNK is set.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "a", -1.0f);
  builder.AddTokenWithScore(1, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder, /*unk_token_id=*/1, /*unk_score=*/-10.0f,
              IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK |
                  IREE_TOKENIZER_UNIGRAM_FLAG_NO_FUSE_UNK);

  // Input: "a\xF0\x9F\x98\x80\xF0\x9F\x98\x80" = "a😀😀"
  // Each 😀 emoji is 4 bytes (F0 9F 98 80), but represents ONE character.
  // With NO_FUSE_UNK: [a, UNK, UNK] - each emoji produces one UNK.
  TestEncode(model(), "a\xF0\x9F\x98\x80\xF0\x9F\x98\x80", {0, 1, 1},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Normalized Whitespace Handling
//===----------------------------------------------------------------------===//

// These tests verify behavior after normalizers have transformed whitespace
// (tabs, newlines) into metaspace or other representations.

TEST_F(UnigramModelTest, NormalizedWhitespace_MixedTypes) {
  // After normalization, different whitespace types may become the same
  // metaspace character. This test verifies consistent tokenization.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "\xE2\x96\x81", -2.0f);  // ▁ (metaspace)
  builder.AddTokenWithScore(1, "text", -1.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  // Simulating normalized input where tab became ▁: "▁text▁"
  // This is what the Unigram model would see after Replace normalizer.
  TestEncode(model(), "\xE2\x96\x81text\xE2\x96\x81", {0, 1, 0},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Segment Boundary Effects
//===----------------------------------------------------------------------===//

// When normalization creates different segment boundaries than expected,
// the Viterbi DP operates on smaller segments and may find different paths.

TEST_F(UnigramModelTest, SingleSegment_ViterbiOptimizesAcrossTokens) {
  // Test that with a single segment, Viterbi can find the globally optimal
  // tokenization across all tokens. This is the normal case.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "\xE2\x96\x81", -2.0f);  // ▁
  builder.AddTokenWithScore(1, "\xE2\x96\x81\xE2\x96\x81",
                            -1.0f);  // ▁▁ (better score!)
  builder.AddTokenWithScore(2, "word", -1.0f);
  builder.AddTokenWithScore(3, "\xE2\x96\x81word", -1.5f);  // ▁word
  builder.AddTokenWithScore(4, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 4);

  CreateModel(builder, /*unk_token_id=*/4);

  // Single segment: "▁▁word" -> should pick ▁▁ + word = [1, 2] (score -2.0)
  // vs ▁ + ▁word = [0, 3] (score -3.5).
  TestEncode(model(), "\xE2\x96\x81\xE2\x96\x81word", {1, 2},
             /*expect_pending_after_encode=*/false);
}

TEST_F(UnigramModelTest, MultipleSegments_EachOptimizedIndependently) {
  // When segments are provided separately, each is tokenized independently.
  // This tests that segment boundaries are respected and don't allow tokens
  // to span across segments.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "\xE2\x96\x81", -2.0f);  // ▁
  builder.AddTokenWithScore(1, "word", -1.0f);
  builder.AddTokenWithScore(2, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, /*unk_token_id=*/2);

  // Two segments: "▁" and "word"
  // Each tokenized independently: [0] and [1]
  std::string text = "\xE2\x96\x81word";
  std::vector<iree_tokenizer_segment_t> segments = {{0, 3}, {3, 7}};
  auto tokens = EncodeAndFinalize(model(), text, segments,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // ▁
  EXPECT_EQ(tokens[1], 1);  // word
}

// Tests ALBERT-like scenario with multiple metaspace-delimited segments.
// This exercises the case where segments don't start at offset 0, which is
// critical for verifying segment offset handling.
TEST_F(UnigramModelTest, MetaspaceSegments_VariousOffsets) {
  // Build a vocab with ALBERT-like tokens.
  // Using string concatenation to avoid hex escape issues with 'a'-'f'.
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0,
                            "\xE2\x96\x81"
                            "multiple",
                            -2.0f);  // ▁multiple
  builder.AddTokenWithScore(1,
                            "\xE2\x96\x81"
                            "spaces",
                            -2.0f);  // ▁spaces
  builder.AddTokenWithScore(2,
                            "\xE2\x96\x81"
                            "and",
                            -2.0f);  // ▁and
  builder.AddTokenWithScore(3,
                            "\xE2\x96\x81"
                            "tabs",
                            -2.0f);  // ▁tabs
  builder.AddTokenWithScore(4,
                            "\xE2\x96\x81"
                            "newlines",
                            -2.0f);  // ▁newlines
  builder.AddTokenWithScore(5,
                            "\xE2\x96\x81"
                            "as",
                            -2.0f);  // ▁as (decoy)
  builder.AddTokenWithScore(6, "[UNK]", -10.0f);
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 6);

  CreateModel(builder, /*unk_token_id=*/6);

  // Normalized text: ▁multiple▁spaces▁and▁tabs▁and▁newlines
  // Each metaspace segment includes the leading ▁.
  std::string text =
      "\xE2\x96\x81"
      "multiple"  // 0-10 (11 bytes)
      "\xE2\x96\x81"
      "spaces"  // 11-19 (9 bytes)
      "\xE2\x96\x81"
      "and"  // 20-25 (6 bytes)
      "\xE2\x96\x81"
      "tabs"  // 26-32 (7 bytes)
      "\xE2\x96\x81"
      "and"  // 33-38 (6 bytes)
      "\xE2\x96\x81"
      "newlines";  // 39-49 (11 bytes)
  ASSERT_EQ(text.size(), 50u);

  // Segments as they would be produced by Metaspace segmenter.
  std::vector<iree_tokenizer_segment_t> segments = {
      {0, 11},   // ▁multiple
      {11, 20},  // ▁spaces
      {20, 26},  // ▁and
      {26, 33},  // ▁tabs
      {33, 39},  // ▁and
      {39, 50},  // ▁newlines
  };

  auto tokens = EncodeAndFinalize(model(), text, segments,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 6u);
  EXPECT_EQ(tokens[0], 0);  // ▁multiple
  EXPECT_EQ(tokens[1], 1);  // ▁spaces
  EXPECT_EQ(tokens[2], 2);  // ▁and
  EXPECT_EQ(tokens[3], 3);  // ▁tabs
  EXPECT_EQ(tokens[4], 2);  // ▁and (same token as position 2)
  EXPECT_EQ(tokens[5], 4);  // ▁newlines
}

}  // namespace
}  // namespace iree::tokenizer
