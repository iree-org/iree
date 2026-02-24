// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/model/wordpiece.h"

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
// Test fixture for WordPiece model tests.
//===----------------------------------------------------------------------===//

class WordPieceModelTest : public ::testing::Test {
 protected:
  // Helper to create model from vocab builder with default settings.
  void CreateModel(ScopedVocabBuilder& builder,
                   iree_string_view_t prefix = IREE_SV("##"),
                   iree_host_size_t max_chars = 100) {
    vocab_ = ScopedVocab(builder.Build());
    iree_tokenizer_model_t* raw_model = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_wordpiece_model_allocate(
        vocab_.get(), prefix, max_chars, IREE_TOKENIZER_WORDPIECE_FLAG_NONE,
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

TEST_F(WordPieceModelTest, CreateAndDestroy) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder);
  EXPECT_NE(model(), nullptr);
}

TEST_F(WordPieceModelTest, StateSizeIsReasonable) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder);

  iree_host_size_t state_size = iree_tokenizer_model_state_size(model());
  EXPECT_GT(state_size, 0u);
  // State = struct (~32 bytes) + pending buffer (100 entries * ~24 bytes).
  EXPECT_LE(state_size, 4096u);
}

TEST_F(WordPieceModelTest, RequiresUnkToken) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  // No UNK token set.
  auto* vocab = builder.Build();
  iree_tokenizer_model_t* raw_model = nullptr;
  iree_status_t status = iree_tokenizer_wordpiece_model_allocate(
      vocab, IREE_SV("##"), 100, IREE_TOKENIZER_WORDPIECE_FLAG_NONE,
      iree_allocator_system(), &raw_model);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_tokenizer_vocab_free(vocab);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(WordPieceModelTest, EmptySegment) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder);

  std::vector<iree_tokenizer_segment_t> segments = {{0, 0}};
  auto tokens = EncodeAndFinalize(model(), "", segments,
                                  /*expect_pending_after_encode=*/false);
  EXPECT_TRUE(tokens.empty());
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

// All expectations verified against HuggingFace tokenizers library:
//   from tokenizers.models import WordPiece
//   model = WordPiece(vocab, unk_token="[UNK]",
//                     continuing_subword_prefix="##",
//                     max_input_chars_per_word=100)
//   model.tokenize(word)

TEST_F(WordPieceModelTest, SingleCharToken) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder);

  TestEncode(model(), "a", {0}, /*expect_pending_after_encode=*/false);
}

TEST_F(WordPieceModelTest, WholeWordMatch) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hello");
  builder.AddToken(1, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder);

  TestEncode(model(), "hello", {0}, /*expect_pending_after_encode=*/false);
}

TEST_F(WordPieceModelTest, SubwordSplit) {
  // HuggingFace: "unaffable" -> ["un", "##aff", "##able"]
  ScopedVocabBuilder builder;
  builder.AddToken(0, "un");
  builder.AddToken(1, "##aff");
  builder.AddToken(2, "##able");
  builder.AddToken(3, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 3);

  CreateModel(builder);

  TestEncode(model(), "unaffable", {0, 1, 2},
             /*expect_pending_after_encode=*/false);
}

TEST_F(WordPieceModelTest, UnknownWord) {
  // Word not in vocab and no valid subword decomposition -> [UNK]
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hello");
  builder.AddToken(1, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder);

  TestEncode(model(), "xyz", {1}, /*expect_pending_after_encode=*/false);
}

TEST_F(WordPieceModelTest, PartiallyUnknownWord) {
  // First subword matches but continuation fails -> entire word is [UNK].
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hel");
  builder.AddToken(1, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);
  // No "##lo" token, so "hello" fails after matching "hel".

  CreateModel(builder);

  TestEncode(model(), "hello", {1}, /*expect_pending_after_encode=*/false);
}

TEST_F(WordPieceModelTest, MaxInputCharsPerWord) {
  // Word exceeding max_input_chars_per_word -> [UNK]
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "##a");
  builder.AddToken(2, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  // Set max to 3 characters.
  CreateModel(builder, IREE_SV("##"), /*max_chars=*/3);

  // "aaaa" is 4 chars, exceeds max of 3 -> [UNK]
  TestEncode(model(), "aaaa", {2}, /*expect_pending_after_encode=*/false);

  // "aaa" is exactly 3 chars -> should tokenize normally.
  TestEncode(model(), "aaa", {0, 1, 1},
             /*expect_pending_after_encode=*/false);
}

TEST_F(WordPieceModelTest, GreedyLongestMatch) {
  // Greedy takes longest match at each position.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "ab");
  builder.AddToken(2, "##c");
  builder.AddToken(3, "##cd");
  builder.AddToken(4, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 4);

  CreateModel(builder);

  // "abcd": greedy picks "ab" (longest at pos 0), then "##cd" (longest at pos
  // 2).
  TestEncode(model(), "abcd", {1, 3}, /*expect_pending_after_encode=*/false);
}

TEST_F(WordPieceModelTest, AllSingleChars) {
  // Each character is its own token.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "h");
  builder.AddToken(1, "##e");
  builder.AddToken(2, "##l");
  builder.AddToken(3, "##o");
  builder.AddToken(4, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 4);

  CreateModel(builder);

  TestEncode(model(), "hello", {0, 1, 2, 2, 3},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Multi-segment Encoding
//===----------------------------------------------------------------------===//

TEST_F(WordPieceModelTest, MultipleSegments) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hello");
  builder.AddToken(1, "world");
  builder.AddToken(2, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder);

  std::string text = "helloworld";
  std::vector<iree_tokenizer_segment_t> segments = {{0, 5}, {5, 10}};
  auto tokens = EncodeAndFinalize(model(), text, segments,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "world"
}

TEST_F(WordPieceModelTest, MultipleSegmentsWithUnknown) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hello");
  builder.AddToken(1, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);

  CreateModel(builder);

  std::string text = "helloxyz";
  std::vector<iree_tokenizer_segment_t> segments = {{0, 5}, {5, 8}};
  auto tokens = EncodeAndFinalize(model(), text, segments,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "xyz" -> [UNK]
}

//===----------------------------------------------------------------------===//
// UTF-8 Handling
//===----------------------------------------------------------------------===//

TEST_F(WordPieceModelTest, MultibyteContinuationPrefix) {
  // Verify character-aware shrinking for multi-byte UTF-8.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "caf");
  builder.AddToken(1, "##\xc3\xa9");  // ##é
  builder.AddToken(2, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder);

  // "café" = "caf" + "é" (2 bytes: 0xC3 0xA9)
  TestEncode(model(), "caf\xc3\xa9", {0, 1},
             /*expect_pending_after_encode=*/false);
}

TEST_F(WordPieceModelTest, MaxInputCharsCountsCharactersNotBytes) {
  // max_input_chars_per_word counts Unicode characters, not bytes.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "\xc3\xa9");  // é (2 bytes, 1 char)
  builder.AddToken(1, "##\xc3\xa9");
  builder.AddToken(2, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  // Max 2 characters.
  CreateModel(builder, IREE_SV("##"), /*max_chars=*/2);

  // "éé" is 2 chars (4 bytes) — should work.
  TestEncode(model(), "\xc3\xa9\xc3\xa9", {0, 1},
             /*expect_pending_after_encode=*/false);

  // "ééé" is 3 chars (6 bytes) — exceeds max of 2.
  TestEncode(model(), "\xc3\xa9\xc3\xa9\xc3\xa9", {2},
             /*expect_pending_after_encode=*/false);
}

//===----------------------------------------------------------------------===//
// Offset Tracking
//===----------------------------------------------------------------------===//

TEST_F(WordPieceModelTest, OffsetsTrackBytePositions) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "un");
  builder.AddToken(1, "##aff");
  builder.AddToken(2, "##able");
  builder.AddToken(3, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 3);

  CreateModel(builder);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "unaffable",
                                   /*expect_pending_after_encode=*/false);
  ASSERT_EQ(result.tokens.size(), 3u);
  EXPECT_EQ(result.tokens[0], 0);  // "un"
  EXPECT_EQ(result.tokens[1], 1);  // "##aff"
  EXPECT_EQ(result.tokens[2], 2);  // "##able"

  // Offsets are byte positions in the input.
  ASSERT_EQ(result.offsets.size(), 3u);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);  // "un"
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 5u);  // "aff"
  EXPECT_EQ(result.offsets[2].start, 5u);
  EXPECT_EQ(result.offsets[2].end, 9u);  // "able"
}

TEST_F(WordPieceModelTest, OffsetsForUnknownWord) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);

  CreateModel(builder);

  auto result =
      EncodeWithOffsetsAndFinalize(model(), "xyz",
                                   /*expect_pending_after_encode=*/false);
  ASSERT_EQ(result.tokens.size(), 1u);
  EXPECT_EQ(result.tokens[0], 0);  // [UNK]

  ASSERT_EQ(result.offsets.size(), 1u);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 3u);  // Covers entire word.
}

//===----------------------------------------------------------------------===//
// Output Buffer Handling
//===----------------------------------------------------------------------===//

TEST_F(WordPieceModelTest, LimitedOutputCapacity) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "h");
  builder.AddToken(1, "##e");
  builder.AddToken(2, "##l");
  builder.AddToken(3, "##o");
  builder.AddToken(4, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 4);

  CreateModel(builder);

  std::vector<iree_tokenizer_segment_t> segments = {{0, 5}};
  TestLimitedOutputCapacity(model(), "hello", segments, {0, 1, 2, 2, 3});
}

TEST_F(WordPieceModelTest, LimitedOutputCapacityMultiSegment) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "##b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "##d");
  builder.AddToken(4, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 4);

  CreateModel(builder);

  std::vector<iree_tokenizer_segment_t> segments = {{0, 2}, {2, 4}};
  TestLimitedOutputCapacity(model(), "abcd", segments, {0, 1, 2, 3});
}

//===----------------------------------------------------------------------===//
// Custom Prefix
//===----------------------------------------------------------------------===//

TEST_F(WordPieceModelTest, EmptyPrefix) {
  // Some models use empty prefix (non-standard but allowed).
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hel");
  builder.AddToken(1, "lo");
  builder.AddToken(2, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, IREE_SV(""));

  TestEncode(model(), "hello", {0, 1},
             /*expect_pending_after_encode=*/false);
}

TEST_F(WordPieceModelTest, LongerPrefix) {
  // Non-standard but valid prefix.
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hel");
  builder.AddToken(1, "###lo");
  builder.AddToken(2, "[UNK]");
  builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  CreateModel(builder, IREE_SV("###"));

  TestEncode(model(), "hello", {0, 1},
             /*expect_pending_after_encode=*/false);
}

}  // namespace
}  // namespace iree::tokenizer
