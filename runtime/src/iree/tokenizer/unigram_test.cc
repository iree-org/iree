// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/unigram.h"

#include <cmath>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/vocab_builder.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

// Helper fixture for Unigram tests.
// Builds a vocabulary with log probability scores for testing Viterbi decoding.
class UnigramVocabFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        /*capacity=*/32, iree_allocator_system(), &builder_));
  }

  void TearDown() override {
    if (state_) iree_tokenizer_unigram_state_free(state_);
    if (vocab_) iree_tokenizer_vocab_free(vocab_);
    if (scores_) iree_allocator_free(iree_allocator_system(), scores_);
  }

  void AddToken(
      int32_t id, const char* text, float score,
      iree_tokenizer_token_attr_t attrs = IREE_TOKENIZER_TOKEN_ATTR_NONE) {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        builder_, id, IREE_SV(text), score, attrs));
    if (id >= static_cast<int32_t>(scores_vec_.size())) {
      scores_vec_.resize(id + 1, -100.0f);
    }
    scores_vec_[id] = score;
  }

  void BuildVocab(float unk_score = -10.0f) {
    unk_score_ = unk_score;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder_, &vocab_));
    builder_ = nullptr;

    // Copy scores to allocated array.
    iree_host_size_t count = scores_vec_.size();
    IREE_ASSERT_OK(iree_allocator_malloc(
        iree_allocator_system(), count * sizeof(float), (void**)&scores_));
    for (size_t i = 0; i < count; ++i) {
      scores_[i] = scores_vec_[i];
    }

    IREE_ASSERT_OK(iree_tokenizer_unigram_state_allocate(
        vocab_, scores_, count, unk_score_, iree_allocator_system(), &state_));
  }

  std::vector<int32_t> Encode(const char* word,
                              iree_host_size_t max_ids = 256) {
    std::vector<int32_t> ids_buffer(max_ids);
    iree_host_size_t count = 0;
    iree_status_t status = iree_tokenizer_unigram_encode_word(
        state_, IREE_SV(word), ids_buffer.data(), max_ids, &count);
    IREE_EXPECT_OK(status) << "Failed to encode: " << word;
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return {};
    }
    return std::vector<int32_t>(ids_buffer.begin(), ids_buffer.begin() + count);
  }

  iree_tokenizer_vocab_builder_t* builder_ = nullptr;
  iree_tokenizer_vocab_t* vocab_ = nullptr;
  iree_tokenizer_unigram_state_t* state_ = nullptr;
  float* scores_ = nullptr;
  float unk_score_ = -10.0f;
  std::vector<float> scores_vec_;
};

//===----------------------------------------------------------------------===//
// Basic Viterbi Algorithm Tests
//===----------------------------------------------------------------------===//

TEST_F(UnigramVocabFixture, SingleCharacterWord) {
  // Vocab: 'a'(0), 'b'(1), 'c'(2)
  AddToken(0, "a", -1.0f);
  AddToken(1, "b", -1.0f);
  AddToken(2, "c", -1.0f);
  BuildVocab();

  auto ids = Encode("a");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);
}

TEST_F(UnigramVocabFixture, MultipleCharactersNoLongerMatch) {
  // Vocab: 'a'(0), 'b'(1), 'c'(2) - no longer tokens
  // "abc" should tokenize as ['a', 'b', 'c']
  AddToken(0, "a", -1.0f);
  AddToken(1, "b", -1.0f);
  AddToken(2, "c", -1.0f);
  BuildVocab();

  auto ids = Encode("abc");
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 0);  // 'a'
  EXPECT_EQ(ids[1], 1);  // 'b'
  EXPECT_EQ(ids[2], 2);  // 'c'
}

TEST_F(UnigramVocabFixture, PreferLongerTokenHigherScore) {
  // Vocab: 'h'(0), 'e'(1), 'l'(2), 'o'(3), "hello"(4)
  // "hello" as single token has higher score than sum of individual chars.
  AddToken(0, "h", -2.0f);
  AddToken(1, "e", -2.0f);
  AddToken(2, "l", -2.0f);
  AddToken(3, "o", -2.0f);
  AddToken(4, "hello", -5.0f);  // -5 > -2*5 = -10
  BuildVocab();

  auto ids = Encode("hello");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 4);  // 'hello'
}

TEST_F(UnigramVocabFixture, PreferShorterTokensLowerScore) {
  // "hello" token has lower score than sum of individual chars.
  AddToken(0, "h", -1.0f);
  AddToken(1, "e", -1.0f);
  AddToken(2, "l", -1.0f);
  AddToken(3, "o", -1.0f);
  AddToken(4, "hello", -10.0f);  // -10 < -1*5 = -5
  BuildVocab();

  auto ids = Encode("hello");
  ASSERT_EQ(ids.size(), 5u);
  EXPECT_EQ(ids[0], 0);  // 'h'
  EXPECT_EQ(ids[1], 1);  // 'e'
  EXPECT_EQ(ids[2], 2);  // 'l'
  EXPECT_EQ(ids[3], 2);  // 'l'
  EXPECT_EQ(ids[4], 3);  // 'o'
}

TEST_F(UnigramVocabFixture, OptimalSegmentation) {
  // Test that Viterbi finds globally optimal segmentation.
  // Vocab: 'un'(0), 'happ'(1), 'y'(2), 'unhappy'(3), 'u'(4), 'n'(5),
  //        'h'(6), 'a'(7), 'p'(8)
  AddToken(0, "un", -2.0f);
  AddToken(1, "happ", -3.0f);
  AddToken(2, "y", -1.0f);
  AddToken(3, "unhappy", -4.0f);  // Best: -4 vs -2 + -3 + -1 = -6
  AddToken(4, "u", -1.5f);
  AddToken(5, "n", -1.5f);
  AddToken(6, "h", -1.5f);
  AddToken(7, "a", -1.5f);
  AddToken(8, "p", -1.5f);
  BuildVocab();

  auto ids = Encode("unhappy");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 3);  // 'unhappy'
}

TEST_F(UnigramVocabFixture, OptimalSegmentationMultipleParts) {
  // "unbreakable" -> "un" + "break" + "able" is optimal
  AddToken(0, "un", -2.0f);
  AddToken(1, "break", -3.0f);
  AddToken(2, "able", -2.5f);
  AddToken(3, "unbreak", -6.0f);    // -6 vs -2 + -3 = -5 (worse)
  AddToken(4, "breakable", -6.0f);  // -6 vs -3 + -2.5 = -5.5 (worse)
  // Individual chars as fallback.
  AddToken(5, "u", -1.5f);
  AddToken(6, "n", -1.5f);
  AddToken(7, "b", -1.5f);
  AddToken(8, "r", -1.5f);
  AddToken(9, "e", -1.5f);
  AddToken(10, "a", -1.5f);
  AddToken(11, "k", -1.5f);
  AddToken(12, "l", -1.5f);
  BuildVocab();

  auto ids = Encode("unbreakable");
  // Optimal: "un"(-2) + "break"(-3) + "able"(-2.5) = -7.5
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 0);  // 'un'
  EXPECT_EQ(ids[1], 1);  // 'break'
  EXPECT_EQ(ids[2], 2);  // 'able'
}

//===----------------------------------------------------------------------===//
// Empty and Edge Case Tests
//===----------------------------------------------------------------------===//

TEST_F(UnigramVocabFixture, EmptyInput) {
  AddToken(0, "a", -1.0f);
  BuildVocab();

  auto ids = Encode("");
  EXPECT_EQ(ids.size(), 0u);
}

TEST_F(UnigramVocabFixture, SingleByteUnknown) {
  // No tokens match, and no byte fallback tokens.
  // Should fall back to UNK if configured.
  AddToken(0, "x", -1.0f);
  AddToken(
      1, "[UNK]", -10.0f,
      IREE_TOKENIZER_TOKEN_ATTR_SPECIAL | IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
      builder_, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1));
  BuildVocab();

  auto ids = Encode("z");  // 'z' not in vocab
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 1);  // [UNK]
}

//===----------------------------------------------------------------------===//
// Unicode Tests
//===----------------------------------------------------------------------===//

TEST_F(UnigramVocabFixture, UnicodeBasic) {
  // Test with UTF-8 encoded tokens.
  AddToken(0, "hello", -2.0f);
  AddToken(1, "\xE4\xB8\x96", -2.0f);              // 世 (U+4E16)
  AddToken(2, "\xE7\x95\x8C", -2.0f);              // 界 (U+754C)
  AddToken(3, "\xE4\xB8\x96\xE7\x95\x8C", -3.0f);  // 世界
  BuildVocab();

  // 世界 as single token (-3) is better than two tokens (-4)
  auto ids = Encode("\xE4\xB8\x96\xE7\x95\x8C");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 3);
}

TEST_F(UnigramVocabFixture, UnicodeMixedWithAscii) {
  // Mix of ASCII and Unicode.
  AddToken(0, "h", -1.0f);
  AddToken(1, "i", -1.0f);
  AddToken(2, "\xE4\xB8\x96", -2.0f);  // 世
  AddToken(3, "hi", -1.5f);            // Better than h + i
  BuildVocab();

  auto ids = Encode("hi\xE4\xB8\x96");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 3);  // 'hi'
  EXPECT_EQ(ids[1], 2);  // 世
}

//===----------------------------------------------------------------------===//
// SentencePiece-style Tests (with ▁ prefix)
//===----------------------------------------------------------------------===//

TEST_F(UnigramVocabFixture, SentencePieceStyle) {
  // SentencePiece uses ▁ (U+2581) as word boundary marker.
  const char* sp = "\xE2\x96\x81";  // ▁

  std::string sp_the = std::string(sp) + "the";
  std::string sp_quick = std::string(sp) + "quick";

  AddToken(0, sp, -2.0f);
  AddToken(1, sp_the.c_str(), -3.0f);
  AddToken(2, sp_quick.c_str(), -4.0f);
  AddToken(3, "t", -1.5f);
  AddToken(4, "h", -1.5f);
  AddToken(5, "e", -1.5f);
  AddToken(6, "q", -1.5f);
  AddToken(7, "u", -1.5f);
  AddToken(8, "i", -1.5f);
  AddToken(9, "c", -1.5f);
  AddToken(10, "k", -1.5f);
  BuildVocab();

  // "▁the" should match as single token.
  auto ids = Encode(sp_the.c_str());
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 1);
}

//===----------------------------------------------------------------------===//
// Score Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(UnigramVocabFixture, ZeroScores) {
  // All tokens have score 0.
  AddToken(0, "a", 0.0f);
  AddToken(1, "b", 0.0f);
  AddToken(2, "ab", 0.0f);
  BuildVocab();

  // "ab" as single token (0) equals "a" + "b" (0 + 0 = 0).
  // Implementation should prefer longer tokens or first found.
  auto ids = Encode("ab");
  // Either [2] or [0, 1] is acceptable with equal scores.
  EXPECT_TRUE(ids.size() == 1 || ids.size() == 2);
}

TEST_F(UnigramVocabFixture, VeryNegativeScores) {
  // Very negative scores (like real SentencePiece).
  // We maximize log probabilities, so higher (less negative) is better.
  AddToken(0, "a", -100.0f);
  AddToken(1, "b", -100.0f);
  AddToken(2, "ab", -250.0f);  // -250 < -200 (a+b), so individual chars win
  BuildVocab();

  auto ids = Encode("ab");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 1);
}

//===----------------------------------------------------------------------===//
// Buffer Overflow Tests
//===----------------------------------------------------------------------===//

TEST_F(UnigramVocabFixture, OutputBufferTooSmall) {
  AddToken(0, "a", -1.0f);
  AddToken(1, "b", -1.0f);
  AddToken(2, "c", -1.0f);
  BuildVocab();

  int32_t ids[2];  // Too small for "abc" (needs 3)
  iree_host_size_t count = 0;
  iree_status_t status = iree_tokenizer_unigram_encode_word(
      state_, IREE_SVL("abc"), ids, /*max_ids=*/2, &count);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));
}

TEST_F(UnigramVocabFixture, LongInputWord) {
  // Create single-char tokens for a longer word.
  for (int i = 0; i < 26; ++i) {
    char c[2] = {static_cast<char>('a' + i), '\0'};
    AddToken(i, c, -1.0f);
  }
  BuildVocab();

  // 100-character input.
  std::string long_word(100, 'a');
  auto ids = Encode(long_word.c_str());
  EXPECT_EQ(ids.size(), 100u);
}

//===----------------------------------------------------------------------===//
// Repeated Characters
//===----------------------------------------------------------------------===//

TEST_F(UnigramVocabFixture, RepeatedCharacters) {
  AddToken(0, "a", -1.0f);
  AddToken(1, "aa", -1.5f);   // Better than 2x 'a' (-2)
  AddToken(2, "aaa", -2.0f);  // Better than 3x 'a' (-3)
  BuildVocab();

  // "aaaa" -> "aaa" + "a" = -3.0, or "aa" + "aa" = -3.0, or 4x "a" = -4.0
  auto ids = Encode("aaaa");
  // Either ["aaa", "a"] or ["aa", "aa"] is optimal.
  EXPECT_EQ(ids.size(), 2u);
  float total_score = 0.0f;
  for (int32_t id : ids) {
    total_score += scores_vec_[id];
  }
  EXPECT_FLOAT_EQ(total_score, -3.0f);
}

//===----------------------------------------------------------------------===//
// API Safety Tests
//===----------------------------------------------------------------------===//

TEST(UnigramApiSafetyTest, ScoreCountMismatchReturnsError) {
  // Build a simple vocab with 3 tokens.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/8, iree_allocator_system(), &builder));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SV("a"), -1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SV("b"), -1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SV("c"), -1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // Allocate scores array that's too small (only 2 elements, but vocab has 3).
  float scores[2] = {-1.0f, -1.0f};

  // Attempting to allocate with mismatched score count should fail.
  iree_tokenizer_unigram_state_t* state = nullptr;
  iree_status_t status = iree_tokenizer_unigram_state_allocate(
      vocab, scores, /*score_count=*/2, /*unk_score=*/-10.0f,
      iree_allocator_system(), &state);

  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(state, nullptr);

  iree_tokenizer_vocab_free(vocab);
}

}  // namespace
