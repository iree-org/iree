// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/unigram_json.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

class UnigramJsonTest : public ::testing::Test {
 protected:
  void TearDown() override {
    if (vocab_) iree_tokenizer_vocab_free(vocab_);
    if (scores_) iree_allocator_free(iree_allocator_system(), scores_);
    if (tokenizer_) iree_tokenizer_free(tokenizer_);
  }

  iree_tokenizer_vocab_t* vocab_ = nullptr;
  float* scores_ = nullptr;
  iree_host_size_t score_count_ = 0;
  float unk_score_ = 0.0f;
  iree_tokenizer_t* tokenizer_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Vocab Import Tests
//===----------------------------------------------------------------------===//

TEST_F(UnigramJsonTest, MinimalVocab) {
  // Minimal Unigram tokenizer.json structure.
  const char* json = R"({
    "model": {
      "type": "Unigram",
      "unk_id": 0,
      "vocab": [
        ["<unk>", 0.0],
        ["a", -1.0],
        ["b", -2.0]
      ]
    }
  })";

  IREE_ASSERT_OK(iree_tokenizer_vocab_import_unigram_json(
      IREE_SV(json), iree_allocator_system(), &vocab_, &scores_, &score_count_,
      &unk_score_));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab_), 3u);
  EXPECT_FLOAT_EQ(scores_[0], 0.0f);
  EXPECT_FLOAT_EQ(scores_[1], -1.0f);
  EXPECT_FLOAT_EQ(scores_[2], -2.0f);
}

TEST_F(UnigramJsonTest, UnicodeTokens) {
  // Vocab with Unicode tokens (SentencePiece style).
  const char* json = R"({
    "model": {
      "type": "Unigram",
      "unk_id": 0,
      "vocab": [
        ["<unk>", 0.0],
        ["\u2581", -2.0],
        ["\u2581the", -3.5],
        ["\u4e16\u754c", -4.0]
      ]
    }
  })";

  IREE_ASSERT_OK(iree_tokenizer_vocab_import_unigram_json(
      IREE_SV(json), iree_allocator_system(), &vocab_, &scores_, &score_count_,
      &unk_score_));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab_), 4u);

  // Verify token text is correctly unescaped.
  iree_string_view_t text1 = iree_tokenizer_vocab_token_text(vocab_, 1);
  EXPECT_EQ(text1.size, 3u);  // ▁ is 3 bytes in UTF-8.

  iree_string_view_t text2 = iree_tokenizer_vocab_token_text(vocab_, 2);
  EXPECT_EQ(text2.size, 6u);  // ▁the is 3 + 3 = 6 bytes.

  iree_string_view_t text3 = iree_tokenizer_vocab_token_text(vocab_, 3);
  EXPECT_EQ(text3.size, 6u);  // 世界 is 3 + 3 = 6 bytes.
}

TEST_F(UnigramJsonTest, NegativeScores) {
  // Verify negative floating point scores are parsed correctly.
  const char* json = R"({
    "model": {
      "type": "Unigram",
      "unk_id": 0,
      "vocab": [
        ["<unk>", -10.5],
        ["hello", -5.123456],
        ["world", -3.14159]
      ]
    }
  })";

  IREE_ASSERT_OK(iree_tokenizer_vocab_import_unigram_json(
      IREE_SV(json), iree_allocator_system(), &vocab_, &scores_, &score_count_,
      &unk_score_));

  EXPECT_FLOAT_EQ(scores_[0], -10.5f);
  EXPECT_NEAR(scores_[1], -5.123456f, 0.0001f);
  EXPECT_NEAR(scores_[2], -3.14159f, 0.0001f);
  EXPECT_FLOAT_EQ(unk_score_, -10.5f);
}

TEST_F(UnigramJsonTest, LargeVocab) {
  // Build a larger vocabulary programmatically.
  std::string json = R"({"model": {"type": "Unigram", "unk_id": 0, "vocab": [)";
  json += R"(["<unk>", 0.0])";
  for (int i = 1; i < 100; ++i) {
    json += ",\n[\"token" + std::to_string(i) + "\", " +
            std::to_string(-static_cast<float>(i)) + "]";
  }
  json += "]}}";

  IREE_ASSERT_OK(iree_tokenizer_vocab_import_unigram_json(
      IREE_SV(json.c_str()), iree_allocator_system(), &vocab_, &scores_,
      &score_count_, &unk_score_));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab_), 100u);
}

//===----------------------------------------------------------------------===//
// Error Handling Tests
//===----------------------------------------------------------------------===//

TEST_F(UnigramJsonTest, WrongModelType) {
  const char* json = R"({
    "model": {
      "type": "BPE",
      "vocab": {}
    }
  })";

  iree_status_t status = iree_tokenizer_vocab_import_unigram_json(
      IREE_SV(json), iree_allocator_system(), &vocab_, &scores_, &score_count_,
      &unk_score_);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(UnigramJsonTest, MissingVocab) {
  const char* json = R"({
    "model": {
      "type": "Unigram",
      "unk_id": 0
    }
  })";

  iree_status_t status = iree_tokenizer_vocab_import_unigram_json(
      IREE_SV(json), iree_allocator_system(), &vocab_, &scores_, &score_count_,
      &unk_score_);
  EXPECT_THAT(Status(std::move(status)), StatusIs(StatusCode::kNotFound));
}

TEST_F(UnigramJsonTest, InvalidVocabEntry) {
  // Vocab entry with wrong number of elements.
  const char* json = R"({
    "model": {
      "type": "Unigram",
      "unk_id": 0,
      "vocab": [
        ["<unk>", 0.0, "extra"]
      ]
    }
  })";

  iree_status_t status = iree_tokenizer_vocab_import_unigram_json(
      IREE_SV(json), iree_allocator_system(), &vocab_, &scores_, &score_count_,
      &unk_score_);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kInvalidArgument));
}

//===----------------------------------------------------------------------===//
// Full Tokenizer Factory Tests
//===----------------------------------------------------------------------===//

TEST_F(UnigramJsonTest, CreateTokenizer) {
  const char* json = R"({
    "model": {
      "type": "Unigram",
      "unk_id": 0,
      "vocab": [
        ["<unk>", 0.0],
        ["hello", -2.0],
        ["world", -2.0],
        ["h", -1.0],
        ["e", -1.0],
        ["l", -1.0],
        ["o", -1.0]
      ]
    }
  })";

  IREE_ASSERT_OK(iree_tokenizer_from_unigram_json(
      IREE_SV(json), iree_allocator_system(), &tokenizer_));

  EXPECT_NE(tokenizer_, nullptr);
}

TEST_F(UnigramJsonTest, TokenizerWithNormalizer) {
  // Tokenizer with normalizer configuration.
  const char* json = R"({
    "normalizer": {
      "type": "Lowercase"
    },
    "model": {
      "type": "Unigram",
      "unk_id": 0,
      "vocab": [
        ["<unk>", 0.0],
        ["hello", -2.0]
      ]
    }
  })";

  IREE_ASSERT_OK(iree_tokenizer_from_unigram_json(
      IREE_SV(json), iree_allocator_system(), &tokenizer_));

  EXPECT_NE(tokenizer_, nullptr);
}

TEST_F(UnigramJsonTest, TokenizerWithAddedTokens) {
  // Tokenizer with added_tokens (special tokens).
  const char* json = R"({
    "added_tokens": [
      {"id": 0, "content": "<pad>", "special": true},
      {"id": 1, "content": "</s>", "special": true},
      {"id": 2, "content": "<unk>", "special": true}
    ],
    "model": {
      "type": "Unigram",
      "unk_id": 2,
      "vocab": [
        ["<pad>", 0.0],
        ["</s>", 0.0],
        ["<unk>", 0.0],
        ["hello", -2.0]
      ]
    }
  })";

  IREE_ASSERT_OK(iree_tokenizer_from_unigram_json(
      IREE_SV(json), iree_allocator_system(), &tokenizer_));

  EXPECT_NE(tokenizer_, nullptr);

  // Verify special tokens are set correctly.
  iree_tokenizer_special_ids_t special_ids =
      iree_tokenizer_vocab_special_ids(tokenizer_->vocab);
  EXPECT_EQ(special_ids.unk, 2);
}

//===----------------------------------------------------------------------===//
// Real-world Format Tests
//===----------------------------------------------------------------------===//

TEST_F(UnigramJsonTest, T5StyleFormat) {
  // T5-style tokenizer.json structure.
  const char* json = R"({
    "added_tokens": [
      {"id": 0, "content": "<pad>", "special": true},
      {"id": 1, "content": "</s>", "special": true},
      {"id": 2, "content": "<unk>", "special": true}
    ],
    "normalizer": null,
    "pre_tokenizer": {
      "type": "Metaspace",
      "replacement": "\u2581",
      "add_prefix_space": true
    },
    "decoder": {
      "type": "Metaspace",
      "replacement": "\u2581",
      "add_prefix_space": true
    },
    "model": {
      "type": "Unigram",
      "unk_id": 2,
      "vocab": [
        ["<pad>", 0.0],
        ["</s>", 0.0],
        ["<unk>", 0.0],
        ["\u2581", -2.0],
        ["\u2581the", -3.5],
        ["X", -2.5],
        [".", -3.0]
      ]
    }
  })";

  IREE_ASSERT_OK(iree_tokenizer_from_unigram_json(
      IREE_SV(json), iree_allocator_system(), &tokenizer_));

  EXPECT_NE(tokenizer_, nullptr);
}

}  // namespace
