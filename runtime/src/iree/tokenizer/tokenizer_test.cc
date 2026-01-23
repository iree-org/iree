// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/tokenizer.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/bpe.h"
#include "iree/tokenizer/decoder.h"
#include "iree/tokenizer/literals.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/postprocessor.h"
#include "iree/tokenizer/transforms/transform.h"
#include "iree/tokenizer/vocab_builder.h"
#include "iree/tokenizer/wordpiece.h"

namespace {

//===----------------------------------------------------------------------===//
// BPE Tokenizer Tests
//===----------------------------------------------------------------------===//

class BpeTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a simple BPE vocab: a, b, c, ab, bc, abc
    // Merge rules: (a,b)->ab rank 0, (ab,c)->abc rank 1
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        10, iree_allocator_system(), &builder));

    // Add tokens: IDs will be sequential 0, 1, 2, ...
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("a"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("b"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("c"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("ab"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("abc"), 0, 0));

    // Add merge rules.
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 0, 1));  // a+b -> ab
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 3, 2));  // ab+c -> abc

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

    // Create tokenizer. Allocate consumes vocab.
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    // Set whitespace pre-tokenizer (the low-level allocate defaults to none).
    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);
  }

  void TearDown() override {
    // tokenizer_free also frees the vocab (owned by tokenizer).
    iree_tokenizer_free(tokenizer_);
  }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(BpeTokenizerTest, EncodeSimple) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("abc"), options, ids,
                                       IREE_ARRAYSIZE(ids), &count));

  // "abc" should encode to a single token [4] (abc)
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 4);
}

TEST_F(BpeTokenizerTest, EncodeMultipleWords) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("ab c"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // "ab" -> [3], "c" -> [2]
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 3);  // ab
  EXPECT_EQ(ids[1], 2);  // c
}

TEST_F(BpeTokenizerTest, DecodeSimple) {
  int32_t ids[] = {4};  // abc
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_EQ(length, 3u);
  EXPECT_STREQ(text, "abc");
}

TEST_F(BpeTokenizerTest, DecodeMultiple) {
  int32_t ids[] = {0, 1, 2};  // a, b, c
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_EQ(length, 3u);
  EXPECT_STREQ(text, "abc");
}

TEST_F(BpeTokenizerTest, VocabAccessor) {
  const iree_tokenizer_vocab_t* v = iree_tokenizer_vocab(tokenizer_);
  EXPECT_NE(v, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(v), 5u);
}

//===----------------------------------------------------------------------===//
// WordPiece Tokenizer Tests
//===----------------------------------------------------------------------===//

class WordPieceTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a WordPiece vocab with continuation prefix ##.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        10, iree_allocator_system(), &builder));

    // [UNK] at id 0.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[UNK]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0));

    // Regular tokens.
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("un"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("##happy"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("happy"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("test"), 0, 0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

    // Create tokenizer with default config. Allocate consumes vocab.
    IREE_ASSERT_OK(iree_tokenizer_wordpiece_allocate(
        vocab, /*config=*/nullptr, /*prefix_storage=*/nullptr,
        iree_allocator_system(), &tokenizer_));

    // Set whitespace pre-tokenizer (the low-level allocate defaults to none).
    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);
  }

  void TearDown() override {
    // tokenizer_free also frees the vocab (owned by tokenizer).
    iree_tokenizer_free(tokenizer_);
  }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(WordPieceTokenizerTest, EncodeSimple) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("happy"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // "happy" -> [3] (happy is a whole word)
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 3);
}

TEST_F(WordPieceTokenizerTest, EncodeSubword) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("unhappy"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // "unhappy" -> "un" + "##happy" = [1, 2]
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 1);  // un
  EXPECT_EQ(ids[1], 2);  // ##happy
}

TEST_F(WordPieceTokenizerTest, EncodeMultipleWords) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("happy test"),
                                       options, ids, IREE_ARRAYSIZE(ids),
                                       &count));

  // "happy" -> [3], "test" -> [4]
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 3);  // happy
  EXPECT_EQ(ids[1], 4);  // test
}

TEST_F(WordPieceTokenizerTest, DecodeSimple) {
  int32_t ids[] = {1, 2};  // un, ##happy
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Note: decode concatenates raw token text, so "un" + "##happy" = "un##happy"
  EXPECT_EQ(length, 9u);
  EXPECT_STREQ(text, "un##happy");
}

TEST_F(WordPieceTokenizerTest, VocabAccessor) {
  const iree_tokenizer_vocab_t* v = iree_tokenizer_vocab(tokenizer_);
  EXPECT_NE(v, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(v), 5u);
}

//===----------------------------------------------------------------------===//
// Empty and Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(BpeTokenizerTest, EncodeEmptyString) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV(""), options, ids,
                                       IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 0u);
}

TEST_F(BpeTokenizerTest, DecodeEmptyArray) {
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, nullptr, 0,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_EQ(length, 0u);
  EXPECT_STREQ(text, "");
}

//===----------------------------------------------------------------------===//
// Special Token Handling Tests
//===----------------------------------------------------------------------===//

class SpecialTokenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        10, iree_allocator_system(), &builder));

    // ID 0: [BOS]
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[BOS]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_BOS, 0));

    // ID 1: [EOS]
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[EOS]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_EOS, 1));

    // Regular tokens: a, b, c, ab.
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("a"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("b"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("c"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("ab"), 0, 0));

    // Merge rule: a+b -> ab.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder, 2, 3));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    // Set whitespace pre-tokenizer (the low-level allocate defaults to none).
    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);

    // Set up a BOS/EOS template postprocessor: [BOS] $A [EOS].
    iree_tokenizer_template_piece_t* templates = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc(
        iree_allocator_system(), 3 * sizeof(iree_tokenizer_template_piece_t),
        (void**)&templates));
    templates[0] = {0, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};      // BOS
    templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};  // $A
    templates[2] = {1, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};      // EOS
    IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
        templates, 3, 0, iree_allocator_system(), &tokenizer_->postprocessor));
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(SpecialTokenTest, EncodeAddsSpecialTokens) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("ab"), options, ids,
                                       IREE_ARRAYSIZE(ids), &count));

  // Should be: [BOS] ab [EOS] = [0, 5, 1]
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 0);  // [BOS]
  EXPECT_EQ(ids[1], 5);  // ab
  EXPECT_EQ(ids[2], 1);  // [EOS]
}

TEST_F(SpecialTokenTest, EncodeWithoutSpecialTokens) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_DEFAULT,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("ab"), options, ids,
                                       IREE_ARRAYSIZE(ids), &count));

  // Should be just: ab = [5]
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 5);  // ab
}

TEST_F(SpecialTokenTest, EncodeEmptyWithSpecialTokens) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV(""), options, ids,
                                       IREE_ARRAYSIZE(ids), &count));

  // Empty text with special tokens: [BOS] [EOS] = [0, 1]
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 0);  // [BOS]
  EXPECT_EQ(ids[1], 1);  // [EOS]
}

TEST_F(SpecialTokenTest, DecodeSkipsSpecialTokens) {
  int32_t ids[] = {0, 5, 1};  // [BOS] ab [EOS]
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(
      iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                            IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
                            text, sizeof(text), &length));

  // Should skip [BOS] and [EOS], output just "ab".
  EXPECT_EQ(length, 2u);
  EXPECT_STREQ(text, "ab");
}

TEST_F(SpecialTokenTest, DecodeIncludesSpecialTokens) {
  int32_t ids[] = {0, 5, 1};  // [BOS] ab [EOS]
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Should include special tokens: "[BOS]ab[EOS]".
  EXPECT_EQ(length, 12u);
  EXPECT_STREQ(text, "[BOS]ab[EOS]");
}

//===----------------------------------------------------------------------===//
// CLS/SEP Priority Tests
//===----------------------------------------------------------------------===//
// When a tokenizer has BOTH BOS/EOS and CLS/SEP defined, CLS/SEP should take
// precedence. This matches HuggingFace behavior for models like
// funnel-transformer that define both sets of tokens.

class ClsSepPriorityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        20, iree_allocator_system(), &builder));

    // ID 0: <s> (BOS) - like funnel-transformer's ID 96.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("<s>"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_BOS, 0));

    // ID 1: </s> (EOS) - like funnel-transformer's ID 97.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("</s>"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_EOS, 1));

    // ID 2: <cls> (CLS) - like funnel-transformer's ID 101.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("<cls>"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_CLS, 2));

    // ID 3: <sep> (SEP) - like funnel-transformer's ID 102.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("<sep>"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_SEP, 3));

    // Character tokens (IDs 4-7).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("h"),
                                                          0, 0));  // ID 4
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("e"),
                                                          0, 0));  // ID 5
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("l"),
                                                          0, 0));  // ID 6
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("o"),
                                                          0, 0));  // ID 7

    // Intermediate merged tokens (IDs 8-10).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("he"), 0, 0));  // ID 8
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("ll"), 0, 0));  // ID 9
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("llo"), 0, 0));  // ID 10

    // Final merged token (ID 11).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));  // ID 11

    // Merge rules: BPE builds up step-by-step.
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 4, 5));  // h+e -> he
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 6, 6));  // l+l -> ll
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 9, 7));  // ll+o -> llo
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(
        builder, 8, 10));  // he+llo -> hello

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));
    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);

    // Set up a CLS/SEP template postprocessor: [CLS] $A [SEP].
    iree_tokenizer_template_piece_t* templates = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc(
        iree_allocator_system(), 3 * sizeof(iree_tokenizer_template_piece_t),
        (void**)&templates));
    templates[0] = {2, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};      // CLS
    templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};  // $A
    templates[2] = {3, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};      // SEP
    IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
        templates, 3, 0, iree_allocator_system(), &tokenizer_->postprocessor));
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(ClsSepPriorityTest, ClsTakesPriorityOverBos) {
  // When both BOS (ID 0) and CLS (ID 2) are defined, CLS should be used.
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // Should be: [CLS] hello [SEP] = [2, 11, 3]
  // NOT: [BOS] hello [EOS] = [0, 11, 1]
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 2);   // <cls> (NOT <s>)
  EXPECT_EQ(ids[1], 11);  // hello
  EXPECT_EQ(ids[2], 3);   // <sep> (NOT </s>)
}

TEST_F(ClsSepPriorityTest, EmptyInputWithClsSep) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV(""), options, ids,
                                       IREE_ARRAYSIZE(ids), &count));

  // Empty with special tokens: [CLS] [SEP] = [2, 3]
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 2);  // <cls>
  EXPECT_EQ(ids[1], 3);  // <sep>
}

//===----------------------------------------------------------------------===//
// BOS/EOS Fallback Tests
//===----------------------------------------------------------------------===//
// When CLS/SEP are NOT defined, BOS/EOS should be used.

class BosEosFallbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        20, iree_allocator_system(), &builder));

    // Only BOS/EOS, no CLS/SEP.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("<s>"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_BOS, 0));

    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("</s>"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_EOS, 1));

    // Character tokens (IDs 2-5).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("h"),
                                                          0, 0));  // ID 2
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("e"),
                                                          0, 0));  // ID 3
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("l"),
                                                          0, 0));  // ID 4
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("o"),
                                                          0, 0));  // ID 5

    // Intermediate merged tokens (IDs 6-8).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("he"), 0, 0));  // ID 6
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("ll"), 0, 0));  // ID 7
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("llo"), 0, 0));  // ID 8

    // Final merged token (ID 9).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));  // ID 9

    // Merge rules: BPE builds up step-by-step.
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 2, 3));  // h+e -> he
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 4, 4));  // l+l -> ll
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 7, 5));  // ll+o -> llo
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(
        builder, 6, 8));  // he+llo -> hello

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));
    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);

    // Set up a BOS/EOS template postprocessor: [BOS] $A [EOS].
    iree_tokenizer_template_piece_t* templates = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc(
        iree_allocator_system(), 3 * sizeof(iree_tokenizer_template_piece_t),
        (void**)&templates));
    templates[0] = {0, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};      // BOS
    templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};  // $A
    templates[2] = {1, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};      // EOS
    IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
        templates, 3, 0, iree_allocator_system(), &tokenizer_->postprocessor));
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(BosEosFallbackTest, UsesBosEosWhenNoClsSep) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // With only BOS/EOS defined: [BOS] hello [EOS] = [0, 9, 1]
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 0);  // <s> (BOS)
  EXPECT_EQ(ids[1], 9);  // hello
  EXPECT_EQ(ids[2], 1);  // </s> (EOS)
}

//===----------------------------------------------------------------------===//
// Truncation Tests
//===----------------------------------------------------------------------===//

TEST_F(SpecialTokenTest, TruncateRight) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode "a b c" with max_length=3, add_special_tokens=true.
  // Without truncation: [BOS] a b c [EOS] = 5 tokens.
  // With truncation to 3: [BOS] a b = first 3 tokens.
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/3,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("a b c"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 0);  // [BOS]
  EXPECT_EQ(ids[1], 2);  // a
  EXPECT_EQ(ids[2], 3);  // b
}

TEST_F(SpecialTokenTest, TruncateLeft) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode "a b c" with max_length=3, truncate_left=true, add_special_tokens.
  // Without truncation: [BOS] a b c [EOS] = 5 tokens.
  // With left truncation to 3: c [EOS] and then shift = last 3 tokens.
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS |
          IREE_TOKENIZER_ENCODE_FLAG_TRUNCATE_LEFT,
      /*max_length=*/3,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("a b c"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 3);  // b
  EXPECT_EQ(ids[1], 4);  // c
  EXPECT_EQ(ids[2], 1);  // [EOS]
}

TEST_F(SpecialTokenTest, NoTruncationNeeded) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode "ab" with max_length=10 (larger than needed).
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/10,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("ab"), options, ids,
                                       IREE_ARRAYSIZE(ids), &count));

  // No truncation: [BOS] ab [EOS] = 3 tokens.
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 0);  // [BOS]
  EXPECT_EQ(ids[1], 5);  // ab
  EXPECT_EQ(ids[2], 1);  // [EOS]
}

TEST_F(SpecialTokenTest, TruncateWithoutSpecialTokens) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode "a b c" with max_length=2, no special tokens.
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_DEFAULT,
      /*max_length=*/2,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("a b c"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // Without special tokens: a b c = 3 tokens, truncated to 2.
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 2);  // a
  EXPECT_EQ(ids[1], 3);  // b
}

//===----------------------------------------------------------------------===//
// Error Path Tests
//===----------------------------------------------------------------------===//

TEST_F(SpecialTokenTest, EncodeBufferTooSmall) {
  int32_t ids[2];  // Only space for 2 tokens.
  iree_host_size_t count = 0;

  // "a b c" without special tokens produces 3 tokens, but buffer only has 2.
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_DEFAULT,
      /*max_length=*/0,
  };
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, IREE_SV("a b c"), options, ids, IREE_ARRAYSIZE(ids), &count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);
}

TEST_F(SpecialTokenTest, EncodeBufferTooSmallForSpecialTokens) {
  int32_t ids[1];  // Only space for 1 token.
  iree_host_size_t count = 0;

  // With add_special_tokens, even "ab" needs 3 tokens: [BOS] ab [EOS].
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, IREE_SV("ab"), options, ids, IREE_ARRAYSIZE(ids), &count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);
}

TEST_F(SpecialTokenTest, DecodeBufferTooSmall) {
  int32_t ids[] = {5};  // "ab" (2 chars)
  char text[2];         // Only space for 1 char + null.
  iree_host_size_t length = 0;

  iree_status_t status = iree_tokenizer_decode(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      text, sizeof(text), &length);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);
}

TEST_F(SpecialTokenTest, DecodeOnlySpecialTokensWithSkip) {
  int32_t ids[] = {0, 1};  // [BOS] [EOS] only
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(
      iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                            IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
                            text, sizeof(text), &length));

  // All tokens are special, so output should be empty.
  EXPECT_EQ(length, 0u);
  EXPECT_STREQ(text, "");
}

TEST_F(SpecialTokenTest, DecodeInvalidTokenId) {
  int32_t ids[] = {999};  // ID that doesn't exist.
  char text[64];
  iree_host_size_t length = 0;

  // Out-of-range IDs should return an error (fail fast, no silent failures).
  iree_status_t status = iree_tokenizer_decode(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      text, sizeof(text), &length);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_free(status);
}

TEST_F(SpecialTokenTest, ExtremeTruncationToOne) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // max_length=1 with add_special_tokens: only room for BOS.
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/1,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("a b c"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 0);  // [BOS]
}

TEST_F(SpecialTokenTest, ExtremeTruncationToTwo) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // max_length=2 with add_special_tokens: BOS + first token (a).
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/2,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("a b c"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 0);  // [BOS]
  EXPECT_EQ(ids[1], 2);  // a
}

TEST_F(SpecialTokenTest, DecodeZeroBufferSize) {
  int32_t ids[] = {5};  // "ab"
  char text[1];
  iree_host_size_t length = 0;

  // Zero buffer size is explicitly invalid.
  iree_status_t status =
      iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                            IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                            /*max_text=*/0, &length);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
}

TEST_F(SpecialTokenTest, EncodeEmptyWithSpecialTokensBufferTooSmall) {
  int32_t ids[1];  // Only space for 1 token.
  iree_host_size_t count = 0;

  // Empty input with add_special_tokens needs 2 tokens: [BOS] [EOS].
  // Buffer only has room for 1, should return RESOURCE_EXHAUSTED.
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, IREE_SV(""), options, ids, IREE_ARRAYSIZE(ids), &count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);
}

TEST_F(SpecialTokenTest, EncodeEmptyWithSpecialTokensZeroBuffer) {
  int32_t ids[1];
  iree_host_size_t count = 0;

  // Zero buffer should fail for empty input with special tokens.
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  iree_status_t status = iree_tokenizer_encode(tokenizer_, IREE_SV(""), options,
                                               ids, /*max_ids=*/0, &count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);
}

TEST_F(SpecialTokenTest, DecodeNegativeTokenId) {
  int32_t ids[] = {-1, 5, -100};  // Negative IDs are invalid.
  char text[64];
  iree_host_size_t length = 0;

  // Out-of-range IDs (including negative) should return an error.
  iree_status_t status = iree_tokenizer_decode(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      text, sizeof(text), &length);
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_free(status);
}

//===----------------------------------------------------------------------===//
// Normalizer Integration Tests
//===----------------------------------------------------------------------===//

// Tests tokenizer with BertNormalizer (lowercase, strip accents).
class NormalizerIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a WordPiece vocab with tokens for testing normalization.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        15, iree_allocator_system(), &builder));

    // Special tokens.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[CLS]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_CLS, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[SEP]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_SEP, 1));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[UNK]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2));

    // Lowercase tokens (what normalized text will match).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("world"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("cafe"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("test"), 0, 0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

    // Create tokenizer with BertNormalizer.
    IREE_ASSERT_OK(iree_tokenizer_wordpiece_allocate(
        vocab, /*config=*/nullptr, /*prefix_storage=*/nullptr,
        iree_allocator_system(), &tokenizer_));

    // Set up BertNormalizer (lowercase + strip accents) with Bert
    // pre-tokenizer.
    iree_tokenizer_text_transform_initialize_bert(&tokenizer_->transform);
    iree_tokenizer_normalizer_initialize_bert(
        IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE |
            IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS,
        &tokenizer_->transform.normalizer);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(NormalizerIntegrationTest, LowercasesInput) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // "HELLO" should be normalized to "hello" and match token 3.
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("HELLO"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 3);  // hello
}

TEST_F(NormalizerIntegrationTest, StripsAccents) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // "café" should be normalized to "cafe" and match token 5.
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("café"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 5);  // cafe
}

TEST_F(NormalizerIntegrationTest, MixedCaseAndAccents) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // "CAFÉ" should normalize to "cafe".
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("CAFÉ"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 5);  // cafe
}

TEST_F(NormalizerIntegrationTest, MultipleWordsNormalized) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // "HELLO WORLD" -> "hello world" -> [hello, world]
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("HELLO WORLD"),
                                       options, ids, IREE_ARRAYSIZE(ids),
                                       &count));

  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 3);  // hello
  EXPECT_EQ(ids[1], 4);  // world
}

//===----------------------------------------------------------------------===//
// Decoder Integration Tests
//===----------------------------------------------------------------------===//

// Tests tokenizer with WordPiece decoder (subword cleanup).
class DecoderIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create vocab with subword tokens.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        10, iree_allocator_system(), &builder));

    // UNK token required for WordPiece.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[UNK]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0));

    // Regular tokens and subwords (IDs 1-5).
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("un"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("##believ"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("##able"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("test"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("##ing"), 0, 0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

    // Create tokenizer.
    IREE_ASSERT_OK(iree_tokenizer_wordpiece_allocate(
        vocab, /*config=*/nullptr, /*prefix_storage=*/nullptr,
        iree_allocator_system(), &tokenizer_));

    // Set up WordPiece decoder.
    iree_tokenizer_decoder_initialize_wordpiece(
        IREE_SVL("##"), IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_CLEANUP_SPACES,
        &tokenizer_->decoder);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(DecoderIntegrationTest, JoinsSubwords) {
  // "un" + "##believ" + "##able" should decode to "unbelievable".
  int32_t ids[] = {1, 2, 3};  // un, ##believ, ##able (0 is [UNK])
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_EQ(length, 12u);
  EXPECT_STREQ(text, "unbelievable");
}

TEST_F(DecoderIntegrationTest, SingleToken) {
  int32_t ids[] = {4};  // test (0 is [UNK])
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_EQ(length, 4u);
  EXPECT_STREQ(text, "test");
}

TEST_F(DecoderIntegrationTest, MixedTokensAndSubwords) {
  // "test" + "##ing" should decode to "testing".
  int32_t ids[] = {4, 5};  // test, ##ing (0 is [UNK])
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_EQ(length, 7u);
  EXPECT_STREQ(text, "testing");
}

//===----------------------------------------------------------------------===//
// Full Pipeline Integration Tests (Normalizer + Encoder + Decoder)
//===----------------------------------------------------------------------===//

// Tests complete roundtrip with normalizer and decoder.
class FullPipelineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        15, iree_allocator_system(), &builder));

    // Special tokens.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[CLS]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_CLS, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[SEP]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_SEP, 1));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("[UNK]"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2));

    // Lowercase tokens with subwords.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("world"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("test"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("##ing"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("cafe"), 0, 0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

    IREE_ASSERT_OK(iree_tokenizer_wordpiece_allocate(
        vocab, /*config=*/nullptr, /*prefix_storage=*/nullptr,
        iree_allocator_system(), &tokenizer_));

    // Set up BertNormalizer (lowercase + strip accents) with Bert
    // pre-tokenizer.
    iree_tokenizer_text_transform_initialize_bert(&tokenizer_->transform);
    iree_tokenizer_normalizer_initialize_bert(
        IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE |
            IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS,
        &tokenizer_->transform.normalizer);

    // Set up WordPiece decoder.
    iree_tokenizer_decoder_initialize_wordpiece(
        IREE_SVL("##"), IREE_TOKENIZER_WORDPIECE_DECODER_FLAG_CLEANUP_SPACES,
        &tokenizer_->decoder);

    // Set up a CLS/SEP template postprocessor: [CLS] $A [SEP].
    iree_tokenizer_template_piece_t* templates = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc(
        iree_allocator_system(), 3 * sizeof(iree_tokenizer_template_piece_t),
        (void**)&templates));
    templates[0] = {0, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};  // [CLS]
    templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};  // $A
    templates[2] = {1, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};  // [SEP]
    IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
        templates, 3, 0, iree_allocator_system(), &tokenizer_->postprocessor));
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(FullPipelineTest, RoundtripSimple) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode "HELLO" (normalized to "hello").
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("HELLO"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 3);  // hello

  // Decode back.
  char text[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, count,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Should get lowercase "hello" back.
  EXPECT_EQ(length, 5u);
  EXPECT_STREQ(text, "hello");
}

TEST_F(FullPipelineTest, RoundtripWithAccents) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode "CAFÉ" (normalized to "cafe").
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("CAFÉ"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 7);  // cafe

  // Decode back.
  char text[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, count,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Accent is stripped, so we get "cafe".
  EXPECT_EQ(length, 4u);
  EXPECT_STREQ(text, "cafe");
}

TEST_F(FullPipelineTest, RoundtripWithSpecialTokens) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode with special tokens.
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("HELLO"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // Should be [CLS] hello [SEP].
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 0);  // [CLS]
  EXPECT_EQ(ids[1], 3);  // hello
  EXPECT_EQ(ids[2], 1);  // [SEP]

  // Decode with skip_special_tokens.
  char text[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode(
      tokenizer_, ids, count, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
      text, sizeof(text), &length));

  // Should get just "hello".
  EXPECT_EQ(length, 5u);
  EXPECT_STREQ(text, "hello");
}

TEST_F(FullPipelineTest, RoundtripSubwordDecoding) {
  // Manually create subword sequence and decode.
  int32_t ids[] = {5, 6};  // test, ##ing
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // WordPiece decoder should join to "testing".
  EXPECT_EQ(length, 7u);
  EXPECT_STREQ(text, "testing");
}

//===----------------------------------------------------------------------===//
// ByteLevel (GPT-2 Style) Round-Trip Tests
//===----------------------------------------------------------------------===//

// Tests GPT-2 style tokenizer with ByteLevel pre-tokenizer and decoder.
// These tests verify that encode->decode produces the original text.
class ByteLevelRoundTripTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a BPE vocab with GPT-2 style byte-level encoding.
    // Ġ (U+0120) represents space in GPT-2.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        20, iree_allocator_system(), &builder));

    // Special tokens.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("<|endoftext|>"), 0,
        IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_EOS, 0));

    // Byte-level tokens (GPT-2 uses Ġ for space).
    // Ġ is UTF-8 encoded as C4 A0.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xC4\xA0"), 0, 0));  // ID 1: Ġ (space)
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("h"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("e"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("l"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("o"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("w"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("r"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("d"), 0, 0));

    // Merged tokens.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("he"), 0, 0));  // ID 9
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("ll"), 0, 0));  // ID 10
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("llo"), 0, 0));  // ID 11
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));  // ID 12
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xC4\xA0world"), 0, 0));  // ID 13: Ġworld

    // Merge rules (using correct IDs: h=2, e=3, l=4, o=5, he=9, ll=10, llo=11).
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 2, 3));  // h+e -> he
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 4, 4));  // l+l -> ll
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 10, 5));  // ll+o -> llo
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder, 9,
                                               11));  // he+llo -> hello

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    // Set up ByteLevel pre-tokenizer with ADD_PREFIX_SPACE.
    iree_tokenizer_text_transform_initialize_byte_level(
        IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE,
        &tokenizer_->transform);

    // Set up ByteLevel decoder with ADD_PREFIX_SPACE flag.
    iree_tokenizer_decoder_initialize_byte_level(
        IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_ADD_PREFIX_SPACE,
        &tokenizer_->decoder);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(ByteLevelRoundTripTest, SimpleWord) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode "hello".
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // Decode back.
  char text[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, count,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_STREQ(text, "hello");
}

TEST_F(ByteLevelRoundTripTest, MultipleWords) {
  // Test decoding token IDs directly (simulating GPT-2 output).
  // "Ġworld" is a single token in GPT-2 style tokenizers.
  int32_t ids[] = {12, 13};  // hello (ID 12), Ġworld (ID 13)
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // ByteLevel decoder reverses the Ġ->space mapping.
  EXPECT_STREQ(text, "hello world");
}

TEST_F(ByteLevelRoundTripTest, PrefixSpaceStripping) {
  // Decode a single token that starts with Ġ (space).
  // With ADD_PREFIX_SPACE flag, the leading space should be stripped.
  int32_t ids[] = {13};  // Ġworld (ID 13)
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Leading space stripped because ADD_PREFIX_SPACE is set.
  EXPECT_STREQ(text, "world");
}

//===----------------------------------------------------------------------===//
// Sequence[ByteLevel] + ByteLevel Decoder Double-Decode Prevention
//===----------------------------------------------------------------------===//

// Regression test: when a Sequence transform contains ByteLevel and the decoder
// is also ByteLevel, the code should detect this and skip transform decode
// (decoder handles the inverse). Previously, only top-level transform type was
// checked, causing double-decode with corrupted output.
class SequenceByteLevelDoubleDecodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        20, iree_allocator_system(), &builder));

    // GPT-2 style tokens. Ġ (U+0120 = 288 = 256 + 32) represents space.
    // UTF-8 encoding: C4 A0.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder,
                                                          IREE_SV("hello"), 0,
                                                          0));  // ID 0
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xC4\xA0world"), 0, 0));  // ID 1: Ġworld

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    // Create a Sequence transform containing ByteLevel (simulates complex
    // tokenizer.json configs where ByteLevel is wrapped in a Sequence).
    iree_tokenizer_text_transform_t byte_level_child;
    iree_tokenizer_text_transform_initialize_byte_level(
        IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE, &byte_level_child);

    IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
        &byte_level_child, 1, iree_allocator_system(), &tokenizer_->transform));

    // ByteLevel decoder - same inverse as the ByteLevel in Sequence.
    iree_tokenizer_decoder_initialize_byte_level(
        IREE_TOKENIZER_BYTE_LEVEL_DECODER_FLAG_ADD_PREFIX_SPACE,
        &tokenizer_->decoder);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(SequenceByteLevelDoubleDecodeTest, NoDoubleDecodeOnSingleToken) {
  // Token "hello" has no special bytes, so double-decode wouldn't corrupt it.
  // But this verifies the basic path works.
  int32_t ids[] = {0};  // hello
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_STREQ(text, "hello");
}

TEST_F(SequenceByteLevelDoubleDecodeTest, NoDoubleDecodeOnGPT2Space) {
  // Token "Ġworld" contains Ġ (encoded space). If double-decoded:
  // - First decode (decoder): Ġ -> space
  // - Second decode (transform): space stays space (no change for space char)
  // But the issue is that if the transform ALSO tried to decode, it would
  // interpret the already-decoded bytes incorrectly.
  //
  // With the fix: decoder handles inverse, transform decode is skipped.
  int32_t ids[] = {0, 1};  // hello, Ġworld
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Correct output: "hello world" (Ġ decoded to space by decoder).
  // Without fix: could get corrupted output or double-decoded garbage.
  EXPECT_STREQ(text, "hello world");
}

//===----------------------------------------------------------------------===//
// Metaspace (SentencePiece Style) Round-Trip Tests
//===----------------------------------------------------------------------===//

// Tests SentencePiece style tokenizer with Metaspace pre-tokenizer and decoder.
class MetaspaceRoundTripTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create vocab with SentencePiece style tokens (▁ prefix for word start).
    // ▁ is UTF-8 encoded as E2 96 81.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        15, iree_allocator_system(), &builder));

    // Special tokens.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("<s>"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_BOS, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("</s>"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_EOS, 1));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("<unk>"), 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2));

    // Metaspace tokens (▁ = U+2581 = E2 96 81 in UTF-8).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xE2\x96\x81hello"), 0, 0));  // ID 3: ▁hello
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xE2\x96\x81world"), 0, 0));  // ID 4: ▁world
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xE2\x96\x81the"), 0, 0));  // ID 5: ▁the
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("ing"), 0, 0));  // ID 6: ing (subword)
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xE2\x96\x81test"), 0, 0));  // ID 7: ▁test

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    // Set up Metaspace pre-tokenizer with PREPEND_ALWAYS and SPLIT.
    iree_tokenizer_text_transform_initialize_metaspace(
        0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
        IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &tokenizer_->transform);

    // Set up Metaspace decoder with strip leading (matches prepend_scheme).
    iree_tokenizer_decoder_initialize_metaspace(
        0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING,
        &tokenizer_->decoder);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(MetaspaceRoundTripTest, SingleWord) {
  // Decode a single word with leading metaspace.
  int32_t ids[] = {3};  // ▁hello
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Metaspace at start is stripped, so just "hello".
  EXPECT_STREQ(text, "hello");
}

TEST_F(MetaspaceRoundTripTest, MultipleWords) {
  // "▁hello ▁world" -> "hello world"
  int32_t ids[] = {3, 4};  // ▁hello, ▁world
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Metaspace becomes space between words.
  EXPECT_STREQ(text, "hello world");
}

TEST_F(MetaspaceRoundTripTest, SubwordContinuation) {
  // "▁test" + "ing" -> "testing"
  int32_t ids[] = {7, 6};  // ▁test, ing
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Subword without ▁ joins directly.
  EXPECT_STREQ(text, "testing");
}

TEST_F(MetaspaceRoundTripTest, WithSpecialTokens) {
  // "<s> ▁hello ▁world </s>" with skip_special_tokens.
  int32_t ids[] = {0, 3, 4, 1};  // <s>, ▁hello, ▁world, </s>
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(
      iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                            IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
                            text, sizeof(text), &length));

  // Special tokens skipped, metaspaces converted.
  EXPECT_STREQ(text, "hello world");
}

//===----------------------------------------------------------------------===//
// Streaming API Tests - Callback Batching
//===----------------------------------------------------------------------===//

// Context for counting callback invocations and collecting data.
struct StreamingTestContext {
  std::vector<std::vector<int32_t>> batches;
  size_t total_tokens;
};

static iree_status_t StreamingTestCallback(void* user_data,
                                           iree_tokenizer_id_list_t ids) {
  auto* ctx = static_cast<StreamingTestContext*>(user_data);
  std::vector<int32_t> batch(ids.values, ids.values + ids.count);
  ctx->batches.push_back(batch);
  ctx->total_tokens += ids.count;
  return iree_ok_status();
}

class StreamingEncodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a simple vocab where each character is a separate token.
    // BPE without merge rules acts as a character-level tokenizer.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        256, iree_allocator_system(), &builder));

    // Add single-char tokens a-z.
    for (char c = 'a'; c <= 'z'; ++c) {
      char str[2] = {c, 0};
      IREE_ASSERT_OK(
          iree_tokenizer_vocab_builder_add_token(builder, IREE_SV(str), 0, 0));
    }

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));
    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(StreamingEncodeTest, SmallInput) {
  StreamingTestContext ctx = {{}, 0};

  IREE_ASSERT_OK(iree_tokenizer_encode_streaming(
      tokenizer_, IREE_SV("abc"), IREE_TOKENIZER_ENCODE_FLAG_DEFAULT,
      StreamingTestCallback, &ctx));

  EXPECT_EQ(ctx.total_tokens, 3u);
  // Small input should come in one batch.
  EXPECT_EQ(ctx.batches.size(), 1u);
  EXPECT_EQ(ctx.batches[0].size(), 3u);
}

TEST_F(StreamingEncodeTest, EmptyInput) {
  StreamingTestContext ctx = {{}, 0};

  IREE_ASSERT_OK(iree_tokenizer_encode_streaming(
      tokenizer_, IREE_SV(""), IREE_TOKENIZER_ENCODE_FLAG_DEFAULT,
      StreamingTestCallback, &ctx));

  EXPECT_EQ(ctx.total_tokens, 0u);
  EXPECT_EQ(ctx.batches.size(), 0u);
}

TEST_F(StreamingEncodeTest, MultipleWords) {
  StreamingTestContext ctx = {{}, 0};

  // "abc def" = 6 chars (space is not in vocab, so skipped by whitespace
  // tokenizer).
  IREE_ASSERT_OK(iree_tokenizer_encode_streaming(
      tokenizer_, IREE_SV("abc def"), IREE_TOKENIZER_ENCODE_FLAG_DEFAULT,
      StreamingTestCallback, &ctx));

  EXPECT_EQ(ctx.total_tokens, 6u);
}

//===----------------------------------------------------------------------===//
// Streaming Decode Tests - Boundary Conditions
//===----------------------------------------------------------------------===//

// Context for streaming decode tests.
struct StreamingDecodeContext {
  std::vector<std::string> chunks;
  std::string concatenated;
};

static iree_status_t StreamingDecodeCallback(void* user_data,
                                             iree_string_view_list_t strings) {
  auto* ctx = static_cast<StreamingDecodeContext*>(user_data);
  for (iree_host_size_t i = 0; i < strings.count; ++i) {
    std::string chunk(strings.values[i].data, strings.values[i].size);
    ctx->chunks.push_back(chunk);
    ctx->concatenated += chunk;
  }
  return iree_ok_status();
}

class StreamingDecodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a vocab with tokens of varying lengths.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        100, iree_allocator_system(), &builder));

    // Single char tokens.
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("a"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("b"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("c"), 0, 0));

    // Multi-char tokens.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("world"), 0, 0));

    // Long token for boundary testing.
    std::string long_token(1000, 'x');
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, iree_make_string_view(long_token.data(), long_token.size()), 0,
        0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    // No transform, no decoder - raw concatenation.
    iree_tokenizer_text_transform_initialize_none(&tokenizer_->transform);
    iree_tokenizer_decoder_initialize_none(&tokenizer_->decoder);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(StreamingDecodeTest, SmallInput) {
  StreamingDecodeContext ctx;

  int32_t ids[] = {0, 1, 2};  // a, b, c
  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  EXPECT_EQ(ctx.concatenated, "abc");
}

TEST_F(StreamingDecodeTest, EmptyInput) {
  StreamingDecodeContext ctx;

  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, nullptr, 0, IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  EXPECT_EQ(ctx.concatenated, "");
  EXPECT_EQ(ctx.chunks.size(), 0u);
}

TEST_F(StreamingDecodeTest, SingleToken) {
  StreamingDecodeContext ctx;

  int32_t ids[] = {3};  // hello
  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  EXPECT_EQ(ctx.concatenated, "hello");
}

TEST_F(StreamingDecodeTest, ManySmallTokens) {
  StreamingDecodeContext ctx;

  // Create many small tokens to test batching.
  std::vector<int32_t> ids;
  for (int i = 0; i < 100; ++i) {
    ids.push_back(i % 3);  // Rotate through a, b, c.
  }

  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, ids.data(), ids.size(), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  EXPECT_EQ(ctx.concatenated.size(), 100u);
  // Should have multiple chunks due to batching.
  EXPECT_GT(ctx.chunks.size(), 0u);
}

TEST_F(StreamingDecodeTest, LongToken) {
  StreamingDecodeContext ctx;

  int32_t ids[] = {5};  // 1000-char token
  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  EXPECT_EQ(ctx.concatenated.size(), 1000u);
}

TEST_F(StreamingDecodeTest, MixedTokenLengths) {
  StreamingDecodeContext ctx;

  // Mix of short and long tokens.
  int32_t ids[] = {0, 5, 1, 3, 2};  // a, 1000x, b, hello, c
  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  std::string expected = "a";
  expected += std::string(1000, 'x');
  expected += "bhelloc";
  EXPECT_EQ(ctx.concatenated, expected);
}

TEST_F(StreamingDecodeTest, InvalidTokenIdReturnsError) {
  StreamingDecodeContext ctx;

  // Mix of valid and invalid token IDs.
  int32_t ids[] = {0, 999, 1, -1, 2};  // a, INVALID, b, INVALID, c
  iree_status_t status = iree_tokenizer_decode_streaming(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx);

  // Invalid IDs should return an error (fail fast, no silent failures).
  EXPECT_TRUE(iree_status_is_invalid_argument(status));
  iree_status_free(status);
}

//===----------------------------------------------------------------------===//
// Decoder Streaming State Tests
//===----------------------------------------------------------------------===//

// Test that decoder state (is_first_token) is preserved across flush
// boundaries.
class DecoderStreamingStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create vocab with metaspace tokens for testing is_first_token state.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        100, iree_allocator_system(), &builder));

    // Metaspace tokens (▁ = E2 96 81).
    // Note: String concatenation needed to avoid \x81a being interpreted as a
    // single hex escape (since 'a' is a valid hex digit).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder,
                                                          IREE_SV("\xE2\x96\x81"
                                                                  "a"),
                                                          0, 0));  // ID 0: ▁a
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder,
                                                          IREE_SV("\xE2\x96\x81"
                                                                  "b"),
                                                          0, 0));  // ID 1: ▁b
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder,
                                                          IREE_SV("\xE2\x96\x81"
                                                                  "c"),
                                                          0, 0));  // ID 2: ▁c

    // Long token to force flush.
    std::string long_token = "\xE2\x96\x81";
    long_token += std::string(4000, 'x');
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, iree_make_string_view(long_token.data(), long_token.size()), 0,
        0));  // ID 3: ▁xxxx...

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    // No transform (we're testing decoder directly).
    iree_tokenizer_text_transform_initialize_none(&tokenizer_->transform);
    // Metaspace decoder strips leading ▁ on first token.
    iree_tokenizer_decoder_initialize_metaspace(
        0, IREE_TOKENIZER_METASPACE_DECODER_FLAG_STRIP_LEADING,
        &tokenizer_->decoder);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(DecoderStreamingStateTest, FirstTokenStripsMetaspace) {
  StreamingDecodeContext ctx;

  // Single token - metaspace at start should be stripped.
  int32_t ids[] = {0};  // ▁a
  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  EXPECT_EQ(ctx.concatenated, "a");  // Leading ▁ stripped.
}

TEST_F(DecoderStreamingStateTest, SecondTokenKeepsMetaspace) {
  StreamingDecodeContext ctx;

  // Two tokens - first ▁ stripped, second becomes space.
  int32_t ids[] = {0, 1};  // ▁a, ▁b
  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  EXPECT_EQ(ctx.concatenated, "a b");  // First ▁ stripped, second -> space.
}

TEST_F(DecoderStreamingStateTest, StatePreservedAcrossFlush) {
  StreamingDecodeContext ctx;

  // Long token forces a flush, then short token.
  // The is_first_token state should be preserved across the flush.
  int32_t ids[] = {3, 1};  // ▁xxxx..., ▁b
  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, ids, IREE_ARRAYSIZE(ids), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  // First token (long) has its leading ▁ stripped.
  // Second token's ▁ becomes space.
  std::string expected = std::string(4000, 'x') + " b";
  EXPECT_EQ(ctx.concatenated, expected);
}

TEST_F(DecoderStreamingStateTest, ManyTokensAcrossFlushes) {
  StreamingDecodeContext ctx;

  // Create enough tokens to span multiple flushes.
  std::vector<int32_t> ids;
  ids.push_back(3);  // Long token first (forces flush).
  for (int i = 0; i < 50; ++i) {
    ids.push_back(i % 3);  // Rotate through ▁a, ▁b, ▁c.
  }

  IREE_ASSERT_OK(iree_tokenizer_decode_streaming(
      tokenizer_, ids.data(), ids.size(), IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
      StreamingDecodeCallback, &ctx));

  // First token's ▁ stripped, all subsequent become spaces.
  std::string expected = std::string(4000, 'x');
  for (int i = 0; i < 50; ++i) {
    expected += " ";
    expected += ('a' + (i % 3));
  }
  EXPECT_EQ(ctx.concatenated, expected);
}

//===----------------------------------------------------------------------===//
// Buffer Boundary Edge Cases
//===----------------------------------------------------------------------===//

class BufferBoundaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        10, iree_allocator_system(), &builder));

    // Create tokens of specific sizes to test boundaries.
    // Token 0: exactly 1 byte.
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("x"), 0, 0));

    // Token 1: exactly 100 bytes.
    std::string token100(100, 'y');
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, iree_make_string_view(token100.data(), token100.size()), 0,
        0));

    // Token 2: exactly 1000 bytes.
    std::string token1000(1000, 'z');
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, iree_make_string_view(token1000.data(), token1000.size()), 0,
        0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));
    iree_tokenizer_text_transform_initialize_none(&tokenizer_->transform);
    iree_tokenizer_decoder_initialize_none(&tokenizer_->decoder);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(BufferBoundaryTest, ExactlyOneByte) {
  char text[64];
  iree_host_size_t length = 0;
  int32_t ids[] = {0};

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, 1,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));
  EXPECT_EQ(length, 1u);
  EXPECT_EQ(text[0], 'x');
}

TEST_F(BufferBoundaryTest, ExactBufferFit) {
  // Buffer exactly fits the output.
  char text[101];  // 100 bytes + null terminator.
  iree_host_size_t length = 0;
  int32_t ids[] = {1};  // 100-byte token.

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, 1,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));
  EXPECT_EQ(length, 100u);
}

TEST_F(BufferBoundaryTest, BufferOneByteShort) {
  // Buffer is one byte short.
  char text[100];  // Need 101 (100 + null), have 100.
  iree_host_size_t length = 0;
  int32_t ids[] = {1};  // 100-byte token.

  iree_status_t status = iree_tokenizer_decode(
      tokenizer_, ids, 1, IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
      sizeof(text), &length);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);
}

TEST_F(BufferBoundaryTest, MultipleTokensExactFit) {
  // Multiple tokens that exactly fit.
  char text[201];  // 100 + 100 + null.
  iree_host_size_t length = 0;
  int32_t ids[] = {1, 1};  // Two 100-byte tokens.

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, 2,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));
  EXPECT_EQ(length, 200u);
}

TEST_F(BufferBoundaryTest, ZeroSizeBuffer) {
  // Zero-size buffer is an invalid argument.
  char text[1];  // Minimal buffer.
  iree_host_size_t length = 0;
  int32_t ids[] = {0};  // 1-byte token.

  iree_status_t status = iree_tokenizer_decode(
      tokenizer_, ids, 1, IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text, 0, &length);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_status_free(status);
}

TEST_F(BufferBoundaryTest, LargeInputSmallBuffer) {
  // Large input that doesn't fit in buffer.
  char text[10];  // Way too small for 1000-byte token.
  iree_host_size_t length = 0;
  int32_t ids[] = {2};  // 1000-byte token.

  iree_status_t status = iree_tokenizer_decode(
      tokenizer_, ids, 1, IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
      sizeof(text), &length);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_free(status);
}

//===----------------------------------------------------------------------===//
// Invalid UTF-8 Handling Tests
//===----------------------------------------------------------------------===//
// These tests verify security-sensitive UTF-8 validation. Invalid sequences
// should be handled gracefully without crashes, buffer overflows, or undefined
// behavior.

class InvalidUtf8Test : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a simple BPE tokenizer for testing.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        10, iree_allocator_system(), &builder));

    // Add basic ASCII tokens.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("world"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("a"), 0, 0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));
    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(InvalidUtf8Test, OverlongEncoding) {
  // Overlong encoding of NUL: C0 80 should be 00, but encoded as 2 bytes.
  // This is a security concern (null byte injection).
  int32_t ids[32];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  // Attempt to encode text with overlong NUL.
  const char overlong_nul[] = "hello\xC0\x80world";
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(overlong_nul, sizeof(overlong_nul) - 1),
      options, ids, IREE_ARRAYSIZE(ids), &count);

  // Should handle without crash. May succeed with replacement or fail.
  if (iree_status_is_ok(status)) {
    EXPECT_GT(count, 0u);  // Some tokens produced.
  } else {
    // Error is acceptable for invalid UTF-8.
    iree_status_free(status);
  }
}

TEST_F(InvalidUtf8Test, TruncatedSequence) {
  // Truncated multi-byte sequence: E5 alone (should start 3-byte sequence).
  int32_t ids[32];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  const char truncated[] = "hello\xE5world";  // E5 is incomplete.
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(truncated, sizeof(truncated) - 1),
      options, ids, IREE_ARRAYSIZE(ids), &count);

  // Should handle without crash.
  if (iree_status_is_ok(status)) {
    EXPECT_GT(count, 0u);
  } else {
    iree_status_free(status);
  }
}

TEST_F(InvalidUtf8Test, InvalidContinuationByte) {
  // Invalid continuation: E5 followed by non-continuation byte.
  int32_t ids[32];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  const char invalid_cont[] =
      "hello\xE5\x40world";  // 0x40 is not continuation.
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(invalid_cont, sizeof(invalid_cont) - 1),
      options, ids, IREE_ARRAYSIZE(ids), &count);

  // Should handle without crash.
  if (iree_status_is_ok(status)) {
    EXPECT_GT(count, 0u);
  } else {
    iree_status_free(status);
  }
}

TEST_F(InvalidUtf8Test, SurrogatePairs) {
  // UTF-16 surrogate pair encoded in UTF-8 (invalid).
  // High surrogate U+D800: ED A0 80.
  int32_t ids[32];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  const char surrogate[] = "hello\xED\xA0\x80world";
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(surrogate, sizeof(surrogate) - 1),
      options, ids, IREE_ARRAYSIZE(ids), &count);

  // Should handle without crash.
  if (iree_status_is_ok(status)) {
    EXPECT_GT(count, 0u);
  } else {
    iree_status_free(status);
  }
}

TEST_F(InvalidUtf8Test, OutOfRangeCodepoint) {
  // Code point > U+10FFFF: F4 90 80 80 encodes U+110000.
  int32_t ids[32];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  const char out_of_range[] = "hello\xF4\x90\x80\x80world";
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(out_of_range, sizeof(out_of_range) - 1),
      options, ids, IREE_ARRAYSIZE(ids), &count);

  // Should handle without crash.
  if (iree_status_is_ok(status)) {
    EXPECT_GT(count, 0u);
  } else {
    iree_status_free(status);
  }
}

TEST_F(InvalidUtf8Test, MixedValidInvalid) {
  // Mix of valid and invalid UTF-8 sequences.
  int32_t ids[32];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  // Valid: "hello ", Invalid: FF, Valid: "world", Invalid: FE.
  const char mixed[] = "hello \xFFworld\xFE";
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(mixed, sizeof(mixed) - 1), options, ids,
      IREE_ARRAYSIZE(ids), &count);

  // Should handle without crash.
  if (iree_status_is_ok(status)) {
    EXPECT_GT(count, 0u);
  } else {
    iree_status_free(status);
  }
}

//===----------------------------------------------------------------------===//
// Exact Unicode Range Tests
//===----------------------------------------------------------------------===//

// Tests that exact Unicode codepoint ranges in Split pre-tokenizer patterns
// work correctly through the full tokenization pipeline. This would have
// caught the bug where ranges/num_ranges weren't propagated through split
// config (fixed in transform.h, split.c, split.h).

class ExactUnicodeRangeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a BPE vocab with CJK characters: 世 (U+4E16) and 界 (U+754C).
    // These are within the CJK Unified Ideographs range U+4E00-U+9FA5.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        20, iree_allocator_system(), &builder));

    // Add tokens.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));  // ID 0
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("test"), 0, 0));  // ID 1
    // CJK tokens (UTF-8 encoded).
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SVL("世"), 0,
                                               0));  // ID 2 (U+4E16 = E4 B8 96)
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SVL("界"), 0,
                                               0));  // ID 3 (U+754C = E7 95 8C)
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SVL("世界"), 0, 0));  // ID 4 (merged)

    // Add merge rule for 世+界 -> 世界.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder, 2, 3));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    // Set up a Split pre-tokenizer that uses EXACT Unicode codepoint ranges.
    // Pattern: [一-龥]+ matches CJK Unified Ideographs U+4E00-U+9FA5.
    // This exercises the range transition propagation through split config.
    //
    // ISOLATED: CJK runs become separate segments.
    // invert = false: non-CJK text stays, CJK is isolated.
    IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_split(
        IREE_SVL("[一-龥]+"), IREE_TOKENIZER_REGEX_SPLIT_ISOLATED,
        /*invert=*/false, iree_allocator_system(), &tokenizer_->transform));
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(ExactUnicodeRangeTest, CjkRangePatternMatches) {
  // Test that exact CJK range [一-龥] correctly segments the text.
  // "hello世界test" -> segments: ["hello", "世界", "test"]
  // The CJK characters 世 (U+4E16) and 界 (U+754C) are within U+4E00-U+9FA5.
  int32_t ids[32];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  // This would have crashed before the fix because ranges weren't propagated
  // through split config to the DFA executor.
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SVL("hello世界test"),
                                       options, ids, IREE_ARRAYSIZE(ids),
                                       &count));

  // Expected: [hello, 世界 (merged), test] = IDs [0, 4, 1].
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 0);  // hello
  EXPECT_EQ(ids[1], 4);  // 世界 (merged from 世+界)
  EXPECT_EQ(ids[2], 1);  // test
}

TEST_F(ExactUnicodeRangeTest, CjkOnlyInput) {
  // Test input that is entirely CJK characters.
  int32_t ids[32];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SVL("世界"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // Expected: [世界] = ID [4] (merged token).
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 4);  // 世界
}

TEST_F(ExactUnicodeRangeTest, NoCjkInput) {
  // Test input without CJK characters (pattern doesn't match).
  int32_t ids[32];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SVL("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // Expected: [hello] = ID [0].
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 0);  // hello
}

//===----------------------------------------------------------------------===//
// Streaming Chunk Boundary Tests
//
// Tests for the low-level streaming API (stream_initialize/feed/finalize)
// that verify correct handling of:
// - UTF-8 multi-byte sequences split across chunks
// - Literal tokens spanning chunk boundaries
// - Word boundaries for transform carryover
//===----------------------------------------------------------------------===//

// Helper to collect tokens from streaming encode.
struct StreamingChunkContext {
  std::vector<int32_t> all_ids;
};

static iree_status_t StreamingChunkCallback(void* user_data,
                                            iree_tokenizer_id_list_t ids) {
  auto* ctx = static_cast<StreamingChunkContext*>(user_data);
  for (iree_host_size_t i = 0; i < ids.count; ++i) {
    ctx->all_ids.push_back(ids.values[i]);
  }
  return iree_ok_status();
}

// Helper to encode with streaming API using specified chunk size.
static std::vector<int32_t> EncodeWithChunks(iree_tokenizer_t* tokenizer,
                                             const std::string& text,
                                             size_t chunk_size) {
  StreamingChunkContext ctx;
  iree_tokenizer_encode_stream_state_t state;
  iree_tokenizer_encode_stream_initialize(&state, tokenizer,
                                          IREE_TOKENIZER_ENCODE_FLAG_DEFAULT);

  // Feed in chunks.
  size_t offset = 0;
  while (offset < text.size()) {
    size_t remaining = text.size() - offset;
    size_t this_chunk = std::min(remaining, chunk_size);
    iree_string_view_t chunk =
        iree_make_string_view(text.data() + offset, this_chunk);
    iree_status_t status = iree_tokenizer_encode_stream_feed(
        &state, chunk, StreamingChunkCallback, &ctx);
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      return {};  // Return empty on error.
    }
    offset += this_chunk;
  }

  // Finalize.
  iree_status_t status = iree_tokenizer_encode_stream_finalize(
      &state, StreamingChunkCallback, &ctx);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return {};
  }

  return ctx.all_ids;
}

// Helper to encode with one-shot API.
static std::vector<int32_t> EncodeOneShot(iree_tokenizer_t* tokenizer,
                                          const std::string& text) {
  int32_t ids[1024];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};

  iree_status_t status = iree_tokenizer_encode(
      tokenizer, iree_make_string_view(text.data(), text.size()), options, ids,
      IREE_ARRAYSIZE(ids), &count);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return {};
  }

  return std::vector<int32_t>(ids, ids + count);
}

//===----------------------------------------------------------------------===//
// UTF-8 Boundary Tests
//===----------------------------------------------------------------------===//

class StreamingUtf8BoundaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a tokenizer with single-character tokens plus multi-byte chars.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        100, iree_allocator_system(), &builder));

    // ASCII characters a-z.
    for (char c = 'a'; c <= 'z'; ++c) {
      char str[2] = {c, 0};
      IREE_ASSERT_OK(
          iree_tokenizer_vocab_builder_add_token(builder, IREE_SV(str), 0, 0));
    }

    // Multi-byte UTF-8 tokens.
    // ▁ (U+2581) = E2 96 81 (3 bytes) - metaspace character.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xE2\x96\x81"), 0, 0));
    // é (U+00E9) = C3 A9 (2 bytes).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xC3\xA9"), 0, 0));
    // 😀 (U+1F600) = F0 9F 98 80 (4 bytes).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xF0\x9F\x98\x80"), 0, 0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));
    iree_tokenizer_text_transform_initialize_none(&tokenizer_->transform);
    iree_tokenizer_decoder_initialize_none(&tokenizer_->decoder);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(StreamingUtf8BoundaryTest, TwoByteSequenceSplit) {
  // é (C3 A9) split at byte boundary.
  // "café" = 4 ASCII bytes + 2-byte é.
  std::string input = "caf\xC3\xA9";  // "café"

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Split at various positions, including mid-UTF-8.
  // Split after 'f' (position 3), leaving \xC3 in first chunk.
  auto streaming_split_3 = EncodeWithChunks(tokenizer_, input, 3);
  EXPECT_EQ(streaming_split_3, oneshot)
      << "Split at position 3 (mid-UTF-8) should match one-shot";

  // Split after first byte of é (position 4).
  auto streaming_split_4 = EncodeWithChunks(tokenizer_, input, 4);
  EXPECT_EQ(streaming_split_4, oneshot)
      << "Split at position 4 (mid-2-byte) should match one-shot";
}

TEST_F(StreamingUtf8BoundaryTest, ThreeByteSequenceSplit) {
  // ▁ (E2 96 81) split at each byte boundary.
  // "a▁b" = 1 byte + 3-byte + 1 byte.
  std::string input =
      "a\xE2\x96\x81"
      "b";

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Split leaving 1 byte of ▁ in first chunk (after \xE2).
  auto streaming_split_2 = EncodeWithChunks(tokenizer_, input, 2);
  EXPECT_EQ(streaming_split_2, oneshot)
      << "Split after E2 should match one-shot";

  // Split leaving 2 bytes of ▁ in first chunk (after \xE2\x96).
  auto streaming_split_3 = EncodeWithChunks(tokenizer_, input, 3);
  EXPECT_EQ(streaming_split_3, oneshot)
      << "Split after E2 96 should match one-shot";
}

TEST_F(StreamingUtf8BoundaryTest, FourByteSequenceSplit) {
  // 😀 (F0 9F 98 80) split at each byte boundary.
  std::string input =
      "a\xF0\x9F\x98\x80"
      "b";

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Split at each position within the 4-byte sequence.
  for (size_t split_pos = 2; split_pos <= 5; ++split_pos) {
    auto streaming = EncodeWithChunks(tokenizer_, input, split_pos);
    EXPECT_EQ(streaming, oneshot)
        << "Split at position " << split_pos << " should match one-shot";
  }
}

TEST_F(StreamingUtf8BoundaryTest, MultipleSplitsByteByte) {
  // Extreme case: feed one byte at a time.
  std::string input =
      "a\xE2\x96\x81"
      "b\xC3\xA9"
      "c";

  auto oneshot = EncodeOneShot(tokenizer_, input);
  auto streaming = EncodeWithChunks(tokenizer_, input, 1);

  EXPECT_EQ(streaming, oneshot)
      << "Byte-by-byte streaming should match one-shot";
}

//===----------------------------------------------------------------------===//
// Literal Lookahead Boundary Tests
//===----------------------------------------------------------------------===//

class StreamingLiteralBoundaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create tokenizer with special literal tokens.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        100, iree_allocator_system(), &builder));

    // ASCII characters a-z and common symbols.
    for (char c = 'a'; c <= 'z'; ++c) {
      char str[2] = {c, 0};
      IREE_ASSERT_OK(
          iree_tokenizer_vocab_builder_add_token(builder, IREE_SV(str), 0, 0));
    }
    // Add < and > and | for partial literal matching.
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("<"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV(">"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("|"), 0, 0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));
    iree_tokenizer_text_transform_initialize_none(&tokenizer_->transform);
    iree_tokenizer_decoder_initialize_none(&tokenizer_->decoder);

    // Add literal tokens that should be matched as whole units.
    // Use LSTRIP flag to trigger interception without requiring word
    // boundaries. <mask> - common BERT token.
    IREE_ASSERT_OK(iree_tokenizer_literals_add(
        &tokenizer_->literals, 100, IREE_SV("<mask>"),
        IREE_TOKENIZER_LITERAL_FLAG_SPECIAL |
            IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
        (iree_tokenizer_special_token_t)-1));
    // <|begin|> - longer literal.
    IREE_ASSERT_OK(iree_tokenizer_literals_add(
        &tokenizer_->literals, 101, IREE_SV("<|begin|>"),
        IREE_TOKENIZER_LITERAL_FLAG_SPECIAL |
            IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
        (iree_tokenizer_special_token_t)-1));
    // Finalize the literals collection for matching.
    IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&tokenizer_->literals));
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(StreamingLiteralBoundaryTest, LiteralNotSplit) {
  // When literal is entirely in one chunk, it should match.
  std::string input = "a<mask>b";

  auto oneshot = EncodeOneShot(tokenizer_, input);
  auto streaming = EncodeWithChunks(tokenizer_, input, 100);

  EXPECT_EQ(streaming, oneshot);
  // Should contain the literal token 100.
  EXPECT_NE(std::find(streaming.begin(), streaming.end(), 100), streaming.end())
      << "Should contain <mask> literal token";
}

TEST_F(StreamingLiteralBoundaryTest, LiteralSplitAtVariousPositions) {
  // "<mask>" split at various positions.
  std::string input = "a<mask>b";

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Split after "<" (position 2).
  auto streaming_2 = EncodeWithChunks(tokenizer_, input, 2);
  EXPECT_EQ(streaming_2, oneshot) << "Split after '<' should match one-shot";

  // Split after "<m" (position 3).
  auto streaming_3 = EncodeWithChunks(tokenizer_, input, 3);
  EXPECT_EQ(streaming_3, oneshot) << "Split after '<m' should match one-shot";

  // Split after "<mas" (position 5).
  auto streaming_5 = EncodeWithChunks(tokenizer_, input, 5);
  EXPECT_EQ(streaming_5, oneshot) << "Split after '<mas' should match one-shot";
}

TEST_F(StreamingLiteralBoundaryTest, LongerLiteralSplit) {
  // "<|begin|>" (9 bytes) split at various positions.
  std::string input = "x<|begin|>y";

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Split in the middle of the literal.
  auto streaming_5 = EncodeWithChunks(tokenizer_, input, 5);
  EXPECT_EQ(streaming_5, oneshot) << "Split mid-literal should match one-shot";

  // Byte-by-byte feeding.
  auto streaming_1 = EncodeWithChunks(tokenizer_, input, 1);
  EXPECT_EQ(streaming_1, oneshot) << "Byte-by-byte should match one-shot";
}

TEST_F(StreamingLiteralBoundaryTest, PartialLiteralAtEnd) {
  // Partial literal at end of stream (no match).
  std::string input = "abc<mas";  // Partial "<mask>".

  auto oneshot = EncodeOneShot(tokenizer_, input);
  auto streaming = EncodeWithChunks(tokenizer_, input, 3);

  EXPECT_EQ(streaming, oneshot)
      << "Partial literal at end should match one-shot";
  // Should NOT contain the literal token 100.
  EXPECT_EQ(std::find(streaming.begin(), streaming.end(), 100), streaming.end())
      << "Should NOT contain <mask> literal token (incomplete)";
}

//===----------------------------------------------------------------------===//
// Word Boundary / Transform Carryover Tests
//===----------------------------------------------------------------------===//

class StreamingWordBoundaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create tokenizer with whole-word tokens to test word boundaries.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        100, iree_allocator_system(), &builder));

    // Whole words.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("world"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("test"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("the"), 0, 0));

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    // Use whitespace transform (splits on whitespace).
    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);
    iree_tokenizer_decoder_initialize_none(&tokenizer_->decoder);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(StreamingWordBoundaryTest, WordNotSplitAcrossChunks) {
  // "hello world" split between words should work correctly.
  std::string input = "hello world";

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Split right at space (position 5 = after "hello").
  auto streaming_5 = EncodeWithChunks(tokenizer_, input, 5);
  EXPECT_EQ(streaming_5, oneshot)
      << "Split at word boundary should match one-shot";

  // Split right after space (position 6).
  auto streaming_6 = EncodeWithChunks(tokenizer_, input, 6);
  EXPECT_EQ(streaming_6, oneshot) << "Split after space should match one-shot";
}

TEST_F(StreamingWordBoundaryTest, WordSplitMidWord) {
  // "hello world" with split in middle of "hello".
  // The transform carryover should handle this correctly.
  std::string input = "hello world";

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Split after "hel" (position 3).
  auto streaming_3 = EncodeWithChunks(tokenizer_, input, 3);
  EXPECT_EQ(streaming_3, oneshot)
      << "Split mid-word 'hel|lo' should match one-shot";

  // Split after "hell" (position 4).
  auto streaming_4 = EncodeWithChunks(tokenizer_, input, 4);
  EXPECT_EQ(streaming_4, oneshot)
      << "Split mid-word 'hell|o' should match one-shot";
}

TEST_F(StreamingWordBoundaryTest, MultipleWordsSplitVariously) {
  std::string input = "hello world test";

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Various chunk sizes.
  for (size_t chunk_size = 1; chunk_size <= input.size(); ++chunk_size) {
    auto streaming = EncodeWithChunks(tokenizer_, input, chunk_size);
    EXPECT_EQ(streaming, oneshot)
        << "Chunk size " << chunk_size << " should match one-shot";
  }
}

//===----------------------------------------------------------------------===//
// Streaming Equivalence Tests
//
// Comprehensive tests that verify streaming == one-shot for various inputs
// and chunk sizes.
//===----------------------------------------------------------------------===//

class StreamingEquivalenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a realistic tokenizer similar to BERT.
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        200, iree_allocator_system(), &builder));

    // Common words.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));  // 0
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("world"), 0, 0));  // 1
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("the"), 0, 0));  // 2
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("is"), 0, 0));  // 3
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("a"),
                                                          0, 0));  // 4
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("test"), 0, 0));  // 5

    // Subwords.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("##ing"), 0, 0));  // 6
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("##ed"), 0, 0));  // 7

    // Single characters for UNK fallback.
    for (char c = 'a'; c <= 'z'; ++c) {
      char str[2] = {c, 0};
      IREE_ASSERT_OK(
          iree_tokenizer_vocab_builder_add_token(builder, IREE_SV(str), 0, 0));
    }

    // Multi-byte UTF-8.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xC3\xA9"), 0, 0));  // é

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);
    iree_tokenizer_decoder_initialize_none(&tokenizer_->decoder);

    // Add a literal token. Use LSTRIP to trigger interception.
    IREE_ASSERT_OK(iree_tokenizer_literals_add(
        &tokenizer_->literals, 200, IREE_SV("[SEP]"),
        IREE_TOKENIZER_LITERAL_FLAG_SPECIAL |
            IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
        (iree_tokenizer_special_token_t)-1));
    // Finalize the literals collection for matching.
    IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&tokenizer_->literals));
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(StreamingEquivalenceTest, SimpleText) {
  std::string input = "hello world";

  auto oneshot = EncodeOneShot(tokenizer_, input);

  for (size_t chunk_size = 1; chunk_size <= input.size() + 1; ++chunk_size) {
    auto streaming = EncodeWithChunks(tokenizer_, input, chunk_size);
    EXPECT_EQ(streaming, oneshot)
        << "Chunk size " << chunk_size << " should match one-shot";
  }
}

TEST_F(StreamingEquivalenceTest, WithLiteral) {
  std::string input = "hello [SEP] world";

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Test various chunk sizes, especially those that split the literal.
  std::vector<size_t> chunk_sizes = {1, 2, 3, 5, 7, 8, 9, 10, 11, 50};
  for (size_t chunk_size : chunk_sizes) {
    auto streaming = EncodeWithChunks(tokenizer_, input, chunk_size);
    EXPECT_EQ(streaming, oneshot)
        << "Chunk size " << chunk_size << " should match one-shot";
  }
}

TEST_F(StreamingEquivalenceTest, WithUtf8) {
  std::string input = "caf\xC3\xA9 test";  // "café test"

  auto oneshot = EncodeOneShot(tokenizer_, input);

  for (size_t chunk_size = 1; chunk_size <= input.size() + 1; ++chunk_size) {
    auto streaming = EncodeWithChunks(tokenizer_, input, chunk_size);
    EXPECT_EQ(streaming, oneshot)
        << "Chunk size " << chunk_size << " should match one-shot";
  }
}

TEST_F(StreamingEquivalenceTest, EmptyInput) {
  std::string input = "";

  auto oneshot = EncodeOneShot(tokenizer_, input);
  auto streaming = EncodeWithChunks(tokenizer_, input, 10);

  EXPECT_EQ(streaming, oneshot);
  EXPECT_TRUE(streaming.empty());
}

TEST_F(StreamingEquivalenceTest, LargeInput) {
  // Generate a large input to test buffer handling.
  std::string input;
  for (int i = 0; i < 100; ++i) {
    input += "hello world test ";
  }

  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Test with various chunk sizes.
  std::vector<size_t> chunk_sizes = {7, 64, 256, 1024, 4096};
  for (size_t chunk_size : chunk_sizes) {
    auto streaming = EncodeWithChunks(tokenizer_, input, chunk_size);
    EXPECT_EQ(streaming, oneshot) << "Chunk size " << chunk_size
                                  << " should match one-shot for large input";
  }
}

//===----------------------------------------------------------------------===//
// Streaming Stress Tests
//
// Comprehensive edge case testing for streaming encode.
//===----------------------------------------------------------------------===//

class StreamingStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a comprehensive tokenizer with:
    // - Whole words
    // - Single characters (for fallback)
    // - Multi-byte UTF-8 tokens
    // - Multiple literals with shared prefixes
    iree_tokenizer_vocab_builder_t* builder = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        300, iree_allocator_system(), &builder));

    // Whole words (tokens 0-9).
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("hello"), 0, 0));  // 0
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("world"), 0, 0));  // 1
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("test"), 0, 0));  // 2
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("café"), 0, 0));  // 3 - word with UTF-8
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("日本語"), 0, 0));  // 4 - three 3-byte chars

    // ASCII characters a-z (tokens 5-30).
    for (char c = 'a'; c <= 'z'; ++c) {
      char str[2] = {c, 0};
      IREE_ASSERT_OK(
          iree_tokenizer_vocab_builder_add_token(builder, IREE_SV(str), 0, 0));
    }

    // Common symbols (tokens 31-40).
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("<"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV(">"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("|"), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("["), 0, 0));
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_token(builder, IREE_SV("]"), 0, 0));

    // UTF-8 characters.
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xC3\xA9"), 0, 0));  // é (2-byte)
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xE2\x96\x81"), 0, 0));  // ▁ (3-byte)
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV("\xF0\x9F\x98\x80"), 0, 0));  // 😀 (4-byte)

    iree_tokenizer_vocab_t* vocab = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
    IREE_ASSERT_OK(iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(),
                                               &tokenizer_));

    iree_tokenizer_text_transform_initialize_whitespace(&tokenizer_->transform);
    iree_tokenizer_decoder_initialize_none(&tokenizer_->decoder);

    // Add literals with shared prefixes to stress prefix matching.
    // <mask> and <masked> share "<mask" prefix.
    IREE_ASSERT_OK(iree_tokenizer_literals_add(
        &tokenizer_->literals, 200, IREE_SV("<mask>"),
        IREE_TOKENIZER_LITERAL_FLAG_SPECIAL |
            IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
        (iree_tokenizer_special_token_t)-1));
    IREE_ASSERT_OK(iree_tokenizer_literals_add(
        &tokenizer_->literals, 201, IREE_SV("<masked>"),
        IREE_TOKENIZER_LITERAL_FLAG_SPECIAL |
            IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
        (iree_tokenizer_special_token_t)-1));
    // [CLS] and [SEP] - common BERT tokens.
    IREE_ASSERT_OK(iree_tokenizer_literals_add(
        &tokenizer_->literals, 202, IREE_SV("[CLS]"),
        IREE_TOKENIZER_LITERAL_FLAG_SPECIAL |
            IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
        (iree_tokenizer_special_token_t)-1));
    IREE_ASSERT_OK(iree_tokenizer_literals_add(
        &tokenizer_->literals, 203, IREE_SV("[SEP]"),
        IREE_TOKENIZER_LITERAL_FLAG_SPECIAL |
            IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
        (iree_tokenizer_special_token_t)-1));
    // Longer literal for buffer boundary testing.
    IREE_ASSERT_OK(iree_tokenizer_literals_add(
        &tokenizer_->literals, 204, IREE_SV("<|begin_of_text|>"),
        IREE_TOKENIZER_LITERAL_FLAG_SPECIAL |
            IREE_TOKENIZER_LITERAL_FLAG_LSTRIP,
        (iree_tokenizer_special_token_t)-1));

    IREE_ASSERT_OK(iree_tokenizer_literals_finalize(&tokenizer_->literals));
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  // Verify streaming == oneshot for all chunk sizes from 1 to input.size()+1.
  void VerifyAllChunkSizes(const std::string& input) {
    auto oneshot = EncodeOneShot(tokenizer_, input);
    for (size_t chunk_size = 1; chunk_size <= input.size() + 1; ++chunk_size) {
      auto streaming = EncodeWithChunks(tokenizer_, input, chunk_size);
      EXPECT_EQ(streaming, oneshot)
          << "Chunk size " << chunk_size << " failed for input: "
          << (input.size() < 50 ? input : input.substr(0, 50) + "...");
    }
  }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

// UTF-8 Stress Tests

TEST_F(StreamingStressTest, ConsecutiveMultiByteChars) {
  // Three 3-byte characters in a row.
  std::string input = "\xE2\x96\x81\xE2\x96\x81\xE2\x96\x81";  // ▁▁▁
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, MixedUtf8Lengths) {
  // Mix of 1, 2, 3, 4-byte UTF-8 characters.
  std::string input =
      "a\xC3\xA9\xE2\x96\x81\xF0\x9F\x98\x80"
      "b";
  // a + é + ▁ + 😀 + b
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, Utf8AtWordBoundaries) {
  // UTF-8 characters at word boundaries.
  std::string input = "caf\xC3\xA9 \xE2\x96\x81test";  // "café ▁test"
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, OnlyUtf8Characters) {
  // Input with only multi-byte UTF-8 characters.
  std::string input = "\xC3\xA9\xC3\xA9\xC3\xA9";  // ééé
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, JapaneseText) {
  // Three 3-byte Japanese characters.
  std::string input = "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E";  // 日本語
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, EmojiSequence) {
  // Multiple 4-byte emoji.
  std::string input = "\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82";
  VerifyAllChunkSizes(input);
}

// Literal Stress Tests

TEST_F(StreamingStressTest, LiteralAtStart) {
  std::string input = "<mask>hello world";
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, LiteralAtEnd) {
  std::string input = "hello world<mask>";
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, ConsecutiveLiterals) {
  std::string input = "<mask>[CLS][SEP]<masked>";
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, LiteralsWithSharedPrefix) {
  // <mask> and <masked> share prefix - test correct matching.
  std::string input = "a<mask>b<masked>c";
  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Verify correct literals matched.
  EXPECT_NE(std::find(oneshot.begin(), oneshot.end(), 200), oneshot.end())
      << "Should contain <mask> token";
  EXPECT_NE(std::find(oneshot.begin(), oneshot.end(), 201), oneshot.end())
      << "Should contain <masked> token";

  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, FakeLiteralPrefix) {
  // Input that looks like a literal but isn't (e.g., "<masx>" when "<mask>"
  // exists).
  std::string input = "a<masx>b";
  auto oneshot = EncodeOneShot(tokenizer_, input);

  // Should NOT contain the <mask> literal token.
  EXPECT_EQ(std::find(oneshot.begin(), oneshot.end(), 200), oneshot.end())
      << "Should NOT contain <mask> token";

  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, LongLiteralSplit) {
  // 17-byte literal split at every position.
  std::string input = "a<|begin_of_text|>b";
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, MultipleLongLiterals) {
  std::string input = "<|begin_of_text|>hello<|begin_of_text|>";
  VerifyAllChunkSizes(input);
}

// Word Boundary Stress Tests

TEST_F(StreamingStressTest, OnlyWhitespace) {
  std::string input = "     ";  // 5 spaces
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, MultipleConsecutiveSpaces) {
  std::string input = "hello     world";  // 5 spaces between words
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, MixedWhitespace) {
  std::string input = "hello\t\n\rworld";  // tab, newline, carriage return
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, NoWhitespace) {
  // One long "word" with no whitespace.
  std::string input = "abcdefghijklmnopqrstuvwxyz";
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, VeryLongWord) {
  // Word longer than typical buffer sizes.
  std::string input(500, 'a');  // 500 'a' characters
  input = "start " + input + " end";
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, WhitespaceAtChunkBoundaries) {
  // Designed to have whitespace at common chunk sizes.
  std::string input = "ab cd ef gh ij kl mn";
  VerifyAllChunkSizes(input);
}

// Combined Edge Case Tests

TEST_F(StreamingStressTest, Utf8BeforeLiteral) {
  std::string input = "caf\xC3\xA9<mask>test";  // café<mask>test
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, Utf8AfterLiteral) {
  std::string input = "<mask>caf\xC3\xA9";  // <mask>café
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, Utf8BetweenLiterals) {
  std::string input = "<mask>\xC3\xA9[SEP]";  // <mask>é[SEP]
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, ComplexMixedInput) {
  // Complex input with UTF-8, literals, and word boundaries.
  std::string input = "hello caf\xC3\xA9 <mask> \xE2\x96\x81test [SEP] world";
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, Utf8SplitAtWordBoundary) {
  // UTF-8 character where chunk boundary could fall on word boundary.
  std::string input = "test \xE2\x96\x81 hello";
  VerifyAllChunkSizes(input);
}

// Extreme Chunk Size Tests

TEST_F(StreamingStressTest, ChunkSizeLargerThanInput) {
  std::string input = "hello";
  auto oneshot = EncodeOneShot(tokenizer_, input);
  auto streaming = EncodeWithChunks(tokenizer_, input, 1000);
  EXPECT_EQ(streaming, oneshot);
}

TEST_F(StreamingStressTest, SingleByteChunks) {
  // Every single byte as its own chunk - maximum stress.
  std::string input = "hello <mask> caf\xC3\xA9 [SEP] \xE2\x96\x81test";
  auto oneshot = EncodeOneShot(tokenizer_, input);
  auto streaming = EncodeWithChunks(tokenizer_, input, 1);
  EXPECT_EQ(streaming, oneshot);
}

// Buffer Boundary Tests

TEST_F(StreamingStressTest, NearBufferCapacity) {
  // Create input near the 4KB buffer capacity.
  std::string input(4000, 'a');
  input[2000] = ' ';  // Add a space in the middle.
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, ExactlyBufferCapacity) {
  // Create input exactly at buffer boundary.
  std::string input(4096, 'a');
  input[2048] = ' ';
  auto oneshot = EncodeOneShot(tokenizer_, input);
  auto streaming = EncodeWithChunks(tokenizer_, input, 1024);
  EXPECT_EQ(streaming, oneshot);
}

// Degenerate Input Tests

TEST_F(StreamingStressTest, EmptyChunksBetweenNonEmpty) {
  // Simulate empty chunks by feeding input with size 0.
  std::string input = "hello world";
  auto oneshot = EncodeOneShot(tokenizer_, input);

  StreamingChunkContext ctx;
  iree_tokenizer_encode_stream_state_t state;
  iree_tokenizer_encode_stream_initialize(&state, tokenizer_,
                                          IREE_TOKENIZER_ENCODE_FLAG_DEFAULT);

  // Feed with empty chunks interspersed.
  IREE_ASSERT_OK(iree_tokenizer_encode_stream_feed(
      &state, iree_make_string_view("", 0), StreamingChunkCallback, &ctx));
  IREE_ASSERT_OK(iree_tokenizer_encode_stream_feed(
      &state, iree_make_string_view("hello", 5), StreamingChunkCallback, &ctx));
  IREE_ASSERT_OK(iree_tokenizer_encode_stream_feed(
      &state, iree_make_string_view("", 0), StreamingChunkCallback, &ctx));
  IREE_ASSERT_OK(iree_tokenizer_encode_stream_feed(
      &state, iree_make_string_view(" ", 1), StreamingChunkCallback, &ctx));
  IREE_ASSERT_OK(iree_tokenizer_encode_stream_feed(
      &state, iree_make_string_view("", 0), StreamingChunkCallback, &ctx));
  IREE_ASSERT_OK(iree_tokenizer_encode_stream_feed(
      &state, iree_make_string_view("world", 5), StreamingChunkCallback, &ctx));
  IREE_ASSERT_OK(iree_tokenizer_encode_stream_feed(
      &state, iree_make_string_view("", 0), StreamingChunkCallback, &ctx));
  IREE_ASSERT_OK(iree_tokenizer_encode_stream_finalize(
      &state, StreamingChunkCallback, &ctx));

  EXPECT_EQ(ctx.all_ids, oneshot);
}

TEST_F(StreamingStressTest, SingleCharacterInput) { VerifyAllChunkSizes("a"); }

TEST_F(StreamingStressTest, SingleSpaceInput) { VerifyAllChunkSizes(" "); }

TEST_F(StreamingStressTest, SingleUtf8Character) {
  VerifyAllChunkSizes("\xC3\xA9");          // é
  VerifyAllChunkSizes("\xE2\x96\x81");      // ▁
  VerifyAllChunkSizes("\xF0\x9F\x98\x80");  // 😀
}

// Regression Tests (for previously found bugs)

TEST_F(StreamingStressTest, Utf8CompletionBeforeCarryover) {
  // Regression: UTF-8 was being encoded before transform carryover
  // was applied, causing incorrect token order.
  std::string input = "caf\xC3\xA9 test";  // "café test"
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, LiteralLookaheadBypassingWordBoundary) {
  // Regression: Literal lookahead path was encoding directly,
  // bypassing word boundary handling.
  std::string input = "hello [SEP] world";
  VerifyAllChunkSizes(input);
}

TEST_F(StreamingStressTest, TransformCarryoverOverwrite) {
  // Regression: When no whitespace in chunk, carryover was being
  // overwritten instead of appended.
  std::string input = "abcdefghij klmnopqrst";  // Long words
  VerifyAllChunkSizes(input);
}

// BPE overlap edge cases - verify streaming produces identical tokens to
// oneshot even when chunks split at potentially problematic boundaries.

TEST_F(StreamingStressTest, ContractionBoundary) {
  // Contractions like 's could split between ' and s at chunk boundaries.
  // The transform carryover should prevent incorrect tokenization.
  VerifyAllChunkSizes("he's going to the store");
  VerifyAllChunkSizes("don't won't can't shouldn't");
  VerifyAllChunkSizes("it's 2026 and we're testing");
}

TEST_F(StreamingStressTest, Utf8WordBoundaries) {
  // UTF-8 characters at word boundaries - both UTF-8 partial handling
  // and word boundary handling must work together.
  VerifyAllChunkSizes("caf\xC3\xA9 r\xC3\xA9sum\xC3\xA9 na\xC3\xAFve");
  VerifyAllChunkSizes("\xC3\xA9l\xC3\xA8ve ma\xC3\xAEtre");
  VerifyAllChunkSizes("Tokyo \xE6\x9D\xB1\xE4\xBA\xAC Kyoto");
}

TEST_F(StreamingStressTest, VeryLongWordBpeOverlap) {
  // Very long words that might require many BPE merges.
  // Streaming should produce same tokens regardless of chunk boundaries.
  VerifyAllChunkSizes("antidisestablishmentarianism is long");
  VerifyAllChunkSizes("pneumonoultramicroscopicsilicovolcanoconiosis test");
  VerifyAllChunkSizes("supercalifragilisticexpialidocious word");
}

TEST_F(StreamingStressTest, NumbersAndPunctuation) {
  // Numbers and punctuation - regex patterns might split these specially.
  VerifyAllChunkSizes("The price is $123.45 (15% off)");
  VerifyAllChunkSizes("Date: 2026-01-11T10:30:00Z");
  VerifyAllChunkSizes("Email: user@example.com");
  VerifyAllChunkSizes("Version 1.2.3-beta.4+build.567");
}

TEST_F(StreamingStressTest, CodeSnippets) {
  // Code-like text with symbols - tests BPE handling of operators/delimiters.
  VerifyAllChunkSizes("x = foo() + bar()");
  VerifyAllChunkSizes("if (a && b || c) { return d; }");
  VerifyAllChunkSizes("array[0] = map[key].value");
}

TEST_F(StreamingStressTest, MixedScriptsAndEmoji) {
  // Mixed scripts require both UTF-8 and word boundary handling.
  VerifyAllChunkSizes("Hello \xE4\xB8\x96\xE7\x95\x8C World");  // 世界
  VerifyAllChunkSizes("Test \xF0\x9F\x9A\x80 rocket \xF0\x9F\x8C\x8D earth");
  VerifyAllChunkSizes(
      "Mix: English \xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82 "
      "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF");  // привет
                                                                        // こんにちは
}

}  // namespace
