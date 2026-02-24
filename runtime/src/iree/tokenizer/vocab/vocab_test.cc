// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab/vocab.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

namespace {

// Helper to build a simple test vocab.
iree_tokenizer_vocab_t* BuildTestVocab(const std::vector<std::string>& tokens) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_vocab_builder_allocate(
      tokens.size(), iree_allocator_system(), &builder));

  for (size_t i = 0; i < tokens.size(); ++i) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token(
        builder, iree_make_string_view(tokens[i].data(), tokens[i].size()),
        (float)i, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_CHECK_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
  return vocab;
}

TEST(VocabTest, LookupHit) {
  auto vocab = BuildTestVocab({"hello", "world", "test"});
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("hello")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("world")), 1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("test")), 2);
  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabTest, LookupMiss) {
  auto vocab = BuildTestVocab({"hello", "world"});
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("missing")), -1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("HELLO")),
            -1);  // Case-sensitive.
  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabTest, TokenText) {
  auto vocab = BuildTestVocab({"alpha", "beta", "gamma"});
  iree_string_view_t text0 = iree_tokenizer_vocab_token_text(vocab, 0);
  iree_string_view_t text1 = iree_tokenizer_vocab_token_text(vocab, 1);
  iree_string_view_t text2 = iree_tokenizer_vocab_token_text(vocab, 2);

  EXPECT_EQ(iree_string_view_compare(text0, IREE_SVL("alpha")), 0);
  EXPECT_EQ(iree_string_view_compare(text1, IREE_SVL("beta")), 0);
  EXPECT_EQ(iree_string_view_compare(text2, IREE_SVL("gamma")), 0);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabTest, TokenTextOutOfRange) {
  auto vocab = BuildTestVocab({"hello"});

  // Negative ID.
  iree_string_view_t text_neg = iree_tokenizer_vocab_token_text(vocab, -1);
  EXPECT_EQ(text_neg.size, 0u);

  // ID beyond range.
  iree_string_view_t text_oob = iree_tokenizer_vocab_token_text(vocab, 999);
  EXPECT_EQ(text_oob.size, 0u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabTest, TokenScore) {
  auto vocab = BuildTestVocab({"a", "b", "c"});
  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, 0), 0.0f);
  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, 1), 1.0f);
  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, 2), 2.0f);

  // Out of range returns 0.
  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, -1), 0.0f);
  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, 999), 0.0f);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabTest, VocabCapacity) {
  auto vocab0 = BuildTestVocab({});
  auto vocab3 = BuildTestVocab({"a", "b", "c"});

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab0), 0u);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab3), 3u);

  iree_tokenizer_vocab_free(vocab0);
  iree_tokenizer_vocab_free(vocab3);
}

TEST(VocabTest, SpecialIds) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      3, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("[PAD]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("[UNK]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("hello"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_PAD, 0));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.pad, 0);
  EXPECT_EQ(ids.unk, 1);
  EXPECT_EQ(ids.bos, -1);
  EXPECT_EQ(ids.eos, -1);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabTest, FreeNullVocab) {
  // Should not crash.
  iree_tokenizer_vocab_free(nullptr);
}

TEST(VocabTest, NullVocabAccessors) {
  // All accessors should handle null gracefully.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(nullptr, IREE_SVL("x")), -1);
  EXPECT_EQ(iree_tokenizer_vocab_token_text(nullptr, 0).size, 0u);
  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(nullptr, 0), 0.0f);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(nullptr), 0u);

  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(nullptr);
  EXPECT_EQ(ids.unk, -1);
}

}  // namespace
