// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab_builder.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(VocabBuilderTest, EmptyVocab) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      0, iree_allocator_system(), &builder));
  ASSERT_NE(builder, nullptr);

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
  ASSERT_NE(vocab, nullptr);

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 0u);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("hello")), -1);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, SingleToken) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      1, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("hello"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 1u);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("hello")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("world")), -1);

  iree_string_view_t text = iree_tokenizer_vocab_token_text(vocab, 0);
  EXPECT_EQ(iree_string_view_compare(text, IREE_SVL("hello")), 0);

  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, 0), 1.0f);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, MultipleTokens) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      5, iree_allocator_system(), &builder));

  const char* tokens[] = {"the", "quick", "brown", "fox", "jumps"};
  for (int i = 0; i < 5; ++i) {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, IREE_SV(tokens[i]), (float)i, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 5u);

  // Verify lookups.
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SV(tokens[i])), i)
        << "Failed for token: " << tokens[i];
  }

  // Verify text retrieval.
  for (int i = 0; i < 5; ++i) {
    iree_string_view_t text = iree_tokenizer_vocab_token_text(vocab, i);
    EXPECT_EQ(iree_string_view_compare(text, IREE_SV(tokens[i])), 0)
        << "Text mismatch for token " << i;
  }

  // Verify scores.
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, i), (float)i);
  }

  // Non-existent tokens.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("lazy")), -1);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, SpecialTokens) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      5, iree_allocator_system(), &builder));

  // Add tokens: 0=[UNK], 1=[BOS], 2=[EOS], 3=hello, 4=world
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("[UNK]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("[BOS]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("[EOS]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("hello"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("world"), 2.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Set special token IDs.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_BOS, 1));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_EOS, 2));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab);
  EXPECT_EQ(ids.unk, 0);
  EXPECT_EQ(ids.bos, 1);
  EXPECT_EQ(ids.eos, 2);
  EXPECT_EQ(ids.pad, -1);  // Not set.

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, EmptyStringToken) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      2, iree_allocator_system(), &builder));

  // Empty string token.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL(""), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("nonempty"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("nonempty")), 1);

  iree_string_view_t text = iree_tokenizer_vocab_token_text(vocab, 0);
  EXPECT_EQ(text.size, 0u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, CapacityHintGrowth) {
  // Start with small capacity hint, add more tokens.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      2, iree_allocator_system(), &builder));

  // Add 100 tokens (well over capacity hint).
  for (int i = 0; i < 100; ++i) {
    std::string token = "token_" + std::to_string(i);
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder, iree_make_string_view(token.data(), token.size()), (float)i,
        IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 100u);

  // Verify all tokens are accessible.
  for (int i = 0; i < 100; ++i) {
    std::string token = "token_" + std::to_string(i);
    EXPECT_EQ(iree_tokenizer_vocab_lookup(
                  vocab, iree_make_string_view(token.data(), token.size())),
              i)
        << "Failed for token: " << token;
  }

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, FreeWithoutBuild) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      10, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("test"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Free without building (e.g., on error path).
  iree_tokenizer_vocab_builder_free(builder);
  // Should not crash or leak.
}

TEST(VocabBuilderTest, FreeNullBuilder) {
  // Should not crash.
  iree_tokenizer_vocab_builder_free(nullptr);
}

TEST(VocabBuilderTest, AddTokenWithIdAutoSorted) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      5, iree_allocator_system(), &builder));

  // Add tokens out of order: IDs 2, 0, 1.
  // build() will automatically sort them.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 2, IREE_SVL("gamma"), 2.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("alpha"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SVL("beta"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 3u);

  // Verify tokens are in correct order by ID.
  iree_string_view_t text0 = iree_tokenizer_vocab_token_text(vocab, 0);
  iree_string_view_t text1 = iree_tokenizer_vocab_token_text(vocab, 1);
  iree_string_view_t text2 = iree_tokenizer_vocab_token_text(vocab, 2);
  EXPECT_EQ(iree_string_view_compare(text0, IREE_SVL("alpha")), 0);
  EXPECT_EQ(iree_string_view_compare(text1, IREE_SVL("beta")), 0);
  EXPECT_EQ(iree_string_view_compare(text2, IREE_SVL("gamma")), 0);

  // Verify scores follow tokens.
  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, 0), 0.0f);
  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, 1), 1.0f);
  EXPECT_FLOAT_EQ(iree_tokenizer_vocab_token_score(vocab, 2), 2.0f);

  // Verify lookups work.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("alpha")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("beta")), 1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("gamma")), 2);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, AddTokenWithIdReverseOrder) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      0, iree_allocator_system(), &builder));

  // Add 10 tokens in reverse order (auto-sorted on build).
  for (int i = 9; i >= 0; --i) {
    std::string token = "tok" + std::to_string(i);
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        builder, i, iree_make_string_view(token.data(), token.size()), (float)i,
        IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 10u);

  for (int i = 0; i < 10; ++i) {
    std::string expected = "tok" + std::to_string(i);
    iree_string_view_t text = iree_tokenizer_vocab_token_text(vocab, i);
    EXPECT_EQ(
        iree_string_view_compare(
            text, iree_make_string_view(expected.data(), expected.size())),
        0)
        << "Mismatch at ID " << i;
  }

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, BuildDuplicateIdError) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      3, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SVL("b"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("c"), 2.0f,  // Duplicate ID 0.
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // build() auto-sorts and detects duplicate.
  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_builder_build(builder, &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
  // Builder is freed by build() on error.
}

//===----------------------------------------------------------------------===//
// Sparse/Non-Contiguous Token ID Tests
//===----------------------------------------------------------------------===//
// These tests verify support for vocabularies with gaps in the token ID space,
// as found in models like ConvBERT which have duplicate vocab entries.

TEST(VocabBuilderTest, SparseIdsSingleGapMiddle) {
  // IDs: 0, 1, 3 (missing ID 2)
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      3, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("alpha"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SVL("beta"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 3, IREE_SVL("delta"), 3.0f,  // Gap at ID 2.
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // vocab_capacity() returns max_id + 1 (array size).
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 4u);

  // Verify tokens at their assigned IDs.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("alpha")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("beta")), 1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("delta")), 3);

  // Verify text retrieval.
  iree_string_view_t text0 = iree_tokenizer_vocab_token_text(vocab, 0);
  iree_string_view_t text1 = iree_tokenizer_vocab_token_text(vocab, 1);
  iree_string_view_t text3 = iree_tokenizer_vocab_token_text(vocab, 3);
  EXPECT_EQ(iree_string_view_compare(text0, IREE_SVL("alpha")), 0);
  EXPECT_EQ(iree_string_view_compare(text1, IREE_SVL("beta")), 0);
  EXPECT_EQ(iree_string_view_compare(text3, IREE_SVL("delta")), 0);

  // Gap at ID 2 returns empty string.
  iree_string_view_t text2 = iree_tokenizer_vocab_token_text(vocab, 2);
  EXPECT_EQ(text2.size, 0u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, SparseIdsGapAtStart) {
  // IDs: 1, 2, 3 (missing ID 0)
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      3, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SVL("one"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 2, IREE_SVL("two"), 2.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 3, IREE_SVL("three"), 3.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 4u);

  // Gap at ID 0 returns empty.
  iree_string_view_t text0 = iree_tokenizer_vocab_token_text(vocab, 0);
  EXPECT_EQ(text0.size, 0u);

  // Tokens at IDs 1, 2, 3 work.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("one")), 1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("two")), 2);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("three")), 3);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, SparseIdsMultipleGaps) {
  // IDs: 0, 2, 5, 9 (gaps at 1, 3, 4, 6, 7, 8)
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      4, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 2, IREE_SVL("c"), 2.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 5, IREE_SVL("f"), 5.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 9, IREE_SVL("j"), 9.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 10u);

  // Verify existing tokens.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("a")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("c")), 2);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("f")), 5);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("j")), 9);

  // Verify gaps return empty.
  int gaps[] = {1, 3, 4, 6, 7, 8};
  for (int gap : gaps) {
    iree_string_view_t text = iree_tokenizer_vocab_token_text(vocab, gap);
    EXPECT_EQ(text.size, 0u) << "Gap at ID " << gap << " should return empty";
  }

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, SparseIdsLargeGap) {
  // IDs: 0, 1, 1000 (large gap from 2 to 999)
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      3, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("first"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SVL("second"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1000, IREE_SVL("thousandth"), 1000.0f,
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // vocab_capacity = max_id + 1 = 1001
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 1001u);

  // Verify tokens work.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("first")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("second")), 1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("thousandth")), 1000);

  // Verify some gaps.
  EXPECT_EQ(iree_tokenizer_vocab_token_text(vocab, 2).size, 0u);
  EXPECT_EQ(iree_tokenizer_vocab_token_text(vocab, 500).size, 0u);
  EXPECT_EQ(iree_tokenizer_vocab_token_text(vocab, 999).size, 0u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, SparseIdsExcessiveGapRejected) {
  // Verifies that vocabularies with excessive sparsity are rejected
  // to prevent memory exhaustion DoS attacks.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      2, iree_allocator_system(), &builder));

  // Add two tokens with IDs that would create >1M gap.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("first"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 10000000, IREE_SVL("distant"), 1.0f,
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Build should fail with RESOURCE_EXHAUSTED due to excessive sparsity.
  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_builder_build(builder, &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
  // Builder is freed by build() on error.
}

TEST(VocabBuilderTest, SparseIdsOutOfBoundsLookup) {
  // IDs: 0, 1, 3
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      3, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SVL("b"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 3, IREE_SVL("d"), 3.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // IDs beyond max_id return empty.
  EXPECT_EQ(iree_tokenizer_vocab_token_text(vocab, 4).size, 0u);
  EXPECT_EQ(iree_tokenizer_vocab_token_text(vocab, 100).size, 0u);

  // Negative IDs return empty.
  EXPECT_EQ(iree_tokenizer_vocab_token_text(vocab, -1).size, 0u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, SparseIdsConvBERTPattern) {
  // Simulates ConvBERT: N tokens but max_id = N (off-by-one due to gaps).
  // e.g., 5 tokens with IDs 0, 1, 2, 3, 5 (gap at 4).
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      5, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("tok0"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SVL("tok1"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 2, IREE_SVL("tok2"), 2.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 3, IREE_SVL("tok3"), 3.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 5, IREE_SVL("tok5"), 5.0f,  // Gap at ID 4.
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // 5 tokens but vocab_capacity = 6 (max_id + 1).
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 6u);

  // All tokens retrievable.
  for (int i = 0; i < 6; ++i) {
    if (i == 4) continue;  // Gap.
    std::string expected = "tok" + std::to_string(i);
    EXPECT_EQ(
        iree_tokenizer_vocab_lookup(
            vocab, iree_make_string_view(expected.data(), expected.size())),
        i)
        << "Failed for " << expected;
  }

  // Gap at ID 4.
  EXPECT_EQ(iree_tokenizer_vocab_token_text(vocab, 4).size, 0u);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, BuildDuplicateIdStillRejected) {
  // Duplicate IDs should still be rejected even with sparse ID support.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      3, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 5, IREE_SVL("b"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 5, IREE_SVL("c"), 2.0f,  // Duplicate ID 5.
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_builder_build(builder, &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(VocabBuilderTest, BuildNegativeIdRejected) {
  // Negative IDs should still be rejected.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      2, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, -1, IREE_SVL("b"), 1.0f,  // Negative ID.
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  iree_status_t status = iree_tokenizer_vocab_builder_build(builder, &vocab);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(vocab, nullptr);
  iree_status_free(status);
}

TEST(VocabBuilderTest, MixedAddTokenThenAddTokenWithId) {
  // Tests calling add_token() FIRST, then add_token_with_id().
  // This exercises the backfill logic in add_token_with_id().
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      5, iree_allocator_system(), &builder));

  // First add tokens with implicit IDs (0, 1).
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("implicit_zero"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("implicit_one"), 1.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Now add with explicit ID. This triggers target_ids allocation and backfill.
  // Give it ID 4 to create a gap that we'll fill.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 4, IREE_SVL("explicit_four"), 4.0f,
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Fill the gaps with explicit IDs.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 2, IREE_SVL("explicit_two"), 2.0f,
      IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 3, IREE_SVL("explicit_three"), 3.0f,
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 5u);

  // Verify implicit tokens kept their original IDs (0, 1) after sorting.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("implicit_zero")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("implicit_one")), 1);
  // Verify explicit tokens are at their assigned IDs.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("explicit_two")), 2);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("explicit_three")), 3);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SVL("explicit_four")), 4);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, AddTokenAttrsSequential) {
  // Test adding attributes to a token added with sequential (implicit) IDs.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      5, iree_allocator_system(), &builder));

  // Add tokens with no special attribute initially.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("[UNK]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("[BOS]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("hello"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Now add ATTR_SPECIAL to the first two tokens.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_attrs(
      builder, 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_attrs(
      builder, 1, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // Verify [UNK] and [BOS] have ATTR_SPECIAL.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 0) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 1) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  // Verify hello does NOT have ATTR_SPECIAL.
  EXPECT_FALSE(iree_tokenizer_vocab_token_attrs(vocab, 2) &
               IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, AddTokenAttrsWithExplicitIds) {
  // Test adding attributes to tokens added with explicit IDs.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      3, iree_allocator_system(), &builder));

  // Add tokens out of order with explicit IDs.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 2, IREE_SVL("hello"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SVL("[UNK]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SVL("[BOS]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Add ATTR_SPECIAL to IDs 0 and 1 (not 2).
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_attrs(
      builder, 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_attrs(
      builder, 1, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // Verify [UNK] (ID 0) and [BOS] (ID 1) have ATTR_SPECIAL.
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 0) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  EXPECT_TRUE(iree_tokenizer_vocab_token_attrs(vocab, 1) &
              IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  // Verify hello (ID 2) does NOT have ATTR_SPECIAL.
  EXPECT_FALSE(iree_tokenizer_vocab_token_attrs(vocab, 2) &
               IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);

  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, AddTokenAttrsNotFoundError) {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      2, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("hello"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Try to add attrs to non-existent ID.
  iree_status_t status = iree_tokenizer_vocab_builder_add_token_attrs(
      builder, 99, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  iree_status_free(status);

  // Builder should still be usable.
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 1u);
  iree_tokenizer_vocab_free(vocab);
}

TEST(VocabBuilderTest, AddTokenAttrsOrsExistingAttrs) {
  // Test that add_token_attrs ORs with existing attributes rather than
  // replacing.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      1, iree_allocator_system(), &builder));

  // Add token with one attribute already set.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder, IREE_SVL("[UNK]"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_BYTE));

  // Add ATTR_SPECIAL - should OR with existing ATTR_BYTE.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_attrs(
      builder, 0, IREE_TOKENIZER_TOKEN_ATTR_SPECIAL));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // Verify token has both attributes.
  iree_tokenizer_token_attr_t attrs =
      iree_tokenizer_vocab_token_attrs(vocab, 0);
  EXPECT_TRUE(attrs & IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
  EXPECT_TRUE(attrs & IREE_TOKENIZER_TOKEN_ATTR_BYTE);

  iree_tokenizer_vocab_free(vocab);
}

}  // namespace
