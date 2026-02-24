// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/added_tokens_json.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class AddedTokensJsonTest : public ::testing::Test {
 protected:
  iree_allocator_t allocator_ = iree_allocator_system();
};

TEST_F(AddedTokensJsonTest, EmptyJson) {
  // No added_tokens field.
  iree_string_view_t json = IREE_SV(R"({"model": {}})");
  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens));
  EXPECT_EQ(tokens.count, 0u);
  iree_tokenizer_huggingface_added_tokens_free(&tokens);
}

TEST_F(AddedTokensJsonTest, NullAddedTokens) {
  // added_tokens is null.
  iree_string_view_t json = IREE_SV(R"({"added_tokens": null, "model": {}})");
  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens));
  EXPECT_EQ(tokens.count, 0u);
  iree_tokenizer_huggingface_added_tokens_free(&tokens);
}

TEST_F(AddedTokensJsonTest, EmptyArray) {
  // added_tokens is an empty array.
  iree_string_view_t json = IREE_SV(R"({"added_tokens": [], "model": {}})");
  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens));
  EXPECT_EQ(tokens.count, 0u);
  iree_tokenizer_huggingface_added_tokens_free(&tokens);
}

TEST_F(AddedTokensJsonTest, SingleToken) {
  // Single added token with all fields.
  iree_string_view_t json = IREE_SV(R"({
    "added_tokens": [
      {
        "id": 50256,
        "content": "<|endoftext|>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true
      }
    ],
    "model": {}
  })");

  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens));
  ASSERT_EQ(tokens.count, 1u);

  const iree_tokenizer_huggingface_added_token_t* token =
      iree_tokenizer_huggingface_added_tokens_get(&tokens, 0);
  EXPECT_EQ(token->id, 50256);
  EXPECT_EQ(token->flags, IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL);
  EXPECT_EQ(token->content_length, 13u);

  iree_string_view_t content =
      iree_tokenizer_huggingface_added_token_content(&tokens, token);
  EXPECT_TRUE(iree_string_view_equal(content, IREE_SV("<|endoftext|>")));

  iree_tokenizer_huggingface_added_tokens_free(&tokens);
}

TEST_F(AddedTokensJsonTest, MultipleTokens) {
  // Multiple added tokens with various flags.
  iree_string_view_t json = IREE_SV(R"({
    "added_tokens": [
      {
        "id": 0,
        "content": "[PAD]",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true
      },
      {
        "id": 1,
        "content": "[UNK]",
        "single_word": true,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true
      },
      {
        "id": 2,
        "content": "[CLS]",
        "single_word": false,
        "lstrip": true,
        "rstrip": true,
        "normalized": false,
        "special": true
      }
    ],
    "model": {}
  })");

  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens));
  ASSERT_EQ(tokens.count, 3u);

  // Token 0: [PAD]
  const iree_tokenizer_huggingface_added_token_t* token0 =
      iree_tokenizer_huggingface_added_tokens_get(&tokens, 0);
  EXPECT_EQ(token0->id, 0);
  EXPECT_EQ(token0->flags, IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL);
  EXPECT_TRUE(iree_string_view_equal(
      iree_tokenizer_huggingface_added_token_content(&tokens, token0),
      IREE_SV("[PAD]")));

  // Token 1: [UNK] with single_word
  const iree_tokenizer_huggingface_added_token_t* token1 =
      iree_tokenizer_huggingface_added_tokens_get(&tokens, 1);
  EXPECT_EQ(token1->id, 1);
  EXPECT_EQ(token1->flags,
            IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SINGLE_WORD |
                IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL);
  EXPECT_TRUE(iree_string_view_equal(
      iree_tokenizer_huggingface_added_token_content(&tokens, token1),
      IREE_SV("[UNK]")));

  // Token 2: [CLS] with lstrip and rstrip
  const iree_tokenizer_huggingface_added_token_t* token2 =
      iree_tokenizer_huggingface_added_tokens_get(&tokens, 2);
  EXPECT_EQ(token2->id, 2);
  EXPECT_EQ(token2->flags,
            IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_LSTRIP |
                IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_RSTRIP |
                IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL);
  EXPECT_TRUE(iree_string_view_equal(
      iree_tokenizer_huggingface_added_token_content(&tokens, token2),
      IREE_SV("[CLS]")));

  iree_tokenizer_huggingface_added_tokens_free(&tokens);
}

TEST_F(AddedTokensJsonTest, MissingBoolFieldError) {
  // All boolean fields are required (matching HuggingFace serde which has no
  // #[serde(default)] on AddedToken fields). Missing any field is an error.
  iree_string_view_t json = IREE_SV(R"({
    "added_tokens": [
      {
        "id": 0,
        "content": "test",
        "special": false
      }
    ],
    "model": {}
  })");

  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  iree_status_t status = iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

TEST_F(AddedTokensJsonTest, AllBoolFieldsPresent) {
  // Verify all boolean fields are read correctly when explicitly provided.
  iree_string_view_t json = IREE_SV(R"({
    "added_tokens": [
      {
        "id": 0,
        "content": "normal_token",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": true,
        "special": false
      },
      {
        "id": 1,
        "content": "special_token",
        "single_word": true,
        "lstrip": true,
        "rstrip": true,
        "normalized": false,
        "special": true
      }
    ],
    "model": {}
  })");

  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens));
  ASSERT_EQ(tokens.count, 2u);

  // Normal token: only normalized flag set.
  const iree_tokenizer_huggingface_added_token_t* token0 =
      iree_tokenizer_huggingface_added_tokens_get(&tokens, 0);
  EXPECT_EQ(token0->flags,
            IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_NORMALIZED);

  // Special token: all flags except normalized.
  const iree_tokenizer_huggingface_added_token_t* token1 =
      iree_tokenizer_huggingface_added_tokens_get(&tokens, 1);
  EXPECT_EQ(token1->flags,
            IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SINGLE_WORD |
                IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_LSTRIP |
                IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_RSTRIP |
                IREE_TOKENIZER_HUGGINGFACE_ADDED_TOKEN_FLAG_SPECIAL);

  iree_tokenizer_huggingface_added_tokens_free(&tokens);
}

TEST_F(AddedTokensJsonTest, EscapedContent) {
  // Token content with escape sequences.
  iree_string_view_t json = IREE_SV(R"({
    "added_tokens": [
      {
        "id": 0,
        "content": "\u2581",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": false
      }
    ],
    "model": {}
  })");

  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens));
  ASSERT_EQ(tokens.count, 1u);

  const iree_tokenizer_huggingface_added_token_t* token =
      iree_tokenizer_huggingface_added_tokens_get(&tokens, 0);
  // \u2581 is the lower one eighth block (▁), 3 bytes in UTF-8.
  EXPECT_EQ(token->content_length, 3u);
  iree_string_view_t content =
      iree_tokenizer_huggingface_added_token_content(&tokens, token);
  EXPECT_TRUE(iree_string_view_equal(content, IREE_SV("▁")));

  iree_tokenizer_huggingface_added_tokens_free(&tokens);
}

TEST_F(AddedTokensJsonTest, UnknownKeyError) {
  // Unknown key in added token should fail.
  iree_string_view_t json = IREE_SV(R"({
    "added_tokens": [
      {
        "id": 0,
        "content": "test",
        "unknown_field": true
      }
    ],
    "model": {}
  })");

  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  iree_status_t status = iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(AddedTokensJsonTest, MissingIdError) {
  // Missing required id field should fail.
  iree_string_view_t json = IREE_SV(R"({
    "added_tokens": [
      {
        "content": "test"
      }
    ],
    "model": {}
  })");

  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  iree_status_t status = iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

TEST_F(AddedTokensJsonTest, MissingContentError) {
  // Missing required content field should fail.
  iree_string_view_t json = IREE_SV(R"({
    "added_tokens": [
      {
        "id": 0
      }
    ],
    "model": {}
  })");

  iree_tokenizer_huggingface_added_tokens_t tokens = {0};
  iree_status_t status = iree_tokenizer_huggingface_parse_added_tokens_json(
      json, allocator_, &tokens);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

}  // namespace
