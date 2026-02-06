// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/special_tokens.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace tokenizer {
namespace {

//===----------------------------------------------------------------------===//
// Builder Tests
//===----------------------------------------------------------------------===//

TEST(SpecialTokensBuilder, InitializeDeinitialize) {
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);
  EXPECT_EQ(builder.entry_count, 0u);
  EXPECT_EQ(builder.string_size, 0u);

  iree_tokenizer_special_tokens_builder_deinitialize(&builder);
  EXPECT_EQ(builder.entry_count, 0u);
}

TEST(SpecialTokensBuilder, AddSingleToken) {
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);

  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<|endoftext|>"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  EXPECT_EQ(builder.entry_count, 1u);
  EXPECT_EQ(builder.string_size, 13u);

  iree_tokenizer_special_tokens_builder_deinitialize(&builder);
}

TEST(SpecialTokensBuilder, AddMultipleTokens) {
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);

  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[CLS]"), 101, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[SEP]"), 102, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[PAD]"), 0, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[UNK]"), 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[MASK]"), 103,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

  EXPECT_EQ(builder.entry_count, 5u);
  // 5 + 5 + 5 + 5 + 6 = 26
  EXPECT_EQ(builder.string_size, 26u);

  iree_tokenizer_special_tokens_builder_deinitialize(&builder);
}

TEST(SpecialTokensBuilder, RejectsEmptyContent) {
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);

  iree_status_t status = iree_tokenizer_special_tokens_builder_add(
      &builder, iree_string_view_empty(), 1,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  iree_tokenizer_special_tokens_builder_deinitialize(&builder);
}

TEST(SpecialTokensBuilder, DeinitializeNull) {
  // Should not crash.
  iree_tokenizer_special_tokens_builder_deinitialize(NULL);
}

//===----------------------------------------------------------------------===//
// Build Tests
//===----------------------------------------------------------------------===//

TEST(SpecialTokensBuild, EmptyBuilder) {
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);

  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &builder, iree_allocator_system(), &special_tokens));

  EXPECT_TRUE(iree_tokenizer_special_tokens_is_empty(&special_tokens));
  EXPECT_EQ(special_tokens.count, 0u);
  EXPECT_EQ(special_tokens.bucket_count, 0u);
  EXPECT_EQ(special_tokens.slab, nullptr);

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_special_tokens_builder_deinitialize(&builder);
}

TEST(SpecialTokensBuild, SingleToken) {
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<|endoftext|>"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &builder, iree_allocator_system(), &special_tokens));

  EXPECT_FALSE(iree_tokenizer_special_tokens_is_empty(&special_tokens));
  EXPECT_EQ(special_tokens.count, 1u);
  EXPECT_EQ(special_tokens.bucket_count, 1u);
  EXPECT_EQ(special_tokens.min_length, 13u);
  EXPECT_EQ(special_tokens.max_length, 13u);
  EXPECT_NE(special_tokens.slab, nullptr);

  // Verify bucket setup.
  EXPECT_NE(special_tokens.first_byte_to_bucket['<'],
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_BUCKET);
  EXPECT_EQ(special_tokens.first_byte_to_bucket['a'],
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_BUCKET);

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_special_tokens_builder_deinitialize(&builder);
}

TEST(SpecialTokensBuild, MultipleBuckets) {
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);

  // Tokens with different first bytes.
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<|endoftext|>"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[CLS]"), 101, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[SEP]"), 102, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &builder, iree_allocator_system(), &special_tokens));

  EXPECT_EQ(special_tokens.count, 3u);
  EXPECT_EQ(special_tokens.bucket_count, 2u);  // '<' and '['

  // Verify both first bytes have buckets.
  EXPECT_NE(special_tokens.first_byte_to_bucket['<'],
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_BUCKET);
  EXPECT_NE(special_tokens.first_byte_to_bucket['['],
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_BUCKET);

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_special_tokens_builder_deinitialize(&builder);
}

TEST(SpecialTokensBuild, SortsByLengthDescending) {
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);

  // Add tokens with same first byte but different lengths (in arbitrary order).
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<s>"), 1, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<|endoftext|>"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("</s>"), 2, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &builder, iree_allocator_system(), &special_tokens));

  EXPECT_EQ(special_tokens.count, 3u);
  EXPECT_EQ(special_tokens.bucket_count, 1u);  // All start with '<'.
  EXPECT_EQ(special_tokens.min_length, 3u);
  EXPECT_EQ(special_tokens.max_length, 13u);

  // First token in bucket should be longest (for longest-match semantics).
  const iree_tokenizer_special_tokens_bucket_t* bucket =
      &special_tokens.buckets[0];
  EXPECT_EQ(bucket->start, 0u);
  EXPECT_EQ(bucket->end, 3u);

  // Verify longest token's ID is first.
  EXPECT_EQ(special_tokens.ids[0], 50256);  // <|endoftext|> (13 bytes)

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_special_tokens_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Match Tests
//===----------------------------------------------------------------------===//

class SpecialTokensMatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_special_tokens_builder_t builder;
    iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                     &builder);

    // GPT-2 style special tokens.
    IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
        &builder, IREE_SV("<|endoftext|>"), 50256,
        IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
    IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
        &builder, IREE_SV("<|startoftext|>"), 50257,
        IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
    // BERT style.
    IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
        &builder, IREE_SV("[CLS]"), 101,
        IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
    IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
        &builder, IREE_SV("[SEP]"), 102,
        IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
    IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
        &builder, IREE_SV("[MASK]"), 103,
        IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

    IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
        &builder, iree_allocator_system(), &special_tokens_));
    iree_tokenizer_special_tokens_builder_deinitialize(&builder);

    iree_tokenizer_special_tokens_encode_state_initialize(&state_);
  }

  void TearDown() override {
    iree_tokenizer_special_tokens_deinitialize(&special_tokens_);
  }

  // Resets state between tests that need fresh state.
  void ResetState() {
    iree_tokenizer_special_tokens_encode_state_initialize(&state_);
  }

  iree_tokenizer_special_tokens_t special_tokens_;
  iree_tokenizer_special_tokens_encode_state_t state_;
};

TEST_F(SpecialTokensMatchTest, ExactMatch) {
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("<|endoftext|>"), &length, &id, &state_);

  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
  EXPECT_EQ(length, 13u);
  EXPECT_EQ(id, 50256);
}

TEST_F(SpecialTokensMatchTest, MatchWithTrailingText) {
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("[CLS]hello world"), &length, &id, &state_);

  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
  EXPECT_EQ(length, 5u);
  EXPECT_EQ(id, 101);
}

TEST_F(SpecialTokensMatchTest, NoMatchDifferentFirstByte) {
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("hello world"), &length, &id, &state_);

  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

TEST_F(SpecialTokensMatchTest, NoMatchSameFirstByteDifferentContent) {
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  // Starts with '<' but doesn't match any token.
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("<html>"), &length, &id, &state_);

  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

TEST_F(SpecialTokensMatchTest, NeedMorePartialPrefix) {
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  // "<|" could be start of "<|endoftext|>" or "<|startoftext|>".
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("<|"), &length, &id, &state_);

  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);
}

TEST_F(SpecialTokensMatchTest, NeedMorePartialToken) {
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  // Partial "<|endoftext|>" - could match if more data comes.
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("<|endof"), &length, &id, &state_);

  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);
}

TEST_F(SpecialTokensMatchTest, LongestMatchWins) {
  // Build a special tokens collection with overlapping tokens.
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<s>"), 1, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<special>"), 2,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&builder);

  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  // Both "<s>" and "<special>" could match, but longest wins.
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens, IREE_SV("<special>token"), &length, &id, &state_);

  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
  EXPECT_EQ(length, 9u);  // "<special>" is 9 bytes.
  EXPECT_EQ(id, 2);

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
}

TEST_F(SpecialTokensMatchTest, EmptyInput) {
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, iree_string_view_empty(), &length, &id, &state_);

  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

TEST_F(SpecialTokensMatchTest, EmptySpecialTokens) {
  iree_tokenizer_special_tokens_t empty;
  iree_tokenizer_special_tokens_initialize(&empty);

  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  auto result = iree_tokenizer_special_tokens_match(
      &empty, IREE_SV("<|endoftext|>"), &length, &id, &state_);

  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);

  iree_tokenizer_special_tokens_deinitialize(&empty);
}

//===----------------------------------------------------------------------===//
// Streaming Continuation Tests
//===----------------------------------------------------------------------===//

// Tests that match() correctly handles streaming continuation where caller
// provides NEW bytes only (not re-provided bytes) after NEED_MORE.
TEST_F(SpecialTokensMatchTest, StreamingContinuationFullMatch) {
  // Simulate streaming "<|endoftext|>" in 3 chunks: "<|", "endoftext", "|>"
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  // Chunk 1: "<|" (2 bytes)
  ResetState();
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("<|"), &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);
  EXPECT_EQ(state_.match_position, 2u);

  // Chunk 2: "endoftext" (9 bytes) - continuation, state tracks progress
  result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("endoftext"), &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);
  EXPECT_EQ(state_.match_position, 11u);

  // Chunk 3: "|>" (2 bytes) - completes the match
  result = iree_tokenizer_special_tokens_match(&special_tokens_, IREE_SV("|>"),
                                               &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
  EXPECT_EQ(length, 2u);  // Only NEW bytes consumed.
  EXPECT_EQ(id, 50256);
  EXPECT_EQ(state_.match_position, 0u);  // Cleared on match.
}

TEST_F(SpecialTokensMatchTest, StreamingContinuationByteByByte) {
  // Stream "<|endoftext|>" one byte at a time.
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;
  const char* token = "<|endoftext|>";
  size_t token_len = strlen(token);

  ResetState();
  for (size_t i = 0; i < token_len - 1; ++i) {
    auto result = iree_tokenizer_special_tokens_match(
        &special_tokens_, iree_make_string_view(&token[i], 1), &length, &id,
        &state_);
    EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE)
        << "Failed at byte " << i << " ('" << token[i] << "')";
    EXPECT_EQ(state_.match_position, i + 1);
  }

  // Final byte completes the match.
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, iree_make_string_view(&token[token_len - 1], 1),
      &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
  EXPECT_EQ(length, 1u);
  EXPECT_EQ(id, 50256);
}

TEST_F(SpecialTokensMatchTest, StreamingContinuationMismatchMidway) {
  // Start matching "<|endoftext|>" but diverge at byte 7 ("<|endofX").
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  ResetState();
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("<|endof"), &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);
  EXPECT_EQ(state_.match_position, 7u);

  // Now provide "X" which doesn't match "t" (expected next byte).
  result = iree_tokenizer_special_tokens_match(&special_tokens_, IREE_SV("X"),
                                               &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
  // State still has match_position for get_partial() reconstruction.
  EXPECT_EQ(state_.match_position, 7u);
}

TEST_F(SpecialTokensMatchTest, StreamingContinuationGetPartial) {
  // Start a partial match, then verify get_partial() reconstructs correctly.
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  ResetState();
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("<|endo"), &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);
  EXPECT_EQ(state_.match_position, 6u);

  // Reconstruct the partial match bytes.
  uint8_t buffer[32];
  iree_host_size_t partial_size =
      iree_tokenizer_special_tokens_encode_state_get_partial(
          &state_, &special_tokens_, buffer);
  EXPECT_EQ(partial_size, 6u);
  EXPECT_EQ(memcmp(buffer, "<|endo", 6), 0);
}

TEST_F(SpecialTokensMatchTest, StreamingContinuationMatchWithExtraData) {
  // Match completes but caller provides extra bytes beyond the token.
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  ResetState();
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("<|endoftext"), &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);
  EXPECT_EQ(state_.match_position, 11u);

  // Provide "|>extra" - should match and report only 2 bytes consumed.
  result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("|>extra"), &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
  EXPECT_EQ(length, 2u);  // Only "|>" consumed, "extra" remains.
  EXPECT_EQ(id, 50256);
}

TEST_F(SpecialTokensMatchTest, StreamingContinuationEmptyChunk) {
  // Empty chunk during continuation should return NO_MATCH (nothing to check).
  iree_host_size_t length = 0;
  iree_tokenizer_token_id_t id = 0;

  ResetState();
  auto result = iree_tokenizer_special_tokens_match(
      &special_tokens_, IREE_SV("<|"), &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  // Empty continuation.
  result = iree_tokenizer_special_tokens_match(
      &special_tokens_, iree_string_view_empty(), &length, &id, &state_);
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

//===----------------------------------------------------------------------===//
// Encode State Tests
//===----------------------------------------------------------------------===//

TEST(SpecialTokensEncodeState, Initialize) {
  iree_tokenizer_special_tokens_encode_state_t state;
  iree_tokenizer_special_tokens_encode_state_initialize(&state);

  EXPECT_EQ(state.match_position, 0u);
  EXPECT_FALSE(iree_tokenizer_special_tokens_encode_state_has_partial(&state));
}

TEST(SpecialTokensEncodeState, ClearPartial) {
  iree_tokenizer_special_tokens_encode_state_t state;
  state.match_position = 5;
  state.partial_token_index = 1;

  iree_tokenizer_special_tokens_encode_state_clear_partial(&state);

  EXPECT_EQ(state.match_position, 0u);
  EXPECT_FALSE(iree_tokenizer_special_tokens_encode_state_has_partial(&state));
}

//===----------------------------------------------------------------------===//
// Flag Behavior Tests (lstrip/rstrip/single_word)
//===----------------------------------------------------------------------===//

// Test fixture for flag behavior tests. Builds special tokens with specific
// flags to verify boundary checking works correctly.
class SpecialTokensFlagsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_special_tokens_encode_state_initialize(&state_);
  }

  void TearDown() override {
    iree_tokenizer_special_tokens_deinitialize(&special_tokens_);
  }

  // Builds special tokens with a single token having the specified flags.
  void BuildWithFlags(const char* content, iree_tokenizer_token_id_t id,
                      iree_tokenizer_special_token_flags_t flags) {
    iree_tokenizer_special_tokens_builder_t builder;
    iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                     &builder);
    IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
        &builder, iree_make_cstring_view(content), id, flags));
    IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
        &builder, iree_allocator_system(), &special_tokens_));
    iree_tokenizer_special_tokens_builder_deinitialize(&builder);
  }

  // Attempts match and returns the result.
  iree_tokenizer_special_tokens_match_result_t TryMatch(
      iree_string_view_t input, iree_host_size_t* out_length = nullptr,
      iree_tokenizer_token_id_t* out_id = nullptr) {
    iree_host_size_t length = 0;
    iree_tokenizer_token_id_t id = 0;
    auto result = iree_tokenizer_special_tokens_match(&special_tokens_, input,
                                                      &length, &id, &state_);
    if (out_length) *out_length = length;
    if (out_id) *out_id = id;
    return result;
  }

  // Simulates consuming bytes before attempting match (updates prev_byte
  // state).
  void ConsumeBytes(iree_string_view_t bytes) {
    for (iree_host_size_t i = 0; i < bytes.size; ++i) {
      state_.prev_byte_plus_one = (uint8_t)bytes.data[i] + 1;
      state_.at_start_of_input = false;
    }
  }

  void ResetState() {
    iree_tokenizer_special_tokens_encode_state_initialize(&state_);
  }

  iree_tokenizer_special_tokens_t special_tokens_ = {};
  iree_tokenizer_special_tokens_encode_state_t state_;
};

//===----------------------------------------------------------------------===//
// LSTRIP flag tests
//===----------------------------------------------------------------------===//

TEST_F(SpecialTokensFlagsTest, LstripMatchesAtStartOfInput) {
  BuildWithFlags("<|special|>", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP);

  // At start of input, lstrip should allow match.
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
}

TEST_F(SpecialTokensFlagsTest, LstripMatchesAfterWhitespace) {
  BuildWithFlags("<|special|>", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP);

  // After space, lstrip should allow match.
  ResetState();
  ConsumeBytes(IREE_SV(" "));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // After tab.
  ResetState();
  ConsumeBytes(IREE_SV("\t"));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // After newline.
  ResetState();
  ConsumeBytes(IREE_SV("\n"));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
}

TEST_F(SpecialTokensFlagsTest, LstripRejectsAfterNonWhitespace) {
  BuildWithFlags("<|special|>", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP);

  // After letter 'a', lstrip should NOT allow match.
  ResetState();
  ConsumeBytes(IREE_SV("abc"));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);

  // After digit.
  ResetState();
  ConsumeBytes(IREE_SV("123"));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);

  // After punctuation.
  ResetState();
  ConsumeBytes(IREE_SV("!"));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

//===----------------------------------------------------------------------===//
// RSTRIP flag tests
//===----------------------------------------------------------------------===//

TEST_F(SpecialTokensFlagsTest, RstripMatchesAtEndOfInput) {
  BuildWithFlags("<|special|>", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP);

  // Exact match (no trailing bytes) = at end of input.
  ResetState();
  iree_host_size_t length = 0;
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>"), &length),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
  EXPECT_EQ(length, 11u);
}

TEST_F(SpecialTokensFlagsTest, RstripMatchesBeforeWhitespace) {
  BuildWithFlags("<|special|>", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP);

  // Before space, rstrip should allow match.
  ResetState();
  iree_host_size_t length = 0;
  EXPECT_EQ(TryMatch(IREE_SV("<|special|> "), &length),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
  EXPECT_EQ(length, 11u);

  // Before tab.
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>\t")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // Before newline.
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>\n")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
}

TEST_F(SpecialTokensFlagsTest, RstripRejectsBeforeNonWhitespace) {
  BuildWithFlags("<|special|>", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP);

  // Before letter, rstrip should NOT allow match.
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>abc")),
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);

  // Before digit.
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>123")),
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);

  // Before punctuation.
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>!")),
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

//===----------------------------------------------------------------------===//
// SINGLE_WORD flag tests
//===----------------------------------------------------------------------===//

TEST_F(SpecialTokensFlagsTest, SingleWordMatchesAtWordBoundaries) {
  BuildWithFlags("TOKEN", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_SINGLE_WORD);

  // At start (word boundary) and end (word boundary).
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("TOKEN")), IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // After space, before space.
  ResetState();
  ConsumeBytes(IREE_SV(" "));
  EXPECT_EQ(TryMatch(IREE_SV("TOKEN ")), IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // After punctuation, before punctuation.
  ResetState();
  ConsumeBytes(IREE_SV("!"));
  EXPECT_EQ(TryMatch(IREE_SV("TOKEN!")), IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
}

TEST_F(SpecialTokensFlagsTest, SingleWordRejectsWhenNotAtWordBoundary) {
  BuildWithFlags("TOKEN", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_SINGLE_WORD);

  // After letter (not a word boundary).
  ResetState();
  ConsumeBytes(IREE_SV("abc"));
  EXPECT_EQ(TryMatch(IREE_SV("TOKEN")), IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);

  // Before letter (not a word boundary).
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("TOKENabc")),
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);

  // After digit (not a word boundary).
  ResetState();
  ConsumeBytes(IREE_SV("123"));
  EXPECT_EQ(TryMatch(IREE_SV("TOKEN")), IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

//===----------------------------------------------------------------------===//
// Combined flag tests
//===----------------------------------------------------------------------===//

TEST_F(SpecialTokensFlagsTest, LstripAndRstripCombined) {
  BuildWithFlags("<|special|>", 100,
                 IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP |
                     IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP);

  // At start and end - should match.
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // After space, before space - should match.
  ResetState();
  ConsumeBytes(IREE_SV(" "));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|> ")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // After letter, before space - lstrip fails.
  ResetState();
  ConsumeBytes(IREE_SV("abc"));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|> ")),
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);

  // After space, before letter - rstrip fails.
  ResetState();
  ConsumeBytes(IREE_SV(" "));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>abc")),
            IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

TEST_F(SpecialTokensFlagsTest, NoFlagsMatchesAnywhere) {
  BuildWithFlags("<|special|>", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE);

  // At start.
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // After letter.
  ResetState();
  ConsumeBytes(IREE_SV("abc"));
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // Before letter.
  ResetState();
  EXPECT_EQ(TryMatch(IREE_SV("<|special|>abc")),
            IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);
}

//===----------------------------------------------------------------------===//
// Streaming flag tests (token spans chunks)
//===----------------------------------------------------------------------===//

TEST_F(SpecialTokensFlagsTest, LstripStreamingRemembersPrevByte) {
  BuildWithFlags("<|special|>", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP);

  // Scenario 1: Stream token in chunks, starting at input start (should match).
  ResetState();
  auto result = TryMatch(IREE_SV("<|spe"));  // Partial
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  // Continue - prev_byte check uses state set before first chunk.
  result = TryMatch(IREE_SV("cial|>"));  // Complete
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // Scenario 2: Stream after non-whitespace (should NOT match).
  ResetState();
  ConsumeBytes(IREE_SV("abc"));  // Set prev_byte to 'c'.
  result = TryMatch(IREE_SV("<|spe"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  // When completing, lstrip should check the prev_byte from BEFORE the token
  // started. Since 'c' preceded it, should reject.
  result = TryMatch(IREE_SV("cial|>"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

TEST_F(SpecialTokensFlagsTest, RstripStreamingChecksNextByte) {
  BuildWithFlags("<|special|>", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP);

  // Scenario 1: Stream token ending at input end (should match).
  ResetState();
  auto result = TryMatch(IREE_SV("<|spe"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  // Complete - no trailing bytes = at end.
  result = TryMatch(IREE_SV("cial|>"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // Scenario 2: Stream token with whitespace after.
  ResetState();
  result = TryMatch(IREE_SV("<|spe"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  // Complete with space after - should match.
  result = TryMatch(IREE_SV("cial|> "));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // Scenario 3: Stream token with letter after (should NOT match).
  ResetState();
  result = TryMatch(IREE_SV("<|spe"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  // Complete with letter after - should NOT match.
  result = TryMatch(IREE_SV("cial|>abc"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

TEST_F(SpecialTokensFlagsTest, SingleWordStreamingChecksBothBoundaries) {
  BuildWithFlags("TOKEN", 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_SINGLE_WORD);

  // Scenario 1: At word boundaries on both sides (should match).
  ResetState();
  auto result = TryMatch(IREE_SV("TOK"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  result = TryMatch(IREE_SV("EN"));  // Complete at end.
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // Scenario 2: Left boundary violated (should NOT match).
  ResetState();
  ConsumeBytes(IREE_SV("abc"));  // Prev is letter.
  result = TryMatch(IREE_SV("TOK"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  result = TryMatch(IREE_SV("EN"));  // Complete.
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);

  // Scenario 3: Right boundary violated (should NOT match).
  ResetState();
  ConsumeBytes(IREE_SV(" "));  // Prev is space (word boundary).
  result = TryMatch(IREE_SV("TOK"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  result = TryMatch(IREE_SV("ENabc"));  // Next is letter.
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

TEST_F(SpecialTokensFlagsTest, StreamingByteByByteWithFlags) {
  BuildWithFlags("ABC", 100,
                 IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP |
                     IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP);

  // Stream byte by byte after whitespace, before whitespace (should match).
  ResetState();
  ConsumeBytes(IREE_SV(" "));

  auto result = TryMatch(IREE_SV("A"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  result = TryMatch(IREE_SV("B"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  result = TryMatch(IREE_SV("C "));  // Complete with trailing space.
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED);

  // Same but no trailing whitespace at end - should NOT match.
  ResetState();
  ConsumeBytes(IREE_SV(" "));

  result = TryMatch(IREE_SV("A"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  result = TryMatch(IREE_SV("B"));
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE);

  result = TryMatch(IREE_SV("Cx"));  // Complete with 'x' after.
  EXPECT_EQ(result, IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH);
}

//===----------------------------------------------------------------------===//
// Lifecycle Tests
//===----------------------------------------------------------------------===//

TEST(SpecialTokensLifecycle, InitializeDeinitialize) {
  iree_tokenizer_special_tokens_t special_tokens;
  iree_tokenizer_special_tokens_initialize(&special_tokens);

  EXPECT_TRUE(iree_tokenizer_special_tokens_is_empty(&special_tokens));
  EXPECT_EQ(special_tokens.count, 0u);
  EXPECT_EQ(special_tokens.slab, nullptr);

  // All first bytes should map to NO_BUCKET.
  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(special_tokens.first_byte_to_bucket[i],
              IREE_TOKENIZER_SPECIAL_TOKENS_NO_BUCKET);
  }

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);

  // After deinitialize, struct should be zeroed.
  EXPECT_EQ(special_tokens.count, 0u);
}

}  // namespace
}  // namespace tokenizer
}  // namespace iree
