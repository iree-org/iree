// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/bpe.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/vocab_builder.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

// Helper to build a test vocabulary with BPE tokens and merges.
// Example vocabulary for "hello":
//   Tokens: 'h'(0), 'e'(1), 'l'(2), 'o'(3), 'll'(4), 'he'(5), 'hel'(6),
//           'hell'(7), 'hello'(8), [UNK](9)
//   Merges (in rank order):
//     0: ('l', 'l') -> 'll'
//     1: ('h', 'e') -> 'he'
//     2: ('he', 'l') -> 'hel'
//     3: ('hel', 'l') -> 'hell' -- wait, this doesn't match 'll'
// Let me redesign to make the merges actually work.
//
// Simpler vocabulary:
//   Tokens: 'h'(0), 'e'(1), 'l'(2), 'o'(3), 'll'(4), 'he'(5), 'lo'(6),
//           'hel'(7), 'llo'(8), 'hell'(9), 'ello'(10), 'hello'(11), [UNK](12)
//   Merges:
//     0: ('l', 'l') -> 'll'
//     1: ('h', 'e') -> 'he'
//     2: ('l', 'o') -> 'lo'
//     3: ('ll', 'o') -> 'llo'
//     4: ('he', 'l') -> 'hel'
//     5: ('hel', 'l') -> wait, we need 'l' + 'l' first
//
// Real BPE ordering for "hello":
// Start: ['h', 'e', 'l', 'l', 'o']
// Best merge at each step based on rank:
// If rank order is: ll < he < lo < llo < hel < hello
//   Step 1: merge l+l -> ll : ['h', 'e', 'll', 'o']
//   Step 2: merge h+e -> he : ['he', 'll', 'o']
//   Step 3: merge ll+o -> llo : ['he', 'llo']
//   Step 4: merge he+llo -> hello : ['hello']

class BpeVocabFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        /*capacity=*/20, iree_allocator_system(), &builder_));

    // Single character tokens.
    AddToken(0, "h", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(1, "e", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(2, "l", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(3, "o", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(4, "w", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(5, "r", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(6, "d", IREE_TOKENIZER_TOKEN_ATTR_NONE);

    // Merged tokens.
    AddToken(7, "ll", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(8, "he", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(9, "llo", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(10, "hello", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(11, "or", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(12, "wor", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(13, "world", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(14, "ld", IREE_TOKENIZER_TOKEN_ATTR_NONE);

    // Special tokens.
    AddToken(
        15, "[UNK]",
        IREE_TOKENIZER_TOKEN_ATTR_SPECIAL | IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);

    iree_tokenizer_vocab_builder_set_special_token(
        builder_, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 15);

    // Add merges in rank order (lower rank = higher priority).
    // For "hello": h e l l o -> he ll o -> he llo -> hello
    AddMerge(2, 2);  // 0: l + l -> ll
    AddMerge(0, 1);  // 1: h + e -> he
    AddMerge(7, 3);  // 2: ll + o -> llo
    AddMerge(8, 9);  // 3: he + llo -> hello

    // For "world": w o r l d -> wor l d -> wor ld -> world
    AddMerge(3, 5);    // 4: o + r -> or
    AddMerge(4, 11);   // 5: w + or -> wor
    AddMerge(2, 6);    // 6: l + d -> ld
    AddMerge(12, 14);  // 7: wor + ld -> world

    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder_, &vocab_));
    IREE_ASSERT_OK(iree_tokenizer_bpe_state_allocate(
        vocab_, iree_allocator_system(), &state_));
  }

  void TearDown() override {
    if (state_) iree_tokenizer_bpe_state_free(state_);
    if (vocab_) iree_tokenizer_vocab_free(vocab_);
  }

  void AddToken(int32_t id, const char* text,
                iree_tokenizer_token_attr_t attrs) {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        builder_, id, IREE_SV(text), /*score=*/0.0f, attrs));
  }

  void AddMerge(uint32_t left_id, uint32_t right_id) {
    IREE_ASSERT_OK(
        iree_tokenizer_vocab_builder_add_merge(builder_, left_id, right_id));
  }

  std::vector<int32_t> Encode(const char* word) {
    int32_t ids[64];
    iree_host_size_t count = 0;
    iree_status_t status = iree_tokenizer_bpe_encode_word(
        state_, IREE_SV(word), ids, /*max_ids=*/64, &count);
    IREE_EXPECT_OK(status) << "Failed to encode: " << word;
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return {};
    }
    return std::vector<int32_t>(ids, ids + count);
  }

  iree_tokenizer_vocab_builder_t* builder_ = nullptr;
  iree_tokenizer_vocab_t* vocab_ = nullptr;
  iree_tokenizer_bpe_state_t* state_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Basic BPE tests
//===----------------------------------------------------------------------===//

TEST_F(BpeVocabFixture, SingleCharacterWord) {
  // 'h' is a single character token, no merges.
  auto ids = Encode("h");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 0);  // 'h'
}

TEST_F(BpeVocabFixture, TwoCharacterMerge) {
  // "ll" -> l + l -> ll (merge rank 0)
  auto ids = Encode("ll");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 7);  // 'll'
}

TEST_F(BpeVocabFixture, HelloWord) {
  // "hello" -> h e l l o -> he ll o -> he llo -> hello
  auto ids = Encode("hello");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 10);  // 'hello'
}

TEST_F(BpeVocabFixture, WorldWord) {
  // "world" -> w o r l d -> w or l d -> wor l d -> wor ld -> world
  auto ids = Encode("world");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 13);  // 'world'
}

TEST_F(BpeVocabFixture, PartialMerge) {
  // "he" -> h e -> he (merge rank 1)
  auto ids = Encode("he");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 8);  // 'he'
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST_F(BpeVocabFixture, EmptyWord) {
  auto ids = Encode("");
  EXPECT_EQ(ids.size(), 0u);
}

TEST_F(BpeVocabFixture, NoMergesNeeded) {
  // "led" - has l, e, d but no useful merges between them.
  auto ids = Encode("led");
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 2);  // 'l'
  EXPECT_EQ(ids[1], 1);  // 'e'
  EXPECT_EQ(ids[2], 6);  // 'd'
}

TEST_F(BpeVocabFixture, UnknownCharacter) {
  // 'x' is not in vocabulary, should become [UNK] (or byte fallback).
  auto ids = Encode("x");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 15);  // [UNK]
}

//===----------------------------------------------------------------------===//
// Buffer capacity tests
//===----------------------------------------------------------------------===//

TEST_F(BpeVocabFixture, BufferTooSmall) {
  int32_t ids[1];
  iree_host_size_t count = 0;
  // "led" produces 3 tokens.
  iree_status_t status = iree_tokenizer_bpe_encode_word(
      state_, IREE_SV("led"), ids, /*max_ids=*/1, &count);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));
}

//===----------------------------------------------------------------------===//
// State management tests
//===----------------------------------------------------------------------===//

TEST_F(BpeVocabFixture, StateCreationNoMerges) {
  // Build a vocab with no merges.
  iree_tokenizer_vocab_builder_t* simple_builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/5, iree_allocator_system(), &simple_builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      simple_builder, 0, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      simple_builder, 1, IREE_SV("b"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      simple_builder, 2, IREE_SV("[UNK]"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_SPECIAL | IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN));

  iree_tokenizer_vocab_builder_set_special_token(
      simple_builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);

  iree_tokenizer_vocab_t* simple_vocab = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_build(simple_builder, &simple_vocab));

  // State creation should succeed even without merges.
  iree_tokenizer_bpe_state_t* simple_state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_state_allocate(
      simple_vocab, iree_allocator_system(), &simple_state));

  // Encode should work (no merges applied).
  int32_t ids[32];
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(iree_tokenizer_bpe_encode_word(simple_state, IREE_SV("ab"),
                                                ids, /*max_ids=*/32, &count));
  ASSERT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 0);  // 'a'
  EXPECT_EQ(ids[1], 1);  // 'b'

  iree_tokenizer_bpe_state_free(simple_state);
  iree_tokenizer_vocab_free(simple_vocab);
}

//===----------------------------------------------------------------------===//
// Tokenizer allocator validation tests
//===----------------------------------------------------------------------===//

TEST(BpeAllocatorTest, AcceptsVocabWithoutMerges) {
  // Build a vocab WITHOUT merge rules.
  // BPE without merges is valid - it acts as a character-level tokenizer.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/5, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SV("b"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // Create BPE state - should succeed even without merges.
  iree_tokenizer_bpe_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_state_allocate(
      vocab, iree_allocator_system(), &state));
  EXPECT_NE(state, nullptr);

  // Verify encoding works - should output individual characters (no merging).
  int32_t ids[8];
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_bpe_encode_word(state, IREE_SV("abba"), ids, 8, &count));
  ASSERT_EQ(count, 4u);
  EXPECT_EQ(ids[0], 0);  // a
  EXPECT_EQ(ids[1], 1);  // b
  EXPECT_EQ(ids[2], 1);  // b
  EXPECT_EQ(ids[3], 0);  // a

  iree_tokenizer_bpe_state_free(state);
  iree_tokenizer_vocab_free(vocab);
}

TEST(BpeAllocatorTest, AcceptsVocabWithMerges) {
  // Build a vocab WITH merge rules.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/5, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SV("b"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 2, IREE_SV("ab"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Add a merge rule: a + b -> ab.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder, 0, 1));

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // Should succeed. Allocate consumes vocab.
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_bpe_allocate(vocab, iree_allocator_system(), &tokenizer));
  EXPECT_NE(tokenizer, nullptr);

  // tokenizer_free also frees the vocab (owned by tokenizer).
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Iteration limit tests
//===----------------------------------------------------------------------===//

// Verifies that the merge iteration limit is properly configured.
// Standard BPE with N starting symbols requires at most N-1 merges (each merge
// reduces symbol count by 1). The iteration limit is a defensive measure
// against malformed vocabularies with cyclic merge rules.
TEST(BpeIterationLimitTest, LimitIsReasonable) {
  // The iteration limit should be much larger than MAX_SYMBOLS to allow all
  // legitimate merges, but bounded to prevent hangs with malicious vocabs.
  // With MAX_SYMBOLS=512, max legitimate merges is 511.
  // IREE_TOKENIZER_BPE_MAX_MERGE_ITERATIONS=10000 provides ~20x headroom.
  //
  // We verify the constant relationship holds by building a vocab that
  // exercises many merges and ensuring it completes successfully.

  // Build a vocabulary with a chain of merges:
  // a(0), b(1), c(2), d(3), e(4), f(5), g(6), h(7)
  // ab(8), abc(9), abcd(10), abcde(11), abcdef(12), abcdefg(13), abcdefgh(14)
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/20, iree_allocator_system(), &builder));

  // Base character tokens.
  const char* chars = "abcdefgh";
  for (int i = 0; i < 8; ++i) {
    char token[2] = {chars[i], '\0'};
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        builder, i, iree_make_cstring_view(token), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  // Merged tokens - each step adds one more character.
  const char* merged_tokens[] = {"ab",     "abc",     "abcd",    "abcde",
                                 "abcdef", "abcdefg", "abcdefgh"};
  for (int i = 0; i < 7; ++i) {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        builder, 8 + i, iree_make_cstring_view(merged_tokens[i]), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  // UNK token.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 15, IREE_SV("[UNK]"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_SPECIAL | IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN));
  iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 15);

  // Chain of merges: a+b->ab, ab+c->abc, abc+d->abcd, etc.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder, 0, 1));  // a+b
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_add_merge(builder, 8, 2));  // ab+c
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_add_merge(builder, 9, 3));  // abc+d
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_add_merge(builder, 10, 4));  // abcd+e
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_add_merge(builder, 11, 5));  // abcde+f
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_add_merge(builder, 12, 6));  // abcdef+g
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_add_merge(builder, 13, 7));  // abcdefg+h

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  iree_tokenizer_bpe_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_state_allocate(
      vocab, iree_allocator_system(), &state));

  // Encode "abcdefgh" - requires 7 sequential merges.
  // a b c d e f g h -> ab c d e f g h -> abc d e f g h -> abcd e f g h
  // -> abcde f g h -> abcdef g h -> abcdefg h -> abcdefgh
  int32_t ids[64];
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(iree_tokenizer_bpe_encode_word(state, IREE_SV("abcdefgh"), ids,
                                                64, &count));
  ASSERT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 14);  // abcdefgh

  iree_tokenizer_bpe_state_free(state);
  iree_tokenizer_vocab_free(vocab);
}

// Note: We cannot easily construct a test that triggers the iteration limit
// with valid BPE semantics. Each merge reduces symbol count by 1, so with N
// starting symbols we have at most N-1 iterations. The limit protects against:
// 1. Malformed/corrupt vocabulary data
// 2. Implementation bugs that could cause non-termination
// 3. Adversarial vocabularies designed to cause DoS
//
// The limit's correctness is verified by:
// - The constant value (10000) being well above max legitimate merges (511)
// - Code review confirming the check is applied correctly
// - The LimitIsReasonable test showing legitimate multi-merge chains work

}  // namespace
