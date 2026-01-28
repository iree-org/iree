// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/wordpiece.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/vocab_builder.h"

namespace {

// Helper to build a test vocabulary with WordPiece tokens.
class WordPieceVocabFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // Build a small WordPiece-style vocabulary:
    // 0: [PAD], 1: [UNK], 2: [CLS], 3: [SEP], 4: [MASK]
    // 5: "un", 6: "##happy", 7: "##happi", 8: "##ness"
    // 9: "happy", 10: "test", 11: "##ing", 12: "##ed"
    // 13: "hello", 14: "world", 15: "the", 16: "a"
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        /*capacity=*/20, iree_allocator_system(), &builder_));

    // Special tokens.
    AddToken(0, "[PAD]", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
    AddToken(
        1, "[UNK]",
        IREE_TOKENIZER_TOKEN_ATTR_SPECIAL | IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
    AddToken(2, "[CLS]", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
    AddToken(3, "[SEP]", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);
    AddToken(4, "[MASK]", IREE_TOKENIZER_TOKEN_ATTR_SPECIAL);

    // Regular and continuation tokens.
    AddToken(5, "un", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(6, "##happy", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(7, "##happi", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(8, "##ness", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(9, "happy", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(10, "test", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(11, "##ing", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(12, "##ed", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(13, "hello", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(14, "world", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(15, "the", IREE_TOKENIZER_TOKEN_ATTR_NONE);
    AddToken(16, "a", IREE_TOKENIZER_TOKEN_ATTR_NONE);

    // Set special token IDs.
    iree_tokenizer_vocab_builder_set_special_token(
        builder_, IREE_TOKENIZER_SPECIAL_TOKEN_PAD, 0);
    iree_tokenizer_vocab_builder_set_special_token(
        builder_, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);
    iree_tokenizer_vocab_builder_set_special_token(
        builder_, IREE_TOKENIZER_SPECIAL_TOKEN_CLS, 2);
    iree_tokenizer_vocab_builder_set_special_token(
        builder_, IREE_TOKENIZER_SPECIAL_TOKEN_SEP, 3);
    iree_tokenizer_vocab_builder_set_special_token(
        builder_, IREE_TOKENIZER_SPECIAL_TOKEN_MASK, 4);

    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder_, &vocab_));
  }

  void TearDown() override {
    if (vocab_) iree_tokenizer_vocab_free(vocab_);
  }

  void AddToken(int32_t id, const char* text,
                iree_tokenizer_token_attr_t attrs) {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        builder_, id, IREE_SV(text), /*score=*/0.0f, attrs));
  }

  // Encodes a word and returns the token IDs.
  std::vector<int32_t> Encode(const char* word) {
    int32_t ids[32];
    iree_host_size_t count = 0;
    iree_status_t status = iree_tokenizer_wordpiece_encode_word(
        vocab_, /*config=*/NULL, IREE_SV(word), ids,
        /*max_ids=*/32, &count);
    IREE_EXPECT_OK(status) << "Failed to encode: " << word;
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return {};
    }
    return std::vector<int32_t>(ids, ids + count);
  }

  iree_tokenizer_vocab_builder_t* builder_ = nullptr;
  iree_tokenizer_vocab_t* vocab_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Basic tokenization tests
//===----------------------------------------------------------------------===//

TEST_F(WordPieceVocabFixture, SingleTokenWord) {
  // "hello" exists as a single token.
  auto ids = Encode("hello");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 13);  // "hello"
}

TEST_F(WordPieceVocabFixture, MultipleSubwords) {
  // "unhappy" -> ["un", "##happy"]
  auto ids = Encode("unhappy");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 5);  // "un"
  EXPECT_EQ(ids[1], 6);  // "##happy"
}

TEST_F(WordPieceVocabFixture, ThreeSubwords) {
  // "unhappiness" -> ["un", "##happi", "##ness"]
  auto ids = Encode("unhappiness");
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 5);  // "un"
  EXPECT_EQ(ids[1], 7);  // "##happi"
  EXPECT_EQ(ids[2], 8);  // "##ness"
}

TEST_F(WordPieceVocabFixture, WordWithSuffix) {
  // "testing" -> ["test", "##ing"]
  auto ids = Encode("testing");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 10);  // "test"
  EXPECT_EQ(ids[1], 11);  // "##ing"
}

TEST_F(WordPieceVocabFixture, WordWithDifferentSuffix) {
  // "tested" -> ["test", "##ed"]
  auto ids = Encode("tested");
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 10);  // "test"
  EXPECT_EQ(ids[1], 12);  // "##ed"
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST_F(WordPieceVocabFixture, EmptyWord) {
  auto ids = Encode("");
  EXPECT_EQ(ids.size(), 0u);
}

TEST_F(WordPieceVocabFixture, SingleCharacterWord) {
  // "a" exists as a single token.
  auto ids = Encode("a");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 16);  // "a"
}

TEST_F(WordPieceVocabFixture, UnknownWord) {
  // "xyz" doesn't exist and can't be broken down -> [UNK]
  auto ids = Encode("xyz");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 1);  // [UNK]
}

TEST_F(WordPieceVocabFixture, PartiallyUnknownWord) {
  // "helloX" - "hello" exists but "X" can't be tokenized
  // Should produce [UNK] for the whole word.
  auto ids = Encode("helloX");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 1);  // [UNK]
}

//===----------------------------------------------------------------------===//
// Buffer capacity tests
//===----------------------------------------------------------------------===//

TEST_F(WordPieceVocabFixture, BufferTooSmall) {
  int32_t ids[1];  // Only room for 1 token.
  iree_host_size_t count = 0;
  // "unhappy" requires 2 tokens.
  iree_status_t status = iree_tokenizer_wordpiece_encode_word(
      vocab_, /*config=*/NULL, IREE_SV("unhappy"), ids, /*max_ids=*/1, &count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Configuration tests
//===----------------------------------------------------------------------===//

TEST_F(WordPieceVocabFixture, MaxInputCharsLimit) {
  // Set max_input_chars_per_word to 3.
  iree_tokenizer_wordpiece_config_t config = {
      /*max_input_chars_per_word=*/3,
      /*continuing_subword_prefix=*/IREE_SV("##"),
  };

  int32_t ids[32];
  iree_host_size_t count = 0;
  // "hello" has 5 characters, exceeds limit -> [UNK]
  IREE_ASSERT_OK(iree_tokenizer_wordpiece_encode_word(
      vocab_, &config, IREE_SV("hello"), ids, /*max_ids=*/32, &count));
  ASSERT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 1);  // [UNK]
}

TEST_F(WordPieceVocabFixture, CustomContinuationPrefix) {
  // Build a vocab with different continuation prefix.
  iree_tokenizer_vocab_builder_t* custom_builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/10, iree_allocator_system(), &custom_builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      custom_builder, 0, IREE_SV("[UNK]"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_SPECIAL | IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      custom_builder, 1, IREE_SV("test"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      custom_builder, 2, IREE_SV("@@ing"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_builder_set_special_token(
      custom_builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);

  iree_tokenizer_vocab_t* custom_vocab = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_build(custom_builder, &custom_vocab));

  iree_tokenizer_wordpiece_config_t config = {
      /*max_input_chars_per_word=*/200,
      /*continuing_subword_prefix=*/IREE_SV("@@"),
  };

  int32_t ids[32];
  iree_host_size_t count = 0;
  // "testing" with @@ prefix -> ["test", "@@ing"]
  IREE_ASSERT_OK(iree_tokenizer_wordpiece_encode_word(
      custom_vocab, &config, IREE_SV("testing"), ids, /*max_ids=*/32, &count));
  ASSERT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 1);  // "test"
  EXPECT_EQ(ids[1], 2);  // "@@ing"

  iree_tokenizer_vocab_free(custom_vocab);
}

//===----------------------------------------------------------------------===//
// Tokenizer allocator validation tests
//===----------------------------------------------------------------------===//

TEST(WordPieceAllocatorTest, RejectsVocabWithoutUnk) {
  // Build a vocab WITHOUT an UNK token.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/5, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SV("hello"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SV("world"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  // Note: No UNK token set.

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // Attempt to create a WordPiece tokenizer should fail.
  // Note: allocate consumes vocab (frees on failure), so no manual cleanup.
  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_wordpiece_allocate(
      vocab, /*config=*/nullptr, /*prefix_storage=*/nullptr,
      iree_allocator_system(), &tokenizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(WordPieceAllocatorTest, AcceptsVocabWithUnk) {
  // Build a vocab WITH an UNK token.
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/5, iree_allocator_system(), &builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SV("[UNK]"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_SPECIAL | IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 1, IREE_SV("hello"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));

  // Should succeed. Allocate consumes vocab; tokenizer_free cleans up both.
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_wordpiece_allocate(
      vocab, /*config=*/nullptr, /*prefix_storage=*/nullptr,
      iree_allocator_system(), &tokenizer));
  EXPECT_NE(tokenizer, nullptr);

  iree_tokenizer_free(tokenizer);
}

// Helper to build a minimal vocab with UNK for allocator tests.
static iree_tokenizer_vocab_t* BuildMinimalVocabWithUnk() {
  iree_tokenizer_vocab_builder_t* builder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/5, iree_allocator_system(), &builder));
  IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder, 0, IREE_SV("[UNK]"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_SPECIAL | IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN));
  iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_CHECK_OK(iree_tokenizer_vocab_builder_build(builder, &vocab));
  return vocab;
}

TEST(WordPieceAllocatorTest, RejectsConfigWithMaxCharsExceedingScratchBuffer) {
  // The scratch buffer is 1024 bytes. With default "##" prefix (2 bytes),
  // max chars is (1024 - 2) / 4 = 255 (accounting for 4-byte UTF-8).
  // Configuring 256 chars should fail.
  iree_tokenizer_vocab_t* vocab = BuildMinimalVocabWithUnk();

  iree_tokenizer_wordpiece_config_t config = {
      /*max_input_chars_per_word=*/256,  // Exceeds limit of 255.
      /*continuing_subword_prefix=*/IREE_SV("##"),
  };

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_wordpiece_allocate(
      vocab, &config, /*prefix_storage=*/nullptr, iree_allocator_system(),
      &tokenizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(WordPieceAllocatorTest, AcceptsConfigAtMaxCharsLimit) {
  // With default "##" prefix (2 bytes), max chars is exactly 255.
  iree_tokenizer_vocab_t* vocab = BuildMinimalVocabWithUnk();

  iree_tokenizer_wordpiece_config_t config = {
      /*max_input_chars_per_word=*/255,  // Exactly at limit.
      /*continuing_subword_prefix=*/IREE_SV("##"),
  };

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_wordpiece_allocate(
      vocab, &config, /*prefix_storage=*/nullptr, iree_allocator_system(),
      &tokenizer));
  EXPECT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

TEST(WordPieceAllocatorTest, LongerPrefixReducesMaxChars) {
  // With a 10-byte prefix, max chars is (1024 - 10) / 4 = 253.
  // Configuring 254 chars should fail.
  iree_tokenizer_vocab_t* vocab = BuildMinimalVocabWithUnk();

  iree_tokenizer_wordpiece_config_t config = {
      /*max_input_chars_per_word=*/254,  // Exceeds limit of 253.
      /*continuing_subword_prefix=*/IREE_SV("##PREFIX##"),  // 10 bytes.
  };

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_wordpiece_allocate(
      vocab, &config, /*prefix_storage=*/nullptr, iree_allocator_system(),
      &tokenizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(WordPieceAllocatorTest, EmptyPrefixAllowsMoreChars) {
  // With empty prefix (0 bytes), max chars is 1024 / 4 = 256.
  iree_tokenizer_vocab_t* vocab = BuildMinimalVocabWithUnk();

  static const char empty_str[] = "";
  iree_tokenizer_wordpiece_config_t config = {
      /*max_input_chars_per_word=*/256,  // Valid with empty prefix.
      /*continuing_subword_prefix=*/{/*data=*/empty_str, /*size=*/0},
  };

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_wordpiece_allocate(
      vocab, &config, /*prefix_storage=*/nullptr, iree_allocator_system(),
      &tokenizer));
  EXPECT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

TEST(WordPieceAllocatorTest, DefaultConfigIsValid) {
  // Default config (200 chars, "##" prefix) should always succeed.
  // 200 * 4 + 2 = 802 bytes, well under 1024.
  iree_tokenizer_vocab_t* vocab = BuildMinimalVocabWithUnk();

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_wordpiece_allocate(
      vocab, /*config=*/nullptr, /*prefix_storage=*/nullptr,
      iree_allocator_system(), &tokenizer));
  EXPECT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Empty prefix configuration tests
//===----------------------------------------------------------------------===//

TEST_F(WordPieceVocabFixture, ExplicitEmptyPrefix) {
  // Build a vocab where continuation tokens have no prefix.
  iree_tokenizer_vocab_builder_t* custom_builder = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
      /*capacity=*/10, iree_allocator_system(), &custom_builder));

  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      custom_builder, 0, IREE_SV("[UNK]"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_SPECIAL | IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      custom_builder, 1, IREE_SV("test"), 0.0f,
      IREE_TOKENIZER_TOKEN_ATTR_NONE));
  // "ing" with no prefix (explicit empty prefix scheme).
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      custom_builder, 2, IREE_SV("ing"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  iree_tokenizer_vocab_builder_set_special_token(
      custom_builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);

  iree_tokenizer_vocab_t* custom_vocab = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_build(custom_builder, &custom_vocab));

  // Empty string: data points to empty string literal, size=0.
  // This should use explicit empty prefix, NOT the default "##".
  static const char empty_str[] = "";
  iree_tokenizer_wordpiece_config_t config = {
      /*max_input_chars_per_word=*/200,
      /*continuing_subword_prefix=*/{/*data=*/empty_str, /*size=*/0},
  };

  int32_t ids[32];
  iree_host_size_t count = 0;
  // "testing" with empty prefix -> ["test", "ing"]
  IREE_ASSERT_OK(iree_tokenizer_wordpiece_encode_word(
      custom_vocab, &config, IREE_SV("testing"), ids, /*max_ids=*/32, &count));
  ASSERT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 1);  // "test"
  EXPECT_EQ(ids[1], 2);  // "ing" (no prefix)

  iree_tokenizer_vocab_free(custom_vocab);
}

TEST_F(WordPieceVocabFixture, NullPrefixUsesDefault) {
  // data=NULL should use the default "##" prefix.
  iree_tokenizer_wordpiece_config_t config = {
      /*max_input_chars_per_word=*/200,
      /*continuing_subword_prefix=*/{/*data=*/NULL, /*size=*/0},
  };

  int32_t ids[32];
  iree_host_size_t count = 0;
  // "testing" with default "##" prefix -> ["test", "##ing"]
  IREE_ASSERT_OK(iree_tokenizer_wordpiece_encode_word(
      vocab_, &config, IREE_SV("testing"), ids, /*max_ids=*/32, &count));
  ASSERT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 10);  // "test"
  EXPECT_EQ(ids[1], 11);  // "##ing"
}

}  // namespace
