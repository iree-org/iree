// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/tokenizer_json.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/testdata/tokenizer_testdata.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab.h"

namespace {

static iree_string_view_t GetTestFile(const char* name) {
  const struct iree_file_toc_t* file_toc = iree_tokenizer_testdata_create();
  for (size_t i = 0; i < iree_tokenizer_testdata_size(); ++i) {
    if (strcmp(file_toc[i].name, name) == 0) {
      return iree_make_string_view((const char*)file_toc[i].data,
                                   file_toc[i].size);
    }
  }
  return iree_string_view_empty();
}

//===----------------------------------------------------------------------===//
// End-to-End JSON Loading Tests
//===----------------------------------------------------------------------===//

class TokenizerJsonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_view_t json = GetTestFile("wordpiece_bert_minimal.json");
    ASSERT_GT(json.size, 0u);
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        json, iree_allocator_system(), &tokenizer_));
    ASSERT_NE(tokenizer_, nullptr);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(TokenizerJsonTest, VocabLoaded) {
  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer_);
  ASSERT_NE(vocab, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 13u);
}

TEST_F(TokenizerJsonTest, EncodeSimple) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 5);  // hello
}

TEST_F(TokenizerJsonTest, EncodeWithNormalization) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // BertNormalizer should lowercase "HELLO" -> "hello".
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("HELLO"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 5);  // hello (lowercased)
}

TEST_F(TokenizerJsonTest, EncodeWithAccentStripping) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // BertNormalizer with strip_accents should normalize "café" -> "cafe".
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("café"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 9);  // cafe
}

TEST_F(TokenizerJsonTest, EncodeMultipleWords) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("HELLO WORLD"),
                                       options, ids, IREE_ARRAYSIZE(ids),
                                       &count));

  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 5);  // hello
  EXPECT_EQ(ids[1], 6);  // world
}

TEST_F(TokenizerJsonTest, EncodeWithSpecialTokens) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // Should be [CLS] hello [SEP] = [2, 5, 3]
  EXPECT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 2);  // [CLS]
  EXPECT_EQ(ids[1], 5);  // hello
  EXPECT_EQ(ids[2], 3);  // [SEP]
}

TEST_F(TokenizerJsonTest, DecodeSimple) {
  int32_t ids[] = {5};  // hello
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_EQ(length, 5u);
  EXPECT_STREQ(text, "hello");
}

TEST_F(TokenizerJsonTest, DecodeWithWordPieceDecoder) {
  // WordPiece decoder should join subwords by stripping "##" prefix.
  int32_t ids[] = {7, 8};  // test, ##ing
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_EQ(length, 7u);
  EXPECT_STREQ(text, "testing");
}

TEST_F(TokenizerJsonTest, DecodeSubwordSequence) {
  // "un" + "##believ" + "##able" -> "unbelievable"
  int32_t ids[] = {10, 11, 12};  // un, ##believ, ##able
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  EXPECT_EQ(length, 12u);
  EXPECT_STREQ(text, "unbelievable");
}

TEST_F(TokenizerJsonTest, DecodeSkipSpecialTokens) {
  int32_t ids[] = {2, 5, 3};  // [CLS] hello [SEP]
  char text[64];
  iree_host_size_t length = 0;

  IREE_ASSERT_OK(
      iree_tokenizer_decode(tokenizer_, ids, IREE_ARRAYSIZE(ids),
                            IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
                            text, sizeof(text), &length));

  EXPECT_EQ(length, 5u);
  EXPECT_STREQ(text, "hello");
}

TEST_F(TokenizerJsonTest, RoundtripNormalized) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode "CAFÉ" (normalized to "cafe").
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("CAFÉ"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 9);  // cafe

  // Decode back.
  char text[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode(tokenizer_, ids, count,
                                       IREE_TOKENIZER_DECODE_FLAG_DEFAULT, text,
                                       sizeof(text), &length));

  // Normalization is one-way, so we get "cafe" not "CAFÉ".
  EXPECT_EQ(length, 4u);
  EXPECT_STREQ(text, "cafe");
}

TEST_F(TokenizerJsonTest, RoundtripWithSpecialTokens) {
  int32_t ids[10];
  iree_host_size_t count = 0;

  // Encode with special tokens.
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("HELLO WORLD"),
                                       options, ids, IREE_ARRAYSIZE(ids),
                                       &count));

  // [CLS] hello world [SEP]
  EXPECT_EQ(count, 4u);
  EXPECT_EQ(ids[0], 2);  // [CLS]
  EXPECT_EQ(ids[1], 5);  // hello
  EXPECT_EQ(ids[2], 6);  // world
  EXPECT_EQ(ids[3], 3);  // [SEP]

  // Decode with skip_special_tokens.
  char text[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_tokenizer_decode(
      tokenizer_, ids, count, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
      text, sizeof(text), &length));

  EXPECT_STREQ(text, "hello world");
}

//===----------------------------------------------------------------------===//
// BPE/GPT-2 Style JSON Loading Tests (post_processor: null)
//===----------------------------------------------------------------------===//

class BpeJsonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_view_t json = GetTestFile("benchmark_bpe.json");
    ASSERT_GT(json.size, 0u);
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        json, iree_allocator_system(), &tokenizer_));
    ASSERT_NE(tokenizer_, nullptr);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(BpeJsonTest, NullPostProcessorNoSpecialTokens) {
  // GPT-2 style: post_processor is null, so ADD_SPECIAL_TOKENS should be no-op.
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("Hello world"),
                                       options, ids, IREE_ARRAYSIZE(ids),
                                       &count));

  // With post_processor: null, NO special tokens should be added.
  // Just the encoded text tokens.
  EXPECT_GT(count, 0u);

  // Verify first token is NOT <|endoftext|> (ID 0).
  // "Hello world" should encode to something like [224, 209] or similar.
  EXPECT_NE(ids[0], 0) << "First token should NOT be <|endoftext|> special";

  // Last token should also NOT be <|endoftext|>.
  EXPECT_NE(ids[count - 1], 0) << "Last token should NOT be <|endoftext|>";
}

TEST_F(BpeJsonTest, EncodeWithoutSpecialTokens) {
  int32_t ids_with[10];
  int32_t ids_without[10];
  iree_host_size_t count_with = 0;
  iree_host_size_t count_without = 0;

  // Encode with ADD_SPECIAL_TOKENS flag.
  iree_tokenizer_encode_options_t options_with = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("Hello"),
                                       options_with, ids_with,
                                       IREE_ARRAYSIZE(ids_with), &count_with));

  // Encode without ADD_SPECIAL_TOKENS flag.
  iree_tokenizer_encode_options_t options_without = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(
      tokenizer_, IREE_SV("Hello"), options_without, ids_without,
      IREE_ARRAYSIZE(ids_without), &count_without));

  // With post_processor: null, both should produce identical results.
  EXPECT_EQ(count_with, count_without);
  for (iree_host_size_t i = 0; i < count_with; ++i) {
    EXPECT_EQ(ids_with[i], ids_without[i]) << "Mismatch at position " << i;
  }
}

// Callback to capture transform segments.
static iree_status_t CollectSegmentCallback(void* user_data,
                                            iree_string_view_list_t segments) {
  auto* vec = static_cast<std::vector<std::string>*>(user_data);
  for (size_t i = 0; i < segments.count; ++i) {
    vec->push_back(
        std::string(segments.values[i].data, segments.values[i].size));
  }
  return iree_ok_status();
}

// Callback to capture tokens.
static iree_status_t CollectTokenCallback(void* user_data,
                                          iree_tokenizer_id_list_t ids) {
  auto* vec = static_cast<std::vector<int32_t>*>(user_data);
  for (size_t i = 0; i < ids.count; ++i) {
    vec->push_back(ids.values[i]);
  }
  return iree_ok_status();
}

// Tests "hello world" tokenization with GPT-2 style ByteLevel pretokenizer.
// The GPT-2 regex pattern uses space as a word PREFIX, not terminator:
//   Input: "hello world"
//   → ByteLevel prepends space: " hello world"
//   → Split(GPT-2 regex): [" hello", " world"]
//   → ByteLevel encodes: ["Ġhello", "Ġworld"]
//   → BPE: "Ġhello" → [Ġ, he, llo], "Ġworld" → [Ġworld]
//   → Total: 4 tokens
TEST_F(BpeJsonTest, HelloWorldTokenization) {
  // Verify transform segments.
  std::vector<std::string> segments;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_encode(
      NULL, &tokenizer_->transform, IREE_SV("hello world"),
      CollectSegmentCallback, &segments));
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "\xC4\xA0hello");  // Ġhello
  EXPECT_EQ(segments[1], "\xC4\xA0world");  // Ġworld

  // Verify full tokenization.
  std::vector<int32_t> tokens;
  IREE_ASSERT_OK(iree_tokenizer_encode_streaming(
      tokenizer_, IREE_SV("hello world"), /*flags=*/0, CollectTokenCallback,
      &tokens));

  // Vocab: 65=Ġ, 85=he, 167=llo, 209=Ġworld
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0], 65);   // Ġ
  EXPECT_EQ(tokens[1], 85);   // he
  EXPECT_EQ(tokens[2], 167);  // llo
  EXPECT_EQ(tokens[3], 209);  // Ġworld
}

//===----------------------------------------------------------------------===//
// JSON Error Path and Cleanup Tests
//===----------------------------------------------------------------------===//

// Tests that error paths correctly clean up partially-allocated resources.
// These tests verify the fix for uninitialized transform/decoder on error.

TEST(TokenizerJsonErrorTest, InvalidPreTokenizerTypeCleanup) {
  // Valid vocab but invalid pre_tokenizer type.
  // This tests cleanup of vocab when pre_tokenizer parsing fails.
  const char* json = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1},
      "unk_token": "[UNK]"
    },
    "pre_tokenizer": {
      "type": "InvalidType"
    },
    "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}]
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer);

  // Should fail (invalid pre_tokenizer type), but not crash.
  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(TokenizerJsonErrorTest, InvalidNormalizerTypeCleanup) {
  // Valid vocab but invalid normalizer type.
  // This tests cleanup of vocab when normalizer parsing fails.
  const char* json = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1},
      "unk_token": "[UNK]"
    },
    "normalizer": {
      "type": "InvalidNormalizer"
    },
    "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}]
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer);

  // Should fail (invalid normalizer type), but not crash.
  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(TokenizerJsonErrorTest, InvalidDecoderTypeCleanup) {
  // Valid vocab but invalid decoder type.
  // This tests cleanup of vocab + transform when decoder parsing fails.
  const char* json = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1},
      "unk_token": "[UNK]"
    },
    "decoder": {
      "type": "InvalidDecoder"
    },
    "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}]
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer);

  // Should fail (invalid decoder type), but not crash.
  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(TokenizerJsonErrorTest, InvalidPostProcessorTypeCleanup) {
  // Valid vocab but invalid post_processor type.
  // This tests cleanup of vocab + transform + decoder when postprocessor fails.
  const char* json = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1},
      "unk_token": "[UNK]"
    },
    "post_processor": {
      "type": "InvalidPostProcessor"
    },
    "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}]
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer);

  // Should fail (invalid post_processor type), but not crash.
  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(TokenizerJsonErrorTest, MalformedJsonNoLeak) {
  // Completely malformed JSON should fail early without leaks.
  const char* json = "not json at all {{{";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer);

  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(TokenizerJsonErrorTest, MissingVocabNoLeak) {
  // Valid JSON structure but missing vocab should fail cleanly.
  const char* json = R"({
    "model": {
      "type": "WordPiece"
    }
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer);

  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(TokenizerJsonErrorTest, EmptyInputNoLeak) {
  iree_string_view_t json = iree_string_view_empty();

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json, iree_allocator_system(), &tokenizer);

  EXPECT_THAT(status, ::testing::Not(::iree::testing::status::IsOk()));
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

//===----------------------------------------------------------------------===//
// WordPiece JSON Factory Tests
//===----------------------------------------------------------------------===//

TEST(WordPieceJsonFactoryTest, RejectsNegativeMaxInputChars) {
  const char* json = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1},
      "unk_token": "[UNK]",
      "max_input_chars_per_word": -5
    },
    "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}]
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(WordPieceJsonFactoryTest, RejectsZeroMaxInputChars) {
  const char* json = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1},
      "unk_token": "[UNK]",
      "max_input_chars_per_word": 0
    },
    "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}]
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
  iree_status_free(status);
}

TEST(WordPieceJsonFactoryTest, ParsesMaxInputCharsFromJson) {
  const char* json = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "hello": 1, "##world": 2},
      "unk_token": "[UNK]",
      "max_input_chars_per_word": 3
    },
    "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}]
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer));

  // "hello" has 5 chars, exceeds max of 3 -> [UNK]
  int32_t ids[10];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer, IREE_SV("hello"), options,
                                       ids, 10, &count));
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 0);  // [UNK]

  iree_tokenizer_free(tokenizer);
}

TEST(WordPieceJsonFactoryTest, ParsesPrefixFromJson) {
  const char* json = R"({
    "model": {
      "type": "WordPiece",
      "vocab": {"[UNK]": 0, "test": 1, "@@ing": 2},
      "unk_token": "[UNK]",
      "continuing_subword_prefix": "@@"
    },
    "added_tokens": [{"id": 0, "content": "[UNK]", "special": true}]
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer));

  // Verify vocab was imported correctly.
  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer);
  EXPECT_EQ(iree_tokenizer_vocab_capacity(vocab), 3u);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SV("[UNK]")), 0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SV("test")), 1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab, IREE_SV("@@ing")), 2);

  // "testing" with @@ prefix via tokenizer -> ["test", "@@ing"]
  int32_t ids[10];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer, IREE_SV("testing"), options,
                                       ids, 10, &count));
  EXPECT_EQ(count, 2u) << "Tokenizer encode";
  EXPECT_EQ(ids[0], 1) << "Tokenizer encode: test";
  EXPECT_EQ(ids[1], 2) << "Tokenizer encode: @@ing";

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// BPE with Sequence Pre-tokenizer Tests (CLIP-style)
//===----------------------------------------------------------------------===//

class BpeSequencePretokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_view_t json = GetTestFile("bpe_sequence_pretokenizer.json");
    ASSERT_GT(json.size, 0u);
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        json, iree_allocator_system(), &tokenizer_));
    ASSERT_NE(tokenizer_, nullptr);
  }

  void TearDown() override { iree_tokenizer_free(tokenizer_); }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_F(BpeSequencePretokenizerTest, LowercaseNormalizerApplied) {
  // Verifies that the Sequence normalizer is applied correctly when the
  // pre-tokenizer is also a Sequence. The normalizer should be applied once
  // by the first pre-tokenizer child, not re-applied by subsequent children.
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("HELLO"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // "HELLO" normalized to "hello", then "hello</w>" via end_of_word_suffix.
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 15);  // "hello</w>"
}

TEST_F(BpeSequencePretokenizerTest, EndOfWordSuffixApplied) {
  // Verifies end_of_word_suffix is appended before vocab lookup.
  // Input "a" becomes "a</w>" which maps to token 14.
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("a"), options, ids,
                                       IREE_ARRAYSIZE(ids), &count));

  // "a" + "</w>" = "a</w>" -> token 14 (not "a" -> token 13).
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 14);  // "a</w>"
}

TEST_F(BpeSequencePretokenizerTest, WholeWordLookupBeforeSplit) {
  // When end_of_word_suffix is set and the whole word+suffix exists in vocab,
  // it should be returned as a single token without character splitting.
  int32_t ids[10];
  iree_host_size_t count = 0;

  iree_tokenizer_encode_options_t options = {0};
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer_, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // "hello" + "</w>" = "hello</w>" exists as token 15, returned directly.
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(ids[0], 15);  // "hello</w>"
}

//===----------------------------------------------------------------------===//
// Multi-Token Prefix Postprocessor Tests (Whisper-style)
//===----------------------------------------------------------------------===//

// Whisper and similar models use TemplateProcessing with multiple special
// tokens before the input sequence ($A). This tests that pattern:
//   <|startoftranscript|> <|notimestamps|> $A <|endoftext|>
//
// The special tokens are only in added_tokens (not in the main vocab), which
// matches how HuggingFace tokenizers store Whisper tokenizer.json files.
TEST(WhisperStylePostprocessorTest, MultiPrefixTokensEmitted) {
  const char* json = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "!": 0,
        "a": 1,
        "b": 2,
        "hello": 100,
        "world": 101
      },
      "merges": []
    },
    "added_tokens": [
      {"id": 50257, "content": "<|endoftext|>", "special": true},
      {"id": 50258, "content": "<|startoftranscript|>", "special": true},
      {"id": 50364, "content": "<|notimestamps|>", "special": true}
    ],
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "<|startoftranscript|>", "type_id": 0}},
        {"SpecialToken": {"id": "<|notimestamps|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
      ],
      "pair": null,
      "special_tokens": {
        "<|startoftranscript|>": {"id": "<|startoftranscript|>", "ids": [50258], "tokens": ["<|startoftranscript|>"]},
        "<|notimestamps|>": {"id": "<|notimestamps|>", "ids": [50364], "tokens": ["<|notimestamps|>"]},
        "<|endoftext|>": {"id": "<|endoftext|>", "ids": [50257], "tokens": ["<|endoftext|>"]}
      }
    }
  })";
  iree_string_view_t json_span = iree_make_string_view(json, strlen(json));

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      json_span, iree_allocator_system(), &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Verify postprocessor was parsed correctly.
  ASSERT_EQ(tokenizer->postprocessor.type,
            IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE);
  ASSERT_EQ(tokenizer->postprocessor.config.template_.single_count, 4u);

  // Check that ALL template pieces have valid token_ids.
  const iree_tokenizer_template_piece_t* pieces =
      tokenizer->postprocessor.config.template_.templates;

  EXPECT_EQ(pieces[0].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[0].token_id, 50258);  // <|startoftranscript|>

  EXPECT_EQ(pieces[1].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[1].token_id, 50364);  // <|notimestamps|>

  EXPECT_EQ(pieces[2].type, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A);

  EXPECT_EQ(pieces[3].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[3].token_id, 50257);  // <|endoftext|>

  // Encode "hello" with special tokens.
  int32_t ids[10];
  iree_host_size_t count = 0;
  iree_tokenizer_encode_options_t options = {
      /*flags=*/IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      /*max_length=*/0,
  };
  IREE_ASSERT_OK(iree_tokenizer_encode(tokenizer, IREE_SV("hello"), options,
                                       ids, IREE_ARRAYSIZE(ids), &count));

  // Expected: [<|startoftranscript|>, <|notimestamps|>, hello, <|endoftext|>]
  //         = [50258, 50364, 100, 50257]
  ASSERT_EQ(count, 4u) << "Should have 2 prefix + 1 text + 1 suffix tokens";
  EXPECT_EQ(ids[0], 50258);  // <|startoftranscript|>
  EXPECT_EQ(ids[1], 50364);  // <|notimestamps|>
  EXPECT_EQ(ids[2], 100);    // hello
  EXPECT_EQ(ids[3], 50257);  // <|endoftext|>

  iree_tokenizer_free(tokenizer);
}

}  // namespace
