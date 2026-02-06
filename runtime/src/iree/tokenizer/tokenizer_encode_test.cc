// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/model/unigram.h"
#include "iree/tokenizer/model/wordpiece.h"
#include "iree/tokenizer/normalizer/bert.h"
#include "iree/tokenizer/normalizer/lowercase.h"
#include "iree/tokenizer/normalizer/nfc.h"
#include "iree/tokenizer/normalizer/nfd.h"
#include "iree/tokenizer/normalizer/prepend.h"
#include "iree/tokenizer/normalizer/regex_replace.h"
#include "iree/tokenizer/normalizer/replace.h"
#include "iree/tokenizer/normalizer/sequence.h"
#include "iree/tokenizer/normalizer/strip.h"
#include "iree/tokenizer/normalizer/strip_accents.h"
#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/segmenter/bert.h"
#include "iree/tokenizer/segmenter/metaspace.h"
#include "iree/tokenizer/segmenter/punctuation.h"
#include "iree/tokenizer/segmenter/sequence.h"
#include "iree/tokenizer/segmenter/split.h"
#include "iree/tokenizer/segmenter/whitespace.h"
#include "iree/tokenizer/special_tokens.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/tokenizer_test_util.h"

namespace iree {
namespace tokenizer {
namespace {

using testing::BuildTokenizer;
using testing::CreateBertNormalizer;
using testing::CreateBertSegmenter;
using testing::CreateBPEModel;
using testing::CreateBPEModelIgnoreMerges;
using testing::CreatePunctuationSegmenter;
using testing::CreateWhitespaceSegmenter;
using testing::CreateWordPieceModel;
using testing::Encode;
using testing::ScopedBuilder;
using testing::ScopedVocab;
using testing::ScopedVocabBuilder;

//===----------------------------------------------------------------------===//
// Encode-Specific Helpers
//===----------------------------------------------------------------------===//

// Helper to encode multiple strings in batch.
// Returns a vector of token ID vectors, one per input string.
std::vector<std::vector<iree_tokenizer_token_id_t>> EncodeBatch(
    iree_tokenizer_t* tokenizer, const std::vector<std::string>& texts) {
  if (texts.empty()) {
    std::vector<iree_tokenizer_encode_batch_item_t> items;
    IREE_CHECK_OK(iree_tokenizer_encode_batch(
        tokenizer, items.data(), 0, IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_byte_span_empty(), iree_byte_span_empty(),
        iree_tokenizer_offset_run_list_empty()));
    return {};
  }

  // Allocate state and transform buffers.
  iree_host_size_t state_size = 0;
  IREE_CHECK_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);

  // Transform buffer: max text size * 2, minimum 1024.
  iree_host_size_t max_text_size = 0;
  for (const auto& text : texts) {
    max_text_size = std::max(max_text_size, text.size());
  }
  iree_host_size_t transform_size =
      std::max(max_text_size * 2, static_cast<size_t>(1024));
  std::vector<uint8_t> transform_buffer(transform_size);

  // Set up batch items with output buffers.
  std::vector<std::vector<iree_tokenizer_token_id_t>> all_token_ids(
      texts.size());
  std::vector<iree_tokenizer_encode_batch_item_t> items(texts.size());

  for (size_t i = 0; i < texts.size(); ++i) {
    all_token_ids[i].resize(256);  // Pre-allocate capacity.
    items[i].text = iree_make_string_view(texts[i].data(), texts[i].size());
    items[i].output = iree_tokenizer_make_token_output(
        all_token_ids[i].data(), NULL, NULL, all_token_ids[i].size());
    items[i].out_token_count = 0;
  }

  IREE_CHECK_OK(iree_tokenizer_encode_batch(
      tokenizer, items.data(), items.size(), IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty()));

  // Resize to actual token counts.
  for (size_t i = 0; i < texts.size(); ++i) {
    all_token_ids[i].resize(items[i].out_token_count);
  }

  return all_token_ids;
}

// Creates a split segmenter from a regex pattern.
iree_tokenizer_segmenter_t* CreateSplitSegmenter(const char* pattern) {
  iree_tokenizer_regex_dfa_t dfa;
  uint8_t* dfa_storage = nullptr;
  iree_tokenizer_regex_compile_error_t error = {0};
  iree_status_t status = iree_tokenizer_regex_compile_and_load(
      iree_make_cstring_view(pattern),
      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
      &dfa, &dfa_storage, &error);
  if (!iree_status_is_ok(status)) {
    return nullptr;
  }

  iree_tokenizer_segmenter_t* segmenter = nullptr;
  status = iree_tokenizer_segmenter_split_allocate(
      dfa, dfa_storage, IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED, false,
      iree_allocator_system(), &segmenter);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), dfa_storage);
    return nullptr;
  }

  return segmenter;
}

//===----------------------------------------------------------------------===//
// Test fixtures.
//===----------------------------------------------------------------------===//

class TokenizerEncodeTest : public ::testing::Test {};
class TokenizerIntegrationTest : public ::testing::Test {};
class TokenizerUTF8Test : public ::testing::Test {};
class TokenizerBatchEncodeTest : public ::testing::Test {};
class TokenizerStreamingTest : public ::testing::Test {};
class TokenizerPartialSegmentTest : public ::testing::Test {};
class TokenizerPostProcessorTest : public ::testing::Test {};

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(TokenizerEncodeTest, EmptyInput) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "");
  EXPECT_TRUE(tokens.empty());

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(TokenizerEncodeTest, SingleWord) {
  // Whole-word vocab without character tokens requires ignore_merges=true.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "hello");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Basic Functionality
//===----------------------------------------------------------------------===//

TEST_F(TokenizerEncodeTest, EncodeHelloWorld) {
  // Whole-word vocab without character tokens requires ignore_merges=true.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Encode "hello world" - should produce tokens [0, 1].
  auto tokens = Encode(tokenizer, "hello world");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "world"

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, EncodeMultipleWords) {
  // Whole-word vocab without character tokens requires ignore_merges=true.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "the");
  vocab_builder.AddToken(1, "quick");
  vocab_builder.AddToken(2, "brown");
  vocab_builder.AddToken(3, "fox");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "the quick brown fox");
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0], 0);  // "the"
  EXPECT_EQ(tokens[1], 1);  // "quick"
  EXPECT_EQ(tokens[2], 2);  // "brown"
  EXPECT_EQ(tokens[3], 3);  // "fox"

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, EncodeWithMerges) {
  // Test BPE merge behavior through the full pipeline.
  // Merges must form a complete path from characters to final token.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "h");
  vocab_builder.AddToken(1, "e");
  vocab_builder.AddToken(2, "l");
  vocab_builder.AddToken(3, "o");
  vocab_builder.AddToken(4, "he");
  vocab_builder.AddToken(5, "ll");
  vocab_builder.AddToken(6, "hel");
  vocab_builder.AddToken(7, "lo");
  vocab_builder.AddToken(8, "hello");

  // Working merge sequence for "hello":
  //   [h, e, l, l, o]
  //   rank 0: h+e -> [he, l, l, o]
  //   rank 1: l+o -> [he, l, lo]
  //   rank 2: he+l -> [hel, lo]
  //   rank 3: hel+lo -> [hello]
  vocab_builder.AddMerge(0, 1);  // h + e -> he (rank 0)
  vocab_builder.AddMerge(2, 3);  // l + o -> lo (rank 1)
  vocab_builder.AddMerge(4, 2);  // he + l -> hel (rank 2)
  vocab_builder.AddMerge(6, 7);  // hel + lo -> hello (rank 3)
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "hello" merges: [h,e,l,l,o] -> [he,l,l,o] -> [he,l,lo] -> [hel,lo] ->
  // [hello]
  auto tokens = Encode(tokenizer, "hello");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 8);  // "hello"

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Pipeline Behavior
//===----------------------------------------------------------------------===//

TEST_F(TokenizerEncodeTest, NullNormalizer) {
  // Tokenizer without normalizer should work (direct passthrough).
  // Whole-word vocab requires ignore_merges=true.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "test");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  // No normalizer set - should use NULL (passthrough).
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "test");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Output Buffer Handling
//===----------------------------------------------------------------------===//

TEST_F(TokenizerEncodeTest, BufferExactSize) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Buffer exactly fits 2 tokens.
  iree_tokenizer_token_id_t token_ids[2];
  iree_host_size_t token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode(
      tokenizer, iree_make_cstring_view("a b"), IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, 2),
      iree_allocator_system(), &token_count));

  EXPECT_EQ(token_count, 2u);
  EXPECT_EQ(token_ids[0], 0);
  EXPECT_EQ(token_ids[1], 1);

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Integration Patterns
//===----------------------------------------------------------------------===//

TEST_F(TokenizerIntegrationTest, WhitespaceSegmenterWithBPE) {
  // Full integration test: whitespace segmentation + BPE encoding.
  // Whole-word vocab requires ignore_merges=true.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  vocab_builder.AddToken(2, "!");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "hello world");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "world"

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerIntegrationTest, TrailingWhitespace) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "test");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "test ");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerIntegrationTest, LeadingWhitespace) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "test");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, " test");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerIntegrationTest, MultipleWhitespace) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "a    b");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);
  EXPECT_EQ(tokens[1], 1);

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// UTF-8 Boundary Handling
//===----------------------------------------------------------------------===//

TEST_F(TokenizerUTF8Test, MultiByteCharacterSplitAcrossChunks) {
  // Test that the tokenizer correctly handles UTF-8 multi-byte characters
  // that are split across chunk boundaries when using the streaming API.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "café");  // 4 visible chars, 5 bytes (é is 2 bytes)
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Allocate state storage.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  // Feed "café" in two chunks, splitting in the middle of the é (0xC3 0xA9).
  const char* full_text = "café";  // 5 bytes total: c a f 0xC3 0xA9
  iree_string_view_t chunk1 = iree_make_string_view(full_text, 4);  // "caf\xC3"

  std::vector<iree_tokenizer_token_id_t> tokens(16);
  iree_host_size_t total_tokens = 0;

  // Feed first chunk.
  iree_host_size_t bytes_consumed = 0;
  iree_host_size_t token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      state, chunk1,
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      &bytes_consumed, &token_count));
  EXPECT_EQ(bytes_consumed, 4u);
  total_tokens += token_count;

  // Feed continuation byte to complete the codepoint.
  iree_string_view_t chunk2 =
      iree_make_string_view(full_text + 4, 1);  // "\xA9"
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      state, chunk2,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &bytes_consumed, &token_count));
  EXPECT_EQ(bytes_consumed, 1u);
  total_tokens += token_count;

  // Finalize.
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &token_count));
  total_tokens += token_count;

  EXPECT_EQ(total_tokens, 1u);
  if (total_tokens >= 1) {
    EXPECT_EQ(tokens[0], 0);  // "café"
  }

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerUTF8Test, MultiCallFinalizeWithSmallCapacity) {
  // Test that finalize() works correctly when called multiple times with
  // small output capacity.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, "c");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  // Feed input but don't collect tokens during feed.
  const char* text = "a b c";
  iree_string_view_t chunk = iree_make_cstring_view(text);
  iree_tokenizer_token_id_t dummy_token;
  iree_host_size_t bytes_consumed = 0;
  iree_host_size_t token_count = 0;

  while (chunk.size > 0) {
    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(&dummy_token, NULL, NULL, 0),
        &bytes_consumed, &token_count));
    chunk.data += bytes_consumed;
    chunk.size -= bytes_consumed;
    if (bytes_consumed == 0) break;
  }

  // Finalize with capacity=1, collecting tokens one at a time.
  std::vector<iree_tokenizer_token_id_t> tokens;
  iree_tokenizer_token_id_t single_token;

  int iterations = 0;
  const int max_iterations = 100;
  while (iterations++ < max_iterations) {
    bool has_pending = iree_tokenizer_encode_state_has_pending(state);
    IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
        state, iree_tokenizer_make_token_output(&single_token, NULL, NULL, 1),
        &token_count));
    if (token_count > 0) {
      tokens.push_back(single_token);
    }
    if (!has_pending && token_count == 0) break;
  }

  EXPECT_LT(iterations, max_iterations) << "Finalize loop did not terminate";

  EXPECT_EQ(tokens.size(), 3u);
  if (tokens.size() >= 3) {
    EXPECT_EQ(tokens[0], 0);  // "a"
    EXPECT_EQ(tokens[1], 1);  // "b"
    EXPECT_EQ(tokens[2], 2);  // "c"
  }

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerUTF8Test, TransformBufferExhaustionMakesProgress) {
  // Tests that streaming encode with a tiny transform_buffer either returns an
  // error OR makes progress (bytes_consumed > 0).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, "c");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(8);  // Tiny buffer.

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  const char* text = "a b c a b c a b c";
  iree_string_view_t chunk = iree_make_cstring_view(text);
  std::vector<iree_tokenizer_token_id_t> tokens(64);
  iree_host_size_t bytes_consumed = 0;
  iree_host_size_t token_count = 0;

  int iterations = 0;
  const int max_iterations = 1000;
  while (chunk.size > 0 && iterations++ < max_iterations) {
    iree_status_t status = iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                         tokens.size()),
        &bytes_consumed, &token_count);

    if (iree_status_is_ok(status)) {
      EXPECT_GT(bytes_consumed, 0u)
          << "feed() returned OK with bytes_consumed=0 on non-empty input";
    } else {
      iree_status_free(status);
      break;
    }

    chunk.data += bytes_consumed;
    chunk.size -= bytes_consumed;
  }

  EXPECT_LT(iterations, max_iterations)
      << "Feed loop did not terminate - likely infinite loop bug";

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Batch Encode
//===----------------------------------------------------------------------===//

TEST_F(TokenizerBatchEncodeTest, EmptyBatch) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  std::vector<std::string> empty_texts;
  auto results = EncodeBatch(tokenizer, empty_texts);
  EXPECT_TRUE(results.empty());

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerBatchEncodeTest, SingleItem) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  std::vector<std::string> texts = {"hello world"};
  auto results = EncodeBatch(tokenizer, texts);

  ASSERT_EQ(results.size(), 1u);
  ASSERT_EQ(results[0].size(), 2u);
  EXPECT_EQ(results[0][0], 0);  // "hello"
  EXPECT_EQ(results[0][1], 1);  // "world"

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerBatchEncodeTest, MultipleItems) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  vocab_builder.AddToken(2, "test");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  std::vector<std::string> texts = {"hello", "world", "test"};
  auto results = EncodeBatch(tokenizer, texts);

  ASSERT_EQ(results.size(), 3u);

  ASSERT_EQ(results[0].size(), 1u);
  EXPECT_EQ(results[0][0], 0);  // "hello"

  ASSERT_EQ(results[1].size(), 1u);
  EXPECT_EQ(results[1][0], 1);  // "world"

  ASSERT_EQ(results[2].size(), 1u);
  EXPECT_EQ(results[2][0], 2);  // "test"

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerBatchEncodeTest, MixedLengths) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, "c");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  std::vector<std::string> texts = {"a", "a b", "a b c"};
  auto results = EncodeBatch(tokenizer, texts);

  ASSERT_EQ(results.size(), 3u);

  ASSERT_EQ(results[0].size(), 1u);
  EXPECT_EQ(results[0][0], 0);  // "a"

  ASSERT_EQ(results[1].size(), 2u);
  EXPECT_EQ(results[1][0], 0);  // "a"
  EXPECT_EQ(results[1][1], 1);  // "b"

  ASSERT_EQ(results[2].size(), 3u);
  EXPECT_EQ(results[2][0], 0);  // "a"
  EXPECT_EQ(results[2][1], 1);  // "b"
  EXPECT_EQ(results[2][2], 2);  // "c"

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerBatchEncodeTest, EmptyStrings) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "test");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  std::vector<std::string> texts = {"", "test", ""};
  auto results = EncodeBatch(tokenizer, texts);

  ASSERT_EQ(results.size(), 3u);
  EXPECT_TRUE(results[0].empty());
  ASSERT_EQ(results[1].size(), 1u);
  EXPECT_EQ(results[1][0], 0);  // "test"
  EXPECT_TRUE(results[2].empty());

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerBatchEncodeTest, MatchesSingleEncode) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "the");
  vocab_builder.AddToken(1, "quick");
  vocab_builder.AddToken(2, "brown");
  vocab_builder.AddToken(3, "fox");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  std::vector<std::string> texts = {"the quick", "brown fox",
                                    "the quick brown fox"};

  auto batch_results = EncodeBatch(tokenizer, texts);
  ASSERT_EQ(batch_results.size(), 3u);

  for (size_t i = 0; i < texts.size(); ++i) {
    auto single_result = Encode(tokenizer, texts[i].c_str());
    EXPECT_EQ(batch_results[i], single_result)
        << "Mismatch for text[" << i << "]: \"" << texts[i] << "\"";
  }

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Streaming with Regex Segmenter
//===----------------------------------------------------------------------===//

TEST_F(TokenizerStreamingTest, CJKTextWithSmallBuffer) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  const char* gpt2_pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter(gpt2_pattern);
  ASSERT_NE(segmenter, nullptr) << "Failed to create split segmenter";

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  const char* cjk_text =
      "今日は天気がいいですね。"
      "明天会更好吗？"
      "こんにちは世界！"
      "程序员写代码。"
      "テスト文字列です。"
      "人工智能很有趣。"
      "さようなら！";

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(1024);
  iree_host_size_t total_tokens = 0;
  iree_string_view_t remaining = iree_make_cstring_view(cjk_text);

  while (remaining.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    iree_status_t status = iree_tokenizer_encode_state_feed(
        state, remaining,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &token_count);

    IREE_ASSERT_OK(status) << "Feed failed at byte "
                           << (strlen(cjk_text) - remaining.size);

    remaining.data += bytes_consumed;
    remaining.size -= bytes_consumed;
    total_tokens += token_count;

    if (bytes_consumed == 0 && remaining.size > 0) {
      FAIL() << "No progress at byte " << (strlen(cjk_text) - remaining.size);
    }
  }

  iree_host_size_t final_tokens = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &final_tokens));
  total_tokens += final_tokens;

  EXPECT_GT(total_tokens, 0u) << "Expected tokens from CJK text";

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerStreamingTest, LongLetterRunWithoutPunctuation) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  const char* gpt2_pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter(gpt2_pattern);
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Build a string of 500 CJK characters (1500 bytes) with NO punctuation.
  std::string cjk_text;
  const char* cjk_chars[] = {
      "日", "本", "語", "中", "文", "字", "漢", "字", "言", "語",
      "今", "天", "明", "天", "昨", "天", "時", "間", "空", "間",
  };
  for (int i = 0; i < 500; ++i) {
    cjk_text += cjk_chars[i % 20];
  }
  ASSERT_EQ(cjk_text.length(), 1500u);

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(4096);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(4096);
  iree_host_size_t total_tokens = 0;
  iree_string_view_t remaining = {cjk_text.data(), cjk_text.length()};

  while (remaining.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    iree_status_t status = iree_tokenizer_encode_state_feed(
        state, remaining,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &token_count);

    IREE_ASSERT_OK(status) << "Feed failed at byte "
                           << (cjk_text.length() - remaining.size);

    remaining.data += bytes_consumed;
    remaining.size -= bytes_consumed;
    total_tokens += token_count;

    if (bytes_consumed == 0 && remaining.size > 0) {
      FAIL() << "No progress at byte " << (cjk_text.length() - remaining.size)
             << " - ring buffer should handle this input size";
    }
  }

  iree_host_size_t final_tokens = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &final_tokens));
  total_tokens += final_tokens;

  // With byte-level fallback, each CJK character (3 bytes) produces 3 tokens.
  EXPECT_EQ(total_tokens, 1500u)
      << "Expected 1500 byte-level tokens from 500 CJK characters";

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerStreamingTest, LargeCJKTextWithCappedBuffer) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  const char* gpt2_pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter(gpt2_pattern);
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Build ~100KB of CJK text with occasional punctuation.
  std::string cjk_text;
  const char* cjk_words[] = {
      "日本語",
      "中文字",
      "漢字言語",
      "今天明天",
  };
  const char* punctuation[] = {"。", "、", "！", "？"};

  for (int i = 0; i < 10000; ++i) {
    cjk_text += cjk_words[i % 4];
    if ((i + 1) % 5 == 0) {
      cjk_text += punctuation[i % 4];
    }
  }
  ASSERT_GT(cjk_text.size(), 90000u);

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);

  iree_host_size_t buffer_size =
      iree_tokenizer_transform_buffer_recommended_size(cjk_text.size());
  std::vector<uint8_t> transform_buffer(buffer_size);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(cjk_text.size());
  iree_host_size_t total_tokens = 0;
  iree_string_view_t remaining = {cjk_text.data(), cjk_text.length()};

  while (remaining.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    iree_status_t status = iree_tokenizer_encode_state_feed(
        state, remaining,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &token_count);

    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      FAIL() << "Streaming encode failed at byte "
             << (cjk_text.length() - remaining.size);
    }

    remaining.data += bytes_consumed;
    remaining.size -= bytes_consumed;
    total_tokens += token_count;

    if (bytes_consumed == 0 && remaining.size > 0) {
      FAIL() << "Ring buffer stalled at byte "
             << (cjk_text.length() - remaining.size);
    }
  }

  iree_host_size_t final_tokens = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &final_tokens));
  total_tokens += final_tokens;

  EXPECT_GT(total_tokens, 0u);

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerStreamingTest, ChunkedEncodeWithUTF8BoundarySplits) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  const char* gpt2_pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter(gpt2_pattern);
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Build CJK text with 3-byte characters. 1KB chunks will split characters.
  std::string cjk_text;
  const char* cjk_words[] = {"中文", "日本", "韓國", "漢字"};
  const char* punctuation[] = {"。", "、"};
  for (int i = 0; i < 500; ++i) {
    cjk_text += cjk_words[i % 4];
    if ((i + 1) % 10 == 0) {
      cjk_text += punctuation[i % 2];
    }
  }
  ASSERT_GT(cjk_text.size(), 3000u);

  const size_t chunk_size = 1024;
  iree_host_size_t total_tokens = 0;
  std::vector<iree_tokenizer_token_id_t> tokens(cjk_text.size());

  for (size_t offset = 0; offset < cjk_text.size(); offset += chunk_size) {
    size_t length = std::min(chunk_size, cjk_text.size() - offset);
    iree_host_size_t token_count = 0;
    iree_status_t status = iree_tokenizer_encode(
        tokenizer, iree_make_string_view(cjk_text.data() + offset, length),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        iree_allocator_system(), &token_count);
    if (!iree_status_is_ok(status)) {
      std::cerr << "Chunk at offset " << offset << " (length " << length
                << ") failed:\n";
      iree_status_fprint(stderr, status);
      FAIL() << "One-shot encode should handle chunks with split UTF-8";
    }
    total_tokens += token_count;
  }

  EXPECT_GT(total_tokens, 0u);
  iree_tokenizer_free(tokenizer);
}

// Streaming with minimal buffer and varied chunk sizes. This exercises ring
// buffer position tracking under stress: small ring buffer (256 bytes logical),
// data-dependent chunk sizes, and invalid UTF-8 byte patterns. The ring buffer
// must correctly track read_position, segment_position, and write_position
// invariants even when the model's reclaim mechanism advances positions.
TEST_F(TokenizerStreamingTest, SmallBufferDataDependentChunks) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Build test input with mix of ASCII and high bytes (simulates UTF-8 and
  // invalid sequences). This pattern stresses partial segment handling.
  std::string input;
  for (int i = 0; i < 500; ++i) {
    if (i % 7 == 0) {
      // High bytes that look like UTF-8 continuation bytes.
      input += static_cast<char>(0x80 + (i % 64));
    } else if (i % 13 == 0) {
      // Whitespace to create segment boundaries.
      input += ' ';
    } else {
      input += static_cast<char>('a' + (i % 26));
    }
  }

  // Use minimal ring buffer: 256-byte allocation = 128 bytes logical capacity.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(256);  // 128 bytes logical.

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(1024);
  iree_host_size_t total_tokens = 0;
  iree_host_size_t offset = 0;
  size_t chunk_index = 0;

  while (offset < input.size()) {
    // Data-dependent chunk size (1-128 bytes) to stress varied patterns.
    size_t chunk_base =
        static_cast<uint8_t>(input[chunk_index % input.size()]) & 0x7F;
    size_t chunk_size = ((chunk_base % 16) + 1) * 8;
    if (chunk_size > input.size() - offset) {
      chunk_size = input.size() - offset;
    }

    iree_string_view_t chunk = {input.data() + offset, chunk_size};
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    iree_status_t status = iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &token_count);

    IREE_ASSERT_OK(status) << "Feed failed at offset " << offset;

    offset += bytes_consumed;
    total_tokens += token_count;
    ++chunk_index;

    if (bytes_consumed == 0 && token_count == 0 && offset < input.size()) {
      // Deadlock protection - with partial segment handling this shouldn't
      // happen, but the 256-byte buffer is very constrained.
      FAIL() << "No progress at offset " << offset;
    }
    if (chunk_index > 10000) {
      FAIL() << "Too many iterations";
    }
  }

  // Finalize.
  iree_host_size_t final_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &final_count));
  total_tokens += final_count;

  // Input bytes minus whitespace separators should produce tokens.
  // The whitespace segmenter removes spaces between segments.
  EXPECT_GT(total_tokens, input.size() / 2);

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Offset Tracking
//===----------------------------------------------------------------------===//

struct EncodeResult {
  std::vector<iree_tokenizer_token_id_t> token_ids;
  std::vector<iree_tokenizer_offset_t> offsets;
};

// Encodes text and returns both token IDs and byte offsets.
iree::StatusOr<EncodeResult> EncodeWithOffsets(
    iree_tokenizer_t* tokenizer, const char* text,
    iree_tokenizer_encode_flags_t flags = IREE_TOKENIZER_ENCODE_FLAG_NONE) {
  EncodeResult result;
  result.token_ids.resize(256);
  result.offsets.resize(256);
  iree_host_size_t token_count = 0;

  iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
      result.token_ids.data(), result.offsets.data(), NULL,
      result.token_ids.size());

  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode(tokenizer, iree_make_cstring_view(text), flags,
                            output, iree_allocator_system(), &token_count));

  result.token_ids.resize(token_count);
  result.offsets.resize(token_count);
  return result;
}

TEST_F(TokenizerEncodeTest, OffsetsWholeWordSingleSegment) {
  // Whole-word match: entire segment maps to one token.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto result, EncodeWithOffsets(tokenizer, "hello"));
  ASSERT_EQ(result.token_ids.size(), 1u);
  EXPECT_EQ(result.token_ids[0], 0);
  // "hello" is bytes [0, 5).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 5u);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, OffsetsMultipleSegments) {
  // Multiple words → multiple segments, each segment maps to one token.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto result,
                            EncodeWithOffsets(tokenizer, "hello world"));
  ASSERT_EQ(result.token_ids.size(), 2u);
  EXPECT_EQ(result.token_ids[0], 0);  // "hello"
  EXPECT_EQ(result.token_ids[1], 1);  // "world"
  // Whitespace segmenter strips the leading space, so "world" starts at byte 6.
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 5u);
  EXPECT_EQ(result.offsets[1].start, 6u);
  EXPECT_EQ(result.offsets[1].end, 11u);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, OffsetsBPEMergesInSegment) {
  // BPE merges within a segment: backtracking produces multiple tokens,
  // each with sub-byte-ranges within the segment.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, "c");
  vocab_builder.AddToken(3, "d");
  vocab_builder.AddToken(4, "ab");
  vocab_builder.AddToken(5, "cd");
  vocab_builder.AddMerge(0, 1);  // a+b → ab
  vocab_builder.AddMerge(2, 3);  // c+d → cd
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "abcd" → tokens [ab, cd] with offsets [0:2, 2:4].
  IREE_ASSERT_OK_AND_ASSIGN(auto result, EncodeWithOffsets(tokenizer, "abcd"));
  ASSERT_EQ(result.token_ids.size(), 2u);
  EXPECT_EQ(result.token_ids[0], 4);  // "ab"
  EXPECT_EQ(result.token_ids[1], 5);  // "cd"
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 4u);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, OffsetsSingleByteTokens) {
  // No merges apply: each byte becomes its own token.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, "c");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "abc" → tokens [a, b, c] with offsets [0:1, 1:2, 2:3].
  IREE_ASSERT_OK_AND_ASSIGN(auto result, EncodeWithOffsets(tokenizer, "abc"));
  ASSERT_EQ(result.token_ids.size(), 3u);
  EXPECT_EQ(result.token_ids[0], 0);
  EXPECT_EQ(result.token_ids[1], 1);
  EXPECT_EQ(result.token_ids[2], 2);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 1u);
  EXPECT_EQ(result.offsets[1].start, 1u);
  EXPECT_EQ(result.offsets[1].end, 2u);
  EXPECT_EQ(result.offsets[2].start, 2u);
  EXPECT_EQ(result.offsets[2].end, 3u);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, OffsetsEmptyInput) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto result, EncodeWithOffsets(tokenizer, ""));
  EXPECT_TRUE(result.token_ids.empty());
  EXPECT_TRUE(result.offsets.empty());

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, OffsetsChainedMerges) {
  // Multi-level merges: a+b→ab, ab+c→abc. The final merged token covers all
  // bytes of the original characters.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, "c");
  vocab_builder.AddToken(3, "d");
  vocab_builder.AddToken(4, "ab");
  vocab_builder.AddToken(5, "abc");
  vocab_builder.AddMerge(0, 1);  // a+b → ab
  vocab_builder.AddMerge(4, 2);  // ab+c → abc
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "abcd" → tokens [abc, d] with offsets [0:3, 3:4].
  IREE_ASSERT_OK_AND_ASSIGN(auto result, EncodeWithOffsets(tokenizer, "abcd"));
  ASSERT_EQ(result.token_ids.size(), 2u);
  EXPECT_EQ(result.token_ids[0], 5);  // "abc"
  EXPECT_EQ(result.token_ids[1], 3);  // "d"
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 3u);
  EXPECT_EQ(result.offsets[1].start, 3u);
  EXPECT_EQ(result.offsets[1].end, 4u);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, OffsetsMultipleWordsWithMerges) {
  // Multiple segments, each with BPE merges. Offsets must be relative to the
  // original input, not the segment.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "h");
  vocab_builder.AddToken(1, "e");
  vocab_builder.AddToken(2, "l");
  vocab_builder.AddToken(3, "o");
  vocab_builder.AddToken(4, "w");
  vocab_builder.AddToken(5, "r");
  vocab_builder.AddToken(6, "d");
  vocab_builder.AddToken(7, "he");
  vocab_builder.AddToken(8, "ll");
  vocab_builder.AddToken(9, "or");
  vocab_builder.AddToken(10, "ld");
  vocab_builder.AddToken(11, "hell");
  vocab_builder.AddToken(12, "hello");
  vocab_builder.AddToken(13, "orld");
  vocab_builder.AddToken(14, "world");
  vocab_builder.AddMerge(0, 1);   // h+e → he
  vocab_builder.AddMerge(2, 2);   // l+l → ll
  vocab_builder.AddMerge(3, 5);   // o+r → or
  vocab_builder.AddMerge(2, 6);   // l+d → ld
  vocab_builder.AddMerge(7, 8);   // he+ll → hell
  vocab_builder.AddMerge(11, 3);  // hell+o → hello
  vocab_builder.AddMerge(9, 10);  // or+ld → orld
  vocab_builder.AddMerge(4, 13);  // w+orld → world
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "hello world" with whitespace segmenter → segments ["hello", "world"].
  // "hello" → token 12 (merges: h+e=he, l+l=ll, he+ll=hell, hell+o=hello)
  // "world" → token 14 (merges: o+r=or, l+d=ld, or+ld=orld, w+orld=world)
  IREE_ASSERT_OK_AND_ASSIGN(auto result,
                            EncodeWithOffsets(tokenizer, "hello world"));
  ASSERT_EQ(result.token_ids.size(), 2u);
  EXPECT_EQ(result.token_ids[0], 12);  // "hello"
  EXPECT_EQ(result.token_ids[1], 14);  // "world"
  // "hello" is bytes [0, 5), "world" is bytes [6, 11) (space at byte 5 stripped
  // by whitespace segmenter).
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 5u);
  EXPECT_EQ(result.offsets[1].start, 6u);
  EXPECT_EQ(result.offsets[1].end, 11u);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, OffsetsWithSplitSegmenter) {
  // Regex-based segmenter (like GPT-2 pattern) splitting on word boundaries.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, " ");
  vocab_builder.AddToken(3, "c");
  vocab_builder.AddToken(4, "ab");
  vocab_builder.AddMerge(0, 1);  // a+b → ab
  ScopedVocab vocab = vocab_builder.Build();

  // Regex pattern that splits on whitespace (keeping it in the token).
  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter("\\s+");
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "ab c" with split-on-whitespace → segments ["ab", " ", "c"].
  IREE_ASSERT_OK_AND_ASSIGN(auto result, EncodeWithOffsets(tokenizer, "ab c"));
  ASSERT_EQ(result.token_ids.size(), 3u);
  EXPECT_EQ(result.token_ids[0], 4);  // "ab"
  EXPECT_EQ(result.token_ids[1], 2);  // " "
  EXPECT_EQ(result.token_ids[2], 3);  // "c"
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 3u);
  EXPECT_EQ(result.offsets[2].start, 3u);
  EXPECT_EQ(result.offsets[2].end, 4u);

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerEncodeTest, OffsetsNullDoesNotCrash) {
  // Passing NULL for token_offsets should not crash (backwards compatibility).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Standard Encode helper passes NULL for offsets.
  auto tokens = Encode(tokenizer, "hello");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Partial Segment Streaming (split=false ring buffer overflow)
//===----------------------------------------------------------------------===//

// Helper: streaming encode with explicit transform buffer size.
// Uses the streaming API directly so we control the ring buffer capacity.
// The logical capacity is buffer_size/2 (double-buffer mode).
iree::StatusOr<std::vector<iree_tokenizer_token_id_t>> EncodeWithBufferSize(
    iree_tokenizer_t* tokenizer, const std::string& text,
    iree_host_size_t buffer_size) {
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(buffer_size);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(text.size() * 2 + 64);
  iree_host_size_t total_tokens = 0;
  iree_string_view_t remaining = {text.data(), text.size()};

  while (remaining.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;
    iree_status_t status = iree_tokenizer_encode_state_feed(
        state, remaining,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &token_count);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_encode_state_deinitialize(state);
      return status;
    }
    remaining.data += bytes_consumed;
    remaining.size -= bytes_consumed;
    total_tokens += token_count;
    if (bytes_consumed == 0 && remaining.size > 0) {
      iree_tokenizer_encode_state_deinitialize(state);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "no progress at byte %" PRIhsz,
                              text.size() - remaining.size);
    }
  }

  iree_host_size_t final_tokens = 0;
  iree_status_t status = iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &final_tokens);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_encode_state_deinitialize(state);
    return status;
  }
  total_tokens += final_tokens;

  iree_tokenizer_encode_state_deinitialize(state);
  tokens.resize(total_tokens);
  return tokens;
}

// Creates a byte-level tokenizer with GPT2 regex pattern.
// Letter runs match as single segments. No merges, so each byte = one token.
iree_tokenizer_t* CreateByteLevelGPT2Tokenizer() {
  // Build vocab: 256 byte-level tokens.
  ScopedVocabBuilder vb(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vb.AddToken(i, byte_token);
  }
  ScopedVocab v = vb.Build();

  const char* gpt2_pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter(gpt2_pattern);
  if (!segmenter) return nullptr;

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(builder.get(), CreateBPEModel(v.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), v.release());

  return BuildTokenizer(builder.get());
}

TEST_F(TokenizerPartialSegmentTest, BasicActivation) {
  // A long letter run exceeds the tiny ring buffer's logical capacity (64B),
  // forcing the partial segment path to activate. With byte-level vocab and
  // no merges, output should be one token per byte.
  iree_tokenizer_t* tokenizer = CreateByteLevelGPT2Tokenizer();
  ASSERT_NE(tokenizer, nullptr);

  // 200 bytes of ASCII letters, no punctuation = one GPT2 \p{L}+ segment.
  std::string input(200, 'x');

  // Buffer size 128 = 64 bytes logical. Input (200B) exceeds this.
  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithBufferSize(tokenizer, input, 128));
  ASSERT_EQ(tokens.size(), 200u) << "Expected 200 byte-level tokens";
  for (size_t i = 0; i < tokens.size(); ++i) {
    EXPECT_EQ(tokens[i], static_cast<int32_t>('x'))
        << "Token " << i << " should be byte value of 'x'";
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, MatchesLargeBufferResult) {
  // The strongest correctness test: streaming with a tiny buffer (partial mode)
  // must produce identical tokens to one-shot encode with a huge buffer.
  iree_tokenizer_t* tokenizer = CreateByteLevelGPT2Tokenizer();
  ASSERT_NE(tokenizer, nullptr);

  // Mixed letters — varied byte values to catch position-dependent bugs.
  std::string input;
  for (int i = 0; i < 300; ++i) {
    input += static_cast<char>('a' + (i % 26));
  }

  // Large buffer: no partial mode (entire input fits in ring buffer).
  IREE_ASSERT_OK_AND_ASSIGN(auto large_tokens,
                            EncodeWithBufferSize(tokenizer, input, 65536));
  ASSERT_FALSE(large_tokens.empty());

  // Tiny buffer: forces partial mode (many reclaim cycles).
  IREE_ASSERT_OK_AND_ASSIGN(auto small_tokens,
                            EncodeWithBufferSize(tokenizer, input, 128));
  ASSERT_FALSE(small_tokens.empty());

  ASSERT_EQ(small_tokens.size(), large_tokens.size())
      << "Partial segment path produced different token count";
  for (size_t i = 0; i < small_tokens.size(); ++i) {
    EXPECT_EQ(small_tokens[i], large_tokens[i])
        << "Token mismatch at index " << i;
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, ShortInputDoesNotActivatePartialMode) {
  // Input shorter than the ring buffer should NOT enter partial mode.
  // This tests the chunk->size > 0 guard that prevents false activation.
  iree_tokenizer_t* tokenizer = CreateByteLevelGPT2Tokenizer();
  ASSERT_NE(tokenizer, nullptr);

  // 30 bytes, buffer 128 (64B logical). Input fits without overflow.
  std::string input(30, 'a');
  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithBufferSize(tokenizer, input, 128));
  ASSERT_EQ(tokens.size(), 30u);
  for (size_t i = 0; i < tokens.size(); ++i) {
    EXPECT_EQ(tokens[i], static_cast<int32_t>('a'));
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, MultipleFeedCyclesSmallBuffer) {
  // Exercises many reclaim cycles: 2KB input with 128B buffer requires ~30+
  // partial-reclaim rounds. Tests that byte position tracking stays consistent
  // across many reclaim operations.
  iree_tokenizer_t* tokenizer = CreateByteLevelGPT2Tokenizer();
  ASSERT_NE(tokenizer, nullptr);

  std::string input;
  for (int i = 0; i < 2048; ++i) {
    input += static_cast<char>('A' + (i % 26));
  }

  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithBufferSize(tokenizer, input, 128));
  ASSERT_EQ(tokens.size(), 2048u) << "All 2048 bytes should produce tokens";

  // Verify each token matches the expected byte value.
  for (size_t i = 0; i < tokens.size(); ++i) {
    EXPECT_EQ(tokens[i], static_cast<int32_t>('A' + (i % 26)))
        << "Token mismatch at index " << i;
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, WithBPEMerges) {
  // Tests that BPE merges fire correctly in the partial segment BYTE_LOOP path.
  // With merge a+b→ab, the pattern "ababab..." should produce merged tokens
  // even when processed across partial boundaries.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, "c");
  vocab_builder.AddToken(3, "ab");
  vocab_builder.AddMerge(0, 1);  // a+b → ab (rank 0)
  ScopedVocab vocab = vocab_builder.Build();

  const char* gpt2_pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter(gpt2_pattern);
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "ababab..." repeated 100 times = 200 bytes, one GPT2 segment.
  std::string input;
  for (int i = 0; i < 100; ++i) input += "ab";
  ASSERT_EQ(input.size(), 200u);

  // Large buffer: reference result.
  IREE_ASSERT_OK_AND_ASSIGN(auto large_tokens,
                            EncodeWithBufferSize(tokenizer, input, 65536));
  ASSERT_FALSE(large_tokens.empty());
  // Each "ab" pair should merge to token 3.
  EXPECT_EQ(large_tokens.size(), 100u) << "100 'ab' pairs → 100 merged tokens";
  for (size_t i = 0; i < large_tokens.size(); ++i) {
    EXPECT_EQ(large_tokens[i], 3) << "Token " << i << " should be 'ab' (3)";
  }

  // Tiny buffer: partial mode. Must produce identical result.
  IREE_ASSERT_OK_AND_ASSIGN(auto small_tokens,
                            EncodeWithBufferSize(tokenizer, input, 128));
  ASSERT_EQ(small_tokens.size(), large_tokens.size())
      << "Partial mode produced wrong count with merges";
  for (size_t i = 0; i < small_tokens.size(); ++i) {
    EXPECT_EQ(small_tokens[i], large_tokens[i])
        << "Merge mismatch at index " << i;
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, WithChainedMerges) {
  // Tests multi-level merges in partial mode: a+b→ab, ab+c→abc.
  // A string of "abcabc..." should produce "abc" tokens even when partial
  // boundaries split across merge constituents.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, "c");
  vocab_builder.AddToken(3, "ab");
  vocab_builder.AddToken(4, "abc");
  vocab_builder.AddMerge(0, 1);  // a+b → ab (rank 0)
  vocab_builder.AddMerge(3, 2);  // ab+c → abc (rank 1)
  ScopedVocab vocab = vocab_builder.Build();

  const char* gpt2_pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter(gpt2_pattern);
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "abcabcabc..." repeated 80 times = 240 bytes, one segment.
  std::string input;
  for (int i = 0; i < 80; ++i) input += "abc";
  ASSERT_EQ(input.size(), 240u);

  // Reference: large buffer.
  IREE_ASSERT_OK_AND_ASSIGN(auto large_tokens,
                            EncodeWithBufferSize(tokenizer, input, 65536));
  EXPECT_EQ(large_tokens.size(), 80u) << "80 'abc' groups → 80 merged tokens";

  // Partial mode: tiny buffer.
  IREE_ASSERT_OK_AND_ASSIGN(auto small_tokens,
                            EncodeWithBufferSize(tokenizer, input, 128));
  ASSERT_EQ(small_tokens.size(), large_tokens.size())
      << "Chained merges should match across partial boundaries";
  for (size_t i = 0; i < small_tokens.size(); ++i) {
    EXPECT_EQ(small_tokens[i], large_tokens[i])
        << "Chained merge mismatch at index " << i;
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, OutputBufferExhaustion) {
  // Tests behavior when the output token buffer fills mid-streaming.
  // With a capacity of only 10 tokens, the encode must still make progress
  // by reclaiming ring buffer bytes on each call.
  iree_tokenizer_t* tokenizer = CreateByteLevelGPT2Tokenizer();
  ASSERT_NE(tokenizer, nullptr);

  // 200 bytes, buffer 128 (forces partial), output capacity 10.
  std::string input(200, 'z');

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(128);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> all_tokens;
  iree_tokenizer_token_id_t batch[10];
  iree_string_view_t remaining = {input.data(), input.size()};

  int iterations = 0;
  while (remaining.size > 0 && iterations++ < 1000) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state, remaining,
        iree_tokenizer_make_token_output(batch, NULL, NULL, 10),
        &bytes_consumed, &token_count));
    for (iree_host_size_t i = 0; i < token_count; ++i) {
      all_tokens.push_back(batch[i]);
    }
    remaining.data += bytes_consumed;
    remaining.size -= bytes_consumed;
    if (bytes_consumed == 0 && token_count == 0 && remaining.size > 0) {
      FAIL() << "No progress at byte " << (input.size() - remaining.size);
    }
  }
  ASSERT_LT(iterations, 1000) << "Feed loop should terminate";

  // Finalize with small capacity too.
  int finalize_iterations = 0;
  while (finalize_iterations++ < 100) {
    iree_host_size_t token_count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
        state, iree_tokenizer_make_token_output(batch, NULL, NULL, 10),
        &token_count));
    for (iree_host_size_t i = 0; i < token_count; ++i) {
      all_tokens.push_back(batch[i]);
    }
    if (token_count == 0 && !iree_tokenizer_encode_state_has_pending(state)) {
      break;
    }
  }

  EXPECT_EQ(all_tokens.size(), 200u)
      << "All bytes should eventually produce tokens";

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, VaryingBufferSizesProduceSameResult) {
  // Tests multiple buffer sizes (powers of two) all produce identical output.
  // This catches off-by-one errors in ring buffer wrap handling.
  iree_tokenizer_t* tokenizer = CreateByteLevelGPT2Tokenizer();
  ASSERT_NE(tokenizer, nullptr);

  std::string input;
  for (int i = 0; i < 500; ++i) {
    input += static_cast<char>('a' + (i % 26));
  }

  // Reference: large buffer (no partial mode).
  IREE_ASSERT_OK_AND_ASSIGN(auto reference,
                            EncodeWithBufferSize(tokenizer, input, 65536));
  ASSERT_EQ(reference.size(), 500u);

  // Test with progressively smaller buffers.
  iree_host_size_t buffer_sizes[] = {4096, 2048, 1024, 512, 256, 128, 64, 32};
  for (iree_host_size_t buffer_size : buffer_sizes) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeWithBufferSize(tokenizer, input, buffer_size));
    ASSERT_EQ(result.size(), reference.size())
        << "Buffer size " << buffer_size << " produced wrong token count";
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_EQ(result[i], reference[i])
          << "Mismatch at token " << i << " with buffer_size=" << buffer_size;
    }
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, MixedLettersAndPunctuation) {
  // Input with mixed segments: some trigger partial mode (long letter runs),
  // others are short enough to process normally. Tests transition between
  // partial and normal modes within one encode operation.
  iree_tokenizer_t* tokenizer = CreateByteLevelGPT2Tokenizer();
  ASSERT_NE(tokenizer, nullptr);

  // Pattern: 100 letters, then punctuation, then 100 letters, etc.
  // With buffer 128 (64B logical), the 100-letter runs trigger partial mode.
  // The punctuation creates segment boundaries, returning to normal mode.
  std::string input;
  for (int group = 0; group < 5; ++group) {
    for (int i = 0; i < 100; ++i) {
      input += static_cast<char>('a' + (i % 26));
    }
    input += ". ";  // Punctuation forces new segments.
  }

  IREE_ASSERT_OK_AND_ASSIGN(auto reference,
                            EncodeWithBufferSize(tokenizer, input, 65536));
  ASSERT_FALSE(reference.empty());

  IREE_ASSERT_OK_AND_ASSIGN(auto result,
                            EncodeWithBufferSize(tokenizer, input, 128));
  ASSERT_EQ(result.size(), reference.size())
      << "Mixed input: partial/normal transition mismatch";
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], reference[i]) << "Token mismatch at index " << i;
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, SingleByteChunkedFeed) {
  // Extreme: feed input one byte at a time with a tiny buffer.
  // Tests that partial mode handles interleaved feed/reclaim correctly
  // when each feed call provides minimal data.
  iree_tokenizer_t* tokenizer = CreateByteLevelGPT2Tokenizer();
  ASSERT_NE(tokenizer, nullptr);

  std::string input(100, 'm');

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(64);  // 32B logical capacity.

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(200);
  iree_host_size_t total_tokens = 0;

  // Feed one byte at a time.
  for (size_t byte_index = 0; byte_index < input.size(); ++byte_index) {
    iree_string_view_t one_byte = {input.data() + byte_index, 1};
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    int retries = 0;
    while (retries++ < 100) {
      IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
          state, one_byte,
          iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                           NULL, tokens.size() - total_tokens),
          &bytes_consumed, &token_count));
      total_tokens += token_count;
      if (bytes_consumed > 0) break;
      // If bytes_consumed == 0, output buffer was needed for reclaim.
      // Retry until the byte is consumed.
      if (token_count == 0) break;
    }
    ASSERT_GT(bytes_consumed, 0u)
        << "Failed to consume byte " << byte_index << " after retries";
  }

  // Finalize.
  iree_host_size_t final_tokens = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &final_tokens));
  total_tokens += final_tokens;

  EXPECT_EQ(total_tokens, 100u) << "Each byte should produce one token";

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerPartialSegmentTest, LargeInputWithMergesMatchesOneShot) {
  // 10KB input with BPE merges, comparing tiny-buffer streaming to one-shot.
  // This is the "all-spaces" scenario equivalent: pathological input that
  // creates one giant segment with merge-heavy content.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "b");
  vocab_builder.AddToken(2, "c");
  vocab_builder.AddToken(3, "d");
  vocab_builder.AddToken(4, "ab");
  vocab_builder.AddToken(5, "cd");
  vocab_builder.AddToken(6, "abcd");
  vocab_builder.AddMerge(0, 1);  // a+b → ab (rank 0)
  vocab_builder.AddMerge(2, 3);  // c+d → cd (rank 1)
  vocab_builder.AddMerge(4, 5);  // ab+cd → abcd (rank 2)
  ScopedVocab vocab = vocab_builder.Build();

  const char* gpt2_pattern =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
      "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter(gpt2_pattern);
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "abcdabcd..." repeated 2500 times = 10KB, all one segment.
  std::string input;
  for (int i = 0; i < 2500; ++i) input += "abcd";
  ASSERT_EQ(input.size(), 10000u);

  // One-shot with large buffer.
  IREE_ASSERT_OK_AND_ASSIGN(auto reference,
                            EncodeWithBufferSize(tokenizer, input, 65536));
  ASSERT_FALSE(reference.empty());
  // Each "abcd" merges fully: a+b=ab, c+d=cd, ab+cd=abcd.
  EXPECT_EQ(reference.size(), 2500u) << "2500 'abcd' groups → 2500 tokens";

  // Streaming with tiny buffer (forces many partial cycles).
  IREE_ASSERT_OK_AND_ASSIGN(auto result,
                            EncodeWithBufferSize(tokenizer, input, 256));
  ASSERT_EQ(result.size(), reference.size())
      << "10KB input with merges: partial mode mismatch";
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], reference[i])
        << "10KB merge test: mismatch at token " << i;
  }

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Post-Processor Integration (ADD_SPECIAL_TOKENS)
//
// Verifies that the full encode pipeline correctly emits prefix/suffix special
// tokens when IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS is set.
// Ground truth expectations from HuggingFace tokenizers library.
//===----------------------------------------------------------------------===//

// Helper: encodes text with ADD_SPECIAL_TOKENS flag, returns token IDs.
static iree::StatusOr<std::vector<iree_tokenizer_token_id_t>>
EncodeWithSpecialTokens(iree_tokenizer_t* tokenizer, const char* text) {
  std::vector<iree_tokenizer_token_id_t> token_ids(256);
  iree_host_size_t token_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode(tokenizer, iree_make_cstring_view(text),
                            IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
                            iree_tokenizer_make_token_output(
                                token_ids.data(), NULL, NULL, token_ids.size()),
                            iree_allocator_system(), &token_count));
  token_ids.resize(token_count);
  return token_ids;
}

// Helper: encodes text with ADD_SPECIAL_TOKENS and type_ids output.
static iree::StatusOr<std::vector<iree_tokenizer_token_id_t>>
EncodeWithSpecialTokensAndTypeIds(iree_tokenizer_t* tokenizer, const char* text,
                                  std::vector<uint8_t>* out_type_ids) {
  std::vector<iree_tokenizer_token_id_t> token_ids(256);
  out_type_ids->resize(256);
  iree_host_size_t token_count = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode(
      tokenizer, iree_make_cstring_view(text),
      IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      iree_tokenizer_make_token_output(token_ids.data(), NULL,
                                       out_type_ids->data(), token_ids.size()),
      iree_allocator_system(), &token_count));
  token_ids.resize(token_count);
  out_type_ids->resize(token_count);
  return token_ids;
}

// Builds a tokenizer with the given postprocessor template.
// Vocab: tokens 0..vocab_size-1 mapped to single ASCII characters 'a'+i.
// Special tokens get IDs specified in the template.
static iree_tokenizer_t* BuildTokenizerWithPostprocessor(
    const iree_tokenizer_postprocessor_t& postprocessor,
    iree_host_size_t vocab_size = 10) {
  ScopedVocabBuilder vocab_builder;
  // Single-character tokens for 'a'..'a'+vocab_size.
  for (iree_host_size_t i = 0; i < vocab_size; ++i) {
    char token_str[2] = {(char)('a' + i), '\0'};
    vocab_builder.AddToken(static_cast<iree_tokenizer_token_id_t>(i),
                           token_str);
  }
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_postprocessor(builder.get(), postprocessor);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  return BuildTokenizer(builder.get());
}

// RobertaProcessing: prefix=<s>(id=0), suffix=</s>(id=2).
// HuggingFace: encode("Hello") → [0, ..., 2]
TEST_F(TokenizerPostProcessorTest, RobertaPrefixAndSuffix) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  postprocessor.single.prefix_count = 1;
  postprocessor.single.suffix_count = 1;
  postprocessor.single.token_ids[0] = 100;  // BOS prefix
  postprocessor.single.token_ids[1] = 200;  // EOS suffix

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  // "ab" → BPE produces [0, 1], post-processor wraps: [100, 0, 1, 200].
  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithSpecialTokens(tokenizer, "ab"));
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0], 100);  // BOS prefix
  EXPECT_EQ(tokens[1], 0);    // 'a'
  EXPECT_EQ(tokens[2], 1);    // 'b'
  EXPECT_EQ(tokens[3], 200);  // EOS suffix

  iree_tokenizer_free(tokenizer);
}

// TemplateProcessing with prefix only (Mistral/LLaMA 2 style): prefix=<s>(1).
// HuggingFace: encode("Hello") → [1, ...]
TEST_F(TokenizerPostProcessorTest, PrefixOnlyNoSuffix) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  postprocessor.single.prefix_count = 1;
  postprocessor.single.suffix_count = 0;
  postprocessor.single.token_ids[0] = 50;  // BOS prefix, no suffix

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithSpecialTokens(tokenizer, "abc"));
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0], 50);  // BOS prefix
  EXPECT_EQ(tokens[1], 0);   // 'a'
  EXPECT_EQ(tokens[2], 1);   // 'b'
  EXPECT_EQ(tokens[3], 2);   // 'c'

  iree_tokenizer_free(tokenizer);
}

// Multi-token prefix + suffix (Whisper style):
// prefix = [<|startoftranscript|>, <|notimestamps|>], suffix = [<|endoftext|>]
// HuggingFace: encode("Hello") → [50258, 50363, ..., 50257]
TEST_F(TokenizerPostProcessorTest, MultiTokenPrefixAndSuffix) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  postprocessor.single.prefix_count = 2;
  postprocessor.single.suffix_count = 1;
  postprocessor.single.token_ids[0] = 500;  // First prefix token
  postprocessor.single.token_ids[1] = 501;  // Second prefix token
  postprocessor.single.token_ids[2] = 999;  // Suffix token

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithSpecialTokens(tokenizer, "ab"));
  ASSERT_EQ(tokens.size(), 5u);
  EXPECT_EQ(tokens[0], 500);  // First prefix
  EXPECT_EQ(tokens[1], 501);  // Second prefix
  EXPECT_EQ(tokens[2], 0);    // 'a'
  EXPECT_EQ(tokens[3], 1);    // 'b'
  EXPECT_EQ(tokens[4], 999);  // Suffix

  iree_tokenizer_free(tokenizer);
}

// Suffix only (OPT style): suffix=</s>(2), no prefix.
// HuggingFace: encode("Hello") → [..., 2]
TEST_F(TokenizerPostProcessorTest, SuffixOnlyNoPrefix) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  postprocessor.single.prefix_count = 0;
  postprocessor.single.suffix_count = 1;
  postprocessor.single.token_ids[0] = 77;  // Suffix only

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithSpecialTokens(tokenizer, "ab"));
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);   // 'a'
  EXPECT_EQ(tokens[1], 1);   // 'b'
  EXPECT_EQ(tokens[2], 77);  // EOS suffix

  iree_tokenizer_free(tokenizer);
}

// Empty input still gets special tokens.
// HuggingFace: roberta.encode("") → [0, 2] = [<s>, </s>]
TEST_F(TokenizerPostProcessorTest, EmptyInputGetsSpecialTokens) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  postprocessor.single.prefix_count = 1;
  postprocessor.single.suffix_count = 1;
  postprocessor.single.token_ids[0] = 100;  // BOS
  postprocessor.single.token_ids[1] = 200;  // EOS

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithSpecialTokens(tokenizer, ""));
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 100);  // BOS prefix
  EXPECT_EQ(tokens[1], 200);  // EOS suffix

  iree_tokenizer_free(tokenizer);
}

// Without ADD_SPECIAL_TOKENS flag, no special tokens are added even when
// postprocessor is configured.
TEST_F(TokenizerPostProcessorTest, NoFlagMeansNoSpecialTokens) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  postprocessor.single.prefix_count = 1;
  postprocessor.single.suffix_count = 1;
  postprocessor.single.token_ids[0] = 100;  // BOS
  postprocessor.single.token_ids[1] = 200;  // EOS

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  // Use Encode() which passes ENCODE_FLAG_NONE.
  auto tokens = Encode(tokenizer, "ab");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // 'a' only
  EXPECT_EQ(tokens[1], 1);  // 'b' only

  iree_tokenizer_free(tokenizer);
}

// Type IDs are assigned correctly to both special and content tokens.
// RoBERTa: all type_ids = 0 for both special and content tokens.
TEST_F(TokenizerPostProcessorTest, TypeIdsAssignedCorrectly) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  postprocessor.single.prefix_count = 1;
  postprocessor.single.suffix_count = 1;
  postprocessor.single.sequence_a_type_id = 0;
  postprocessor.single.token_ids[0] = 100;  // BOS
  postprocessor.single.token_ids[1] = 200;  // EOS
  postprocessor.single.type_ids[0] = 0;     // BOS type_id
  postprocessor.single.type_ids[1] = 0;     // EOS type_id

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  std::vector<uint8_t> type_ids;
  IREE_ASSERT_OK_AND_ASSIGN(auto tokens, EncodeWithSpecialTokensAndTypeIds(
                                             tokenizer, "ab", &type_ids));
  ASSERT_EQ(tokens.size(), 4u);
  ASSERT_EQ(type_ids.size(), 4u);
  EXPECT_EQ(type_ids[0], 0);  // BOS type_id
  EXPECT_EQ(type_ids[1], 0);  // 'a' sequence_a_type_id
  EXPECT_EQ(type_ids[2], 0);  // 'b' sequence_a_type_id
  EXPECT_EQ(type_ids[3], 0);  // EOS type_id

  iree_tokenizer_free(tokenizer);
}

// Non-zero type_ids for special tokens (BERT-style: suffix SEP gets type_id=1
// in pair mode, but for single we can still test non-zero special type_ids).
TEST_F(TokenizerPostProcessorTest, NonZeroSpecialTokenTypeIds) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  postprocessor.single.prefix_count = 1;
  postprocessor.single.suffix_count = 1;
  postprocessor.single.sequence_a_type_id = 0;
  postprocessor.single.token_ids[0] = 100;  // CLS
  postprocessor.single.token_ids[1] = 200;  // SEP
  postprocessor.single.type_ids[0] = 0;     // CLS type_id=0
  postprocessor.single.type_ids[1] = 1;     // SEP type_id=1

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  std::vector<uint8_t> type_ids;
  IREE_ASSERT_OK_AND_ASSIGN(auto tokens, EncodeWithSpecialTokensAndTypeIds(
                                             tokenizer, "ab", &type_ids));
  ASSERT_EQ(tokens.size(), 4u);
  ASSERT_EQ(type_ids.size(), 4u);
  EXPECT_EQ(type_ids[0], 0);  // CLS type_id
  EXPECT_EQ(type_ids[1], 0);  // 'a' sequence_a_type_id
  EXPECT_EQ(type_ids[2], 0);  // 'b' sequence_a_type_id
  EXPECT_EQ(type_ids[3], 1);  // SEP type_id (non-zero)

  iree_tokenizer_free(tokenizer);
}

// Multi-word input with whitespace segmentation: special tokens wrap the
// full encoded output across multiple segments.
TEST_F(TokenizerPostProcessorTest, MultiWordInputWithSpecialTokens) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  postprocessor.single.prefix_count = 1;
  postprocessor.single.suffix_count = 1;
  postprocessor.single.token_ids[0] = 100;  // BOS
  postprocessor.single.token_ids[1] = 200;  // EOS

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  // "a b c" with whitespace segmenter → segments ["a", "b", "c"] → tokens [0,
  // 1, 2] With special tokens: [100, 0, 1, 2, 200]
  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithSpecialTokens(tokenizer, "a b c"));
  ASSERT_EQ(tokens.size(), 5u);
  EXPECT_EQ(tokens[0], 100);  // BOS prefix
  EXPECT_EQ(tokens[1], 0);    // 'a'
  EXPECT_EQ(tokens[2], 1);    // 'b'
  EXPECT_EQ(tokens[3], 2);    // 'c'
  EXPECT_EQ(tokens[4], 200);  // EOS suffix

  iree_tokenizer_free(tokenizer);
}

// No postprocessor configured (zero-initialized) with ADD_SPECIAL_TOKENS flag:
// should be a no-op.
TEST_F(TokenizerPostProcessorTest, NoPostprocessorIsNoOp) {
  iree_tokenizer_postprocessor_t postprocessor = {};
  // All zeros: prefix_count=0, suffix_count=0.

  iree_tokenizer_t* tokenizer = BuildTokenizerWithPostprocessor(postprocessor);
  ASSERT_NE(tokenizer, nullptr);

  IREE_ASSERT_OK_AND_ASSIGN(auto tokens,
                            EncodeWithSpecialTokens(tokenizer, "ab"));
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // 'a'
  EXPECT_EQ(tokens[1], 1);  // 'b'

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Split Trailing Text: Validates reduced-consumed behavior through the encoder.
// These tests use narrow regex patterns (patterns that DON'T match all input)
// to exercise the code path where process() reduces consumed to last_emit_end,
// leaving trailing text for finalize(). This is the crash scenario that
// DeepSeek-V3 hit: a Sequence pretokenizer with a narrow-match Split child
// (e.g., digit-only pattern) on text input produces zero matches, leaving ALL
// input as trailing text.
//===----------------------------------------------------------------------===//

class TokenizerSplitTrailingTextTest : public ::testing::Test {};

// Creates a Split segmenter with specified behavior.
static iree_tokenizer_segmenter_t* CreateSplitSegmenterWithBehavior(
    const char* pattern, iree_tokenizer_regex_split_behavior_t behavior) {
  iree_tokenizer_regex_dfa_t dfa;
  uint8_t* dfa_storage = nullptr;
  iree_tokenizer_regex_compile_error_t error = {0};
  iree_status_t status = iree_tokenizer_regex_compile_and_load(
      iree_make_cstring_view(pattern),
      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
      &dfa, &dfa_storage, &error);
  if (!iree_status_is_ok(status)) {
    return nullptr;
  }

  iree_tokenizer_segmenter_t* segmenter = nullptr;
  status = iree_tokenizer_segmenter_split_allocate(
      dfa, dfa_storage, behavior, false, iree_allocator_system(), &segmenter);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), dfa_storage);
    return nullptr;
  }

  return segmenter;
}

// Narrow pattern with zero matches: the entire input is trailing text.
// This is the exact crash scenario: process() scans all input, finds no
// matches, and must NOT consume past last_emit_end (which stays at 0).
// Without the fix, finalize() would compute chunk_base = bytes_processed
// (= input.size), then try to emit a trailing gap with underflowed coordinates.
TEST_F(TokenizerSplitTrailingTextTest, NoMatchesAllTrailingText) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  // Digit-only pattern: won't match any letters or spaces.
  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter("\\d+");
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "hello world" has zero digit matches. The entire input is trailing text.
  auto tokens = Encode(tokenizer, "hello world");
  // Should produce byte-level tokens for each character.
  ASSERT_EQ(tokens.size(), 11u);  // "hello world" = 11 bytes
  EXPECT_EQ(tokens[0], static_cast<iree_tokenizer_token_id_t>('h'));
  EXPECT_EQ(tokens[10], static_cast<iree_tokenizer_token_id_t>('d'));

  iree_tokenizer_free(tokenizer);
}

// Partial matches: regex matches in the middle, trailing text at the end.
// Process emits segments for the matched portion, but must reduce consumed
// to last_emit_end so the trailing "def" goes to finalize.
TEST_F(TokenizerSplitTrailingTextTest, PartialMatchTrailingText) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  // Digit pattern: matches "123" in the middle.
  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter("\\d+");
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "abc123def": "abc" is gap before match, "123" is match, "def" is trailing.
  // With ISOLATED: segments are ["abc", "123", "def"].
  auto tokens = Encode(tokenizer, "abc123def");
  ASSERT_EQ(tokens.size(), 9u);  // 9 bytes total
  EXPECT_EQ(tokens[0], static_cast<iree_tokenizer_token_id_t>('a'));
  EXPECT_EQ(tokens[3], static_cast<iree_tokenizer_token_id_t>('1'));
  EXPECT_EQ(tokens[6], static_cast<iree_tokenizer_token_id_t>('d'));
  EXPECT_EQ(tokens[8], static_cast<iree_tokenizer_token_id_t>('f'));

  iree_tokenizer_free(tokenizer);
}

// Sequence[Split, Split]: first child has no matches, second child processes
// everything. This mirrors DeepSeek-V3's pretokenizer structure where the first
// Split child is a narrow pattern (digit sequences) and the second is a broader
// pattern (word boundaries). Input without digits means child0 produces zero
// matches and passes everything through to child1.
TEST_F(TokenizerSplitTrailingTextTest, SequenceNoMatchFirstChild) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  // child0: narrow digit pattern (ISOLATED) — no matches on text input.
  iree_tokenizer_segmenter_t* child0 = CreateSplitSegmenterWithBehavior(
      "\\d+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  ASSERT_NE(child0, nullptr);

  // child1: whitespace pattern (REMOVED) — splits on spaces.
  iree_tokenizer_segmenter_t* child1 = CreateSplitSegmenterWithBehavior(
      "\\s+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  ASSERT_NE(child1, nullptr);

  // Build Sequence[child0, child1].
  iree_tokenizer_segmenter_t* children[] = {child0, child1};
  iree_tokenizer_segmenter_t* sequence = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_sequence_allocate(
      children, 2, iree_allocator_system(), &sequence));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), sequence);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "hello world": child0 has no matches (no digits), passes everything to
  // child1 which splits on whitespace. Segments: ["hello", "world"].
  auto tokens = Encode(tokenizer, "hello world");
  ASSERT_EQ(tokens.size(), 10u);  // "hello" (5) + "world" (5), space removed
  EXPECT_EQ(tokens[0], static_cast<iree_tokenizer_token_id_t>('h'));
  EXPECT_EQ(tokens[4], static_cast<iree_tokenizer_token_id_t>('o'));
  EXPECT_EQ(tokens[5], static_cast<iree_tokenizer_token_id_t>('w'));
  EXPECT_EQ(tokens[9], static_cast<iree_tokenizer_token_id_t>('d'));

  iree_tokenizer_free(tokenizer);
}

// Sequence[Split, Split]: first child matches partially, second child refines.
// This tests the interaction where child0 produces segments for matched
// portions AND has trailing text, which child1 then further segments.
TEST_F(TokenizerSplitTrailingTextTest, SequencePartialMatchFirstChild) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  // child0: digit pattern (ISOLATED) — matches "42" in the middle.
  iree_tokenizer_segmenter_t* child0 = CreateSplitSegmenterWithBehavior(
      "\\d+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  ASSERT_NE(child0, nullptr);

  // child1: whitespace pattern (REMOVED) — splits on spaces.
  iree_tokenizer_segmenter_t* child1 = CreateSplitSegmenterWithBehavior(
      "\\s+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED);
  ASSERT_NE(child1, nullptr);

  // Build Sequence[child0, child1].
  iree_tokenizer_segmenter_t* children[] = {child0, child1};
  iree_tokenizer_segmenter_t* sequence = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_sequence_allocate(
      children, 2, iree_allocator_system(), &sequence));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), sequence);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "abc 42 def": child0 ISOLATED produces ["abc ", "42", " def"].
  // child1 REMOVED splits each on whitespace:
  //   "abc " → ["abc"] (trailing space removed)
  //   "42" → ["42"] (no spaces)
  //   " def" → ["def"] (leading space removed)
  auto tokens = Encode(tokenizer, "abc 42 def");
  ASSERT_EQ(tokens.size(), 8u);  // "abc" (3) + "42" (2) + "def" (3)
  EXPECT_EQ(tokens[0], static_cast<iree_tokenizer_token_id_t>('a'));
  EXPECT_EQ(tokens[2], static_cast<iree_tokenizer_token_id_t>('c'));
  EXPECT_EQ(tokens[3], static_cast<iree_tokenizer_token_id_t>('4'));
  EXPECT_EQ(tokens[4], static_cast<iree_tokenizer_token_id_t>('2'));
  EXPECT_EQ(tokens[5], static_cast<iree_tokenizer_token_id_t>('d'));
  EXPECT_EQ(tokens[7], static_cast<iree_tokenizer_token_id_t>('f'));

  iree_tokenizer_free(tokenizer);
}

// Streaming state API with narrow-match pattern: verifies the encoder's ring
// buffer correctly handles reduced consumed across multiple feed() calls.
// A large input with no matches exercises the complete path: process returns
// consumed=0, feed returns bytes_consumed=0, finalize handles everything.
TEST_F(TokenizerSplitTrailingTextTest, StreamingNoMatchLargeInput) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  // Digit-only pattern on a large text input.
  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter("\\d+");
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // 200 bytes of text with no digit matches.
  std::string text(200, 'x');

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(512);
  iree_host_size_t total_tokens = 0;
  iree_string_view_t remaining = {text.data(), text.length()};

  // Feed all data.
  iree_host_size_t bytes_consumed = 0;
  iree_host_size_t token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      state, remaining,
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      &bytes_consumed, &token_count));
  total_tokens += token_count;

  // Finalize to flush remaining.
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &finalize_count));
  total_tokens += finalize_count;

  // All 200 bytes should produce 200 byte-level tokens.
  EXPECT_EQ(total_tokens, 200u);
  EXPECT_EQ(tokens[0], static_cast<iree_tokenizer_token_id_t>('x'));
  EXPECT_EQ(tokens[199], static_cast<iree_tokenizer_token_id_t>('x'));

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Special Token Streaming Tests
//===----------------------------------------------------------------------===//

class TokenizerSpecialTokenStreamingTest : public ::testing::Test {};

// Tests that partial special token matches are correctly drained when the
// match ultimately fails. When draining, each byte must be emitted in order
// (not the first byte repeated).
//
// Example: input "<|xyz" starts matching "<|endoftext|>" for 2 bytes ("<|"),
// then fails at 'x'. The drain logic must emit '<' then '|' (the actual
// matched bytes), not '<' then '<' (first byte repeated).
TEST_F(TokenizerSpecialTokenStreamingTest, PartialMatchDrainsCorrectBytes) {
  // Create character-level vocab so we can verify exact token output.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  // Add special token with high ID.
  vocab_builder.AddToken(50256, "<|endoftext|>");
  ScopedVocab vocab = vocab_builder.Build();

  // Build special tokens collection.
  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("<|endoftext|>"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  // Build tokenizer with special tokens.
  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Allocate streaming state.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  // Input: "<|xyz" - starts matching "<|endoftext|>" but fails at 'x'.
  // Expected tokens: '<' (60), '|' (124), 'x' (120), 'y' (121), 'z' (122)
  // Bug behavior: '<' (60), '<' (60), 'x' (120), 'y' (121), 'z' (122)
  const char* input = "<|xyz";
  std::vector<iree_tokenizer_token_id_t> tokens(64);
  iree_host_size_t total_tokens = 0;

  // Feed byte-by-byte to trigger streaming partial match logic.
  for (size_t i = 0; i < strlen(input); ++i) {
    iree_string_view_t chunk = iree_make_string_view(input + i, 1);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &token_count));
    total_tokens += token_count;
  }

  // Finalize to flush any pending tokens.
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &finalize_count));
  total_tokens += finalize_count;

  tokens.resize(total_tokens);

  // Verify the correct bytes were drained.
  // Bug: would produce [60, 60, 120, 121, 122] (first byte '<' repeated)
  // Fix: should produce [60, 124, 120, 121, 122] (actual bytes '<|xyz')
  ASSERT_EQ(tokens.size(), 5u) << "Expected 5 tokens for '<|xyz'";
  EXPECT_EQ(tokens[0], 60) << "First byte should be '<' (60)";
  EXPECT_EQ(tokens[1], 124) << "Second byte should be '|' (124), not '<' (60)";
  EXPECT_EQ(tokens[2], 120) << "Third byte should be 'x' (120)";
  EXPECT_EQ(tokens[3], 121) << "Fourth byte should be 'y' (121)";
  EXPECT_EQ(tokens[4], 122) << "Fifth byte should be 'z' (122)";

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

// Tests that long special tokens (>64 bytes) work correctly without buffer
// overflow. Buffers must be 256 bytes to handle max-length special tokens.
TEST_F(TokenizerSpecialTokenStreamingTest, LongSpecialTokenNoOverflow) {
  // Create character-level vocab.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }

  // Create a long special token (100 bytes) - well above the old 64-byte limit.
  std::string long_token = "<|";
  for (int i = 0; i < 96; ++i) {
    long_token += 'x';
  }
  long_token += "|>";  // Total: 2 + 96 + 2 = 100 bytes
  ASSERT_EQ(long_token.size(), 100u);

  vocab_builder.AddToken(50256, long_token.c_str());
  ScopedVocab vocab = vocab_builder.Build();

  // Build special tokens collection with the long token.
  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, iree_make_string_view(long_token.data(), long_token.size()),
      50256, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  // Build tokenizer.
  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Allocate streaming state.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  // Input: long token prefix (98 bytes) + "yz" -> partial match fails.
  // This exercises the drain path with >64 bytes.
  std::string input = long_token.substr(0, 98) + "yz";
  std::vector<iree_tokenizer_token_id_t> tokens(256);
  iree_host_size_t total_tokens = 0;

  // Feed byte-by-byte.
  for (size_t i = 0; i < input.size(); ++i) {
    iree_string_view_t chunk = iree_make_string_view(input.data() + i, 1);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &token_count));
    total_tokens += token_count;
  }

  // Finalize.
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &finalize_count));
  total_tokens += finalize_count;

  tokens.resize(total_tokens);

  // The test passes if we don't crash. Also verify we got the right count.
  // 98 bytes from partial drain + 'y' + 'z' = 100 tokens.
  EXPECT_EQ(tokens.size(), 100u) << "Expected 100 tokens for 100 input bytes";

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

// Tests that finalize does not silently drop partial-match bytes when ring is
// full. If finalize can't write partial bytes, it must either return an error
// OR preserve the state for retry.
TEST_F(TokenizerSpecialTokenStreamingTest, FinalizeDoesNotSilentlyDropPartial) {
  // Create character-level vocab.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50256, "<|endoftext|>");
  ScopedVocab vocab = vocab_builder.Build();

  // Build special tokens.
  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("<|endoftext|>"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  // Build tokenizer.
  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Use a TINY transform buffer to make ring buffer full quickly.
  // Minimum is 256 bytes (128 logical capacity).
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(256);  // Minimal: 128 bytes logical.

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  // Feed input that creates a partial special token match ("<|end").
  // The partial bytes will need to be written during finalize.
  const char* partial_input = "<|end";
  std::vector<iree_tokenizer_token_id_t> tokens(64);
  iree_host_size_t total_tokens = 0;

  // Feed byte-by-byte to create partial match state.
  for (size_t i = 0; i < strlen(partial_input); ++i) {
    iree_string_view_t chunk = iree_make_string_view(partial_input + i, 1);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &token_count));
    total_tokens += token_count;
  }

  // At this point, has_pending should be true due to partial special token.
  EXPECT_TRUE(iree_tokenizer_encode_state_has_pending(state))
      << "Should have pending partial special token match";

  // Call finalize - it must NOT silently drop the partial bytes.
  // Either it succeeds and produces tokens, or returns an error.
  iree_host_size_t finalize_count = 0;
  iree_status_t status = iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &finalize_count);

  if (iree_status_is_ok(status)) {
    // Success path: finalize wrote the partial bytes as tokens.
    total_tokens += finalize_count;
    tokens.resize(total_tokens);

    // Should have 5 tokens for "<|end" -> '<', '|', 'e', 'n', 'd'.
    EXPECT_EQ(tokens.size(), 5u) << "Expected 5 tokens for '<|end'";
    if (tokens.size() >= 5) {
      EXPECT_EQ(tokens[0], 60);   // '<'
      EXPECT_EQ(tokens[1], 124);  // '|'
      EXPECT_EQ(tokens[2], 101);  // 'e'
      EXPECT_EQ(tokens[3], 110);  // 'n'
      EXPECT_EQ(tokens[4], 100);  // 'd'
    }
  } else {
    // Error path: finalize returned RESOURCE_EXHAUSTED because ring was full.
    // This is acceptable - it's NOT a silent failure.
    EXPECT_TRUE(iree_status_is_resource_exhausted(status))
        << "Expected RESOURCE_EXHAUSTED if finalize can't write partial";
    iree_status_free(status);

    // State should still have pending (partial not dropped).
    EXPECT_TRUE(iree_tokenizer_encode_state_has_pending(state))
        << "Partial should be preserved on error";
  }

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Strip Normalizer + Special Token Integration Tests
//===----------------------------------------------------------------------===//

class TokenizerStripNormalizerSpecialTokenTest : public ::testing::Test {};

// Strip normalizer with strip_right=true combined with special tokens requires
// that special token boundaries act as normalizer finalization points. Without
// this, the normalizer's lazy consumption (waiting to see if whitespace is
// trailing) conflicts with special token matching that limits normalizer input.
TEST_F(TokenizerStripNormalizerSpecialTokenTest,
       StripNormalizerWhitespaceBeforeSpecialToken) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50256, "[MASK]");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("[MASK]"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_allocate(
      /*strip_left=*/true, /*strip_right=*/true, iree_allocator_system(),
      &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Whitespace before special token must be stripped, not cause deadlock.
  auto result = Encode(tokenizer, "X [MASK]");
  EXPECT_THAT(result, ::testing::ElementsAre(88, 50256));

  auto result2 = Encode(tokenizer, "The [MASK] is blue.");
  EXPECT_FALSE(result2.empty());

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_free(tokenizer);
}

// Verify strip normalizer resets at_start across segment boundaries so that
// leading whitespace after a special token is correctly stripped.
TEST_F(TokenizerStripNormalizerSpecialTokenTest,
       StripNormalizerStateResetAcrossSegments) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50256, "[MASK]");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("[MASK]"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_allocate(
      /*strip_left=*/true, /*strip_right=*/true, iree_allocator_system(),
      &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "A [MASK]  B": trailing space after A stripped (segment end before [MASK]),
  // leading spaces before B stripped (at_start reset after segment end).
  // Expected: 'A'=65, [MASK]=50256, 'B'=66.
  auto result = Encode(tokenizer, "A [MASK]  B");
  EXPECT_THAT(result, ::testing::ElementsAre(65, 50256, 66));

  // " [MASK] ": all whitespace is leading/trailing around the special token.
  auto result2 = Encode(tokenizer, " [MASK] ");
  EXPECT_THAT(result2, ::testing::ElementsAre(50256));

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_free(tokenizer);
}

// Verify streaming API (feed/finalize) handles trailing whitespace without
// deadlocking. This exercises the retry-with-SEGMENT_END path in the pump.
TEST_F(TokenizerStripNormalizerSpecialTokenTest,
       StripNormalizerStreamingTrailingWhitespace) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_allocate(
      /*strip_left=*/true, /*strip_right=*/true, iree_allocator_system(),
      &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Use streaming API with trailing whitespace.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(64);
  iree_host_size_t total_tokens = 0;

  // Feed "hello   " - trailing whitespace should not deadlock.
  iree_host_size_t bytes_consumed = 0;
  iree_host_size_t token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      state, IREE_SV("hello   "),
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      &bytes_consumed, &token_count));
  total_tokens += token_count;

  // Feed should have consumed all input (retry-with-SEGMENT_END handles the
  // trailing whitespace).
  EXPECT_EQ(bytes_consumed, 8u);

  // Finalize to flush remaining tokens.
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &token_count));
  total_tokens += token_count;

  // Should produce tokens for "hello" (5 byte-level tokens).
  EXPECT_GT(total_tokens, 0u);

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

// Verify Strip normalizer works correctly without special tokens.
TEST_F(TokenizerStripNormalizerSpecialTokenTest,
       StripNormalizerWithoutSpecialTokens) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_allocate(
      /*strip_left=*/true, /*strip_right=*/true, iree_allocator_system(),
      &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Leading/trailing whitespace stripped by normalizer, intermediate by
  // whitespace segmenter.
  auto result = Encode(tokenizer, "  hello  world  ");
  EXPECT_FALSE(result.empty());

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// LSTRIP Flag Integration Tests
//===----------------------------------------------------------------------===//

class TokenizerLstripFlagTest : public ::testing::Test {};

// When a special token has the LSTRIP flag, the whitespace immediately
// preceding it should NOT be tokenized - it should be consumed/stripped
// along with the special token match. This is how RoBERTa/BART handle
// tokens like <mask> that have lstrip=True in their tokenizer.json.
TEST_F(TokenizerLstripFlagTest, LstripStripsPrecedingWhitespace) {
  // Create a byte-level style vocab where space (0x20) would normally
  // produce a token. This simulates GPT-2/RoBERTa where space becomes Ġ.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  // Add special token for <mask>
  vocab_builder.AddToken(50264, "<mask>");
  ScopedVocab vocab = vocab_builder.Build();

  // Create special token with LSTRIP flag
  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("<mask>"), 50264,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Input: "hello <mask>." - the space before <mask> should be stripped.
  // Without LSTRIP stripping: [h,e,l,l,o, ,<mask>,.] = 8 tokens (space = 32)
  // With LSTRIP stripping: [h,e,l,l,o,<mask>,.] = 7 tokens (no space token)
  auto result = Encode(tokenizer, "hello <mask>.");
  // Should NOT contain token 32 (space) immediately before 50264 (<mask>)
  EXPECT_THAT(result, ::testing::ElementsAre(104,    // 'h'
                                             101,    // 'e'
                                             108,    // 'l'
                                             108,    // 'l'
                                             111,    // 'o'
                                             50264,  // '<mask>'
                                             46      // '.'
                                             ));

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_free(tokenizer);
}

// Test LSTRIP with multiple spaces - all trailing whitespace should be stripped
TEST_F(TokenizerLstripFlagTest, LstripStripsMultipleSpaces) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50264, "<mask>");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("<mask>"), 50264,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "A   <mask>" - three spaces before <mask> should all be stripped
  auto result = Encode(tokenizer, "A   <mask>");
  EXPECT_THAT(result, ::testing::ElementsAre(65,    // 'A'
                                             50264  // '<mask>'
                                             ));

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_free(tokenizer);
}

// Test that without LSTRIP flag, whitespace IS tokenized
TEST_F(TokenizerLstripFlagTest, NoLstripPreservesWhitespace) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50264, "<mask>");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  // No LSTRIP flag
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("<mask>"), 50264,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Without LSTRIP, space should be tokenized (whitespace segmenter behavior)
  auto result = Encode(tokenizer, "A <mask>");
  // Whitespace segmenter strips the space, so we only get A and <mask>
  // The difference from LSTRIP is that LSTRIP is a special token behavior,
  // while here the whitespace segmenter is doing the stripping.
  EXPECT_THAT(result, ::testing::ElementsAre(65,    // 'A'
                                             50264  // '<mask>'
                                             ));

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// NFC Normalizer + Special Token Integration Tests
//===----------------------------------------------------------------------===//

class TokenizerNFCNormalizerSpecialTokenTest : public ::testing::Test {};

// NFC normalizer buffers combining character sequences internally until the
// next starter character arrives. When a pre-norm special token is matched
// right after a combining sequence, the pipeline_has_content check must
// account for the normalizer's pending buffered data — otherwise the special
// token is emitted before the normalizer's output, producing wrong token order.
//
// This bug only manifests with the streaming API: when the combining sequence
// and special token arrive in separate feed() calls, the SEGMENT_END flag
// (which forces NFC to flush) is NOT set because no special token start byte
// appears within the first chunk. The one-shot API sets SEGMENT_END because
// the special token is in the same chunk as the combining sequence.
TEST_F(TokenizerNFCNormalizerSpecialTokenTest,
       StreamingNFCPendingDataBeforeSpecialToken) {
  // Byte-level ASCII vocab + composed é + special token.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  // é (U+00E9) in UTF-8: 0xC3 0xA9. NFC composes e + combining acute to this.
  vocab_builder.AddToken(200, "\xC3\xA9");
  vocab_builder.AddToken(50256, "[MASK]");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("[MASK]"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_nfc_allocate(iree_allocator_system(),
                                                        &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Allocate streaming encode state.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(64);
  iree_host_size_t total_tokens = 0;

  // Feed 1: decomposed é (e + combining acute U+0301).
  // No special token start byte in this chunk, so safe_prefix covers the whole
  // chunk and SEGMENT_END is NOT set. NFC buffers e+U+0301 without flushing.
  iree_host_size_t bytes_consumed = 0;
  iree_host_size_t token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      state, IREE_SV("e\xCC\x81"),
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      &bytes_consumed, &token_count));
  total_tokens += token_count;
  EXPECT_EQ(bytes_consumed, 3u);

  // Feed 2: [MASK] special token.
  // Ring buffer is empty (NFC hasn't flushed). pipeline_has_content should
  // detect the normalizer's pending data, but the bug causes it to miss this
  // and emit [MASK] before é.
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      state, IREE_SV("[MASK]"),
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &bytes_consumed, &token_count));
  total_tokens += token_count;
  EXPECT_EQ(bytes_consumed, 6u);

  // Finalize to flush NFC pending data and emit remaining tokens.
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &token_count));
  total_tokens += token_count;

  // Expected: [200 (é), 50256 ([MASK])]
  // Bug:      [50256 ([MASK]), 200 (é)] — special token emitted before
  //           normalizer's pending data.
  tokens.resize(total_tokens);
  EXPECT_THAT(tokens, ::testing::ElementsAre(200, 50256));

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_free(tokenizer);
}

// When text precedes a combining sequence + special token (in streaming mode),
// the ring buffer has content from earlier starters (correctly deferring the
// special token). But when the ring drains, the deferred emission check must
// also wait for the normalizer's pending data to flush.
TEST_F(TokenizerNFCNormalizerSpecialTokenTest,
       StreamingNFCPendingDataWithPrecedingTextBeforeSpecialToken) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(200, "\xC3\xA9");
  vocab_builder.AddToken(50256, "[MASK]");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("[MASK]"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_nfc_allocate(iree_allocator_system(),
                                                        &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(64);
  iree_host_size_t total_tokens = 0;

  // Feed 1: "cafe" + decomposed é (e + combining acute U+0301).
  // NFC flushes c, a, f to ring (each is a starter that forces flush of the
  // previous sequence), but buffers e+U+0301 as the final combining sequence.
  iree_host_size_t bytes_consumed = 0;
  iree_host_size_t token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      state, IREE_SV("cafe\xCC\x81"),
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      &bytes_consumed, &token_count));
  total_tokens += token_count;
  EXPECT_EQ(bytes_consumed, 6u);

  // Feed 2: [MASK] special token.
  // Ring has "caf" so the token is correctly deferred. But when the ring
  // drains, try_emit_pending_special_token should also check the normalizer's
  // pending é before emitting.
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      state, IREE_SV("[MASK]"),
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &bytes_consumed, &token_count));
  total_tokens += token_count;
  EXPECT_EQ(bytes_consumed, 6u);

  // Finalize to flush NFC pending data and emit remaining tokens.
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &token_count));
  total_tokens += token_count;

  // Expected: [99(c), 97(a), 102(f), 200(é), 50256([MASK])]
  // Bug:      [99(c), 97(a), 102(f), 50256([MASK]), 200(é)] — deferred
  //           special token emitted before normalizer's pending é flushes.
  tokens.resize(total_tokens);
  EXPECT_THAT(tokens, ::testing::ElementsAre(99, 97, 102, 200, 50256));

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_free(tokenizer);
}

// One-shot API should also produce correct order. With the entire input in one
// chunk, SEGMENT_END is set before the special token boundary, forcing NFC to
// flush. This verifies the one-shot path is not regressed.
TEST_F(TokenizerNFCNormalizerSpecialTokenTest, OneShotNFCWithSpecialToken) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(200, "\xC3\xA9");
  vocab_builder.AddToken(50256, "[MASK]");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("[MASK]"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_nfc_allocate(iree_allocator_system(),
                                                        &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // One-shot: SEGMENT_END forces NFC to flush before special token match.
  auto result = Encode(tokenizer, "e\xCC\x81[MASK]");
  EXPECT_THAT(result, ::testing::ElementsAre(200, 50256));

  auto result2 = Encode(tokenizer, "cafe\xCC\x81[MASK]");
  EXPECT_THAT(result2, ::testing::ElementsAre(99, 97, 102, 200, 50256));

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Sequence Normalizer + Strip + Special Token Integration Tests
//===----------------------------------------------------------------------===//

class TokenizerSequenceStripSpecialTokenTest : public ::testing::Test {};

// Verifies that pre-norm special tokens are correctly emitted when a Sequence
// normalizer containing Strip(strip_right=true) has pending trailing
// whitespace.
//
// The Strip normalizer buffers trailing whitespace until it knows whether more
// non-whitespace content follows. When a pre-norm special token is matched
// (e.g., "[MASK]" after "The "), the tokenizer must finalize the normalizer
// chain to flush any buffered whitespace before emitting the special token.
//
// This configuration mirrors DeBERTa-v3 models which use:
//   Sequence(Replace, NFC, Strip(strip_right=true))
TEST_F(TokenizerSequenceStripSpecialTokenTest,
       SequenceStripPendingWhitespaceBeforePreNormSpecialToken) {
  // Byte-level ASCII vocab + special token.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50256, "[MASK]");
  ScopedVocab vocab = vocab_builder.Build();

  // Pre-norm special token [MASK] (matched on raw input before normalization).
  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("[MASK]"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  // Build a Sequence normalizer: NFC -> Strip(right).
  // This mimics DeBERTa's normalizer chain where Strip can hold pending
  // trailing whitespace.
  iree_tokenizer_normalizer_t* nfc = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_normalizer_nfc_allocate(iree_allocator_system(), &nfc));

  iree_tokenizer_normalizer_t* strip = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_allocate(
      /*strip_left=*/false, /*strip_right=*/true, iree_allocator_system(),
      &strip));

  iree_tokenizer_normalizer_t* children[] = {nfc, strip};
  iree_tokenizer_normalizer_t* sequence = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_sequence_allocate(
      children, 2, iree_allocator_system(), &sequence));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), sequence);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "The [MASK] is blue." - Strip buffers trailing space after "The" until it
  // knows if more content follows. The [MASK] match triggers normalizer flush.
  auto result = Encode(tokenizer, "The [MASK] is blue.");
  EXPECT_FALSE(result.empty());

  // Verify the special token is in the output.
  bool found_mask = false;
  for (auto t : result) {
    if (t == 50256) found_mask = true;
  }
  EXPECT_TRUE(found_mask) << "Pre-norm [MASK] should be found in output";

  // Simpler case: "X [MASK]" - just one character before the special token.
  auto result2 = Encode(tokenizer, "X [MASK]");
  EXPECT_FALSE(result2.empty());

  // "A [MASK] B" - verify context around special token is preserved.
  auto result3 = Encode(tokenizer, "A [MASK] B");
  EXPECT_THAT(result3, ::testing::Contains(50256));

  iree_tokenizer_special_tokens_deinitialize(&special_tokens);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// DeBERTa-v3-large Configuration Tests
//===----------------------------------------------------------------------===//

class TokenizerDebertaTest : public ::testing::Test {};

// Tests tokenization with DeBERTa-v3-large normalizer chain and leading
// whitespace.
//
// DeBERTa models use a specific normalizer configuration:
//   - RegexReplace: collapse multiple whitespace (\s{2,}|[\n\r\t] -> " ")
//   - NFC: Unicode normalization
//   - Strip: right-strip only
//   - Metaspace: space-to-▁ replacement and ▁ prepending
//
// The full normalizer chain (as constructed by tokenizer_json.c):
//   Seq(Seq(user_normalizer, metaspace_replace), metaspace_prepend)
//
// This test uses leading whitespace which exercises the regex normalizer's
// whitespace collapsing behavior combined with the Metaspace normalizers.
TEST_F(TokenizerDebertaTest, LeadingWhitespaceWithMetaspaceUnigram) {
  // Build minimal vocab for Unigram model that can tokenize "trimmed
  // whitespace" after normalization to "▁trimmed▁whitespace".
  ScopedVocabBuilder vocab_builder;

  // U+2581 LOWER ONE EIGHTH BLOCK = 0xE2 0x96 0x81 in UTF-8.
  static const char kMetaspace[] = "\xE2\x96\x81";

  // Token 0: UNK
  vocab_builder.AddTokenWithScore(0, "[UNK]", -10.0f);

  // Token 1: standalone metaspace (for cases where we get just "▁")
  vocab_builder.AddTokenWithScore(1, kMetaspace, -1.0f);

  // Token 2: "▁trimmed" - word with metaspace prefix
  std::string trimmed_token = std::string(kMetaspace) + "trimmed";
  vocab_builder.AddTokenWithScore(2, trimmed_token.c_str(), -2.0f);

  // Token 3: "▁whitespace" - word with metaspace prefix
  std::string whitespace_token = std::string(kMetaspace) + "whitespace";
  vocab_builder.AddTokenWithScore(3, whitespace_token.c_str(), -2.0f);

  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  // User-specified normalizer sequence: RegexReplace -> NFC -> Strip(right).
  // RegexReplace: collapse multiple whitespace and normalize control chars.
  iree_tokenizer_normalizer_t* regex_replace = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_regex_replace_allocate(
      IREE_SV("\\s{2,}|[\\n\\r\\t]"), IREE_SV(" "), iree_allocator_system(),
      &regex_replace));

  // NFC: Unicode normalization.
  iree_tokenizer_normalizer_t* nfc = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_normalizer_nfc_allocate(iree_allocator_system(), &nfc));

  // Strip: right-strip only.
  iree_tokenizer_normalizer_t* strip_right = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_allocate(
      /*strip_left=*/false, /*strip_right=*/true, iree_allocator_system(),
      &strip_right));

  iree_tokenizer_normalizer_t* user_children[] = {regex_replace, nfc,
                                                  strip_right};
  iree_tokenizer_normalizer_t* user_normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_sequence_allocate(
      user_children, 3, iree_allocator_system(), &user_normalizer));

  // Metaspace normalizers synthesized from pre-tokenizer config.
  // Replace: " " -> "▁".
  iree_tokenizer_normalizer_t* metaspace_replace = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_replace_allocate(
      IREE_SV(" "), iree_make_string_view(kMetaspace, 3),
      iree_allocator_system(), &metaspace_replace));

  // Prepend: prepend "▁" (skip if input already starts with "▁").
  iree_tokenizer_normalizer_t* metaspace_prepend = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_prepend_allocate(
      iree_make_string_view(kMetaspace, 3),
      /*skip_if_prefix_matches=*/true, iree_allocator_system(),
      &metaspace_prepend));

  // Chain: Seq(Seq(user_normalizer, metaspace_replace), metaspace_prepend).
  // This nesting matches tokenizer_json.c's incremental chaining.
  iree_tokenizer_normalizer_t* inner_children[] = {user_normalizer,
                                                   metaspace_replace};
  iree_tokenizer_normalizer_t* inner_seq = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_sequence_allocate(
      inner_children, 2, iree_allocator_system(), &inner_seq));
  iree_tokenizer_normalizer_t* outer_children[] = {inner_seq,
                                                   metaspace_prepend};
  iree_tokenizer_normalizer_t* full_normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_sequence_allocate(
      outer_children, 2, iree_allocator_system(), &full_normalizer));

  // Metaspace segmenter with split=true.
  iree_tokenizer_segmenter_t* metaspace_segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
      IREE_TOKENIZER_METASPACE_DEFAULT_REPLACEMENT, /*split_enabled=*/true,
      iree_allocator_system(), &metaspace_segmenter));

  // Unigram model.
  iree_tokenizer_model_t* unigram = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
      vocab.get(), /*unk_token_id=*/0, /*unk_score=*/-10.0f,
      IREE_TOKENIZER_UNIGRAM_FLAG_NONE, iree_allocator_system(), &unigram));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), full_normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(), metaspace_segmenter);
  iree_tokenizer_builder_set_model(builder.get(), unigram);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Input with leading whitespace exercises the full normalizer chain:
  // "  trimmed whitespace  " -> RegexReplace -> " trimmed whitespace "
  //   -> Strip(right) -> " trimmed whitespace" -> MetaspaceReplace ->
  //   "▁trimmed▁whitespace"
  // Expected tokens: [2, 3] for ["▁trimmed", "▁whitespace"].
  auto tokens = Encode(tokenizer, "  trimmed whitespace  ");
  EXPECT_THAT(tokens, ::testing::ElementsAre(2, 3));

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// P1: WordPiece Encode Tests
//===----------------------------------------------------------------------===//

class WordPieceEncodeTest : public ::testing::Test {};

TEST_F(WordPieceEncodeTest, BasicWordPiece) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "un");
  vocab_builder.AddToken(1, "##aff");
  vocab_builder.AddToken(2, "##able");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "un");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);

  iree_tokenizer_free(tokenizer);
}

TEST_F(WordPieceEncodeTest, SubwordSplit) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "un");
  vocab_builder.AddToken(1, "##aff");
  vocab_builder.AddToken(2, "##able");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "unaffable" -> greedy: "un" + "##aff" + "##able"
  auto tokens = Encode(tokenizer, "unaffable");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);  // "un"
  EXPECT_EQ(tokens[1], 1);  // "##aff"
  EXPECT_EQ(tokens[2], 2);  // "##able"

  iree_tokenizer_free(tokenizer);
}

TEST_F(WordPieceEncodeTest, UnknownWord) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "##b");
  vocab_builder.AddToken(2, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "xyz" has no decomposition -> entire word becomes UNK.
  auto tokens = Encode(tokenizer, "xyz");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 2);

  iree_tokenizer_free(tokenizer);
}

TEST_F(WordPieceEncodeTest, MaxCharsExceeded) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 1);
  ScopedVocab vocab = vocab_builder.Build();

  // Set max_input_chars_per_word to 10 for easy testing.
  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateWordPieceModel(vocab.get(), IREE_SV("##"),
                                          /*max_input_chars_per_word=*/10));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Word with 20 chars exceeds max_input_chars_per_word=10 -> UNK.
  auto tokens = Encode(tokenizer, "abcdefghijklmnopqrst");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 1);

  iree_tokenizer_free(tokenizer);
}

TEST_F(WordPieceEncodeTest, SingleChar) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "a");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);

  iree_tokenizer_free(tokenizer);
}

TEST_F(WordPieceEncodeTest, MultipleWords) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "un");
  vocab_builder.AddToken(1, "##aff");
  vocab_builder.AddToken(2, "##able");
  vocab_builder.AddToken(3, "able");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "un able" -> whitespace splits into ["un", "able"].
  // "un" -> [0], "able" -> [3] (whole-word match, not ##able).
  auto tokens = Encode(tokenizer, "un able");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "un"
  EXPECT_EQ(tokens[1], 3);  // "able"

  iree_tokenizer_free(tokenizer);
}

TEST_F(WordPieceEncodeTest, AllSubwords) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "h");
  vocab_builder.AddToken(1, "##e");
  vocab_builder.AddToken(2, "##l");
  vocab_builder.AddToken(3, "##o");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "hello" -> "h" + "##e" + "##l" + "##l" + "##o"
  auto tokens = Encode(tokenizer, "hello");
  ASSERT_EQ(tokens.size(), 5u);
  EXPECT_EQ(tokens[0], 0);  // "h"
  EXPECT_EQ(tokens[1], 1);  // "##e"
  EXPECT_EQ(tokens[2], 2);  // "##l"
  EXPECT_EQ(tokens[3], 2);  // "##l"
  EXPECT_EQ(tokens[4], 3);  // "##o"

  iree_tokenizer_free(tokenizer);
}

TEST_F(WordPieceEncodeTest, PartialFailMeansFullUnk) {
  // If any sub-token lookup fails, the entire word becomes UNK.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "##b");
  // No "##c" in vocab.
  vocab_builder.AddToken(2, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 2);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "abc" -> "a" matches, "##b" matches, but "##c" does not exist.
  // Any sub-token failure -> entire word becomes UNK.
  auto tokens = Encode(tokenizer, "abc");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 2);

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// P2: BERT Pipeline Tests
//===----------------------------------------------------------------------===//

class BERTPipelineTest : public ::testing::Test {};

TEST_F(BERTPipelineTest, UncasedBasic) {
  // BERT uncased: DEFAULT flags (lowercase + strip_accents + clean_text +
  // handle_chinese_chars) with whitespace segmenter + WordPiece.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), CreateBertNormalizer());
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "Hello World" -> BERT lowercases -> "hello world" -> tokenized.
  auto tokens = Encode(tokenizer, "Hello World");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "world"

  iree_tokenizer_free(tokenizer);
}

TEST_F(BERTPipelineTest, CasedMode) {
  // BERT cased: CLEAN_TEXT + HANDLE_CHINESE_CHARS only (no lowercase, no
  // strip_accents).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  vocab_builder.AddToken(1, "World");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(
      builder.get(),
      CreateBertNormalizer(
          IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT |
          IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS));
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Case is preserved in cased mode.
  auto tokens = Encode(tokenizer, "Hello World");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "Hello"
  EXPECT_EQ(tokens[1], 1);  // "World"

  iree_tokenizer_free(tokenizer);
}

TEST_F(BERTPipelineTest, PunctuationIsolation) {
  // BERT segmenter isolates punctuation into separate segments.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, ",");
  vocab_builder.AddToken(2, "world");
  vocab_builder.AddToken(3, "!");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), CreateBertNormalizer());
  iree_tokenizer_builder_set_segmenter(builder.get(), CreateBertSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "Hello, world!" -> normalize -> "hello, world!"
  // BERT segmenter: ["hello", ",", "world", "!"]
  auto tokens = Encode(tokenizer, "Hello, world!");
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // ","
  EXPECT_EQ(tokens[2], 2);  // "world"
  EXPECT_EQ(tokens[3], 3);  // "!"

  iree_tokenizer_free(tokenizer);
}

TEST_F(BERTPipelineTest, Apostrophe) {
  // BERT segmenter splits on apostrophe (punctuation).
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "don");
  vocab_builder.AddToken(1, "'");
  vocab_builder.AddToken(2, "t");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), CreateBertNormalizer());
  iree_tokenizer_builder_set_segmenter(builder.get(), CreateBertSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "don't" -> BERT seg: ["don", "'", "t"]
  auto tokens = Encode(tokenizer, "don't");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);  // "don"
  EXPECT_EQ(tokens[1], 1);  // "'"
  EXPECT_EQ(tokens[2], 2);  // "t"

  iree_tokenizer_free(tokenizer);
}

TEST_F(BERTPipelineTest, ChineseChars) {
  // BERT normalizer adds spaces around CJK ideographs so that the segmenter
  // treats each CJK character as a separate word-segment.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  // U+4E2D = "中" (3 bytes: E4 B8 AD).
  vocab_builder.AddToken(1, "\xE4\xB8\xAD");
  // U+6587 = "文" (3 bytes: E6 96 87).
  vocab_builder.AddToken(2, "\xE6\x96\x87");
  vocab_builder.AddToken(3, "world");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), CreateBertNormalizer());
  iree_tokenizer_builder_set_segmenter(builder.get(), CreateBertSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "Hello中文World" -> lowercase -> "hello中文world"
  // BERT normalizer spaces CJK -> "hello 中 文 world"
  // Segments: ["hello", "中", "文", "world"]
  auto tokens = Encode(tokenizer, "Hello\xE4\xB8\xAD\xE6\x96\x87World");
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "中"
  EXPECT_EQ(tokens[2], 2);  // "文"
  EXPECT_EQ(tokens[3], 3);  // "world"

  iree_tokenizer_free(tokenizer);
}

TEST_F(BERTPipelineTest, AccentStripping) {
  // BERT uncased DEFAULT includes STRIP_ACCENTS which removes combining marks
  // after NFD decomposition.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "cafe");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), CreateBertNormalizer());
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "café" (precomposed é) -> BERT NFD decompose + strip Mn marks -> "cafe"
  auto tokens = Encode(tokenizer, "caf\xC3\xA9");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "cafe"

  iree_tokenizer_free(tokenizer);
}

TEST_F(BERTPipelineTest, ControlCharRemoval) {
  // BERT CLEAN_TEXT flag removes control characters.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "helloworld");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), CreateBertNormalizer());
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "hello\x00world" -> null removed -> "helloworld"
  auto tokens = Encode(tokenizer, "hello\x01world");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "helloworld"

  iree_tokenizer_free(tokenizer);
}

TEST_F(BERTPipelineTest, FullPipelineEndToEnd) {
  // Complete BERT uncased pipeline: normalize + segment + WordPiece tokenize.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "bert");
  vocab_builder.AddToken(1, "'");
  vocab_builder.AddToken(2, "s");
  vocab_builder.AddToken(3, "model");
  vocab_builder.AddToken(4, "!");
  vocab_builder.AddToken(5, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 5);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), CreateBertNormalizer());
  iree_tokenizer_builder_set_segmenter(builder.get(), CreateBertSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "BERT's model!" -> lowercase -> "bert's model!"
  // BERT segmenter: ["bert", "'", "s", "model", "!"]
  auto tokens = Encode(tokenizer, "BERT's model!");
  ASSERT_EQ(tokens.size(), 5u);
  EXPECT_EQ(tokens[0], 0);  // "bert"
  EXPECT_EQ(tokens[1], 1);  // "'"
  EXPECT_EQ(tokens[2], 2);  // "s"
  EXPECT_EQ(tokens[3], 3);  // "model"
  EXPECT_EQ(tokens[4], 4);  // "!"

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// P3: ByteLevel BPE Encode Tests
//===----------------------------------------------------------------------===//

class ByteLevelBPEEncodeTest : public ::testing::Test {};

TEST_F(ByteLevelBPEEncodeTest, SpaceMapping) {
  // GPT-2 byte-level: space (0x20) maps to U+0120 = Ġ = "\xc4\xa0" in UTF-8.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xc4\xa0");  // Ġ = space byte
  vocab_builder.AddToken(1, "h");
  vocab_builder.AddToken(2, "i");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // " hi" -> byte-level maps space to Ġ -> [Ġ, h, i]
  // Note: whitespace segmenter strips leading space, so feed "hi" segment.
  // Actually with byte-level input the segmenter operates first, then bytes
  // are mapped. For the segment "hi": h -> h, i -> i (identity for printable).
  auto tokens = Encode(tokenizer, "hi");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 1);  // "h"
  EXPECT_EQ(tokens[1], 2);  // "i"

  iree_tokenizer_free(tokenizer);
}

TEST_F(ByteLevelBPEEncodeTest, ASCIIPrintable) {
  // Printable ASCII (0x21-0x7E) maps to identity in byte-level encoding.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "hello");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "hello" identity mapped

  iree_tokenizer_free(tokenizer);
}

TEST_F(ByteLevelBPEEncodeTest, WithMerges) {
  // Byte-level input + BPE merges interaction.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "h");
  vocab_builder.AddToken(1, "i");
  vocab_builder.AddToken(2, "hi");
  vocab_builder.AddMerge(0, 1);  // h + i -> hi
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "hi" -> byte-level identity for printable -> "hi" -> BPE merges h+i -> "hi"
  auto tokens = Encode(tokenizer, "hi");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 2);  // merged "hi"

  iree_tokenizer_free(tokenizer);
}

TEST_F(ByteLevelBPEEncodeTest, NewlineMapping) {
  // Newline (0x0A) maps to U+010A = Ċ = "\xc4\x8a" in UTF-8.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "\xc4\x8a");  // Ċ = newline byte
  vocab_builder.AddToken(1, "h");
  vocab_builder.AddToken(2, "e");
  vocab_builder.AddToken(3, "l");
  vocab_builder.AddToken(4, "o");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  // Use split segmenter that doesn't strip newlines.
  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenter("\\s+");
  ASSERT_NE(segmenter, nullptr);
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "\nhelo" -> byte-level: Ċ + h + e + l + o
  // Split on whitespace -> segments may vary; the key point is the Ċ mapping.
  auto tokens = Encode(tokenizer, "helo");
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0], 1);  // h
  EXPECT_EQ(tokens[1], 2);  // e
  EXPECT_EQ(tokens[2], 3);  // l
  EXPECT_EQ(tokens[3], 4);  // o

  iree_tokenizer_free(tokenizer);
}

TEST_F(ByteLevelBPEEncodeTest, StreamingByteLevel) {
  // Verify streaming ring buffer produces same result as one-shot for
  // byte-level BPE.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "h");
  vocab_builder.AddToken(1, "e");
  vocab_builder.AddToken(2, "l");
  vocab_builder.AddToken(3, "o");
  vocab_builder.AddToken(4, "he");
  vocab_builder.AddToken(5, "ll");
  vocab_builder.AddToken(6, "hello");
  vocab_builder.AddMerge(0, 1);  // h + e -> he
  vocab_builder.AddMerge(2, 2);  // l + l -> ll
  vocab_builder.AddMerge(4, 5);  // he + ll -> hell (need token)
  vocab_builder.AddToken(7, "hell");
  vocab_builder.AddMerge(7, 3);  // hell + o -> hello
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // One-shot encode.
  auto oneshot = Encode(tokenizer, "hello");

  // Streaming encode.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  const char* text = "hello";
  std::vector<iree_tokenizer_token_id_t> streaming_tokens(64);
  iree_host_size_t total_tokens = 0;

  // Feed one byte at a time.
  for (size_t i = 0; i < strlen(text); ++i) {
    iree_string_view_t chunk = iree_make_string_view(text + i, 1);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(
            streaming_tokens.data() + total_tokens, NULL, NULL,
            streaming_tokens.size() - total_tokens),
        &bytes_consumed, &token_count));
    total_tokens += token_count;
  }

  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(streaming_tokens.data() + total_tokens,
                                       NULL, NULL,
                                       streaming_tokens.size() - total_tokens),
      &finalize_count));
  total_tokens += finalize_count;

  streaming_tokens.resize(total_tokens);

  EXPECT_EQ(oneshot, streaming_tokens)
      << "Streaming byte-level BPE must match one-shot encode";

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// P4: Post-Normalization Special Token Tests
//===----------------------------------------------------------------------===//

class PostNormSpecialTokenTest : public ::testing::Test {};

TEST_F(PostNormSpecialTokenTest, MatchAfterLowercase) {
  // Post-norm special tokens match against normalized text. With a lowercase
  // normalizer, a post-norm token "[mask]" will match "[MASK]" in the input
  // because it becomes "[mask]" after normalization.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50256, "[mask]");
  ScopedVocab vocab = vocab_builder.Build();

  // Build post-norm special tokens with lowercase "[mask]".
  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("[mask]"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t post_norm_special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &post_norm_special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  // Lowercase normalizer.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_lowercase_allocate(
      iree_allocator_system(), &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_special_tokens_post_norm(
      builder.get(), &post_norm_special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "hello [MASK] world" -> lowercase -> "hello [mask] world"
  // Post-norm matches "[mask]" -> special token ID 50256.
  auto tokens = Encode(tokenizer, "hello [MASK] world");
  // Expect: tokens for "hello", then 50256, then tokens for "world".
  EXPECT_FALSE(tokens.empty());
  bool found_mask = false;
  for (auto t : tokens) {
    if (t == 50256) found_mask = true;
  }
  EXPECT_TRUE(found_mask) << "Post-norm special token [mask] should match "
                             "after lowercase normalization of [MASK]";

  iree_tokenizer_special_tokens_deinitialize(&post_norm_special_tokens);
  iree_tokenizer_free(tokenizer);
}

TEST_F(PostNormSpecialTokenTest, RawCaseDoesNotMatch) {
  // Post-norm token "[MASK]" (uppercase) will NOT match input "[MASK]" if
  // a lowercase normalizer transforms it to "[mask]" first.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50256, "[MASK]");
  ScopedVocab vocab = vocab_builder.Build();

  // Post-norm special token with UPPERCASE "[MASK]".
  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("[MASK]"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t post_norm_special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &post_norm_special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  // Lowercase normalizer transforms "[MASK]" -> "[mask]" before comparison.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_lowercase_allocate(
      iree_allocator_system(), &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_special_tokens_post_norm(
      builder.get(), &post_norm_special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "[MASK]" normalizes to "[mask]" but post-norm pattern is "[MASK]" ->
  // no match.
  auto tokens = Encode(tokenizer, "[MASK]");
  bool found_special = false;
  for (auto t : tokens) {
    if (t == 50256) found_special = true;
  }
  EXPECT_FALSE(found_special) << "Uppercase post-norm pattern should NOT match "
                                 "after lowercase normalization";

  iree_tokenizer_special_tokens_deinitialize(&post_norm_special_tokens);
  iree_tokenizer_free(tokenizer);
}

TEST_F(PostNormSpecialTokenTest, MixedPreAndPostNorm) {
  // Pre-norm special token "<bos>" matches raw text, post-norm "[mask]"
  // matches normalized text. Both in same tokenizer.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50256, "<bos>");
  vocab_builder.AddToken(50257, "[mask]");
  ScopedVocab vocab = vocab_builder.Build();

  // Pre-norm special token: "<bos>".
  iree_tokenizer_special_tokens_builder_t pre_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &pre_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &pre_builder, IREE_SV("<bos>"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t pre_norm_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &pre_builder, iree_allocator_system(), &pre_norm_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&pre_builder);

  // Post-norm special token: "[mask]".
  iree_tokenizer_special_tokens_builder_t post_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &post_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &post_builder, IREE_SV("[mask]"), 50257,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t post_norm_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &post_builder, iree_allocator_system(), &post_norm_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&post_builder);

  // Lowercase normalizer.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_lowercase_allocate(
      iree_allocator_system(), &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_special_tokens(builder.get(), &pre_norm_tokens);
  iree_tokenizer_builder_set_special_tokens_post_norm(builder.get(),
                                                      &post_norm_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "<bos> hello [MASK]"
  // Pre-norm matches "<bos>" (raw text) -> 50256.
  // Lowercase normalizes remaining: "hello [mask]".
  // Post-norm matches "[mask]" -> 50257.
  auto tokens = Encode(tokenizer, "<bos> hello [MASK]");
  bool found_bos = false, found_mask = false;
  for (auto t : tokens) {
    if (t == 50256) found_bos = true;
    if (t == 50257) found_mask = true;
  }
  EXPECT_TRUE(found_bos) << "Pre-norm <bos> should match raw text";
  EXPECT_TRUE(found_mask)
      << "Post-norm [mask] should match after lowercase normalization";

  iree_tokenizer_special_tokens_deinitialize(&pre_norm_tokens);
  iree_tokenizer_special_tokens_deinitialize(&post_norm_tokens);
  iree_tokenizer_free(tokenizer);
}

TEST_F(PostNormSpecialTokenTest, StreamingPostNorm) {
  // Streaming encode with post-norm special tokens must match one-shot.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  vocab_builder.AddToken(50256, "[mask]");
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("[mask]"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t post_norm_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &post_norm_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_lowercase_allocate(
      iree_allocator_system(), &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_special_tokens_post_norm(builder.get(),
                                                      &post_norm_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  const char* text = "hello [MASK] world test [MASK] end";

  // One-shot encode.
  auto oneshot = Encode(tokenizer, text);

  // Streaming encode.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> streaming_state_storage(state_size);
  std::vector<uint8_t> streaming_transform_buffer(1024);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(streaming_state_storage.data(),
                          streaming_state_storage.size()),
      iree_make_byte_span(streaming_transform_buffer.data(),
                          streaming_transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> streaming_tokens(256);
  iree_host_size_t total_tokens = 0;

  // Feed in small chunks (3 bytes at a time).
  size_t text_length = strlen(text);
  size_t offset = 0;
  while (offset < text_length) {
    size_t chunk_size = std::min(static_cast<size_t>(3), text_length - offset);
    iree_string_view_t chunk = iree_make_string_view(text + offset, chunk_size);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(
            streaming_tokens.data() + total_tokens, NULL, NULL,
            streaming_tokens.size() - total_tokens),
        &bytes_consumed, &token_count));
    offset += bytes_consumed;
    total_tokens += token_count;
    if (bytes_consumed == 0) break;
  }

  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(streaming_tokens.data() + total_tokens,
                                       NULL, NULL,
                                       streaming_tokens.size() - total_tokens),
      &finalize_count));
  total_tokens += finalize_count;

  streaming_tokens.resize(total_tokens);

  EXPECT_EQ(oneshot, streaming_tokens)
      << "Streaming post-norm special tokens must match one-shot encode";

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_special_tokens_deinitialize(&post_norm_tokens);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// P5: BPE Flags Encode Tests
//===----------------------------------------------------------------------===//

class BPEFlagsEncodeTest : public ::testing::Test {};

TEST_F(BPEFlagsEncodeTest, FuseUnkConsecutive) {
  // Within-segment FUSE_UNK: consecutive unknowns fuse to single [UNK].
  // HuggingFace: Tokenizer(BPE(fuse_unk=True)).encode("xyz").ids == [0]
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.AddToken(1, "a");
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_FUSE_UNK |
                                      IREE_TOKENIZER_BPE_FLAG_NO_BYTE_FALLBACK |
                                      IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "xyz" - three unknown bytes within a single segment.
  // FUSE_UNK should merge all consecutive unknowns into one [UNK].
  auto tokens = Encode(tokenizer, "xyz");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // Fused [UNK]

  iree_tokenizer_free(tokenizer);
}

TEST_F(BPEFlagsEncodeTest, FuseUnkAcrossSegments) {
  // FUSE_UNK does not fuse across segment boundaries.
  // Each segment produces its own [UNK] independently.
  // HuggingFace: Tokenizer(BPE(fuse_unk=True)).encode("x y").ids == [0, 0]
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.AddToken(1, "a");
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_FUSE_UNK |
                                      IREE_TOKENIZER_BPE_FLAG_NO_BYTE_FALLBACK |
                                      IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "x y" - two unknown words, whitespace segmenter produces two segments.
  // FUSE_UNK does not cross segment boundaries, so we get two separate UNKs.
  auto tokens = Encode(tokenizer, "x y");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // [UNK] for segment "x"
  EXPECT_EQ(tokens[1], 0);  // [UNK] for segment "y"

  iree_tokenizer_free(tokenizer);
}

TEST_F(BPEFlagsEncodeTest, FuseUnkNonConsecutive) {
  // FUSE_UNK only fuses consecutive unknowns; known tokens break the fusion.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.AddToken(1, "a");
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(),
                     IREE_TOKENIZER_BPE_FLAG_FUSE_UNK |
                         IREE_TOKENIZER_BPE_FLAG_NO_BYTE_FALLBACK));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "xay" -> UNK('x') + known('a') + UNK('y') = 3 tokens.
  // Only consecutive unknowns fuse; 'a' breaks the run.
  auto tokens = Encode(tokenizer, "xay");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);  // UNK for 'x'
  EXPECT_EQ(tokens[1], 1);  // 'a'
  EXPECT_EQ(tokens[2], 0);  // UNK for 'y'

  iree_tokenizer_free(tokenizer);
}

TEST_F(BPEFlagsEncodeTest, FuseUnkDisabled) {
  // Without FUSE_UNK, each unknown character produces its own [UNK].
  // HuggingFace: Tokenizer(BPE(fuse_unk=False)).encode("xyz").ids == [0, 0, 0]
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.AddToken(1, "a");
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  // NO_BYTE_FALLBACK but NOT FUSE_UNK - each unknown byte is separate UNK.
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NO_BYTE_FALLBACK |
                                      IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "xyz" - three unknown bytes, no FUSE_UNK -> three separate [UNK] tokens.
  auto tokens = Encode(tokenizer, "xyz");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);  // [UNK] for 'x'
  EXPECT_EQ(tokens[1], 0);  // [UNK] for 'y'
  EXPECT_EQ(tokens[2], 0);  // [UNK] for 'z'

  iree_tokenizer_free(tokenizer);
}

TEST_F(BPEFlagsEncodeTest, NoByteFallback) {
  // NO_BYTE_FALLBACK disables <0xXX> byte fallback tokens.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.AddToken(1, "a");
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(),
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NO_BYTE_FALLBACK));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // 'x' has no token and no byte fallback -> UNK.
  auto tokens = Encode(tokenizer, "x");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // [UNK]

  iree_tokenizer_free(tokenizer);
}

TEST_F(BPEFlagsEncodeTest, ByteFallbackDefault) {
  // Default BPE (no NO_BYTE_FALLBACK flag) falls back to <0xXX> byte tokens.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  // Add byte fallback token for 'x' (0x78).
  vocab_builder.AddToken(1, "<0x78>", IREE_TOKENIZER_TOKEN_ATTR_BYTE);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // 'x' has no regular token but byte fallback <0x78> exists.
  auto tokens = Encode(tokenizer, "x");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 1);  // <0x78> byte fallback

  iree_tokenizer_free(tokenizer);
}

TEST_F(BPEFlagsEncodeTest, EndOfWordSuffix) {
  // CLIP-style end-of-word suffix: token "hi</w>" should match "hi" at end
  // of a segment.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "h");
  vocab_builder.AddToken(1, "i");
  vocab_builder.AddToken(2, "hi</w>");
  // Add merges: h+i with suffix should produce "hi</w>".
  vocab_builder.AddMerge(0, 1);  // h + i -> hi</w> (with suffix)
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_model_t* model =
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE);
  IREE_ASSERT_OK(
      iree_tokenizer_bpe_model_set_end_of_word_suffix(model, IREE_SV("</w>")));
  iree_tokenizer_builder_set_model(builder.get(), model);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "hi" -> with suffix, becomes "hi</w>" at segment end -> token 2.
  auto tokens = Encode(tokenizer, "hi");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 2);  // "hi</w>"

  iree_tokenizer_free(tokenizer);
}

TEST_F(BPEFlagsEncodeTest, EndOfWordSuffixWithMerges) {
  // End-of-word suffix with multi-step merges.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "h");
  vocab_builder.AddToken(1, "e");
  vocab_builder.AddToken(2, "l");
  vocab_builder.AddToken(3, "o");
  vocab_builder.AddToken(4, "he");
  vocab_builder.AddToken(5, "lo</w>");
  vocab_builder.AddToken(6, "helo</w>");
  vocab_builder.AddMerge(0, 1);  // h + e -> he (rank 0)
  vocab_builder.AddMerge(2, 3);  // l + o -> lo</w> with suffix (rank 1)
  vocab_builder.AddMerge(4, 5);  // he + lo</w> -> helo</w> (rank 2)
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_model_t* model =
      CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE);
  IREE_ASSERT_OK(
      iree_tokenizer_bpe_model_set_end_of_word_suffix(model, IREE_SV("</w>")));
  iree_tokenizer_builder_set_model(builder.get(), model);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "helo" -> suffix appended: "helo</w>" -> BPE merges.
  auto tokens = Encode(tokenizer, "helo");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 6);  // "helo</w>"

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// P6: Normalizer Pipeline Tests
//===----------------------------------------------------------------------===//

class NormalizerPipelineTest : public ::testing::Test {};

TEST_F(NormalizerPipelineTest, LowercaseASCII) {
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "hello");
  vocab_builder.AddToken(1, "world");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_lowercase_allocate(
      iree_allocator_system(), &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto tokens = Encode(tokenizer, "Hello WORLD");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "hello"
  EXPECT_EQ(tokens[1], 1);  // "world"

  iree_tokenizer_free(tokenizer);
}

TEST_F(NormalizerPipelineTest, LowercaseUnicode) {
  // U+00DC (Ü) -> U+00FC (ü) via Unicode case folding.
  ScopedVocabBuilder vocab_builder;
  // "über" in UTF-8: ü = \xc3\xbc, b=b, e=e, r=r.
  vocab_builder.AddToken(0,
                         "\xC3\xBC"
                         "ber");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_lowercase_allocate(
      iree_allocator_system(), &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "Über" -> lowercase -> "über"
  // U+00DC (Ü) = \xc3\x9c in UTF-8.
  auto tokens = Encode(tokenizer,
                       "\xC3\x9C"
                       "ber");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "über"

  iree_tokenizer_free(tokenizer);
}

TEST_F(NormalizerPipelineTest, NFDDecomposition) {
  // NFD decomposes precomposed characters: é (U+00E9) -> e + ◌́ (U+0301).
  ScopedVocabBuilder vocab_builder;
  // "cafe" followed by combining acute accent in vocab.
  // e (U+0065) + combining acute (U+0301) = \x65\xCC\x81 in UTF-8.
  vocab_builder.AddToken(0, "caf");
  vocab_builder.AddToken(1, "##e\xCC\x81");  // "##" + e + combining accent
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_nfd_allocate(iree_allocator_system(),
                                                        &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "café" (precomposed é = \xC3\xA9) -> NFD -> "cafe\xCC\x81"
  // WordPiece: "caf" + "##e\xCC\x81"
  auto tokens = Encode(tokenizer, "caf\xC3\xA9");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "caf"
  EXPECT_EQ(tokens[1], 1);  // "##e" + combining accent

  iree_tokenizer_free(tokenizer);
}

TEST_F(NormalizerPipelineTest, StripAccentsAfterNFD) {
  // Sequence[NFD, StripAccents] decomposes then removes combining marks.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "cafe");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_normalizer_t* nfd = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_normalizer_nfd_allocate(iree_allocator_system(), &nfd));
  iree_tokenizer_normalizer_t* strip_accents = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_accents_allocate(
      iree_allocator_system(), &strip_accents));

  iree_tokenizer_normalizer_t* children[] = {nfd, strip_accents};
  iree_tokenizer_normalizer_t* sequence = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_sequence_allocate(
      children, 2, iree_allocator_system(), &sequence));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), sequence);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "café" -> NFD -> "cafe" + combining accent -> StripAccents -> "cafe"
  auto tokens = Encode(tokenizer, "caf\xC3\xA9");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "cafe"

  iree_tokenizer_free(tokenizer);
}

TEST_F(NormalizerPipelineTest, StripAccentsAlonePrecomposed) {
  // StripAccents alone does not decompose precomposed characters.
  // é (U+00E9) is not a combining mark (Mn) — it's category Ll (lowercase
  // letter). StripAccents only removes Mn/Mc/Me marks, so precomposed
  // characters pass through unchanged.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "caf\xC3\xA9");  // Precomposed "café"
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_normalizer_t* strip_accents = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_accents_allocate(
      iree_allocator_system(), &strip_accents));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), strip_accents);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "café" (precomposed) stays "café" since StripAccents alone doesn't
  // decompose. The precomposed é is category Ll, not Mn.
  auto tokens = Encode(tokenizer, "caf\xC3\xA9");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // Still precomposed "café"

  iree_tokenizer_free(tokenizer);
}

TEST_F(NormalizerPipelineTest, LowercaseTurkishI) {
  // U+0130 (İ, Latin Capital Letter I With Dot Above) is the one codepoint
  // that expands during lowering: İ -> i + combining dot above (U+0307).
  // This tests normalizer expansion through the ring buffer pipeline.
  ScopedVocabBuilder vocab_builder;
  // After lowering: İ (2 UTF-8 bytes: \xC4\xB0) -> i (1 byte) + U+0307
  // (2 bytes: \xCC\x87). We tokenize the result as individual chars.
  vocab_builder.AddToken(0, "i");
  // Combining dot above U+0307 = \xCC\x87 in UTF-8.
  vocab_builder.AddToken(1, "##\xCC\x87");
  vocab_builder.AddToken(99, "[UNK]", IREE_TOKENIZER_TOKEN_ATTR_UNKNOWN);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 99);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_lowercase_allocate(
      iree_allocator_system(), &normalizer));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(),
                                       CreateWhitespaceSegmenter());
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateWordPieceModel(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // İ (\xC4\xB0) -> lowercase -> "i" + combining dot above
  auto tokens = Encode(tokenizer, "\xC4\xB0");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "i"
  EXPECT_EQ(tokens[1], 1);  // combining dot above

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// P7: Punctuation Segmenter Encode Tests
//===----------------------------------------------------------------------===//

class PunctuationSegmenterEncodeTest : public ::testing::Test {};

TEST_F(PunctuationSegmenterEncodeTest, Isolated) {
  // ISOLATED: each punctuation char becomes its own segment.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  vocab_builder.AddToken(1, ",");
  vocab_builder.AddToken(2, " world");
  vocab_builder.AddToken(3, "!");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(
      builder.get(),
      CreatePunctuationSegmenter(IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED));
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "Hello, world!" -> ["Hello", ",", " world", "!"]
  auto tokens = Encode(tokenizer, "Hello, world!");
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0], 0);  // "Hello"
  EXPECT_EQ(tokens[1], 1);  // ","
  EXPECT_EQ(tokens[2], 2);  // " world"
  EXPECT_EQ(tokens[3], 3);  // "!"

  iree_tokenizer_free(tokenizer);
}

TEST_F(PunctuationSegmenterEncodeTest, Removed) {
  // REMOVED: punctuation is discarded.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(
      builder.get(),
      CreatePunctuationSegmenter(IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED));
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "Hello!" -> punctuation "!" removed -> ["Hello"]
  auto tokens = Encode(tokenizer, "Hello!");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);  // "Hello"

  iree_tokenizer_free(tokenizer);
}

TEST_F(PunctuationSegmenterEncodeTest, MergedWithPrevious) {
  // MERGED_WITH_PREVIOUS: punctuation appended to preceding segment.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "Hello.");
  vocab_builder.AddToken(1, " world");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(
      builder.get(), CreatePunctuationSegmenter(
                         IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS));
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "Hello. world" -> punct "." merged with previous -> ["Hello.", " world"]
  auto tokens = Encode(tokenizer, "Hello. world");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);  // "Hello."
  EXPECT_EQ(tokens[1], 1);  // " world"

  iree_tokenizer_free(tokenizer);
}

TEST_F(PunctuationSegmenterEncodeTest, Contiguous) {
  // CONTIGUOUS: consecutive punctuation chars grouped into one segment.
  ScopedVocabBuilder vocab_builder;
  vocab_builder.AddToken(0, "a");
  vocab_builder.AddToken(1, "...");
  vocab_builder.AddToken(2, "b");
  ScopedVocab vocab = vocab_builder.Build();

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(
      builder.get(),
      CreatePunctuationSegmenter(IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS));
  iree_tokenizer_builder_set_model(builder.get(),
                                   CreateBPEModelIgnoreMerges(vocab.get()));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // "a...b" -> ["a", "...", "b"] (consecutive punct grouped).
  auto tokens = Encode(tokenizer, "a...b");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);  // "a"
  EXPECT_EQ(tokens[1], 1);  // "..."
  EXPECT_EQ(tokens[2], 2);  // "b"

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// ALBERT-style Configuration Tests (Sequence Segmenter)
//===----------------------------------------------------------------------===//

// Tests tokenization with ALBERT-style configuration:
//   - Normalizer: Strip(right) + RegexReplace(\s+ → ▁) + Prepend(▁)
//   - Segmenter: Sequence[WhitespaceSplit, Metaspace(split=true)]
//   - Model: Unigram
//
// This configuration differs from DeBERTa in using a Sequence segmenter that
// chains WhitespaceSplit → Metaspace. The key behavior is:
//   1. Strip right: removes trailing whitespace
//   2. RegexReplace: collapses all whitespace runs to ▁
//   3. Prepend: adds leading ▁
//   4. WhitespaceSplit: splits on ASCII whitespace (none left after normalize)
//   5. Metaspace: splits on ▁ boundaries
//
// After normalization, the text contains only ▁ as word separators (no ASCII
// whitespace), so WhitespaceSplit passes through unchanged and Metaspace does
// the actual segmentation.

class TokenizerAlbertTest : public ::testing::Test {};

TEST_F(TokenizerAlbertTest, MultipleWhitespaceWithMetaspaceUnigram) {
  // Build minimal vocab for Unigram model that can tokenize the test input.
  // After normalization: "▁multiple▁spaces▁and▁tabs▁and▁newlines"
  ScopedVocabBuilder vocab_builder;

  // U+2581 LOWER ONE EIGHTH BLOCK = 0xE2 0x96 0x81 in UTF-8.
  static const char kMetaspace[] = "\xE2\x96\x81";

  // Token 0: UNK
  vocab_builder.AddTokenWithScore(0, "[UNK]", -10.0f);

  // Tokens for the test input (with metaspace prefix).
  std::string t_multiple = std::string(kMetaspace) + "multiple";
  std::string t_spaces = std::string(kMetaspace) + "spaces";
  std::string t_and = std::string(kMetaspace) + "and";
  std::string t_tabs = std::string(kMetaspace) + "tabs";
  std::string t_newlines = std::string(kMetaspace) + "newlines";

  vocab_builder.AddTokenWithScore(1, t_multiple.c_str(), -2.0f);
  vocab_builder.AddTokenWithScore(2, t_spaces.c_str(), -2.0f);
  vocab_builder.AddTokenWithScore(3, t_and.c_str(), -2.0f);
  vocab_builder.AddTokenWithScore(4, t_tabs.c_str(), -2.0f);
  vocab_builder.AddTokenWithScore(5, t_newlines.c_str(), -2.0f);

  // Add a distractor token that shouldn't be selected.
  std::string t_as = std::string(kMetaspace) + "as";
  vocab_builder.AddTokenWithScore(6, t_as.c_str(), -2.0f);

  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  // Build ALBERT-style normalizer chain:
  // Strip(right) → RegexReplace(\s+ → ▁) → Prepend(▁)

  // Strip: right-strip only (removes trailing whitespace).
  iree_tokenizer_normalizer_t* strip_right = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_allocate(
      /*strip_left=*/false, /*strip_right=*/true, iree_allocator_system(),
      &strip_right));

  // RegexReplace: collapse consecutive whitespace to single ▁.
  // This matches HuggingFace's behavior when WhitespaceSplit + Metaspace are
  // both present: all whitespace (spaces, tabs, newlines) becomes ▁.
  iree_tokenizer_normalizer_t* regex_replace = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_regex_replace_allocate(
      IREE_SV("\\s+"), iree_make_string_view(kMetaspace, 3),
      iree_allocator_system(), &regex_replace));

  // Prepend: add leading ▁ (skip if already starts with ▁).
  iree_tokenizer_normalizer_t* prepend = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_prepend_allocate(
      iree_make_string_view(kMetaspace, 3),
      /*skip_if_prefix_matches=*/true, iree_allocator_system(), &prepend));

  // Chain: Strip → RegexReplace → Prepend.
  iree_tokenizer_normalizer_t* normalizer_children[] = {strip_right,
                                                        regex_replace, prepend};
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_sequence_allocate(
      normalizer_children, 3, iree_allocator_system(), &normalizer));

  // Build ALBERT-style segmenter: Sequence[WhitespaceSplit, Metaspace].
  iree_tokenizer_segmenter_t* whitespace = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_whitespace_allocate(
      iree_allocator_system(), &whitespace));

  iree_tokenizer_segmenter_t* metaspace = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
      IREE_TOKENIZER_METASPACE_DEFAULT_REPLACEMENT, /*split_enabled=*/true,
      iree_allocator_system(), &metaspace));

  iree_tokenizer_segmenter_t* segmenter_children[] = {whitespace, metaspace};
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_sequence_allocate(
      segmenter_children, 2, iree_allocator_system(), &segmenter));

  // Unigram model.
  iree_tokenizer_model_t* model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
      vocab.get(), /*unk_token_id=*/0, /*unk_score=*/-10.0f,
      IREE_TOKENIZER_UNIGRAM_FLAG_NONE, iree_allocator_system(), &model));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(builder.get(), model);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Test input with multiple whitespace types (the ALBERT smoketest failure).
  // Input: "  multiple   spaces   and\ttabs\nand\nnewlines  "
  // After Strip(right): "  multiple   spaces   and\ttabs\nand\nnewlines"
  // After RegexReplace(\s+ → ▁): "▁multiple▁spaces▁and▁tabs▁and▁newlines"
  // After Prepend(▁, skip if starts with ▁): unchanged (already starts with ▁)
  // Segments: [▁multiple, ▁spaces, ▁and, ▁tabs, ▁and, ▁newlines]
  // Expected tokens: [1, 2, 3, 4, 3, 5]
  auto tokens =
      Encode(tokenizer, "  multiple   spaces   and\ttabs\nand\nnewlines  ");
  EXPECT_THAT(tokens, ::testing::ElementsAre(1, 2, 3, 4, 3, 5))
      << "Expected: ▁multiple(1) ▁spaces(2) ▁and(3) ▁tabs(4) ▁and(3) "
         "▁newlines(5)";

  iree_tokenizer_free(tokenizer);
}

// Tests with smaller transform buffer to exercise ring buffer wrapping.
TEST_F(TokenizerAlbertTest, SmallBufferRingWrap) {
  // Same setup as above but with explicit small buffer to force ring wrap.
  ScopedVocabBuilder vocab_builder;
  static const char kMetaspace[] = "\xE2\x96\x81";

  vocab_builder.AddTokenWithScore(0, "[UNK]", -10.0f);
  std::string t_a = std::string(kMetaspace) + "a";
  std::string t_b = std::string(kMetaspace) + "b";
  vocab_builder.AddTokenWithScore(1, t_a.c_str(), -2.0f);
  vocab_builder.AddTokenWithScore(2, t_b.c_str(), -2.0f);
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  // Minimal normalizer chain for this test.
  iree_tokenizer_normalizer_t* strip_right = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_strip_allocate(
      /*strip_left=*/false, /*strip_right=*/true, iree_allocator_system(),
      &strip_right));

  iree_tokenizer_normalizer_t* regex_replace = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_regex_replace_allocate(
      IREE_SV("\\s+"), iree_make_string_view(kMetaspace, 3),
      iree_allocator_system(), &regex_replace));

  iree_tokenizer_normalizer_t* prepend = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_prepend_allocate(
      iree_make_string_view(kMetaspace, 3),
      /*skip_if_prefix_matches=*/true, iree_allocator_system(), &prepend));

  iree_tokenizer_normalizer_t* normalizer_children[] = {strip_right,
                                                        regex_replace, prepend};
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_sequence_allocate(
      normalizer_children, 3, iree_allocator_system(), &normalizer));

  // Sequence[WhitespaceSplit, Metaspace] segmenter.
  iree_tokenizer_segmenter_t* whitespace = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_whitespace_allocate(
      iree_allocator_system(), &whitespace));

  iree_tokenizer_segmenter_t* metaspace = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
      IREE_TOKENIZER_METASPACE_DEFAULT_REPLACEMENT, /*split_enabled=*/true,
      iree_allocator_system(), &metaspace));

  iree_tokenizer_segmenter_t* segmenter_children[] = {whitespace, metaspace};
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_sequence_allocate(
      segmenter_children, 2, iree_allocator_system(), &segmenter));

  iree_tokenizer_model_t* model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
      vocab.get(), /*unk_token_id=*/0, /*unk_score=*/-10.0f,
      IREE_TOKENIZER_UNIGRAM_FLAG_NONE, iree_allocator_system(), &model));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_normalizer(builder.get(), normalizer);
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(builder.get(), model);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Use small buffer (64 bytes allocation = 32 bytes usable capacity).
  // Input "a b a b a b" → normalized "▁a▁b▁a▁b▁a▁b" = 24 bytes.
  // This should still fit without wrap, but tests the streaming path.
  constexpr size_t kBufferSize = 64;
  std::vector<uint8_t> transform_buffer(kBufferSize);

  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  // Feed input and collect tokens.
  const char* input = "a b a b a b";
  std::vector<iree_tokenizer_token_id_t> token_ids(32);
  iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
      token_ids.data(), nullptr, nullptr, token_ids.size());

  iree_host_size_t consumed = 0;
  iree_host_size_t token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
      state, iree_make_string_view(input, strlen(input)), output, &consumed,
      &token_count));

  // Finalize.
  iree_tokenizer_token_output_t remaining_output = {
      .capacity = token_ids.size() - token_count,
      .token_ids = token_ids.data() + token_count,
      .token_offsets = nullptr,
      .type_ids = nullptr,
  };
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(state, remaining_output,
                                                      &finalize_count));

  token_count += finalize_count;
  token_ids.resize(token_count);

  // Expected: ▁a(1) ▁b(2) ▁a(1) ▁b(2) ▁a(1) ▁b(2)
  EXPECT_THAT(token_ids, ::testing::ElementsAre(1, 2, 1, 2, 1, 2));

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Offset Tracking Tests
//
// Verifies that token offsets form a valid coverage of the input:
//   - Monotonically non-decreasing: each token's start >= prior max end
//   - Complete coverage: union of [start, end) covers [0, input_length)
//   - No overlaps: no input byte claimed by more than one token
//
// These properties must hold regardless of segment batch boundaries. The
// Sequence segmenter's output-full resumption path is the primary risk: it
// must correctly track cumulative consumed bytes so that subsequent segments
// don't overlap with already-processed input.
//===----------------------------------------------------------------------===//

class TokenizerOffsetTest : public ::testing::Test {
 protected:
  // Encodes text and returns token IDs + offsets.
  struct EncodeResult {
    std::vector<iree_tokenizer_token_id_t> token_ids;
    std::vector<iree_tokenizer_offset_t> offsets;
  };

  static EncodeResult EncodeWithOffsets(iree_tokenizer_t* tokenizer,
                                        iree_string_view_t text) {
    EncodeResult result;
    result.token_ids.resize(4096);
    result.offsets.resize(4096);

    iree_host_size_t token_count = 0;
    iree_status_t status = iree_tokenizer_encode(
        tokenizer, text, IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(result.token_ids.data(),
                                         result.offsets.data(), NULL,
                                         result.token_ids.size()),
        iree_allocator_system(), &token_count);

    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      result.token_ids.clear();
      result.offsets.clear();
      return result;
    }

    result.token_ids.resize(token_count);
    result.offsets.resize(token_count);
    return result;
  }

  // Verifies offset monotonicity: each token's start must be >= the max end
  // of all prior tokens. Returns the index of the first violation, or -1.
  static int FindFirstBackwardsJump(
      const std::vector<iree_tokenizer_offset_t>& offsets) {
    iree_host_size_t max_end = 0;
    for (size_t i = 0; i < offsets.size(); ++i) {
      if (i > 0 && offsets[i].start < max_end) {
        return static_cast<int>(i);
      }
      if (offsets[i].end > max_end) max_end = offsets[i].end;
    }
    return -1;
  }
};

// Exercises offset correctness across the segment batch boundary (64 segments).
// Uses Sequence[Split(\s+), Split(.)] where child1 expands each child0 segment
// into per-character segments.
//
// The key is choosing word lengths that MISALIGN with the 64-slot output
// capacity. With 5-char words + 1-char spaces, each word+space pair produces
// 6 sub-segments. After 10 pairs (60 sub-segs), the 11th word starts expanding
// 5 sub-segments into 4 remaining slots — triggering the output-full
// mid-expansion path. A broken comparison here (cumulative output_count vs
// per-expansion total_expanded) silently drops segments.
TEST_F(TokenizerOffsetTest, SequenceExpansionAcrossBatchBoundary) {
  // Byte-level vocab: one token per byte, no merges.
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  // child0: whitespace splitter (ISOLATED) — produces word and space segments.
  iree_tokenizer_segmenter_t* child0 = CreateSplitSegmenterWithBehavior(
      "\\s+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  ASSERT_NE(child0, nullptr);

  // child1: per-character splitter (ISOLATED) — expands each segment into
  // individual byte segments. This causes the Sequence output to fill faster
  // than child0 segments are consumed, exercising the resumption path.
  iree_tokenizer_segmenter_t* child1 = CreateSplitSegmenterWithBehavior(
      ".", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  ASSERT_NE(child1, nullptr);

  iree_tokenizer_segmenter_t* children[] = {child0, child1};
  iree_tokenizer_segmenter_t* sequence = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_sequence_allocate(
      children, 2, iree_allocator_system(), &sequence));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), sequence);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // 13 five-letter words separated by spaces = 13*5 + 12 = 77 bytes.
  // child0 produces 25 segments (13 words + 12 spaces).
  // child1 expands: 13*5 + 12*1 = 77 character-segments, exceeding batch=64.
  //
  // After 10 word+space pairs (10*6=60 sub-segs), the 11th word needs 5 slots
  // but only 4 remain. This triggers the output-full mid-expansion path.
  // Crucially: the 11th word has a trailing space (word 12 follows it), so
  // child0 emits it during process() rather than deferring to finalize().
  std::string text =
      "aaaaa bbbbb ccccc ddddd eeeee "
      "fffff ggggg hhhhh iiiii jjjjj "
      "kkkkk lllll mmmmm";

  auto result = EncodeWithOffsets(
      tokenizer, iree_make_string_view(text.data(), text.size()));
  ASSERT_FALSE(result.token_ids.empty()) << "Encode should succeed";
  ASSERT_EQ(result.token_ids.size(), text.size())
      << "Byte-level vocab should produce one token per byte (" << text.size()
      << "), got " << result.token_ids.size()
      << " — likely segments dropped at batch boundary";

  // Verify offset monotonicity: no backwards jumps.
  int backwards_at = FindFirstBackwardsJump(result.offsets);
  if (backwards_at >= 0) {
    iree_host_size_t max_end = 0;
    for (int i = 0; i < backwards_at; ++i) {
      if (result.offsets[i].end > max_end) max_end = result.offsets[i].end;
    }
    FAIL() << "Backwards jump at token " << backwards_at << ": offset=["
           << result.offsets[backwards_at].start << ","
           << result.offsets[backwards_at].end
           << "), previous max_end=" << max_end;
  }

  // Verify each token's offset covers exactly 1 byte (byte-level vocab).
  for (size_t i = 0; i < result.offsets.size(); ++i) {
    EXPECT_EQ(result.offsets[i].end - result.offsets[i].start, 1u)
        << "Token " << i << " should span exactly 1 byte";
  }

  // Verify coverage: offsets should cover [0, text.size()) with no gaps.
  for (size_t i = 0; i < result.offsets.size(); ++i) {
    EXPECT_EQ(result.offsets[i].start, i)
        << "Token " << i << " should start at byte " << i;
    EXPECT_EQ(result.offsets[i].end, i + 1)
        << "Token " << i << " should end at byte " << i + 1;
  }

  iree_tokenizer_free(tokenizer);
}

// Larger input to ensure multiple batch boundary crossings. Uses 5-char words
// (same misalignment as above) to trigger the output-full path at EVERY batch
// boundary, not just the first.
//
// With 5-char words + 1-char spaces = 6 sub-segs per pair:
//   Batch 1: 10 pairs (60) + 4 of word 11 = 64. Word 11 needs resumption.
//   Batch 2: 1 remaining from word 11, space, then pairs... misaligns again.
// This exercises the resumption path repeatedly across many batches.
TEST_F(TokenizerOffsetTest, SequenceExpansionMultipleBatches) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_segmenter_t* child0 = CreateSplitSegmenterWithBehavior(
      "\\s+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  ASSERT_NE(child0, nullptr);
  iree_tokenizer_segmenter_t* child1 = CreateSplitSegmenterWithBehavior(
      ".", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  ASSERT_NE(child1, nullptr);

  iree_tokenizer_segmenter_t* children[] = {child0, child1};
  iree_tokenizer_segmenter_t* sequence = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_sequence_allocate(
      children, 2, iree_allocator_system(), &sequence));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), sequence);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // 40 five-letter words + 39 spaces = 200 + 39 = 239 bytes.
  // Expansion: 40*5 + 39 = 239 sub-segments, crossing ~4 batch boundaries.
  std::string text;
  for (int i = 0; i < 40; ++i) {
    if (i > 0) text += ' ';
    text += std::string(5, 'a' + (i % 26));
  }

  auto result = EncodeWithOffsets(
      tokenizer, iree_make_string_view(text.data(), text.size()));
  ASSERT_FALSE(result.token_ids.empty()) << "Encode should succeed";
  ASSERT_EQ(result.token_ids.size(), text.size())
      << "Byte-level vocab should produce one token per byte (" << text.size()
      << "), got " << result.token_ids.size()
      << " — likely segments dropped at batch boundaries";

  // Monotonicity: no backwards jumps across multiple batch boundaries.
  EXPECT_EQ(FindFirstBackwardsJump(result.offsets), -1)
      << "No backwards jumps should occur across multiple batch boundaries";

  // Full per-byte coverage: every byte has exactly one token.
  for (size_t i = 0; i < result.offsets.size(); ++i) {
    EXPECT_EQ(result.offsets[i].start, i)
        << "Token " << i << " should start at byte " << i;
    EXPECT_EQ(result.offsets[i].end, i + 1)
        << "Token " << i << " should end at byte " << i + 1;
  }

  iree_tokenizer_free(tokenizer);
}

// Test with a single child (no expansion) to verify the non-expansion path
// also produces correct offsets. This is a baseline sanity check.
TEST_F(TokenizerOffsetTest, SingleSplitSegmenterOffsets) {
  ScopedVocabBuilder vocab_builder(256);
  for (int i = 0; i < 256; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddToken(i, byte_token);
  }
  ScopedVocab vocab = vocab_builder.Build();

  // Simple whitespace splitter — no Sequence wrapper.
  iree_tokenizer_segmenter_t* segmenter = CreateSplitSegmenterWithBehavior(
      "\\s+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED);
  ASSERT_NE(segmenter, nullptr);

  ScopedBuilder builder;
  iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
  iree_tokenizer_builder_set_model(
      builder.get(), CreateBPEModel(vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE));
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  std::string text = "hello world foo bar baz";
  auto result = EncodeWithOffsets(
      tokenizer, iree_make_string_view(text.data(), text.size()));
  ASSERT_EQ(result.token_ids.size(), text.size());

  // Monotonicity and sequential 1-byte coverage.
  EXPECT_EQ(FindFirstBackwardsJump(result.offsets), -1);
  for (size_t i = 0; i < result.offsets.size(); ++i) {
    EXPECT_EQ(result.offsets[i].start, i);
    EXPECT_EQ(result.offsets[i].end, i + 1);
  }

  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Partial Segment + Special Token Interaction
//===----------------------------------------------------------------------===//

// Verifies that a pending special token forces the partial segment handler to
// finalize ring content. When the ring has fewer bytes than max_token_length,
// the model's holdback zone would normally cover the entire ring. A pending
// special token is a segment boundary, so the model must process all bytes
// without holdback.
TEST_F(TokenizerPartialSegmentTest,
       UnigramSpecialTokenDoesNotDeadlockPartialMode) {
  // Vocab with a 48-byte token so max_token_length exceeds the ring data size
  // when partial mode activates near a special token boundary.
  ScopedVocabBuilder vocab_builder;

  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddTokenWithScore(i, byte_token, -5.0f);
  }

  // Add some multi-char tokens to increase max_token_length.
  vocab_builder.AddTokenWithScore(128, "hello", -1.0f);
  vocab_builder.AddTokenWithScore(129, "world", -1.0f);
  vocab_builder.AddTokenWithScore(
      130, "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuv", -8.0f);
  vocab_builder.AddToken(131, "</s>");
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("</s>"), 131,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  iree_tokenizer_model_t* unigram = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
      vocab.get(), /*unk_token_id=*/0, /*unk_score=*/-10.0f,
      IREE_TOKENIZER_UNIGRAM_FLAG_NONE, iree_allocator_system(), &unigram));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_model(builder.get(), unigram);
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  auto result = EncodeWithBufferSize(tokenizer, "hello world</s>more text",
                                     IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE);
  IREE_ASSERT_OK(result.status());

  bool found_special = false;
  for (auto token_id : result.value()) {
    if (token_id == 131) {
      found_special = true;
      break;
    }
  }
  EXPECT_TRUE(found_special)
      << "special token </s> (id=131) not found in output";

  iree_tokenizer_free(tokenizer);
}

// Verifies special token emission with repeated partial mode transitions.
TEST_F(TokenizerPartialSegmentTest, UnigramMultipleSpecialTokensSmallBuffer) {
  ScopedVocabBuilder vocab_builder;
  for (int i = 0; i < 128; ++i) {
    char byte_token[2] = {static_cast<char>(i), '\0'};
    vocab_builder.AddTokenWithScore(i, byte_token, -5.0f);
  }
  vocab_builder.AddTokenWithScore(128, "hello", -1.0f);
  vocab_builder.AddTokenWithScore(129, "world", -1.0f);
  // Long token for high max_token_length.
  vocab_builder.AddTokenWithScore(
      130, "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuv", -8.0f);
  vocab_builder.AddToken(131, "</s>");
  vocab_builder.SetSpecialToken(IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  ScopedVocab vocab = vocab_builder.Build();

  iree_tokenizer_special_tokens_builder_t st_builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &st_builder);
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_add(
      &st_builder, IREE_SV("</s>"), 131,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  iree_tokenizer_special_tokens_t special_tokens;
  IREE_ASSERT_OK(iree_tokenizer_special_tokens_builder_build(
      &st_builder, iree_allocator_system(), &special_tokens));
  iree_tokenizer_special_tokens_builder_deinitialize(&st_builder);

  iree_tokenizer_model_t* unigram = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
      vocab.get(), /*unk_token_id=*/0, /*unk_score=*/-10.0f,
      IREE_TOKENIZER_UNIGRAM_FLAG_NONE, iree_allocator_system(), &unigram));

  ScopedBuilder builder;
  iree_tokenizer_builder_set_model(builder.get(), unigram);
  iree_tokenizer_builder_set_special_tokens(builder.get(), &special_tokens);
  iree_tokenizer_builder_set_vocab(builder.get(), vocab.release());

  iree_tokenizer_t* tokenizer = BuildTokenizer(builder.get());
  ASSERT_NE(tokenizer, nullptr);

  // Build input with multiple </s> tokens separated by text that will fill
  // the ring and trigger partial mode repeatedly.
  std::string input;
  for (int i = 0; i < 20; ++i) {
    input += "hello world this is some text";
    input += "</s>";
  }
  input += "trailing text after last special token";

  auto result = EncodeWithBufferSize(tokenizer, input,
                                     IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE);
  IREE_ASSERT_OK(result.status());

  // Count special tokens in output.
  iree_host_size_t special_count = 0;
  for (auto token_id : result.value()) {
    if (token_id == 131) ++special_count;
  }
  EXPECT_EQ(special_count, 20u)
      << "expected 20 </s> special tokens in output, got " << special_count;

  iree_tokenizer_free(tokenizer);
}

}  // namespace
}  // namespace tokenizer
}  // namespace iree
