// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Streaming encode consistency tests.
//
// These tests verify the fundamental invariant: streaming encode produces
// IDENTICAL output to batch encode for the same input text. This invariant must
// hold regardless of:
//   - How input is chunked (byte-by-byte, random sizes, all-at-once)
//   - Transform buffer size
//   - Input content (ASCII, UTF-8, special tokens, edge cases)
//
// The tests use synthetic tokenizers with known behavior to isolate streaming
// correctness from e.g. HuggingFace format compatibility.

#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/segmenter/split.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/tokenizer_test_util.h"

namespace iree {
namespace tokenizer {
namespace {

using testing::BuildTokenizerOr;
using testing::CreateBPEModelIgnoreMerges;
using testing::CreateWhitespaceSegmenter;
using testing::EncodeStateStorage;
using testing::ScopedBuilder;
using testing::ScopedVocab;
using testing::ScopedVocabBuilder;
using testing::TokenizerPtr;

//===----------------------------------------------------------------------===//
// Test Utilities
//===----------------------------------------------------------------------===//

// Creates a split segmenter from a regex pattern.
static iree::StatusOr<iree_tokenizer_segmenter_t*> CreateSplitSegmenter(
    const char* pattern) {
  iree_tokenizer_regex_dfa_t dfa;
  uint8_t* dfa_storage = nullptr;
  iree_tokenizer_regex_compile_error_t error = {0};
  IREE_RETURN_IF_ERROR(iree_tokenizer_regex_compile_and_load(
      iree_make_cstring_view(pattern),
      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
      &dfa, &dfa_storage, &error));

  iree_tokenizer_segmenter_t* segmenter = nullptr;
  iree_status_t status = iree_tokenizer_segmenter_split_allocate(
      dfa, dfa_storage, IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED,
      /*invert=*/false, iree_allocator_system(), &segmenter);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), dfa_storage);
    return status;
  }
  return segmenter;
}

// Encodes text using the batch (one-shot) API.
static iree::StatusOr<std::vector<iree_tokenizer_token_id_t>> EncodeBatch(
    iree_tokenizer_t* tokenizer, const std::string& text) {
  std::vector<iree_tokenizer_token_id_t> tokens(text.size() + 256);
  iree_host_size_t token_count = 0;

  IREE_RETURN_IF_ERROR(iree_tokenizer_encode(
      tokenizer, iree_make_string_view(text.data(), text.size()),
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      iree_allocator_system(), &token_count));

  tokens.resize(token_count);
  return tokens;
}

// Encodes text using the streaming API with specified chunk sizes.
// |chunk_sizes| defines how to split the input. Each element is a chunk size.
// If the sum of chunk_sizes is less than text.size(), remaining bytes use the
// last chunk size (or 1 if chunk_sizes is empty).
static iree::StatusOr<std::vector<iree_tokenizer_token_id_t>> EncodeStreaming(
    iree_tokenizer_t* tokenizer, const std::string& text,
    const std::vector<size_t>& chunk_sizes, size_t transform_buffer_size) {
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));

  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_buffer(transform_buffer_size);

  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  std::vector<iree_tokenizer_token_id_t> tokens(text.size() + 256);
  iree_host_size_t total_tokens = 0;
  size_t offset = 0;
  size_t chunk_index = 0;

  while (offset < text.size()) {
    // Determine chunk size for this iteration.
    size_t chunk_size;
    if (chunk_sizes.empty()) {
      chunk_size = 1;  // Default: byte-by-byte.
    } else if (chunk_index < chunk_sizes.size()) {
      chunk_size = chunk_sizes[chunk_index];
    } else {
      chunk_size = chunk_sizes.back();  // Use last size for remaining.
    }
    chunk_size = std::min(chunk_size, text.size() - offset);

    iree_string_view_t chunk =
        iree_make_string_view(text.data() + offset, chunk_size);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    iree_status_t status = iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &token_count);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_encode_state_deinitialize(state);
      return status;
    }

    offset += bytes_consumed;
    total_tokens += token_count;
    ++chunk_index;

    // Safety: ensure progress or detect deadlock.
    if (bytes_consumed == 0 && chunk_size > 0) {
      // No progress - buffer may be full. Try with remaining output capacity.
      if (total_tokens >= tokens.size()) {
        iree_tokenizer_encode_state_deinitialize(state);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "output buffer exhausted");
      }
    }
  }

  // Finalize.
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

// Encodes text byte-by-byte using the streaming API.
static iree::StatusOr<std::vector<iree_tokenizer_token_id_t>> EncodeByteByByte(
    iree_tokenizer_t* tokenizer, const std::string& text,
    size_t transform_buffer_size) {
  return EncodeStreaming(tokenizer, text, {1}, transform_buffer_size);
}

// Encodes text with random chunk sizes using the streaming API.
static iree::StatusOr<std::vector<iree_tokenizer_token_id_t>>
EncodeRandomChunks(iree_tokenizer_t* tokenizer, const std::string& text,
                   size_t transform_buffer_size, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> dist(1, 64);

  std::vector<size_t> chunk_sizes;
  size_t total = 0;
  while (total < text.size()) {
    size_t size = std::min(dist(rng), text.size() - total);
    chunk_sizes.push_back(size);
    total += size;
  }

  return EncodeStreaming(tokenizer, text, chunk_sizes, transform_buffer_size);
}

// Helper to format token vectors for comparison output.
static std::string FormatTokens(
    const std::vector<iree_tokenizer_token_id_t>& tokens) {
  std::string result = "[";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(tokens[i]);
    if (i > 20) {
      result += ", ...(" + std::to_string(tokens.size() - i - 1) + " more)";
      break;
    }
  }
  result += "]";
  return result;
}

//===----------------------------------------------------------------------===//
// Test Fixture: Byte-Level BPE (No Merges)
//===----------------------------------------------------------------------===//

// Simple byte-level tokenizer: each byte becomes a token (ID = byte value).
// This isolates streaming correctness from BPE merge complexity.
class ByteLevelStreamingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ScopedVocabBuilder vocab_builder(256);
    for (int i = 0; i < 256; ++i) {
      char byte_token[2] = {static_cast<char>(i), '\0'};
      vocab_builder.AddToken(i, byte_token);
    }
    vocab_ = vocab_builder.Build();

    ScopedBuilder builder;
    iree_tokenizer_builder_set_segmenter(builder.get(),
                                         CreateWhitespaceSegmenter());
    iree_tokenizer_builder_set_model(builder.get(),
                                     CreateBPEModelIgnoreMerges(vocab_.get()));
    iree_tokenizer_builder_set_vocab(builder.get(), vocab_.release());

    IREE_ASSERT_OK_AND_ASSIGN(tokenizer_, BuildTokenizerOr(builder.get()));
  }

  void TearDown() override { tokenizer_.reset(); }

  TokenizerPtr tokenizer_;
  ScopedVocab vocab_;
};

TEST_F(ByteLevelStreamingTest, EmptyInput) {
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), ""));
  IREE_ASSERT_OK_AND_ASSIGN(auto streaming,
                            EncodeByteByByte(tokenizer_.get(), "", 4096));
  EXPECT_EQ(batch, streaming);
  EXPECT_TRUE(batch.empty());
}

TEST_F(ByteLevelStreamingTest, SingleByte) {
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), "a"));
  IREE_ASSERT_OK_AND_ASSIGN(auto streaming,
                            EncodeByteByByte(tokenizer_.get(), "a", 4096));
  EXPECT_EQ(batch, streaming) << "Batch: " << FormatTokens(batch)
                              << ", Streaming: " << FormatTokens(streaming);
}

TEST_F(ByteLevelStreamingTest, SimpleASCII) {
  std::string text = "hello world";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));
  IREE_ASSERT_OK_AND_ASSIGN(auto streaming,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  EXPECT_EQ(batch, streaming) << "Batch: " << FormatTokens(batch)
                              << ", Streaming: " << FormatTokens(streaming);
}

TEST_F(ByteLevelStreamingTest, MultipleWords) {
  std::string text = "the quick brown fox jumps over the lazy dog";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  // Test with various chunking strategies.
  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto all_at_once,
      EncodeStreaming(tokenizer_.get(), text, {text.size()}, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(auto fixed_chunks,
                            EncodeStreaming(tokenizer_.get(), text, {5}, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto random_chunks,
      EncodeRandomChunks(tokenizer_.get(), text, 4096, 12345));

  EXPECT_EQ(batch, byte_by_byte) << "byte-by-byte mismatch";
  EXPECT_EQ(batch, all_at_once) << "all-at-once mismatch";
  EXPECT_EQ(batch, fixed_chunks) << "fixed-chunks mismatch";
  EXPECT_EQ(batch, random_chunks) << "random-chunks mismatch";
}

TEST_F(ByteLevelStreamingTest, UTF8MultiByteCharacters) {
  // UTF-8 text with 2-byte (é), 3-byte (中), and 4-byte (🎉) characters.
  std::string text = "café 中文 emoji: 🎉";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  // Byte-by-byte will split UTF-8 sequences - streaming must handle this.
  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto random_chunks, EncodeRandomChunks(tokenizer_.get(), text, 4096, 42));

  EXPECT_EQ(batch, byte_by_byte)
      << "UTF-8 byte-by-byte mismatch\n"
      << "Batch: " << FormatTokens(batch)
      << ", Streaming: " << FormatTokens(byte_by_byte);
  EXPECT_EQ(batch, random_chunks) << "UTF-8 random-chunks mismatch";
}

TEST_F(ByteLevelStreamingTest, CJKText) {
  std::string text = "今日は天気がいいですね。明天会更好吗？";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(auto small_buffer,
                            EncodeByteByByte(tokenizer_.get(), text, 256));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto random_chunks, EncodeRandomChunks(tokenizer_.get(), text, 4096, 99));

  EXPECT_EQ(batch, byte_by_byte) << "CJK byte-by-byte mismatch";
  EXPECT_EQ(batch, small_buffer) << "CJK small-buffer mismatch";
  EXPECT_EQ(batch, random_chunks) << "CJK random-chunks mismatch";
}

TEST_F(ByteLevelStreamingTest, WhitespaceVariations) {
  std::string text = "  hello   world  \t\n  foo  ";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto random_chunks,
      EncodeRandomChunks(tokenizer_.get(), text, 4096, 777));

  EXPECT_EQ(batch, byte_by_byte) << "Whitespace byte-by-byte mismatch";
  EXPECT_EQ(batch, random_chunks) << "Whitespace random-chunks mismatch";
}

TEST_F(ByteLevelStreamingTest, SmallTransformBuffer) {
  // Test with minimal transform buffer to stress ring buffer handling.
  std::string text = "hello world this is a test with longer text";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  // 64-byte buffer (32 bytes logical capacity).
  IREE_ASSERT_OK_AND_ASSIGN(auto small_buffer,
                            EncodeByteByByte(tokenizer_.get(), text, 64));
  EXPECT_EQ(batch, small_buffer)
      << "Small buffer (64) mismatch\n"
      << "Batch: " << FormatTokens(batch)
      << ", Streaming: " << FormatTokens(small_buffer);
}

TEST_F(ByteLevelStreamingTest, VaryingBufferSizes) {
  std::string text = "The quick brown fox jumps over the lazy dog.";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  // Test with power-of-two buffer sizes.
  for (size_t buffer_size : {64, 128, 256, 512, 1024, 4096, 65536}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto streaming, EncodeByteByByte(tokenizer_.get(), text, buffer_size));
    EXPECT_EQ(batch, streaming)
        << "Buffer size " << buffer_size << " mismatch\n"
        << "Batch: " << FormatTokens(batch)
        << ", Streaming: " << FormatTokens(streaming);
  }
}

//===----------------------------------------------------------------------===//
// Test Fixture: GPT-2 Style Regex Segmenter
//===----------------------------------------------------------------------===//

// Tests with GPT-2 style regex segmentation (more realistic tokenization).
class GPT2StyleStreamingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ScopedVocabBuilder vocab_builder(256);
    for (int i = 0; i < 256; ++i) {
      char byte_token[2] = {static_cast<char>(i), '\0'};
      vocab_builder.AddToken(i, byte_token);
    }
    vocab_ = vocab_builder.Build();

    const char* gpt2_pattern =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
        "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    IREE_ASSERT_OK_AND_ASSIGN(iree_tokenizer_segmenter_t * segmenter,
                              CreateSplitSegmenter(gpt2_pattern));

    ScopedBuilder builder;
    iree_tokenizer_builder_set_segmenter(builder.get(), segmenter);
    iree_tokenizer_builder_set_model(builder.get(),
                                     CreateBPEModelIgnoreMerges(vocab_.get()));
    iree_tokenizer_builder_set_vocab(builder.get(), vocab_.release());

    IREE_ASSERT_OK_AND_ASSIGN(tokenizer_, BuildTokenizerOr(builder.get()));
  }

  void TearDown() override { tokenizer_.reset(); }

  TokenizerPtr tokenizer_;
  ScopedVocab vocab_;
};

TEST_F(GPT2StyleStreamingTest, SimpleText) {
  std::string text = "Hello, world!";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto random_chunks,
      EncodeRandomChunks(tokenizer_.get(), text, 4096, 123));

  EXPECT_EQ(batch, byte_by_byte) << "GPT2 simple text mismatch";
  EXPECT_EQ(batch, random_chunks) << "GPT2 random chunks mismatch";
}

TEST_F(GPT2StyleStreamingTest, Contractions) {
  // GPT-2 regex has special handling for contractions.
  std::string text = "I'm don't won't can't shouldn't";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto random_chunks,
      EncodeRandomChunks(tokenizer_.get(), text, 4096, 456));

  EXPECT_EQ(batch, byte_by_byte) << "Contractions byte-by-byte mismatch";
  EXPECT_EQ(batch, random_chunks) << "Contractions random-chunks mismatch";
}

TEST_F(GPT2StyleStreamingTest, MixedContent) {
  // Mix of letters, numbers, punctuation, whitespace.
  std::string text = "Hello123!  Multiple   spaces\t\ttabs\n\nnewlines";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(auto small_buffer,
                            EncodeByteByByte(tokenizer_.get(), text, 128));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto random_chunks,
      EncodeRandomChunks(tokenizer_.get(), text, 4096, 789));

  EXPECT_EQ(batch, byte_by_byte) << "Mixed content byte-by-byte mismatch";
  EXPECT_EQ(batch, small_buffer) << "Mixed content small-buffer mismatch";
  EXPECT_EQ(batch, random_chunks) << "Mixed content random-chunks mismatch";
}

TEST_F(GPT2StyleStreamingTest, LongInput) {
  // Generate a longer input to stress streaming.
  std::string text;
  for (int i = 0; i < 100; ++i) {
    text += "The quick brown fox jumps over the lazy dog. ";
  }

  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto fixed_chunks, EncodeStreaming(tokenizer_.get(), text, {100}, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto random_chunks,
      EncodeRandomChunks(tokenizer_.get(), text, 4096, 999));

  EXPECT_EQ(batch, byte_by_byte) << "Long input byte-by-byte mismatch";
  EXPECT_EQ(batch, fixed_chunks) << "Long input fixed-chunks mismatch";
  EXPECT_EQ(batch, random_chunks) << "Long input random-chunks mismatch";
}

TEST_F(GPT2StyleStreamingTest, UTF8BoundarySplits) {
  // Text where random chunking will likely split UTF-8 sequences.
  std::string text = "café résumé naïve 中文 日本語 한국어";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  // Test multiple random seeds to exercise different split points.
  for (uint32_t seed : {1, 22, 333, 4444, 55555}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto random_chunks,
        EncodeRandomChunks(tokenizer_.get(), text, 4096, seed));
    EXPECT_EQ(batch, random_chunks)
        << "UTF-8 boundary split mismatch (seed=" << seed << ")\n"
        << "Batch: " << FormatTokens(batch)
        << ", Streaming: " << FormatTokens(random_chunks);
  }
}

//===----------------------------------------------------------------------===//
// Edge Case Tests
//===----------------------------------------------------------------------===//

class StreamingEdgeCaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ScopedVocabBuilder vocab_builder(256);
    for (int i = 0; i < 256; ++i) {
      char byte_token[2] = {static_cast<char>(i), '\0'};
      vocab_builder.AddToken(i, byte_token);
    }
    vocab_ = vocab_builder.Build();

    ScopedBuilder builder;
    iree_tokenizer_builder_set_segmenter(builder.get(),
                                         CreateWhitespaceSegmenter());
    iree_tokenizer_builder_set_model(builder.get(),
                                     CreateBPEModelIgnoreMerges(vocab_.get()));
    iree_tokenizer_builder_set_vocab(builder.get(), vocab_.release());

    IREE_ASSERT_OK_AND_ASSIGN(tokenizer_, BuildTokenizerOr(builder.get()));
  }

  void TearDown() override { tokenizer_.reset(); }

  TokenizerPtr tokenizer_;
  ScopedVocab vocab_;
};

TEST_F(StreamingEdgeCaseTest, SingleCharacterChunks) {
  // Each character as its own chunk.
  std::vector<std::string> inputs = {
      "a", "ab", "abc", "a b c", "hello", "hello world",
  };

  for (const auto& text : inputs) {
    IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));
    IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                              EncodeByteByByte(tokenizer_.get(), text, 4096));
    EXPECT_EQ(batch, byte_by_byte)
        << "Single char chunks failed for: \"" << text << "\"";
  }
}

TEST_F(StreamingEdgeCaseTest, ChunkAtEveryPosition) {
  // Test splitting "hello world" at every possible position.
  std::string text = "hello world";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  for (size_t split_pos = 1; split_pos < text.size(); ++split_pos) {
    std::vector<size_t> chunks = {split_pos, text.size() - split_pos};
    IREE_ASSERT_OK_AND_ASSIGN(
        auto streaming, EncodeStreaming(tokenizer_.get(), text, chunks, 4096));
    EXPECT_EQ(batch, streaming)
        << "Split at position " << split_pos << " failed\n"
        << "Batch: " << FormatTokens(batch)
        << ", Streaming: " << FormatTokens(streaming);
  }
}

TEST_F(StreamingEdgeCaseTest, ZeroLengthChunks) {
  // Verify behavior when some chunks would be zero-length (at end of input).
  std::string text = "hello";
  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));

  // Request more chunks than input length.
  std::vector<size_t> chunks = {1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1};  // 10 chunks of 1
  IREE_ASSERT_OK_AND_ASSIGN(
      auto streaming, EncodeStreaming(tokenizer_.get(), text, chunks, 4096));
  EXPECT_EQ(batch, streaming) << "Zero-length chunk handling failed";
}

TEST_F(StreamingEdgeCaseTest, HighByteValues) {
  // Test bytes 128-255 which are UTF-8 continuation bytes.
  std::string text;
  for (int i = 128; i < 256; ++i) {
    text += static_cast<char>(i);
  }

  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));
  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));
  IREE_ASSERT_OK_AND_ASSIGN(
      auto random_chunks,
      EncodeRandomChunks(tokenizer_.get(), text, 4096, 111));

  EXPECT_EQ(batch, byte_by_byte) << "High bytes byte-by-byte mismatch";
  EXPECT_EQ(batch, random_chunks) << "High bytes random-chunks mismatch";
}

TEST_F(StreamingEdgeCaseTest, AllByteValues) {
  // Every possible byte value 0-255.
  std::string text;
  for (int i = 0; i < 256; ++i) {
    text += static_cast<char>(i);
  }

  IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));
  IREE_ASSERT_OK_AND_ASSIGN(auto byte_by_byte,
                            EncodeByteByByte(tokenizer_.get(), text, 4096));

  EXPECT_EQ(batch, byte_by_byte) << "All bytes consistency mismatch\n"
                                 << "Batch size: " << batch.size()
                                 << ", Streaming size: " << byte_by_byte.size();
}

TEST_F(StreamingEdgeCaseTest, RepeatedFeedFinalizeCycles) {
  // Verify state reset works correctly for multiple encode cycles.
  std::vector<std::string> inputs = {"hello", "world", "test input"};

  for (const auto& text : inputs) {
    IREE_ASSERT_OK_AND_ASSIGN(auto batch, EncodeBatch(tokenizer_.get(), text));
    IREE_ASSERT_OK_AND_ASSIGN(auto streaming,
                              EncodeByteByByte(tokenizer_.get(), text, 4096));
    EXPECT_EQ(batch, streaming) << "Cycle failed for: \"" << text << "\"";
  }
}

}  // namespace
}  // namespace tokenizer
}  // namespace iree
