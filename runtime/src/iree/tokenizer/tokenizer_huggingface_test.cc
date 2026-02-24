// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HuggingFace JSON format tests for tokenizer.
//
// These tests load tokenizers from embedded HuggingFace JSON configs and verify
// correct behavior. They complement the builder-API tests in
// tokenizer_encode_test.cc by testing the full JSON loading pipeline with
// realistic configurations.
//
// Buffer/chunk invariance tests verify that tokenizer produces identical
// output across different transform buffer sizes and chunk feeding patterns.

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/testdata/streaming_testdata.h"
#include "iree/tokenizer/tokenizer.h"

namespace iree::tokenizer {
namespace {

//===----------------------------------------------------------------------===//
// Test Utilities
//===----------------------------------------------------------------------===//

static iree_string_view_t GetEmbeddedFile(const char* name) {
  const struct iree_file_toc_t* toc =
      iree_tokenizer_streaming_testdata_create();
  for (size_t i = 0; i < iree_tokenizer_streaming_testdata_size(); ++i) {
    if (strcmp(toc[i].name, name) == 0) {
      return iree_make_string_view(toc[i].data, toc[i].size);
    }
  }
  return iree_string_view_empty();
}

struct TokenizerDeleter {
  void operator()(iree_tokenizer_t* t) const { iree_tokenizer_free(t); }
};
using ScopedTokenizer = std::unique_ptr<iree_tokenizer_t, TokenizerDeleter>;

static iree::StatusOr<ScopedTokenizer> LoadTokenizer(const char* filename) {
  iree_string_view_t json = GetEmbeddedFile(filename);
  if (json.size == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "file '%s' not in testdata",
                            filename);
  }
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_RETURN_IF_ERROR(iree_tokenizer_from_huggingface_json(
      json, iree_allocator_system(), &tokenizer));
  return ScopedTokenizer(tokenizer);
}

// Encodes text using the one-shot API.
static StatusOr<std::vector<iree_tokenizer_token_id_t>> EncodeOneShot(
    iree_tokenizer_t* tokenizer, iree_string_view_t text) {
  std::vector<iree_tokenizer_token_id_t> ids(256);
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode(tokenizer, text, IREE_TOKENIZER_ENCODE_FLAG_NONE,
                            iree_tokenizer_make_token_output(
                                ids.data(), nullptr, nullptr, ids.size()),
                            iree_allocator_system(), &count));
  ids.resize(count);
  return ids;
}

// Encodes text using streaming API with specified transform buffer size.
static StatusOr<std::vector<iree_tokenizer_token_id_t>> EncodeStreaming(
    iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_host_size_t buffer_size, iree_host_size_t chunk_size = 0) {
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

  std::vector<iree_tokenizer_token_id_t> all_ids;
  std::vector<iree_tokenizer_token_id_t> batch_ids(256);
  iree_host_size_t offset = 0;

  while (offset < text.size) {
    iree_host_size_t remaining = text.size - offset;
    iree_host_size_t feed_size =
        (chunk_size > 0 && chunk_size < remaining) ? chunk_size : remaining;

    iree_string_view_t chunk =
        iree_make_string_view(text.data + offset, feed_size);

    iree_host_size_t consumed = 0;
    iree_host_size_t produced = 0;
    iree_status_t status = iree_tokenizer_encode_state_feed(
        state, chunk,
        iree_tokenizer_make_token_output(batch_ids.data(), nullptr, nullptr,
                                         batch_ids.size()),
        &consumed, &produced);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_encode_state_deinitialize(state);
      return status;
    }

    for (iree_host_size_t i = 0; i < produced; ++i) {
      all_ids.push_back(batch_ids[i]);
    }
    offset += consumed;

    if (consumed == 0 && produced == 0 && feed_size > 0) {
      iree_tokenizer_encode_state_deinitialize(state);
      return iree_make_status(IREE_STATUS_INTERNAL, "no progress at offset %zu",
                              offset);
    }
  }

  // Drain finalize - may produce multiple batches for large outputs.
  iree_host_size_t finalize_count = 0;
  do {
    iree_status_t status = iree_tokenizer_encode_state_finalize(
        state,
        iree_tokenizer_make_token_output(batch_ids.data(), nullptr, nullptr,
                                         batch_ids.size()),
        &finalize_count);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_encode_state_deinitialize(state);
      return status;
    }
    for (iree_host_size_t i = 0; i < finalize_count; ++i) {
      all_ids.push_back(batch_ids[i]);
    }
  } while (finalize_count == batch_ids.size());

  iree_tokenizer_encode_state_deinitialize(state);
  return all_ids;
}

//===----------------------------------------------------------------------===//
// BPE ByteLevel Tests
//===----------------------------------------------------------------------===//

class BPEByteLevelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto result = LoadTokenizer("bpe_bytelevel_minimal.json");
    if (!result.ok()) {
      GTEST_SKIP() << result.status().ToString();
    }
    tokenizer_ = std::move(result.value());
  }

  iree_tokenizer_t* tokenizer() { return tokenizer_.get(); }

 private:
  ScopedTokenizer tokenizer_;
};

TEST_F(BPEByteLevelTest, LoadsSuccessfully) { ASSERT_NE(tokenizer(), nullptr); }

TEST_F(BPEByteLevelTest, EncodeHelloWorld) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ids,
                            EncodeOneShot(tokenizer(), IREE_SV("hello world")));
  // Correct BPE merge order: "hello" -> ["he", "llo"], " world" -> ["Ġworld"]
  // The merge "l l" at rank 1 fires before "he l" at rank 9, so "hello"
  // tokenizes as ["he", "llo"] not ["hello"].
  EXPECT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 98);   // he
  EXPECT_EQ(ids[1], 105);  // llo
  EXPECT_EQ(ids[2], 110);  // Ġworld
}

TEST_F(BPEByteLevelTest, BufferInvarianceHelloWorld) {
  iree_string_view_t text = IREE_SV("hello world");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // Test various buffer sizes - all should produce same output.
  for (iree_host_size_t buffer_size : {256, 512, 1024, 4096, 16384}) {
    IREE_ASSERT_OK_AND_ASSIGN(auto result,
                              EncodeStreaming(tokenizer(), text, buffer_size));
    EXPECT_EQ(result, reference)
        << "Buffer size " << buffer_size << " differs from one-shot";
  }
}

TEST_F(BPEByteLevelTest, ChunkInvarianceHelloWorld) {
  iree_string_view_t text = IREE_SV("hello world");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // Test various chunk sizes with fixed buffer.
  for (iree_host_size_t chunk_size : {1, 2, 3, 4, 5, 7, 11}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size << " differs from one-shot";
  }
}

TEST_F(BPEByteLevelTest, BufferInvarianceTheQuickBrown) {
  iree_string_view_t text = IREE_SV("the quick brown fox");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  for (iree_host_size_t buffer_size : {256, 512, 1024, 4096}) {
    IREE_ASSERT_OK_AND_ASSIGN(auto result,
                              EncodeStreaming(tokenizer(), text, buffer_size));
    EXPECT_EQ(result, reference)
        << "Buffer size " << buffer_size << " differs from one-shot";
  }
}

TEST_F(BPEByteLevelTest, ChunkInvarianceTheQuickBrown) {
  iree_string_view_t text = IREE_SV("the quick brown fox");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  for (iree_host_size_t chunk_size : {1, 2, 3, 5, 7}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size << " differs from one-shot";
  }
}

TEST_F(BPEByteLevelTest, EmptyInput) {
  IREE_ASSERT_OK_AND_ASSIGN(
      auto ids, EncodeOneShot(tokenizer(), iree_string_view_empty()));
  EXPECT_TRUE(ids.empty());

  IREE_ASSERT_OK_AND_ASSIGN(
      auto streaming,
      EncodeStreaming(tokenizer(), iree_string_view_empty(), 4096));
  EXPECT_TRUE(streaming.empty());
}

TEST_F(BPEByteLevelTest, SingleCharacter) {
  iree_string_view_t text = IREE_SV("a");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  IREE_ASSERT_OK_AND_ASSIGN(auto streaming,
                            EncodeStreaming(tokenizer(), text, 256));
  EXPECT_EQ(streaming, reference);
}

TEST_F(BPEByteLevelTest, RepeatedCharacters) {
  iree_string_view_t text = IREE_SV("aaaaaaaaaa");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  for (iree_host_size_t buffer_size : {256, 512, 1024}) {
    IREE_ASSERT_OK_AND_ASSIGN(auto result,
                              EncodeStreaming(tokenizer(), text, buffer_size));
    EXPECT_EQ(result, reference)
        << "Buffer size " << buffer_size << " differs for repeated chars";
  }
}

TEST_F(BPEByteLevelTest, MultipleSpaces) {
  iree_string_view_t text = IREE_SV("hello  world");  // Two spaces.
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  for (iree_host_size_t chunk_size : {1, 2, 3}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size << " differs for multiple spaces";
  }
}

// Tests that " the" correctly merges through the chain: Ġ+t -> Ġt -> Ġth ->
// Ġthe even when the space and word are split across chunk boundaries.
TEST_F(BPEByteLevelTest, SpacePrefixedWordMerge) {
  iree_string_view_t text = IREE_SV("hello the world");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // Chunk sizes that force " the" to split at different points:
  // size 5: "hello" | " the " | "world" - space at chunk boundary
  // size 6: "hello " | "the wo" | "rld" - after space
  // size 7: "hello t" | "he worl" | "d" - mid-word
  for (iree_host_size_t chunk_size : {1, 2, 3, 5, 6, 7}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size << " differs for space-prefixed word";
  }
}

// Tests special token recognition when the token spans chunk boundaries.
// "<|endoftext|>" is 13 characters - splitting it should still produce one
// token.
TEST_F(BPEByteLevelTest, SpecialTokenSpanning) {
  iree_string_view_t text = IREE_SV("<|endoftext|>");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));
  // Special token should be recognized as a single token (ID 111).
  EXPECT_EQ(reference.size(), 1u);
  EXPECT_EQ(reference[0], 111);

  // Feed the special token character by character.
  for (iree_host_size_t chunk_size : {1, 2, 3, 4, 5, 6, 7}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size << " differs for special token spanning";
  }
}

//===----------------------------------------------------------------------===//
// Special Token Negative Tests
//===----------------------------------------------------------------------===//

// Tests that text around special tokens doesn't incorrectly match.
// "x<|endoftext|>y" should NOT match the special token.
TEST_F(BPEByteLevelTest, SpecialTokenNegative) {
  iree_string_view_t text = IREE_SV("x<|endoftext|>y");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));
  // Should NOT be a single token - the surrounding chars prevent special match.
  EXPECT_GT(reference.size(), 1u);

  for (iree_host_size_t chunk_size : {1, 2, 3}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size << " differs for special token negative";
  }
}

// Tests behavior at exact buffer boundaries - fill buffer exactly then
// overflow.
TEST_F(BPEByteLevelTest, BufferBoundaryExact) {
  // 64 'a' characters to test exact buffer fill and slight overflow.
  std::string text_64(64, 'a');
  std::string text_65(65, 'a');
  std::string text_128(128, 'a');
  std::string text_129(129, 'a');

  for (const auto& text : {text_64, text_65, text_128, text_129}) {
    iree_string_view_t sv = iree_make_string_view(text.data(), text.size());
    IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), sv));

    // Test with buffer sizes that match or nearly match input size.
    for (iree_host_size_t buffer_size : {64, 128, 256}) {
      IREE_ASSERT_OK_AND_ASSIGN(auto result,
                                EncodeStreaming(tokenizer(), sv, buffer_size));
      EXPECT_EQ(result, reference)
          << "Buffer size " << buffer_size << " differs for " << text.size()
          << " chars";
    }
  }
}

// Tests streaming with empty chunks interspersed between content.
TEST_F(BPEByteLevelTest, EmptyChunks) {
  iree_string_view_t text = IREE_SV("hello");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // Feed character by character - the implementation should handle this.
  IREE_ASSERT_OK_AND_ASSIGN(auto streaming,
                            EncodeStreaming(tokenizer(), text, 4096, 1));
  EXPECT_EQ(streaming, reference);
}

//===----------------------------------------------------------------------===//
// Post-Normalization Special Token Tests
//===----------------------------------------------------------------------===//

// Post-normalization special tokens have normalized=true and are matched AFTER
// the normalizer runs. This test suite verifies correct behavior when these
// tokens appear in various positions and streaming scenarios.

class PostNormSpecialTokenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Minimal BPE tokenizer with:
    // - Regular vocab tokens for basic text
    // - A post-normalization special token (normalized=true)
    const char* json = R"({
      "model": {
        "type": "BPE",
        "vocab": {
          "h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7,
          "he": 8, "ll": 9, "lo": 10,
          "<|special|>": 100
        },
        "merges": ["h e", "l l", "l o"]
      },
      "added_tokens": [{
        "id": 100,
        "content": "<|special|>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": true,
        "special": true
      }],
      "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
    })";
    iree_tokenizer_t* tokenizer = nullptr;
    iree_status_t status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view(json, strlen(json)), iree_allocator_system(),
        &tokenizer);
    IREE_CHECK_OK(status);
    tokenizer_.reset(tokenizer);
  }

  iree_tokenizer_t* tokenizer() { return tokenizer_.get(); }

 private:
  ScopedTokenizer tokenizer_;
};

// Verifies that post-norm special tokens appearing after regular text are
// correctly matched. When streaming, the special token may be encountered while
// segments from earlier text are still pending for the model. The
// implementation must not corrupt ring buffer state by advancing read_position
// prematurely.
TEST_F(PostNormSpecialTokenTest, SpecialTokenAfterText) {
  iree_string_view_t text = IREE_SV("hello <|special|>");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // The last token should be the special token (id=100).
  EXPECT_EQ(reference.back(), 100u);

  // Streaming with various chunk sizes must produce identical results.
  for (iree_host_size_t chunk_size : {1, 2, 3, 5, 7}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size
        << " produced different result for post-norm special token after text";
  }
}

// Verifies that post-norm special tokens at the start of input work correctly.
TEST_F(PostNormSpecialTokenTest, SpecialTokenAtStart) {
  iree_string_view_t text = IREE_SV("<|special|>hello");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // The first token should be the special token (id=100).
  EXPECT_EQ(reference.front(), 100u);

  for (iree_host_size_t chunk_size : {1, 2, 3}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size
        << " produced different result for post-norm special token at start";
  }
}

// Verifies that multiple post-norm special tokens in sequence work correctly.
TEST_F(PostNormSpecialTokenTest, MultipleSpecialTokens) {
  iree_string_view_t text = IREE_SV("<|special|><|special|>");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // Should be exactly two special tokens.
  EXPECT_EQ(reference.size(), 2u);
  EXPECT_EQ(reference[0], 100u);
  EXPECT_EQ(reference[1], 100u);

  for (iree_host_size_t chunk_size : {1, 2, 3}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size
        << " produced different result for multiple post-norm special tokens";
  }
}

// Verifies that streaming with very small buffers handles post-norm tokens
// correctly when interleaved with regular text. This stresses the ring buffer
// management when segments are pending.
TEST_F(PostNormSpecialTokenTest, SmallBufferWithInterleavedTokens) {
  iree_string_view_t text = IREE_SV("hello <|special|> world");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // Use a small buffer to stress the ring buffer management.
  // The special token should appear somewhere in the middle.
  bool found_special = false;
  for (auto id : reference) {
    if (id == 100u) found_special = true;
  }
  EXPECT_TRUE(found_special) << "Special token not found in reference output";

  // Small buffer sizes stress the segment queuing logic.
  for (iree_host_size_t buffer_size : {64, 128, 256}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, buffer_size, 1));
    EXPECT_EQ(result, reference)
        << "Buffer size " << buffer_size
        << " with 1-byte chunks produced different result";
  }
}

//===----------------------------------------------------------------------===//
// Whitespace Pre-Tokenizer Tests
//===----------------------------------------------------------------------===//

// Tests that the Whitespace pre_tokenizer works correctly with streaming
// encoding, including proper word boundary detection across chunk boundaries.

class WhitespacePreTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // BPE tokenizer with Whitespace pre_tokenizer.
    // Vocab and merges must be consistent: every merge operand must exist.
    // "hello" merges: h+e→he, l+l→ll, ll+o→llo, he+llo→hello
    // "world" merges: w+o→wo, r+l→rl, rl+d→rld, wo+rld→world
    const char* json = R"({
      "model": {
        "type": "BPE",
        "vocab": {
          "h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6,
          "he": 10, "ll": 11, "wo": 12, "rl": 13,
          "llo": 20, "rld": 21,
          "hello": 30, "world": 31
        },
        "merges": [
          "h e", "l l", "w o", "r l",
          "ll o", "rl d",
          "he llo", "wo rld"
        ]
      },
      "pre_tokenizer": {"type": "Whitespace"}
    })";
    iree_tokenizer_t* tokenizer = nullptr;
    iree_status_t status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view(json, strlen(json)), iree_allocator_system(),
        &tokenizer);
    IREE_CHECK_OK(status);
    tokenizer_.reset(tokenizer);
  }

  iree_tokenizer_t* tokenizer() { return tokenizer_.get(); }

 private:
  ScopedTokenizer tokenizer_;
};

TEST_F(WhitespacePreTokenizerTest, LoadsSuccessfully) {
  EXPECT_NE(tokenizer(), nullptr);
}

TEST_F(WhitespacePreTokenizerTest, SingleWord) {
  iree_string_view_t text = IREE_SV("hello");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));
  // "hello" should merge to a single token (ID 30).
  EXPECT_EQ(reference.size(), 1u);
  EXPECT_EQ(reference[0], 30);
}

TEST_F(WhitespacePreTokenizerTest, TwoWords) {
  iree_string_view_t text = IREE_SV("hello world");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));
  // "hello world" -> ["hello", "world"] -> [30, 31].
  EXPECT_EQ(reference.size(), 2u);
  EXPECT_EQ(reference[0], 30);
  EXPECT_EQ(reference[1], 31);
}

TEST_F(WhitespacePreTokenizerTest, StreamingChunkInvariance) {
  iree_string_view_t text = IREE_SV("hello world");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // Test various chunk sizes that split at different positions:
  // "h|ello world", "he|llo world", "hel|lo world", "hell|o world",
  // "hello| world", "hello |world", etc.
  for (iree_host_size_t chunk_size = 1; chunk_size <= 11; ++chunk_size) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size << " produced different result";
  }
}

TEST_F(WhitespacePreTokenizerTest, StreamingBufferInvariance) {
  iree_string_view_t text = IREE_SV("hello world");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));

  // Test various buffer sizes that affect ring buffer behavior.
  for (iree_host_size_t buffer_size : {16, 32, 64, 128, 256, 4096}) {
    IREE_ASSERT_OK_AND_ASSIGN(auto result,
                              EncodeStreaming(tokenizer(), text, buffer_size));
    EXPECT_EQ(result, reference)
        << "Buffer size " << buffer_size << " produced different result";
  }
}

TEST_F(WhitespacePreTokenizerTest, MultipleSpaces) {
  iree_string_view_t text = IREE_SV("hello    world");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));
  // Multiple spaces between words should still produce two tokens.
  EXPECT_EQ(reference.size(), 2u);

  for (iree_host_size_t chunk_size : {1, 2, 3, 5}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference)
        << "Chunk size " << chunk_size << " differs for multiple spaces";
  }
}

TEST_F(WhitespacePreTokenizerTest, LeadingAndTrailingWhitespace) {
  iree_string_view_t text = IREE_SV("  hello world  ");
  IREE_ASSERT_OK_AND_ASSIGN(auto reference, EncodeOneShot(tokenizer(), text));
  // Leading and trailing whitespace should be ignored.
  EXPECT_EQ(reference.size(), 2u);

  for (iree_host_size_t chunk_size : {1, 2, 3, 5}) {
    IREE_ASSERT_OK_AND_ASSIGN(
        auto result, EncodeStreaming(tokenizer(), text, 4096, chunk_size));
    EXPECT_EQ(result, reference) << "Chunk size " << chunk_size
                                 << " differs for leading/trailing whitespace";
  }
}

TEST_F(WhitespacePreTokenizerTest, WhitespaceSplitAlias) {
  // WhitespaceSplit is an alias for Whitespace - verify it loads successfully.
  const char* json = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0},
      "merges": []
    },
    "pre_tokenizer": {"type": "WhitespaceSplit"}
  })";
  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      iree_make_string_view(json, strlen(json)), iree_allocator_system(),
      &tokenizer);
  IREE_EXPECT_OK(status);
  if (tokenizer) iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// HuggingFace Ground Truth Validation Tests
//===----------------------------------------------------------------------===//
// These tests verify that IREE's tokenizer produces IDENTICAL output to
// HuggingFace across multiple buffer sizes. Expected token IDs are generated
// from the HuggingFace tokenizers library:
//   uv run --with tokenizers python generate_testdata_expected_ids.py

// Buffer sizes to test. Smaller sizes force partial-segment mode.
constexpr iree_host_size_t kGroundTruthBufferSizes[] = {64, 128, 256, 512,
                                                        65536};

// Expected token IDs from HuggingFace tokenizers library.
// "hello" -> ['he', 'llo']
static constexpr iree_tokenizer_token_id_t kHelloExpected[] = {98, 105};

// "hello world" -> ['he', 'llo', 'Ġworld']
static constexpr iree_tokenizer_token_id_t kHelloWorldExpected[] = {98, 105,
                                                                    110};

// "The quick brown fox" -> individual chars with Ġ for spaces
static constexpr iree_tokenizer_token_id_t kSimpleASCIIExpected[] = {
    51, 98, 94, 80, 84, 72, 66, 74, 94, 65, 81, 78, 86, 77, 94, 69, 78, 87};

// "hello<|endoftext|>world" -> ['he', 'llo', '<|endoftext|>', 'w', 'orld']
static constexpr iree_tokenizer_token_id_t kWithSpecialExpected[] = {
    98, 105, 111, 86, 108};

// "  hello   world  " -> spaces become Ġ tokens
static constexpr iree_tokenizer_token_id_t kWhitespaceExpected[] = {
    94, 94, 98, 105, 94, 94, 110, 94, 94};

// "a" -> ['a']
static constexpr iree_tokenizer_token_id_t kSingleCharExpected[] = {64};

// "12345" -> ['1', '2', '3', '4', '5']
static constexpr iree_tokenizer_token_id_t kNumbersExpected[] = {16, 17, 18, 19,
                                                                 20};

// "!@#$%^&*()" -> individual punctuation tokens
static constexpr iree_tokenizer_token_id_t kPunctuationExpected[] = {
    0, 31, 2, 3, 4, 61, 5, 9, 7, 8};

// Parameterized test fixture for ground truth validation.
class GroundTruthTest : public ::testing::TestWithParam<iree_host_size_t> {
 protected:
  void SetUp() override {
    iree_string_view_t json_str = GetEmbeddedFile("bpe_bytelevel_minimal.json");
    ASSERT_GT(json_str.size, 0u) << "Tokenizer JSON not found";
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        json_str, iree_allocator_system(), &tokenizer_));
  }

  void TearDown() override {
    if (tokenizer_) {
      iree_tokenizer_free(tokenizer_);
    }
  }

  // Encode input text using the current buffer size (from parameterized test).
  std::vector<iree_tokenizer_token_id_t> EncodeWithBufferSize(
      const char* input, iree_host_size_t buffer_size) {
    iree_host_size_t state_size = 0;
    IREE_EXPECT_OK(
        iree_tokenizer_encode_state_calculate_size(tokenizer_, &state_size));

    std::vector<uint8_t> state_storage(state_size);
    std::vector<uint8_t> transform_buffer(buffer_size);

    iree_tokenizer_encode_state_t* state = nullptr;
    IREE_EXPECT_OK(iree_tokenizer_encode_state_initialize(
        tokenizer_,
        iree_make_byte_span(state_storage.data(), state_storage.size()),
        iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
        iree_tokenizer_offset_run_list_empty(),
        IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

    std::vector<iree_tokenizer_token_id_t> tokens(1024);
    iree_host_size_t total_tokens = 0;

    iree_string_view_t input_view = iree_make_string_view(input, strlen(input));
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;

    IREE_EXPECT_OK(iree_tokenizer_encode_state_feed(
        state, input_view,
        iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                         tokens.size()),
        &bytes_consumed, &token_count));
    total_tokens += token_count;

    iree_host_size_t finalize_count = 0;
    IREE_EXPECT_OK(iree_tokenizer_encode_state_finalize(
        state,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &finalize_count));
    total_tokens += finalize_count;

    tokens.resize(total_tokens);
    iree_tokenizer_encode_state_deinitialize(state);

    return tokens;
  }

  // Encode using byte-by-byte feeding (most stressful streaming test).
  std::vector<iree_tokenizer_token_id_t> EncodeByteByByte(
      const char* input, iree_host_size_t buffer_size) {
    iree_host_size_t state_size = 0;
    IREE_EXPECT_OK(
        iree_tokenizer_encode_state_calculate_size(tokenizer_, &state_size));

    std::vector<uint8_t> state_storage(state_size);
    std::vector<uint8_t> transform_buffer(buffer_size);

    iree_tokenizer_encode_state_t* state = nullptr;
    IREE_EXPECT_OK(iree_tokenizer_encode_state_initialize(
        tokenizer_,
        iree_make_byte_span(state_storage.data(), state_storage.size()),
        iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
        iree_tokenizer_offset_run_list_empty(),
        IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

    std::vector<iree_tokenizer_token_id_t> tokens(1024);
    iree_host_size_t total_tokens = 0;
    size_t input_length = strlen(input);

    for (size_t i = 0; i < input_length; ++i) {
      iree_string_view_t chunk = iree_make_string_view(input + i, 1);
      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t token_count = 0;

      IREE_EXPECT_OK(iree_tokenizer_encode_state_feed(
          state, chunk,
          iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                           NULL, tokens.size() - total_tokens),
          &bytes_consumed, &token_count));
      total_tokens += token_count;
    }

    iree_host_size_t finalize_count = 0;
    IREE_EXPECT_OK(iree_tokenizer_encode_state_finalize(
        state,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &finalize_count));
    total_tokens += finalize_count;

    tokens.resize(total_tokens);
    iree_tokenizer_encode_state_deinitialize(state);

    return tokens;
  }

  iree_tokenizer_t* tokenizer_ = nullptr;
};

TEST_P(GroundTruthTest, HelloWorld) {
  iree_host_size_t buffer_size = GetParam();
  auto tokens = EncodeWithBufferSize("hello world", buffer_size);

  std::vector<iree_tokenizer_token_id_t> expected(
      kHelloWorldExpected,
      kHelloWorldExpected +
          sizeof(kHelloWorldExpected) / sizeof(kHelloWorldExpected[0]));

  EXPECT_EQ(tokens, expected)
      << "Buffer size: " << buffer_size << ", input: 'hello world'";
}

TEST_P(GroundTruthTest, HelloWorldByteByByte) {
  iree_host_size_t buffer_size = GetParam();
  auto tokens = EncodeByteByByte("hello world", buffer_size);

  std::vector<iree_tokenizer_token_id_t> expected(
      kHelloWorldExpected,
      kHelloWorldExpected +
          sizeof(kHelloWorldExpected) / sizeof(kHelloWorldExpected[0]));

  EXPECT_EQ(tokens, expected) << "Buffer size: " << buffer_size
                              << ", input: 'hello world' (byte-by-byte)";
}

TEST_P(GroundTruthTest, SimpleASCII) {
  iree_host_size_t buffer_size = GetParam();
  auto tokens = EncodeWithBufferSize("The quick brown fox", buffer_size);

  std::vector<iree_tokenizer_token_id_t> expected(
      kSimpleASCIIExpected,
      kSimpleASCIIExpected +
          sizeof(kSimpleASCIIExpected) / sizeof(kSimpleASCIIExpected[0]));

  EXPECT_EQ(tokens, expected)
      << "Buffer size: " << buffer_size << ", input: 'The quick brown fox'";
}

TEST_P(GroundTruthTest, WithSpecialToken) {
  iree_host_size_t buffer_size = GetParam();
  auto tokens = EncodeWithBufferSize("hello<|endoftext|>world", buffer_size);

  std::vector<iree_tokenizer_token_id_t> expected(
      kWithSpecialExpected,
      kWithSpecialExpected +
          sizeof(kWithSpecialExpected) / sizeof(kWithSpecialExpected[0]));

  EXPECT_EQ(tokens, expected)
      << "Buffer size: " << buffer_size << ", input: 'hello<|endoftext|>world'";
}

TEST_P(GroundTruthTest, Whitespace) {
  iree_host_size_t buffer_size = GetParam();
  auto tokens = EncodeWithBufferSize("  hello   world  ", buffer_size);

  std::vector<iree_tokenizer_token_id_t> expected(
      kWhitespaceExpected,
      kWhitespaceExpected +
          sizeof(kWhitespaceExpected) / sizeof(kWhitespaceExpected[0]));

  EXPECT_EQ(tokens, expected)
      << "Buffer size: " << buffer_size << ", input: '  hello   world  '";
}

TEST_P(GroundTruthTest, SingleChar) {
  iree_host_size_t buffer_size = GetParam();
  auto tokens = EncodeWithBufferSize("a", buffer_size);

  std::vector<iree_tokenizer_token_id_t> expected(
      kSingleCharExpected,
      kSingleCharExpected +
          sizeof(kSingleCharExpected) / sizeof(kSingleCharExpected[0]));

  EXPECT_EQ(tokens, expected)
      << "Buffer size: " << buffer_size << ", input: 'a'";
}

TEST_P(GroundTruthTest, Numbers) {
  iree_host_size_t buffer_size = GetParam();
  auto tokens = EncodeWithBufferSize("12345", buffer_size);

  std::vector<iree_tokenizer_token_id_t> expected(
      kNumbersExpected, kNumbersExpected + sizeof(kNumbersExpected) /
                                               sizeof(kNumbersExpected[0]));

  EXPECT_EQ(tokens, expected)
      << "Buffer size: " << buffer_size << ", input: '12345'";
}

TEST_P(GroundTruthTest, Punctuation) {
  iree_host_size_t buffer_size = GetParam();
  auto tokens = EncodeWithBufferSize("!@#$%^&*()", buffer_size);

  std::vector<iree_tokenizer_token_id_t> expected(
      kPunctuationExpected,
      kPunctuationExpected +
          sizeof(kPunctuationExpected) / sizeof(kPunctuationExpected[0]));

  EXPECT_EQ(tokens, expected)
      << "Buffer size: " << buffer_size << ", input: '!@#$%^&*()'";
}

TEST_P(GroundTruthTest, EmptyInput) {
  iree_host_size_t buffer_size = GetParam();
  auto tokens = EncodeWithBufferSize("", buffer_size);

  EXPECT_TRUE(tokens.empty()) << "Buffer size: " << buffer_size
                              << ", expected empty output for empty input";
}

// Run all parameterized tests with each buffer size.
INSTANTIATE_TEST_SUITE_P(
    BufferSizes, GroundTruthTest, ::testing::ValuesIn(kGroundTruthBufferSizes),
    [](const ::testing::TestParamInfo<iree_host_size_t>& info) {
      return "BufferSize" + std::to_string(info.param);
    });

// Helper to format strings for test output.
static std::string repr(const char* s) {
  std::string result = "'";
  for (const char* p = s; *p; ++p) {
    if (*p == '\n')
      result += "\\n";
    else if (*p == '\t')
      result += "\\t";
    else if (*p == '\'')
      result += "\\'";
    else if (*p < 32)
      result += "\\x" + std::to_string((unsigned char)*p);
    else
      result += *p;
  }
  result += "'";
  return result;
}

// Verifies that ALL buffer sizes produce IDENTICAL output.
TEST(GroundTruthInvarianceTest, AllBufferSizesProduceIdenticalOutput) {
  iree_string_view_t json_str = GetEmbeddedFile("bpe_bytelevel_minimal.json");
  ASSERT_GT(json_str.size, 0u);

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      json_str, iree_allocator_system(), &tokenizer));

  const char* test_inputs[] = {
      "hello world",
      "The quick brown fox jumps over the lazy dog",
      "hello<|endoftext|>world",
      "  lots   of   whitespace  ",
      "!@#$%^&*()",
      "12345",
      "",
  };

  for (const char* input : test_inputs) {
    std::vector<std::vector<iree_tokenizer_token_id_t>> results_per_size;

    for (iree_host_size_t buffer_size : kGroundTruthBufferSizes) {
      iree_host_size_t state_size = 0;
      IREE_EXPECT_OK(
          iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));

      std::vector<uint8_t> state_storage(state_size);
      std::vector<uint8_t> transform_buffer(buffer_size);

      iree_tokenizer_encode_state_t* state = nullptr;
      IREE_EXPECT_OK(iree_tokenizer_encode_state_initialize(
          tokenizer,
          iree_make_byte_span(state_storage.data(), state_storage.size()),
          iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
          iree_tokenizer_offset_run_list_empty(),
          IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

      std::vector<iree_tokenizer_token_id_t> tokens(1024);
      iree_host_size_t total_tokens = 0;

      iree_string_view_t input_view =
          iree_make_string_view(input, strlen(input));
      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t token_count = 0;

      IREE_EXPECT_OK(iree_tokenizer_encode_state_feed(
          state, input_view,
          iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                           tokens.size()),
          &bytes_consumed, &token_count));
      total_tokens += token_count;

      iree_host_size_t finalize_count = 0;
      IREE_EXPECT_OK(iree_tokenizer_encode_state_finalize(
          state,
          iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                           NULL, tokens.size() - total_tokens),
          &finalize_count));
      total_tokens += finalize_count;

      tokens.resize(total_tokens);
      iree_tokenizer_encode_state_deinitialize(state);

      results_per_size.push_back(tokens);
    }

    // Verify all buffer sizes produced identical results.
    for (size_t i = 1; i < results_per_size.size(); ++i) {
      EXPECT_EQ(results_per_size[0], results_per_size[i])
          << "Input: " << repr(input) << "\n"
          << "Buffer size " << kGroundTruthBufferSizes[0] << " produced "
          << results_per_size[0].size() << " tokens\n"
          << "Buffer size " << kGroundTruthBufferSizes[i] << " produced "
          << results_per_size[i].size() << " tokens";
    }
  }

  iree_tokenizer_free(tokenizer);
}

}  // namespace
}  // namespace iree::tokenizer
