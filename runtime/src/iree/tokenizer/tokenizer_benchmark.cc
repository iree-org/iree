// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks for tokenizer encode/decode performance.
//
// These benchmarks measure:
// - Single encode throughput at various text lengths
// - Batch encode throughput with state reuse
// - Streaming encode overhead vs batch
// - Cold start latency (including tokenizer construction)
// - Decode throughput via pre-decoded fast path
//
// Run with: iree-bazel-run //runtime/src/iree/tokenizer:tokenizer_benchmark
// Filter:   --benchmark_filter='Encode'
//
// Custom text file (via environment variable):
//   IREE_BENCHMARK_TEXT_FILE=/path/to/file.txt iree-bazel-run ...
//   When set, uses the file contents instead of generated text.
//   The text_bytes argument becomes a limit on how much of the file to use.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/tokenizer/decoder/byte_level.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/segmenter/whitespace.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

namespace {

//===----------------------------------------------------------------------===//
// Text Generation
//===----------------------------------------------------------------------===//

// Common English words for realistic text generation.
// Top 100 words by frequency, covering ~50% of typical English text.
static const char* const kCommonWords[] = {
    "the",   "of",    "and",  "to",    "a",     "in",    "is",   "it",
    "you",   "that",  "he",   "was",   "for",   "on",    "are",  "with",
    "as",    "his",   "they", "be",    "at",    "one",   "have", "this",
    "from",  "or",    "had",  "by",    "not",   "word",  "but",  "what",
    "some",  "we",    "can",  "out",   "other", "were",  "all",  "there",
    "when",  "up",    "use",  "your",  "how",   "said",  "an",   "each",
    "she",   "which", "do",   "their", "time",  "if",    "will", "way",
    "about", "many",  "then", "them",  "write", "would", "like", "so",
    "these", "her",   "long", "make",  "thing", "see",   "him",  "two",
    "has",   "look",  "more", "day",   "could", "go",    "come", "did",
    "no",    "most",  "my",   "over",  "such",  "our",   "man",  "me",
    "even",  "new",   "just", "only",  "any",   "know",  "take", "into",
    "year",  "good",  "give", "after",
};
static constexpr size_t kCommonWordCount =
    sizeof(kCommonWords) / sizeof(kCommonWords[0]);

// Generates a deterministic pseudo-random word (for rare tokens).
static std::string GenerateSyntheticWord(uint32_t seed) {
  size_t length = 3 + (seed >> 24) % 8;
  std::string word;
  word.reserve(length);
  uint32_t state = seed;
  for (size_t i = 0; i < length; ++i) {
    state = state * 1103515245 + 12345;
    word.push_back('a' + (state >> 16) % 26);
  }
  return word;
}

// Generates deterministic pseudo-English text of approximately target_length.
static std::string GenerateText(uint32_t seed, size_t target_length) {
  std::string result;
  result.reserve(target_length + 20);
  uint32_t state = seed;

  while (result.size() < target_length) {
    state = state * 1103515245 + 12345;

    // 70% common words, 30% synthetic (mimics rare tokens).
    if ((state >> 16) % 10 < 7) {
      result += kCommonWords[(state >> 8) % kCommonWordCount];
    } else {
      result += GenerateSyntheticWord(state);
    }
    result += " ";
  }

  return result;
}

// Loads text from file specified by IREE_BENCHMARK_TEXT_FILE env var,
// truncated to target_length. Returns empty string if env var not set or
// file cannot be read.
static std::string LoadTextFromEnvFile(size_t target_length) {
  const char* path = std::getenv("IREE_BENCHMARK_TEXT_FILE");
  if (!path || path[0] == '\0') return "";

  std::ifstream file(path, std::ios::binary);
  if (!file) return "";

  // Read up to target_length bytes.
  std::string result(target_length, '\0');
  file.read(result.data(), static_cast<std::streamsize>(target_length));
  result.resize(static_cast<size_t>(file.gcount()));
  return result;
}

// Returns text for benchmarking: from IREE_BENCHMARK_TEXT_FILE if set,
// otherwise generates pseudo-random text.
static std::string GetBenchmarkText(uint32_t seed, size_t target_length) {
  std::string text = LoadTextFromEnvFile(target_length);
  if (!text.empty()) return text;
  return GenerateText(seed, target_length);
}

//===----------------------------------------------------------------------===//
// Tokenizer Factory
//===----------------------------------------------------------------------===//

// Builds a tokenizer with the specified vocabulary size.
// Structure:
//   - 256 byte tokens (indices 0-255)
//   - Common English words as whole tokens
//   - Synthetic subword tokens to reach target size
//   - BPE merges connecting byte tokens to form common patterns
static iree_tokenizer_t* BuildBenchmarkTokenizer(size_t vocab_size) {
  iree_tokenizer_vocab_builder_t* vocab_builder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_vocab_builder_allocate(
      vocab_size, iree_allocator_system(), &vocab_builder));

  iree_tokenizer_token_id_t next_id = 0;

  // Add byte tokens (0-255).
  for (int byte_value = 0; byte_value < 256 && next_id < vocab_size;
       ++byte_value) {
    char byte_str[2] = {static_cast<char>(byte_value), '\0'};
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder, next_id++, iree_make_string_view(byte_str, 1), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  // Add common English words as single tokens.
  for (size_t i = 0; i < kCommonWordCount && next_id < vocab_size; ++i) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder, next_id++, iree_make_cstring_view(kCommonWords[i]), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  // Add BPE merge result tokens.
  const char* common_pairs[] = {"th", "he", "in", "er", "an",
                                "re", "on", "at", "en", "nd"};
  const size_t pair_count = sizeof(common_pairs) / sizeof(common_pairs[0]);
  for (size_t i = 0; i < pair_count && next_id < vocab_size; ++i) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder, next_id++, iree_make_string_view(common_pairs[i], 2),
        0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  // Fill remaining slots with synthetic tokens.
  uint32_t seed = 0x12345678;
  while (next_id < vocab_size) {
    seed = seed * 1103515245 + 12345;
    std::string token = GenerateSyntheticWord(seed);
    iree_status_t status = iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder, next_id,
        iree_make_string_view(token.data(), token.size()), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE);
    if (iree_status_is_ok(status)) {
      ++next_id;
    } else {
      iree_status_ignore(status);
    }
  }

  // Add merge rules linking byte tokens to merged tokens.
  for (size_t i = 0; i < pair_count; ++i) {
    iree_tokenizer_token_id_t left =
        static_cast<iree_tokenizer_token_id_t>(common_pairs[i][0]);
    iree_tokenizer_token_id_t right =
        static_cast<iree_tokenizer_token_id_t>(common_pairs[i][1]);
    IREE_CHECK_OK(
        iree_tokenizer_vocab_builder_add_merge(vocab_builder, left, right));
  }

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_CHECK_OK(iree_tokenizer_vocab_builder_build(vocab_builder, &vocab));

  // Create BPE model.
  iree_tokenizer_model_t* model = nullptr;
  IREE_CHECK_OK(iree_tokenizer_bpe_model_allocate(
      vocab, IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(), &model));

  // Create whitespace segmenter.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_CHECK_OK(iree_tokenizer_segmenter_whitespace_allocate(
      iree_allocator_system(), &segmenter));

  // Build tokenizer.
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(iree_allocator_system(), &builder);
  iree_tokenizer_builder_set_segmenter(&builder, segmenter);
  iree_tokenizer_builder_set_model(&builder, model);
  iree_tokenizer_builder_set_vocab(&builder, vocab);

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_builder_build(&builder, &tokenizer));
  iree_tokenizer_builder_deinitialize(&builder);

  return tokenizer;
}

// Builds a tokenizer with a ByteLevel decoder, enabling pre-decoded fast path.
// The ByteLevel decoder has STATELESS capability, so all vocab tokens get
// pre-decoded at build time — decode becomes a memcpy from a flat table.
static iree_tokenizer_t* BuildBenchmarkTokenizerWithDecoder(size_t vocab_size) {
  iree_tokenizer_vocab_builder_t* vocab_builder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_vocab_builder_allocate(
      vocab_size, iree_allocator_system(), &vocab_builder));

  iree_tokenizer_token_id_t next_id = 0;

  // Add byte tokens (0-255).
  for (int byte_value = 0; byte_value < 256 && next_id < vocab_size;
       ++byte_value) {
    char byte_str[2] = {static_cast<char>(byte_value), '\0'};
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder, next_id++, iree_make_string_view(byte_str, 1), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  // Add common English words as single tokens.
  for (size_t i = 0; i < kCommonWordCount && next_id < vocab_size; ++i) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder, next_id++, iree_make_cstring_view(kCommonWords[i]), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  // Add BPE merge result tokens.
  const char* common_pairs[] = {"th", "he", "in", "er", "an",
                                "re", "on", "at", "en", "nd"};
  const size_t pair_count = sizeof(common_pairs) / sizeof(common_pairs[0]);
  for (size_t i = 0; i < pair_count && next_id < vocab_size; ++i) {
    IREE_CHECK_OK(iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder, next_id++, iree_make_string_view(common_pairs[i], 2),
        0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }

  // Fill remaining slots with synthetic tokens.
  uint32_t seed = 0x12345678;
  while (next_id < vocab_size) {
    seed = seed * 1103515245 + 12345;
    std::string token = GenerateSyntheticWord(seed);
    iree_status_t status = iree_tokenizer_vocab_builder_add_token_with_id(
        vocab_builder, next_id,
        iree_make_string_view(token.data(), token.size()), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE);
    if (iree_status_is_ok(status)) {
      ++next_id;
    } else {
      iree_status_ignore(status);
    }
  }

  // Add merge rules linking byte tokens.
  for (size_t i = 0; i < pair_count; ++i) {
    iree_tokenizer_token_id_t left =
        static_cast<iree_tokenizer_token_id_t>(common_pairs[i][0]);
    iree_tokenizer_token_id_t right =
        static_cast<iree_tokenizer_token_id_t>(common_pairs[i][1]);
    IREE_CHECK_OK(
        iree_tokenizer_vocab_builder_add_merge(vocab_builder, left, right));
  }

  iree_tokenizer_vocab_t* vocab = nullptr;
  IREE_CHECK_OK(iree_tokenizer_vocab_builder_build(vocab_builder, &vocab));

  // Create BPE model.
  iree_tokenizer_model_t* model = nullptr;
  IREE_CHECK_OK(iree_tokenizer_bpe_model_allocate(
      vocab, IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(), &model));

  // Create whitespace segmenter.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_CHECK_OK(iree_tokenizer_segmenter_whitespace_allocate(
      iree_allocator_system(), &segmenter));

  // Create ByteLevel decoder (STATELESS — enables pre-decode fast path).
  iree_tokenizer_decoder_t* decoder = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decoder_byte_level_allocate(
      iree_allocator_system(), &decoder));

  // Build tokenizer.
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(iree_allocator_system(), &builder);
  iree_tokenizer_builder_set_segmenter(&builder, segmenter);
  iree_tokenizer_builder_set_model(&builder, model);
  iree_tokenizer_builder_set_decoder(&builder, decoder);
  iree_tokenizer_builder_set_vocab(&builder, vocab);

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_builder_build(&builder, &tokenizer));
  iree_tokenizer_builder_deinitialize(&builder);

  return tokenizer;
}

//===----------------------------------------------------------------------===//
// Encode Benchmarks
//===----------------------------------------------------------------------===//

// Fixture for encode benchmarks. Caches tokenizers across iterations.
// Args: state.range(0) = vocab_size, state.range(1) = text_length.
//
// Uses a pool of different texts to prevent artificial cache warming from
// processing the same text repeatedly. Each iteration uses a different text.
class EncodeBenchmark : public benchmark::Fixture {
 public:
  static constexpr size_t kTextPoolSize = 64;

  void SetUp(benchmark::State& state) override {
    size_t vocab_size = static_cast<size_t>(state.range(0));
    size_t text_length = static_cast<size_t>(state.range(1));

    tokenizer_ = GetCachedTokenizer(vocab_size);

    // Generate pool of different texts to rotate through.
    texts_.resize(kTextPoolSize);
    for (size_t i = 0; i < kTextPoolSize; ++i) {
      texts_[i] =
          GetBenchmarkText(static_cast<uint32_t>(0x42420000 + i), text_length);
    }
    text_index_ = 0;
    token_ids_.resize(text_length);
  }

  void TearDown(benchmark::State&) override {
    texts_.clear();
    texts_.shrink_to_fit();
    token_ids_.clear();
    token_ids_.shrink_to_fit();
  }

 public:
  static iree_tokenizer_t* GetCachedTokenizer(size_t vocab_size) {
    for (auto& entry : cache_.entries) {
      if (entry.vocab_size == vocab_size) return entry.tokenizer;
    }
    iree_tokenizer_t* tokenizer = BuildBenchmarkTokenizer(vocab_size);
    cache_.entries.push_back({vocab_size, tokenizer});
    return tokenizer;
  }

 protected:
  iree_tokenizer_t* tokenizer_ = nullptr;
  std::vector<std::string> texts_;
  size_t text_index_ = 0;
  std::vector<iree_tokenizer_token_id_t> token_ids_;

 private:
  struct CacheEntry {
    size_t vocab_size;
    iree_tokenizer_t* tokenizer;
  };
  struct Cache {
    std::vector<CacheEntry> entries;
    ~Cache() {
      for (auto& entry : entries) {
        iree_tokenizer_free(entry.tokenizer);
      }
    }
  };
  static Cache cache_;
};

EncodeBenchmark::Cache EncodeBenchmark::cache_;

// Single text encode: measures throughput in bytes/s and tokens/s.
// Rotates through a pool of different texts to prevent artificial cache warming
// that would occur from encoding the same text repeatedly.
BENCHMARK_DEFINE_F(EncodeBenchmark, Encode)(benchmark::State& state) {
  for (auto _ : state) {
    const std::string& text = texts_[text_index_];
    text_index_ = (text_index_ + 1) % kTextPoolSize;

    iree_host_size_t token_count = 0;
    IREE_CHECK_OK(iree_tokenizer_encode(
        tokenizer_, iree_make_string_view(text.data(), text.size()),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(token_ids_.data(), NULL, NULL,
                                         token_ids_.size()),
        iree_allocator_system(), &token_count));
    benchmark::DoNotOptimize(token_count);
  }
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(texts_[0].size()));
}

BENCHMARK_REGISTER_F(EncodeBenchmark, Encode)
    ->ArgNames({"vocab", "text_bytes"})
    // Short text (100B) - tweet-sized.
    ->Args({1000, 100})
    ->Args({10000, 100})
    ->Args({50000, 100})
    ->Args({128000, 100})
    // Medium text (1KB) - paragraph-sized.
    ->Args({1000, 1024})
    ->Args({10000, 1024})
    ->Args({50000, 1024})
    ->Args({128000, 1024})
    // Long text (10KB) - article-sized.
    ->Args({1000, 10240})
    ->Args({10000, 10240})
    ->Args({50000, 10240})
    ->Args({128000, 10240})
    // Very long text (100KB) - document-sized.
    ->Args({10000, 102400})
    ->Args({50000, 102400});

//===----------------------------------------------------------------------===//
// Batch Encode Benchmarks
//===----------------------------------------------------------------------===//

// Fixture for batch encode benchmarks.
// Args: state.range(0) = vocab_size, state.range(1) = batch_size,
//       state.range(2) = text_length.
class BatchEncodeBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    size_t vocab_size = static_cast<size_t>(state.range(0));
    batch_size_ = static_cast<size_t>(state.range(1));
    size_t text_length = static_cast<size_t>(state.range(2));

    tokenizer_ = EncodeBenchmark::GetCachedTokenizer(vocab_size);

    // Generate different text for each batch item.
    texts_.resize(batch_size_);
    for (size_t i = 0; i < batch_size_; ++i) {
      texts_[i] =
          GetBenchmarkText(static_cast<uint32_t>(0x12340000 + i), text_length);
    }

    // Allocate state storage (reused across batch items).
    iree_host_size_t state_size = 0;
    IREE_CHECK_OK(
        iree_tokenizer_encode_state_calculate_size(tokenizer_, &state_size));
    state_storage_.resize(state_size);
    transform_buffer_.resize(
        iree_tokenizer_transform_buffer_recommended_size(text_length));

    // Allocate output buffers.
    all_token_ids_.resize(batch_size_);
    items_.resize(batch_size_);
    for (size_t i = 0; i < batch_size_; ++i) {
      all_token_ids_[i].resize(text_length);
      items_[i].text =
          iree_make_string_view(texts_[i].data(), texts_[i].size());
      items_[i].output = iree_tokenizer_make_token_output(
          all_token_ids_[i].data(), NULL, NULL, all_token_ids_[i].size());
      items_[i].out_token_count = 0;
    }

    bytes_per_batch_ = 0;
    for (const auto& text : texts_) {
      bytes_per_batch_ += text.size();
    }
  }

  void TearDown(benchmark::State&) override {
    texts_.clear();
    texts_.shrink_to_fit();
    state_storage_.clear();
    state_storage_.shrink_to_fit();
    transform_buffer_.clear();
    transform_buffer_.shrink_to_fit();
    all_token_ids_.clear();
    all_token_ids_.shrink_to_fit();
    items_.clear();
    items_.shrink_to_fit();
  }

 protected:
  iree_tokenizer_t* tokenizer_ = nullptr;
  size_t batch_size_ = 0;
  std::vector<std::string> texts_;
  std::vector<uint8_t> state_storage_;
  std::vector<uint8_t> transform_buffer_;
  std::vector<std::vector<iree_tokenizer_token_id_t>> all_token_ids_;
  std::vector<iree_tokenizer_encode_batch_item_t> items_;
  size_t bytes_per_batch_ = 0;
};

BENCHMARK_DEFINE_F(BatchEncodeBenchmark, Batch)(benchmark::State& state) {
  for (auto _ : state) {
    // Reset output counts.
    for (size_t i = 0; i < batch_size_; ++i) {
      items_[i].out_token_count = 0;
    }

    IREE_CHECK_OK(iree_tokenizer_encode_batch(
        tokenizer_, items_.data(), items_.size(),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_make_byte_span(state_storage_.data(), state_storage_.size()),
        iree_make_byte_span(transform_buffer_.data(), transform_buffer_.size()),
        iree_tokenizer_offset_run_list_empty()));

    iree_host_size_t batch_tokens = 0;
    for (size_t i = 0; i < batch_size_; ++i) {
      batch_tokens += items_[i].out_token_count;
    }
    benchmark::DoNotOptimize(batch_tokens);
  }
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(bytes_per_batch_));
}

BENCHMARK_REGISTER_F(BatchEncodeBenchmark, Batch)
    ->ArgNames({"vocab", "batch", "text_bytes"})
    ->Args({10000, 8, 1024})
    ->Args({50000, 8, 1024})
    ->Args({10000, 32, 100})
    ->Args({50000, 32, 100})
    ->Args({10000, 128, 100});

//===----------------------------------------------------------------------===//
// Streaming Encode Benchmarks
//===----------------------------------------------------------------------===//

// Fixture for streaming encode benchmarks.
// Args: state.range(0) = vocab_size, state.range(1) = text_length,
//       state.range(2) = chunk_size.
//
// Uses a pool of different texts to prevent artificial cache warming.
class StreamingEncodeBenchmark : public benchmark::Fixture {
 public:
  static constexpr size_t kTextPoolSize = 64;

  void SetUp(benchmark::State& state) override {
    size_t vocab_size = static_cast<size_t>(state.range(0));
    size_t text_length = static_cast<size_t>(state.range(1));
    chunk_size_ = static_cast<size_t>(state.range(2));

    tokenizer_ = EncodeBenchmark::GetCachedTokenizer(vocab_size);

    // Generate pool of different texts to rotate through.
    texts_.resize(kTextPoolSize);
    for (size_t i = 0; i < kTextPoolSize; ++i) {
      texts_[i] =
          GetBenchmarkText(static_cast<uint32_t>(0x55550000 + i), text_length);
    }
    text_index_ = 0;

    iree_host_size_t state_size = 0;
    IREE_CHECK_OK(
        iree_tokenizer_encode_state_calculate_size(tokenizer_, &state_size));
    state_storage_.resize(state_size);
    transform_buffer_.resize(
        iree_tokenizer_transform_buffer_recommended_size(chunk_size_));
    token_ids_.resize(text_length);
  }

  void TearDown(benchmark::State&) override {
    texts_.clear();
    texts_.shrink_to_fit();
    state_storage_.clear();
    state_storage_.shrink_to_fit();
    transform_buffer_.clear();
    transform_buffer_.shrink_to_fit();
    token_ids_.clear();
    token_ids_.shrink_to_fit();
  }

 protected:
  iree_tokenizer_t* tokenizer_ = nullptr;
  size_t chunk_size_ = 0;
  std::vector<std::string> texts_;
  size_t text_index_ = 0;
  std::vector<uint8_t> state_storage_;
  std::vector<uint8_t> transform_buffer_;
  std::vector<iree_tokenizer_token_id_t> token_ids_;
};

// Rotates through a pool of different texts to prevent artificial cache
// warming.
BENCHMARK_DEFINE_F(StreamingEncodeBenchmark, Stream)
(benchmark::State& state) {
  for (auto _ : state) {
    const std::string& text = texts_[text_index_];
    text_index_ = (text_index_ + 1) % kTextPoolSize;

    iree_tokenizer_encode_state_t* encode_state = nullptr;
    IREE_CHECK_OK(iree_tokenizer_encode_state_initialize(
        tokenizer_,
        iree_make_byte_span(state_storage_.data(), state_storage_.size()),
        iree_make_byte_span(transform_buffer_.data(), transform_buffer_.size()),
        iree_tokenizer_offset_run_list_empty(),
        IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &encode_state));

    // Feed text in chunks.
    iree_host_size_t total_tokens = 0;
    iree_string_view_t remaining =
        iree_make_string_view(text.data(), text.size());

    while (remaining.size > 0) {
      size_t this_chunk =
          remaining.size < chunk_size_ ? remaining.size : chunk_size_;
      iree_string_view_t chunk =
          iree_make_string_view(remaining.data, this_chunk);

      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t tokens_written = 0;
      IREE_CHECK_OK(iree_tokenizer_encode_state_feed(
          encode_state, chunk,
          iree_tokenizer_make_token_output(token_ids_.data() + total_tokens,
                                           NULL, NULL,
                                           token_ids_.size() - total_tokens),
          &bytes_consumed, &tokens_written));
      total_tokens += tokens_written;
      remaining.data += bytes_consumed;
      remaining.size -= bytes_consumed;
    }

    // Finalize.
    iree_host_size_t final_tokens = 0;
    IREE_CHECK_OK(iree_tokenizer_encode_state_finalize(
        encode_state,
        iree_tokenizer_make_token_output(token_ids_.data() + total_tokens, NULL,
                                         NULL,
                                         token_ids_.size() - total_tokens),
        &final_tokens));

    iree_tokenizer_encode_state_deinitialize(encode_state);
    benchmark::DoNotOptimize(total_tokens + final_tokens);
  }
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(texts_[0].size()));
}

BENCHMARK_REGISTER_F(StreamingEncodeBenchmark, Stream)
    ->ArgNames({"vocab", "text_bytes", "chunk_bytes"})
    // Various chunk sizes on 10KB text.
    ->Args({10000, 10240, 64})
    ->Args({10000, 10240, 256})
    ->Args({10000, 10240, 1024})
    ->Args({10000, 10240, 4096})
    // Streaming vs batch comparison on 100KB.
    ->Args({10000, 102400, 1024});

//===----------------------------------------------------------------------===//
// Cold Start Benchmarks
//===----------------------------------------------------------------------===//

// Measures full pipeline: tokenizer construction + encode.
// No fixture caching — intentionally rebuilds each iteration.
// Args: state.range(0) = vocab_size, state.range(1) = text_length.
void BM_ColdStart(benchmark::State& state) {
  size_t vocab_size = static_cast<size_t>(state.range(0));
  size_t text_length = static_cast<size_t>(state.range(1));

  std::string text = GetBenchmarkText(0xDEADBEEF, text_length);
  std::vector<iree_tokenizer_token_id_t> token_ids(text_length);

  for (auto _ : state) {
    // Build tokenizer (not cached — measuring cold start).
    iree_tokenizer_t* tokenizer = BuildBenchmarkTokenizer(vocab_size);

    iree_host_size_t token_count = 0;
    iree_status_t status = iree_tokenizer_encode(
        tokenizer, iree_make_string_view(text.data(), text.size()),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(token_ids.data(), NULL, NULL,
                                         token_ids.size()),
        iree_allocator_system(), &token_count);
    benchmark::DoNotOptimize(token_count);
    iree_status_ignore(status);

    iree_tokenizer_free(tokenizer);
  }
}

BENCHMARK(BM_ColdStart)
    ->ArgNames({"vocab", "text_bytes"})
    ->Args({1000, 1024})
    ->Args({10000, 1024})
    ->Args({50000, 1024})
    ->Args({128000, 1024});

//===----------------------------------------------------------------------===//
// Decode Benchmarks (Pre-decoded Fast Path)
//===----------------------------------------------------------------------===//

// Fixture for decode benchmarks. Caches tokenizers with ByteLevel decoder.
// Args: state.range(0) = vocab_size, state.range(1) = text_length.
//
// Uses a pool of different token sequences to prevent artificial cache warming
// from decoding the same tokens repeatedly. Each iteration uses different
// tokens.
class DecodeBenchmark : public benchmark::Fixture {
 public:
  static constexpr size_t kTokenPoolSize = 64;

  void SetUp(benchmark::State& state) override {
    size_t vocab_size = static_cast<size_t>(state.range(0));
    size_t text_length = static_cast<size_t>(state.range(1));

    tokenizer_ = GetCachedDecodeTokenizer(vocab_size);

    // Generate pool of different token sequences to rotate through.
    token_sequences_.resize(kTokenPoolSize);
    size_t max_tokens = 0;
    for (size_t i = 0; i < kTokenPoolSize; ++i) {
      std::string text =
          GetBenchmarkText(static_cast<uint32_t>(0xAAAA0000 + i), text_length);
      std::vector<iree_tokenizer_token_id_t> encode_ids(text_length);
      iree_host_size_t token_count = 0;
      IREE_CHECK_OK(iree_tokenizer_encode(
          tokenizer_, iree_make_string_view(text.data(), text.size()),
          IREE_TOKENIZER_ENCODE_FLAG_NONE,
          iree_tokenizer_make_token_output(encode_ids.data(), NULL, NULL,
                                           encode_ids.size()),
          iree_allocator_system(), &token_count));
      token_sequences_[i].assign(encode_ids.begin(),
                                 encode_ids.begin() + token_count);
      max_tokens = std::max(max_tokens, static_cast<size_t>(token_count));
    }
    token_index_ = 0;

    // Allocate decode state storage.
    iree_host_size_t state_size = 0;
    IREE_CHECK_OK(
        iree_tokenizer_decode_state_calculate_size(tokenizer_, &state_size));
    state_storage_.resize(state_size);

    // Allocate output buffer (generous — max ~16x token count for multi-byte).
    output_.resize(max_tokens * 16 + 256);
  }

  void TearDown(benchmark::State&) override {
    token_sequences_.clear();
    token_sequences_.shrink_to_fit();
    state_storage_.clear();
    state_storage_.shrink_to_fit();
    output_.clear();
    output_.shrink_to_fit();
  }

 public:
  static iree_tokenizer_t* GetCachedDecodeTokenizer(size_t vocab_size) {
    for (auto& entry : cache_.entries) {
      if (entry.vocab_size == vocab_size) return entry.tokenizer;
    }
    iree_tokenizer_t* tokenizer =
        BuildBenchmarkTokenizerWithDecoder(vocab_size);
    cache_.entries.push_back({vocab_size, tokenizer});
    return tokenizer;
  }

 protected:
  iree_tokenizer_t* tokenizer_ = nullptr;
  std::vector<std::vector<iree_tokenizer_token_id_t>> token_sequences_;
  size_t token_index_ = 0;
  std::vector<uint8_t> state_storage_;
  std::vector<char> output_;

 private:
  struct CacheEntry {
    size_t vocab_size;
    iree_tokenizer_t* tokenizer;
  };
  struct Cache {
    std::vector<CacheEntry> entries;
    ~Cache() {
      for (auto& entry : entries) {
        iree_tokenizer_free(entry.tokenizer);
      }
    }
  };
  static Cache cache_;
};

DecodeBenchmark::Cache DecodeBenchmark::cache_;

// Rotates through a pool of different token sequences to prevent artificial
// cache warming that would occur from decoding the same tokens repeatedly.
BENCHMARK_DEFINE_F(DecodeBenchmark, Decode)(benchmark::State& state) {
  for (auto _ : state) {
    const auto& tokens = token_sequences_[token_index_];
    token_index_ = (token_index_ + 1) % kTokenPoolSize;

    iree_tokenizer_token_id_list_t id_list = {
        /*.count=*/tokens.size(),
        /*.values=*/reinterpret_cast<const int32_t*>(tokens.data()),
    };

    iree_tokenizer_decode_state_t* decode_state = nullptr;
    IREE_CHECK_OK(iree_tokenizer_decode_state_initialize(
        tokenizer_, IREE_TOKENIZER_DECODE_FLAG_NONE,
        iree_make_byte_span(state_storage_.data(), state_storage_.size()),
        &decode_state));

    iree_host_size_t tokens_consumed = 0;
    iree_host_size_t text_length_out = 0;
    iree_status_t status = iree_tokenizer_decode_state_feed(
        decode_state, id_list,
        iree_make_mutable_string_view(output_.data(), output_.size()),
        &tokens_consumed, &text_length_out);
    iree_status_ignore(status);

    iree_host_size_t finalize_length = 0;
    status = iree_tokenizer_decode_state_finalize(
        decode_state,
        iree_make_mutable_string_view(output_.data() + text_length_out,
                                      output_.size() - text_length_out),
        &finalize_length);
    iree_status_ignore(status);

    iree_tokenizer_decode_state_deinitialize(decode_state);

    benchmark::DoNotOptimize(text_length_out + finalize_length);
  }
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(token_sequences_[0].size()));
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(token_sequences_[0].size()));
}

BENCHMARK_REGISTER_F(DecodeBenchmark, Decode)
    ->ArgNames({"vocab", "text_bytes"})
    // Short token sequence (~25 tokens from 100B text).
    ->Args({1000, 100})
    ->Args({10000, 100})
    ->Args({50000, 100})
    // Medium token sequence (~250 tokens from 1KB text).
    ->Args({1000, 1024})
    ->Args({10000, 1024})
    ->Args({50000, 1024})
    // Long token sequence (~2500 tokens from 10KB text).
    ->Args({10000, 10240})
    ->Args({50000, 10240})
    // Very long token sequence (~25000 tokens from 100KB text).
    ->Args({10000, 102400})
    ->Args({50000, 102400});

//===----------------------------------------------------------------------===//
// Streaming Decode Benchmarks
//===----------------------------------------------------------------------===//

// Fixture for streaming decode benchmarks. Feeds tokens in small chunks,
// simulating autoregressive LLM inference where tokens arrive one at a time.
// Args: state.range(0) = vocab_size, state.range(1) = text_length,
//       state.range(2) = tokens_per_call.
//
// Uses a pool of different token sequences to prevent artificial cache warming.
class StreamingDecodeBenchmark : public benchmark::Fixture {
 public:
  static constexpr size_t kTokenPoolSize = 64;

  void SetUp(benchmark::State& state) override {
    size_t vocab_size = static_cast<size_t>(state.range(0));
    size_t text_length = static_cast<size_t>(state.range(1));
    tokens_per_call_ = static_cast<size_t>(state.range(2));

    tokenizer_ = DecodeBenchmark::GetCachedDecodeTokenizer(vocab_size);

    // Generate pool of different token sequences to rotate through.
    token_sequences_.resize(kTokenPoolSize);
    size_t max_tokens = 0;
    for (size_t i = 0; i < kTokenPoolSize; ++i) {
      std::string text =
          GetBenchmarkText(static_cast<uint32_t>(0xBBBB0000 + i), text_length);
      std::vector<iree_tokenizer_token_id_t> encode_ids(text_length);
      iree_host_size_t token_count = 0;
      IREE_CHECK_OK(iree_tokenizer_encode(
          tokenizer_, iree_make_string_view(text.data(), text.size()),
          IREE_TOKENIZER_ENCODE_FLAG_NONE,
          iree_tokenizer_make_token_output(encode_ids.data(), NULL, NULL,
                                           encode_ids.size()),
          iree_allocator_system(), &token_count));
      token_sequences_[i].assign(encode_ids.begin(),
                                 encode_ids.begin() + token_count);
      max_tokens = std::max(max_tokens, static_cast<size_t>(token_count));
    }
    token_index_ = 0;

    // Allocate decode state storage.
    iree_host_size_t state_size = 0;
    IREE_CHECK_OK(
        iree_tokenizer_decode_state_calculate_size(tokenizer_, &state_size));
    state_storage_.resize(state_size);

    // Allocate output buffer.
    output_.resize(max_tokens * 16 + 256);
  }

  void TearDown(benchmark::State&) override {
    token_sequences_.clear();
    token_sequences_.shrink_to_fit();
    state_storage_.clear();
    state_storage_.shrink_to_fit();
    output_.clear();
    output_.shrink_to_fit();
  }

 protected:
  iree_tokenizer_t* tokenizer_ = nullptr;
  size_t tokens_per_call_ = 1;
  std::vector<std::vector<iree_tokenizer_token_id_t>> token_sequences_;
  size_t token_index_ = 0;
  std::vector<uint8_t> state_storage_;
  std::vector<char> output_;
};

// Rotates through a pool of different token sequences to prevent artificial
// cache warming that would occur from decoding the same tokens repeatedly.
BENCHMARK_DEFINE_F(StreamingDecodeBenchmark, Stream)
(benchmark::State& state) {
  for (auto _ : state) {
    const auto& tokens = token_sequences_[token_index_];
    token_index_ = (token_index_ + 1) % kTokenPoolSize;

    iree_tokenizer_decode_state_t* decode_state = nullptr;
    IREE_CHECK_OK(iree_tokenizer_decode_state_initialize(
        tokenizer_, IREE_TOKENIZER_DECODE_FLAG_NONE,
        iree_make_byte_span(state_storage_.data(), state_storage_.size()),
        &decode_state));

    iree_host_size_t total_text = 0;
    size_t token_position = 0;
    while (token_position < tokens.size()) {
      size_t chunk = tokens.size() - token_position;
      if (chunk > tokens_per_call_) chunk = tokens_per_call_;

      iree_tokenizer_token_id_list_t id_list = {
          /*.count=*/chunk,
          /*.values=*/
          reinterpret_cast<const int32_t*>(tokens.data() + token_position),
      };

      iree_host_size_t tokens_consumed = 0;
      iree_host_size_t text_written = 0;
      iree_status_t status = iree_tokenizer_decode_state_feed(
          decode_state, id_list,
          iree_make_mutable_string_view(output_.data() + total_text,
                                        output_.size() - total_text),
          &tokens_consumed, &text_written);
      iree_status_ignore(status);

      token_position += tokens_consumed;
      total_text += text_written;

      // Avoid infinite loop if no progress.
      if (tokens_consumed == 0) break;
    }

    iree_host_size_t finalize_length = 0;
    iree_status_t status = iree_tokenizer_decode_state_finalize(
        decode_state,
        iree_make_mutable_string_view(output_.data() + total_text,
                                      output_.size() - total_text),
        &finalize_length);
    iree_status_ignore(status);

    iree_tokenizer_decode_state_deinitialize(decode_state);
    benchmark::DoNotOptimize(total_text + finalize_length);
  }
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(token_sequences_[0].size()));
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(token_sequences_[0].size()));
}

BENCHMARK_REGISTER_F(StreamingDecodeBenchmark, Stream)
    ->ArgNames({"vocab", "text_bytes", "tokens_per_call"})
    // Autoregressive: 1 token at a time (LLM inference hot path).
    ->Args({10000, 10240, 1})
    ->Args({50000, 10240, 1})
    // Small batch: speculative decoding or beam search.
    ->Args({10000, 10240, 10})
    ->Args({50000, 10240, 10})
    // Medium batch.
    ->Args({10000, 10240, 100})
    ->Args({50000, 10240, 100})
    // Large batch (comparable to full-batch Decode benchmark).
    ->Args({10000, 10240, 10000})
    ->Args({50000, 10240, 10000});

}  // namespace
