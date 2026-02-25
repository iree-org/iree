// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Comprehensive tokenizer benchmark using Google Benchmark.
// Tests encode and decode throughput across ASCII, CJK, and code text
// for any HuggingFace tokenizer.json.
//
// Quick start:
//   iree-bazel-run --copt=-O3 --copt=-march=native --features=thin_lto \
//     //runtime/src/iree/tokenizer/tools:comprehensive_benchmark -- \
//     --tokenizer_json=tokenizer.json --benchmark_min_time=1s
//
// See --help for model download instructions and benchmark flags.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/internal/unicode.h"
#include "iree/base/tooling/flags.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

IREE_FLAG(string, tokenizer_json, "",
          "Path to HuggingFace tokenizer.json file (required).");
IREE_FLAG(
    string, ascii_text, "",
    "Path to ASCII/English text file. If empty, uses embedded default (~256KB "
    "repeated pattern).");
IREE_FLAG(string, cjk_text, "",
          "Path to CJK text file. If empty, uses embedded default (~256KB "
          "repeated pattern).");
IREE_FLAG(string, code_text, "",
          "Path to source code text file. If empty, uses embedded default "
          "(~256KB repeated pattern).");
IREE_FLAG(bool, rotate, false,
          "Allocate 64 full copies of each text corpus at separate heap "
          "addresses and cycle through them. The total working set (64x text "
          "size) exceeds L3 cache, defeating cache warming that can inflate "
          "throughput by 20-40%%. Uses significant memory (~64x corpus size "
          "per corpus), so not suitable for memory-constrained environments.");

namespace {

//===----------------------------------------------------------------------===//
// Embedded default seed texts
//===----------------------------------------------------------------------===//
// Each ~1KB of representative content, repeated to ~256KB for benchmarks.
// These provide reasonable default inputs when no text files are specified.
// For more representative results, provide natural text via the flags above.

static constexpr size_t kDefaultRepeatTargetSize = 256 * 1024;

// clang-format off
static const char kAsciiSeed[] =
R"seed(The quick brown fox jumps over the lazy dog near the riverbank. In the distance,
mountains rose above the misty valley, their peaks dusted with fresh snow from
the overnight storm. A gentle breeze carried the scent of pine and wildflowers.

"Have you considered," asked the professor, "that language models process text
in fundamentally different ways than humans do?" She paused for a moment before
continuing. "Tokenization is the critical first step."

Technical specs: CPU freq 3.2GHz, memory bandwidth 51.2GB/s, latency p50=12ms
p99=45ms. Error rates: 0.003% (HTTP 500), 0.1% (timeout). Configuration:
max_batch_size=64, num_workers=8, learning_rate=1e-4, dropout=0.1.

function processInput(data: Record<string, unknown>): Promise<Result> {
  const validated = schema.parse(data);
  return db.transaction(async (tx) => {
    await tx.insert(records).values(validated);
    return { status: 'ok', count: validated.length };
  });
}

Mixed: @user #hashtag $9.99 100EUR 50GBP 1000JPY (C)2024 brand(R) product(TM)
Paths: /usr/local/bin/python3 C:\Users\admin ~/.config/settings.json
Email: user@example.com | Website: https://example.com/api/v2?key=abc&fmt=json
The 2024 report showed 47.3% growth in Q4, compared to -2.1% in Q3.
)seed";

static const char kCjkSeed[] =
R"seed(吾輩は猫である。名前はまだない。どこで生まれたか頓と見当がつかぬ。何でも薄暗い
じめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて
人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族
であったそうだ。

プログラミング言語の歴史は1940年代に始まった。最初のプログラミング言語は機械語
であり、高水準言語の登場により人間が理解しやすい形式でプログラムを記述できるよう
になった。現代のソフトウェア開発では多くの言語が使われている。

中华人民共和国是工人阶级领导的、以工农联盟为基础的人民民主专政的社会主义国家。
社会主义制度是中华人民共和国的根本制度。禁止任何组织或者个人破坏社会主义制度。

대한민국은 민주공화국이다. 대한민국의 주권은 국민에게 있고, 모든 권력은
국민으로부터 나온다. 대한민국의 영토는 한반도와 그 부속도서로 한다.

東京特許許可局局長今日急遽休暇許可却下。すもももももももものうち。
カタカナ：コンピュータサイエンス、アルゴリズム、データベース。
数字混合：令和6年12月25日、価格は9800円（税込10780円）です。
)seed";

static const char kCodeSeed[] =
R"seed(static iree_status_t hash_table_insert(hash_table_t* table, const char* key,
                                       size_t key_length, void* value) {
  if (IREE_UNLIKELY(!table || !key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table and key must be non-NULL");
  }
  uint32_t hash = 5381;
  for (size_t i = 0; i < key_length; ++i) {
    hash = ((hash << 5) + hash) + (unsigned char)key[i];
  }
  size_t index = hash % table->capacity;
  for (size_t probe = 0; probe < table->capacity; ++probe) {
    entry_t* entry = &table->entries[index];
    if (entry->key == NULL) {
      *entry = (entry_t){
          .key = key, .key_length = key_length,
          .hash = hash, .value = value,
      };
      table->count++;
      return iree_ok_status();
    }
    if (entry->hash == hash && entry->key_length == key_length &&
        memcmp(entry->key, key, key_length) == 0) {
      entry->value = value;
      return iree_ok_status();
    }
    index = (index + 1) % table->capacity;
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "table full");
}

#define MAX_BUFFER_SIZE 4096
typedef struct {
  char data[MAX_BUFFER_SIZE];
  size_t length;
  size_t capacity;
  int32_t flags;
} string_buffer_t;

// Initialize a buffer with the given capacity.
void string_buffer_init(string_buffer_t* buffer, size_t capacity) {
  buffer->length = 0;
  buffer->capacity = capacity < MAX_BUFFER_SIZE ? capacity : MAX_BUFFER_SIZE;
  buffer->flags = 0;
  memset(buffer->data, 0, buffer->capacity);
}
)seed";
// clang-format on

// Repeat seed text using whole copies until reaching the target size.
// Always copies complete seeds to preserve UTF-8 codepoint boundaries.
static std::string RepeatSeed(const char* seed, size_t seed_length) {
  std::string result;
  size_t repetitions =
      (kDefaultRepeatTargetSize + seed_length - 1) / seed_length;
  result.reserve(repetitions * seed_length);
  for (size_t i = 0; i < repetitions; ++i) {
    result.append(seed, seed_length);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Input pool for cache defeat
//===----------------------------------------------------------------------===//
// When rotation is enabled, the pool contains kPoolSize independent copies of
// the full text at different heap addresses. Each benchmark iteration uses a
// different copy, so the total working set (kPoolSize * text_size) exceeds L3
// cache, preventing cache warming from artificially inflating throughput.
//
// Without rotation, a single copy of the full text is used (cache-hot,
// matching typical microbenchmark methodology).

static constexpr size_t kPoolSize = 64;

struct TextPool {
  // Text chunks for encode benchmarks. Without rotation, one entry containing
  // the full text. With rotation, kPoolSize copies at different heap addresses.
  std::vector<std::string> text_chunks;

  // Pre-encoded token sequences for decode benchmarks (one per text chunk).
  std::vector<std::vector<int32_t>> token_chunks;

  // Size of each pool entry in bytes.
  size_t chunk_size = 0;

  // Returns the next text chunk, advancing the rotation index.
  iree_string_view_t next_text(size_t& index) const {
    const auto& text = text_chunks[index++ % text_chunks.size()];
    return iree_make_string_view(text.data(), text.size());
  }

  // Returns the next token sequence for decode benchmarks, advancing the
  // rotation index.
  const std::vector<int32_t>& next_tokens(size_t& index) const {
    return token_chunks[index++ % token_chunks.size()];
  }
};

// Build a pool from the given text. When rotation is enabled, creates
// kPoolSize independent copies at different heap addresses. When disabled,
// stores a single copy of the full text.
static void BuildPool(std::string full_text, TextPool& pool) {
  pool.chunk_size = full_text.size();
  if (!FLAG_rotate) {
    pool.text_chunks.resize(1);
    pool.text_chunks[0] = std::move(full_text);
    return;
  }
  pool.text_chunks.resize(kPoolSize);
  for (size_t i = 0; i < kPoolSize; ++i) {
    pool.text_chunks[i] = full_text;
  }
}

//===----------------------------------------------------------------------===//
// Global state shared across all benchmarks
//===----------------------------------------------------------------------===//

struct GlobalState {
  iree_tokenizer_t* tokenizer = nullptr;
  std::string json_string;

  // Input pools (one per corpus).
  TextPool ascii_pool;
  TextPool cjk_pool;
  TextPool code_pool;

  // Pre-allocated encode buffers.
  iree_host_size_t state_size = 0;
  std::vector<uint8_t> state_storage;
  std::vector<uint8_t> transform_buffer;
  std::vector<int32_t> output_tokens;

  // Pre-allocated decode buffers.
  iree_host_size_t decode_state_size = 0;
  std::vector<uint8_t> decode_state_storage;
  std::vector<char> output_text;

  bool initialized = false;
};

GlobalState g_state;

// Load text from a file path, or generate from an embedded seed if the flag
// is empty.
static bool LoadText(benchmark::State& state, const char* flag_path,
                     const char* seed, size_t seed_length, std::string& out) {
  if (flag_path[0] != '\0') {
    std::ifstream file(flag_path);
    if (!file.is_open()) {
      state.SkipWithError(flag_path);
      return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    out = buffer.str();
  } else {
    out = RepeatSeed(seed, seed_length);
  }
  return true;
}

bool EnsureInitialized(benchmark::State& state) {
  if (g_state.initialized) {
    return true;
  }

  // Require a tokenizer.
  if (FLAG_tokenizer_json[0] == '\0') {
    state.SkipWithError("No --tokenizer_json specified");
    return false;
  }

  // Load tokenizer JSON.
  std::ifstream json_file(FLAG_tokenizer_json);
  if (!json_file.is_open()) {
    state.SkipWithError("Failed to open tokenizer JSON file");
    return false;
  }
  std::stringstream json_buffer;
  json_buffer << json_file.rdbuf();
  g_state.json_string = json_buffer.str();

  // Create tokenizer.
  IREE_CHECK_OK(iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(g_state.json_string.c_str()),
      iree_allocator_system(), &g_state.tokenizer));

  // Load text corpora and build input pools.
  std::string raw_text;
  if (!LoadText(state, FLAG_ascii_text, kAsciiSeed, sizeof(kAsciiSeed) - 1,
                raw_text)) {
    return false;
  }
  BuildPool(std::move(raw_text), g_state.ascii_pool);

  if (!LoadText(state, FLAG_cjk_text, kCjkSeed, sizeof(kCjkSeed) - 1,
                raw_text)) {
    return false;
  }
  BuildPool(std::move(raw_text), g_state.cjk_pool);

  if (!LoadText(state, FLAG_code_text, kCodeSeed, sizeof(kCodeSeed) - 1,
                raw_text)) {
    return false;
  }
  BuildPool(std::move(raw_text), g_state.code_pool);

  // Report rotation mode and memory usage.
  if (FLAG_rotate) {
    size_t total_bytes = g_state.ascii_pool.chunk_size +
                         g_state.cjk_pool.chunk_size +
                         g_state.code_pool.chunk_size;
    fprintf(stderr,
            "Pool rotation: ON (%zu copies, "
            "~%zuKB ASCII + ~%zuKB CJK + ~%zuKB Code, ~%zuMB total)\n",
            kPoolSize, g_state.ascii_pool.chunk_size / 1024,
            g_state.cjk_pool.chunk_size / 1024,
            g_state.code_pool.chunk_size / 1024,
            (kPoolSize * total_bytes) / (1024 * 1024));
  } else {
    fprintf(stderr, "Pool rotation: OFF (single copy, cache-hot)\n");
  }

  // Calculate encode state size.
  IREE_CHECK_OK(iree_tokenizer_encode_state_calculate_size(
      g_state.tokenizer, &g_state.state_size));

  // Find max chunk size for buffer allocation.
  size_t max_chunk =
      std::max({g_state.ascii_pool.chunk_size, g_state.cjk_pool.chunk_size,
                g_state.code_pool.chunk_size});

  // Allocate encode buffers.
  g_state.state_storage.resize(g_state.state_size);
  g_state.transform_buffer.resize(
      iree_tokenizer_transform_buffer_recommended_size(max_chunk));
  g_state.output_tokens.resize(max_chunk);  // Conservative upper bound.

  // Byte-level tokenizers (GPT-2 style) can produce MORE decoded bytes than
  // input text: each raw byte maps to a Unicode codepoint that may be 2 bytes
  // in UTF-8 (non-ASCII bytes 0x80-0xFF all map to 2-byte codepoints). For
  // pure CJK text (all multi-byte), the decoded output can be ~2x the input.
  // 4x provides safe headroom for any text composition.
  g_state.output_text.resize(max_chunk * 4);

  // Allocate decode state buffer.
  IREE_CHECK_OK(iree_tokenizer_decode_state_calculate_size(
      g_state.tokenizer, &g_state.decode_state_size));
  g_state.decode_state_storage.resize(g_state.decode_state_size);

  // Pre-encode each pool chunk for decode benchmarks.
  auto pre_encode_pool = [](TextPool& pool) {
    pool.token_chunks.resize(pool.text_chunks.size());
    for (size_t i = 0; i < pool.text_chunks.size(); ++i) {
      const auto& text = pool.text_chunks[i];
      auto& tokens = pool.token_chunks[i];
      tokens.resize(text.size());
      iree_host_size_t count = 0;
      IREE_CHECK_OK(iree_tokenizer_encode(
          g_state.tokenizer, iree_make_string_view(text.data(), text.size()),
          IREE_TOKENIZER_ENCODE_FLAG_NONE,
          iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                           tokens.size()),
          iree_allocator_system(), &count));
      tokens.resize(count);
    }
  };

  pre_encode_pool(g_state.ascii_pool);
  pre_encode_pool(g_state.cjk_pool);
  pre_encode_pool(g_state.code_pool);

  g_state.initialized = true;
  return true;
}

//===----------------------------------------------------------------------===//
// Encode helpers
//===----------------------------------------------------------------------===//

// Streaming encode using pre-allocated buffers.
iree_host_size_t DoEncodeStreaming(iree_string_view_t text,
                                   std::vector<int32_t>& tokens) {
  iree_tokenizer_encode_state_t* encode_state = nullptr;
  IREE_CHECK_OK(iree_tokenizer_encode_state_initialize(
      g_state.tokenizer,
      iree_make_byte_span(g_state.state_storage.data(),
                          g_state.state_storage.size()),
      iree_make_byte_span(g_state.transform_buffer.data(),
                          g_state.transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &encode_state));

  iree_string_view_t remaining = text;
  iree_host_size_t total_tokens = 0;
  while (remaining.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t tokens_written = 0;
    IREE_CHECK_OK(iree_tokenizer_encode_state_feed(
        encode_state, remaining,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &tokens_written));
    remaining.data += bytes_consumed;
    remaining.size -= bytes_consumed;
    total_tokens += tokens_written;
  }

  iree_host_size_t final_tokens = 0;
  IREE_CHECK_OK(iree_tokenizer_encode_state_finalize(
      encode_state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &final_tokens));
  iree_tokenizer_encode_state_deinitialize(encode_state);
  return total_tokens + final_tokens;
}

// One-shot encode.
iree_host_size_t DoEncodeOneShot(iree_string_view_t text,
                                 std::vector<int32_t>& tokens) {
  iree_host_size_t count = 0;
  IREE_CHECK_OK(iree_tokenizer_encode(
      g_state.tokenizer, text, IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      iree_allocator_system(), &count));
  return count;
}

// Chunked encode - walks forward through text, not cache-hot.
// Each chunk is aligned to UTF-8 codepoint boundaries to avoid splitting
// multi-byte characters. The batch encode API requires complete UTF-8 input.
iree_host_size_t DoEncodeChunked(iree_string_view_t text, size_t chunk_size,
                                 std::vector<int32_t>& tokens) {
  iree_host_size_t total_tokens = 0;
  size_t offset = 0;
  while (offset < text.size) {
    size_t length = std::min(chunk_size, text.size - offset);
    // Back off to a complete UTF-8 codepoint boundary. If the chunk ends
    // mid-codepoint, trim the incomplete trailing bytes. They'll be included
    // in the next chunk.
    if (offset + length < text.size) {
      iree_host_size_t incomplete =
          iree_unicode_utf8_incomplete_tail_length(text.data + offset, length);
      length -= incomplete;
    }
    if (length == 0) {
      // Pathological case: chunk_size < max UTF-8 codepoint length (4 bytes).
      // Just fail rather than infinite loop.
      return 0;
    }
    iree_host_size_t count = 0;
    IREE_CHECK_OK(iree_tokenizer_encode(
        g_state.tokenizer, iree_make_string_view(text.data + offset, length),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                         tokens.size()),
        iree_allocator_system(), &count));
    total_tokens += count;
    offset += length;
  }
  return total_tokens;
}

//===----------------------------------------------------------------------===//
// Encode benchmarks
//===----------------------------------------------------------------------===//

// --- ASCII ---

void BM_ASCII_EncodeStreaming(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.ascii_pool;
  size_t pool_index = 0;
  iree_host_size_t tokens = 0;
  for (auto _ : state) {
    tokens =
        DoEncodeStreaming(pool.next_text(pool_index), g_state.output_tokens);
    if (tokens == 0) {
      state.SkipWithError("Encode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["bytes/token"] = (double)pool.chunk_size / tokens;
}
BENCHMARK(BM_ASCII_EncodeStreaming)->Unit(benchmark::kMillisecond);

void BM_ASCII_EncodeOneShot(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.ascii_pool;
  size_t pool_index = 0;
  iree_host_size_t tokens = 0;
  for (auto _ : state) {
    tokens = DoEncodeOneShot(pool.next_text(pool_index), g_state.output_tokens);
    if (tokens == 0) {
      state.SkipWithError("Encode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["bytes/token"] = (double)pool.chunk_size / tokens;
}
BENCHMARK(BM_ASCII_EncodeOneShot)->Unit(benchmark::kMillisecond);

void BM_ASCII_Encode1KB(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.ascii_pool;
  size_t pool_index = 0;
  iree_host_size_t tokens = 0;
  size_t chunk_count = (pool.chunk_size + 1023) / 1024;
  for (auto _ : state) {
    tokens = DoEncodeChunked(pool.next_text(pool_index), 1024,
                             g_state.output_tokens);
    if (tokens == 0) {
      state.SkipWithError("Encode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["chunks"] = chunk_count;
}
BENCHMARK(BM_ASCII_Encode1KB)->Unit(benchmark::kMillisecond);

// --- CJK ---

void BM_CJK_EncodeStreaming(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.cjk_pool;
  size_t pool_index = 0;
  iree_host_size_t tokens = 0;
  for (auto _ : state) {
    tokens =
        DoEncodeStreaming(pool.next_text(pool_index), g_state.output_tokens);
    if (tokens == 0) {
      state.SkipWithError("Encode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["bytes/token"] = (double)pool.chunk_size / tokens;
}
BENCHMARK(BM_CJK_EncodeStreaming)->Unit(benchmark::kMillisecond);

void BM_CJK_EncodeOneShot(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.cjk_pool;
  size_t pool_index = 0;
  iree_host_size_t tokens = 0;
  for (auto _ : state) {
    tokens = DoEncodeOneShot(pool.next_text(pool_index), g_state.output_tokens);
    if (tokens == 0) {
      state.SkipWithError("Encode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["bytes/token"] = (double)pool.chunk_size / tokens;
}
BENCHMARK(BM_CJK_EncodeOneShot)->Unit(benchmark::kMillisecond);

void BM_CJK_Encode1KB(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.cjk_pool;
  size_t pool_index = 0;
  iree_host_size_t tokens = 0;
  size_t chunk_count = (pool.chunk_size + 1023) / 1024;
  for (auto _ : state) {
    tokens = DoEncodeChunked(pool.next_text(pool_index), 1024,
                             g_state.output_tokens);
    if (tokens == 0) {
      state.SkipWithError("Encode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["chunks"] = chunk_count;
}
BENCHMARK(BM_CJK_Encode1KB)->Unit(benchmark::kMillisecond);

// --- Code ---

void BM_Code_EncodeStreaming(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.code_pool;
  size_t pool_index = 0;
  iree_host_size_t tokens = 0;
  for (auto _ : state) {
    tokens =
        DoEncodeStreaming(pool.next_text(pool_index), g_state.output_tokens);
    if (tokens == 0) {
      state.SkipWithError("Encode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["bytes/token"] = (double)pool.chunk_size / tokens;
}
BENCHMARK(BM_Code_EncodeStreaming)->Unit(benchmark::kMillisecond);

void BM_Code_EncodeOneShot(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.code_pool;
  size_t pool_index = 0;
  iree_host_size_t tokens = 0;
  for (auto _ : state) {
    tokens = DoEncodeOneShot(pool.next_text(pool_index), g_state.output_tokens);
    if (tokens == 0) {
      state.SkipWithError("Encode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["bytes/token"] = (double)pool.chunk_size / tokens;
}
BENCHMARK(BM_Code_EncodeOneShot)->Unit(benchmark::kMillisecond);

void BM_Code_Encode1KB(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.code_pool;
  size_t pool_index = 0;
  iree_host_size_t tokens = 0;
  size_t chunk_count = (pool.chunk_size + 1023) / 1024;
  for (auto _ : state) {
    tokens = DoEncodeChunked(pool.next_text(pool_index), 1024,
                             g_state.output_tokens);
    if (tokens == 0) {
      state.SkipWithError("Encode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["chunks"] = chunk_count;
}
BENCHMARK(BM_Code_Encode1KB)->Unit(benchmark::kMillisecond);

//===----------------------------------------------------------------------===//
// Transform buffer size sweep
//===----------------------------------------------------------------------===//

// Streaming encode with a specific transform buffer size.
// Measures how buffer size affects encode throughput (smaller buffers cause
// more frequent flushes but use less memory).
iree_host_size_t DoEncodeStreamingWithBuffer(iree_string_view_t text,
                                             std::vector<int32_t>& tokens,
                                             size_t buffer_size) {
  std::vector<uint8_t> local_buffer(buffer_size);
  iree_tokenizer_encode_state_t* encode_state = nullptr;
  IREE_CHECK_OK(iree_tokenizer_encode_state_initialize(
      g_state.tokenizer,
      iree_make_byte_span(g_state.state_storage.data(),
                          g_state.state_storage.size()),
      iree_make_byte_span(local_buffer.data(), local_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &encode_state));

  iree_string_view_t remaining = text;
  iree_host_size_t total_tokens = 0;
  while (remaining.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t tokens_written = 0;
    IREE_CHECK_OK(iree_tokenizer_encode_state_feed(
        encode_state, remaining,
        iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL,
                                         NULL, tokens.size() - total_tokens),
        &bytes_consumed, &tokens_written));
    remaining.data += bytes_consumed;
    remaining.size -= bytes_consumed;
    total_tokens += tokens_written;
  }

  iree_host_size_t final_tokens = 0;
  IREE_CHECK_OK(iree_tokenizer_encode_state_finalize(
      encode_state,
      iree_tokenizer_make_token_output(tokens.data() + total_tokens, NULL, NULL,
                                       tokens.size() - total_tokens),
      &final_tokens));
  iree_tokenizer_encode_state_deinitialize(encode_state);
  return total_tokens + final_tokens;
}

void BM_ASCII_EncodeBufferSweep(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.ascii_pool;
  size_t pool_index = 0;
  size_t buffer_size = static_cast<size_t>(state.range(0));
  iree_host_size_t tokens = 0;
  for (auto _ : state) {
    tokens = DoEncodeStreamingWithBuffer(pool.next_text(pool_index),
                                         g_state.output_tokens, buffer_size);
    if (tokens == 0) {
      state.SkipWithError("Buffer too small for this tokenizer");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["buffer_kb"] = buffer_size / 1024.0;
}
// Start at IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE (4096 bytes = 2048 logical
// capacity). This accommodates all practical tokenizers — the largest
// max_token_length across our test suite is BLOOM at 900.
BENCHMARK(BM_ASCII_EncodeBufferSweep)
    ->Unit(benchmark::kMillisecond)
    ->Arg(4096)
    ->Arg(8192)
    ->Arg(16384)
    ->Arg(32768)
    ->Arg(65536);

void BM_Code_EncodeBufferSweep(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.code_pool;
  size_t pool_index = 0;
  size_t buffer_size = static_cast<size_t>(state.range(0));
  iree_host_size_t tokens = 0;
  for (auto _ : state) {
    tokens = DoEncodeStreamingWithBuffer(pool.next_text(pool_index),
                                         g_state.output_tokens, buffer_size);
    if (tokens == 0) {
      state.SkipWithError("Buffer too small for this tokenizer");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["buffer_kb"] = buffer_size / 1024.0;
}
BENCHMARK(BM_Code_EncodeBufferSweep)
    ->Unit(benchmark::kMillisecond)
    ->Arg(4096)
    ->Arg(8192)
    ->Arg(16384)
    ->Arg(32768)
    ->Arg(65536);

void BM_CJK_EncodeBufferSweep(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.cjk_pool;
  size_t pool_index = 0;
  size_t buffer_size = static_cast<size_t>(state.range(0));
  iree_host_size_t tokens = 0;
  for (auto _ : state) {
    tokens = DoEncodeStreamingWithBuffer(pool.next_text(pool_index),
                                         g_state.output_tokens, buffer_size);
    if (tokens == 0) {
      state.SkipWithError("Buffer too small for this tokenizer");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * pool.chunk_size);
  state.SetItemsProcessed(state.iterations() * tokens);
  state.counters["tokens"] = tokens;
  state.counters["buffer_kb"] = buffer_size / 1024.0;
}
BENCHMARK(BM_CJK_EncodeBufferSweep)
    ->Unit(benchmark::kMillisecond)
    ->Arg(4096)
    ->Arg(8192)
    ->Arg(16384)
    ->Arg(32768)
    ->Arg(65536);

//===----------------------------------------------------------------------===//
// Decode helpers
//===----------------------------------------------------------------------===//

// Streaming decode using pre-allocated state.
iree_host_size_t DoDecodeStreaming(const std::vector<int32_t>& tokens,
                                   std::vector<char>& output) {
  iree_tokenizer_decode_state_t* decode_state = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decode_state_initialize(
      g_state.tokenizer, IREE_TOKENIZER_DECODE_FLAG_NONE,
      iree_make_byte_span(g_state.decode_state_storage.data(),
                          g_state.decode_state_storage.size()),
      &decode_state));

  iree_tokenizer_token_id_list_t remaining =
      iree_tokenizer_make_token_id_list(tokens.data(), tokens.size());
  iree_mutable_string_view_t out_remaining =
      iree_make_mutable_string_view(output.data(), output.size());
  iree_host_size_t total_bytes = 0;

  while (remaining.count > 0) {
    iree_host_size_t tokens_consumed = 0;
    iree_host_size_t text_written = 0;
    IREE_CHECK_OK(
        iree_tokenizer_decode_state_feed(decode_state, remaining, out_remaining,
                                         &tokens_consumed, &text_written));
    remaining.values += tokens_consumed;
    remaining.count -= tokens_consumed;
    out_remaining.data += text_written;
    out_remaining.size -= text_written;
    total_bytes += text_written;
    // Zero progress means output buffer exhausted.
    if (tokens_consumed == 0 && text_written == 0) {
      break;
    }
  }

  iree_host_size_t final_bytes = 0;
  if (remaining.count == 0) {
    IREE_CHECK_OK(iree_tokenizer_decode_state_finalize(
        decode_state, out_remaining, &final_bytes));
  }
  iree_tokenizer_decode_state_deinitialize(decode_state);
  return total_bytes + final_bytes;
}

// One-shot decode (allocates state per call via high-level API).
iree_host_size_t DoDecodeOneShot(const std::vector<int32_t>& tokens,
                                 std::vector<char>& output) {
  iree_host_size_t text_length = 0;
  iree_mutable_string_view_t out =
      iree_make_mutable_string_view(output.data(), output.size());
  IREE_CHECK_OK(iree_tokenizer_decode(
      g_state.tokenizer,
      iree_tokenizer_make_token_id_list(tokens.data(), tokens.size()),
      IREE_TOKENIZER_DECODE_FLAG_NONE, out, iree_allocator_system(),
      &text_length));
  return text_length;
}

// Chunked decode - 256 tokens per chunk (= 1KB of token ID input data).
// Each chunk gets its own state (allocates per chunk via high-level API).
iree_host_size_t DoDecodeChunked(const std::vector<int32_t>& tokens,
                                 size_t chunk_size, std::vector<char>& output) {
  iree_host_size_t total_bytes = 0;
  for (size_t offset = 0; offset < tokens.size(); offset += chunk_size) {
    size_t length = std::min(chunk_size, tokens.size() - offset);
    iree_host_size_t text_length = 0;
    iree_mutable_string_view_t out =
        iree_make_mutable_string_view(output.data(), output.size());
    IREE_CHECK_OK(iree_tokenizer_decode(
        g_state.tokenizer,
        iree_tokenizer_make_token_id_list(tokens.data() + offset, length),
        IREE_TOKENIZER_DECODE_FLAG_NONE, out, iree_allocator_system(),
        &text_length));
    total_bytes += text_length;
  }
  return total_bytes;
}

//===----------------------------------------------------------------------===//
// Decode benchmarks
//===----------------------------------------------------------------------===//

// --- ASCII ---

void BM_ASCII_DecodeStreaming(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.ascii_pool;
  size_t pool_index = 0;
  iree_host_size_t bytes = 0;
  for (auto _ : state) {
    bytes =
        DoDecodeStreaming(pool.next_tokens(pool_index), g_state.output_text);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["output_bytes"] = bytes;
}
BENCHMARK(BM_ASCII_DecodeStreaming)->Unit(benchmark::kMillisecond);

void BM_ASCII_DecodeOneShot(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.ascii_pool;
  size_t pool_index = 0;
  iree_host_size_t bytes = 0;
  for (auto _ : state) {
    bytes = DoDecodeOneShot(pool.next_tokens(pool_index), g_state.output_text);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["output_bytes"] = bytes;
}
BENCHMARK(BM_ASCII_DecodeOneShot)->Unit(benchmark::kMillisecond);

void BM_ASCII_Decode1KB(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.ascii_pool;
  size_t pool_index = 0;
  iree_host_size_t bytes = 0;
  size_t chunk_count = (pool.token_chunks[0].size() + 255) / 256;
  for (auto _ : state) {
    bytes =
        DoDecodeChunked(pool.next_tokens(pool_index), 256, g_state.output_text);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["chunks"] = chunk_count;
}
BENCHMARK(BM_ASCII_Decode1KB)->Unit(benchmark::kMillisecond);

// --- CJK ---

void BM_CJK_DecodeStreaming(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.cjk_pool;
  size_t pool_index = 0;
  iree_host_size_t bytes = 0;
  for (auto _ : state) {
    bytes =
        DoDecodeStreaming(pool.next_tokens(pool_index), g_state.output_text);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["output_bytes"] = bytes;
}
BENCHMARK(BM_CJK_DecodeStreaming)->Unit(benchmark::kMillisecond);

void BM_CJK_DecodeOneShot(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.cjk_pool;
  size_t pool_index = 0;
  iree_host_size_t bytes = 0;
  for (auto _ : state) {
    bytes = DoDecodeOneShot(pool.next_tokens(pool_index), g_state.output_text);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["output_bytes"] = bytes;
}
BENCHMARK(BM_CJK_DecodeOneShot)->Unit(benchmark::kMillisecond);

void BM_CJK_Decode1KB(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.cjk_pool;
  size_t pool_index = 0;
  iree_host_size_t bytes = 0;
  size_t chunk_count = (pool.token_chunks[0].size() + 255) / 256;
  for (auto _ : state) {
    bytes =
        DoDecodeChunked(pool.next_tokens(pool_index), 256, g_state.output_text);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["chunks"] = chunk_count;
}
BENCHMARK(BM_CJK_Decode1KB)->Unit(benchmark::kMillisecond);

// --- Code ---

void BM_Code_DecodeStreaming(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.code_pool;
  size_t pool_index = 0;
  iree_host_size_t bytes = 0;
  for (auto _ : state) {
    bytes =
        DoDecodeStreaming(pool.next_tokens(pool_index), g_state.output_text);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["output_bytes"] = bytes;
}
BENCHMARK(BM_Code_DecodeStreaming)->Unit(benchmark::kMillisecond);

void BM_Code_DecodeOneShot(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.code_pool;
  size_t pool_index = 0;
  iree_host_size_t bytes = 0;
  for (auto _ : state) {
    bytes = DoDecodeOneShot(pool.next_tokens(pool_index), g_state.output_text);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["output_bytes"] = bytes;
}
BENCHMARK(BM_Code_DecodeOneShot)->Unit(benchmark::kMillisecond);

void BM_Code_Decode1KB(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.code_pool;
  size_t pool_index = 0;
  iree_host_size_t bytes = 0;
  size_t chunk_count = (pool.token_chunks[0].size() + 255) / 256;
  for (auto _ : state) {
    bytes =
        DoDecodeChunked(pool.next_tokens(pool_index), 256, g_state.output_text);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["chunks"] = chunk_count;
}
BENCHMARK(BM_Code_Decode1KB)->Unit(benchmark::kMillisecond);

//===----------------------------------------------------------------------===//
// Decode output buffer size sweep
//===----------------------------------------------------------------------===//

// Streaming decode with a limited output buffer per feed call.
// Measures how output buffer size affects decode throughput (smaller buffers
// cause more frequent feed calls and output-full resumptions).
iree_host_size_t DoDecodeStreamingWithBuffer(const std::vector<int32_t>& tokens,
                                             std::vector<char>& output,
                                             size_t output_chunk_size) {
  iree_tokenizer_decode_state_t* decode_state = nullptr;
  IREE_CHECK_OK(iree_tokenizer_decode_state_initialize(
      g_state.tokenizer, IREE_TOKENIZER_DECODE_FLAG_NONE,
      iree_make_byte_span(g_state.decode_state_storage.data(),
                          g_state.decode_state_storage.size()),
      &decode_state));

  iree_tokenizer_token_id_list_t remaining =
      iree_tokenizer_make_token_id_list(tokens.data(), tokens.size());
  iree_host_size_t total_bytes = 0;

  while (remaining.count > 0) {
    size_t available = std::min(output_chunk_size, output.size() - total_bytes);
    if (available == 0) {
      break;
    }

    iree_mutable_string_view_t chunk_output =
        iree_make_mutable_string_view(output.data() + total_bytes, available);

    iree_host_size_t tokens_consumed = 0;
    iree_host_size_t text_written = 0;
    IREE_CHECK_OK(
        iree_tokenizer_decode_state_feed(decode_state, remaining, chunk_output,
                                         &tokens_consumed, &text_written));
    remaining.values += tokens_consumed;
    remaining.count -= tokens_consumed;
    total_bytes += text_written;
    if (tokens_consumed == 0 && text_written == 0) {
      break;
    }
  }

  iree_host_size_t final_bytes = 0;
  size_t final_available =
      std::min(output_chunk_size, output.size() - total_bytes);
  IREE_CHECK_OK(iree_tokenizer_decode_state_finalize(
      decode_state,
      iree_make_mutable_string_view(output.data() + total_bytes,
                                    final_available),
      &final_bytes));
  iree_tokenizer_decode_state_deinitialize(decode_state);
  return total_bytes + final_bytes;
}

void BM_ASCII_DecodeBufferSweep(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  auto& pool = g_state.ascii_pool;
  size_t pool_index = 0;
  size_t chunk_size = static_cast<size_t>(state.range(0));
  iree_host_size_t bytes = 0;
  for (auto _ : state) {
    bytes = DoDecodeStreamingWithBuffer(pool.next_tokens(pool_index),
                                        g_state.output_text, chunk_size);
    if (bytes == 0) {
      state.SkipWithError("Decode failed");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * bytes);
  state.SetItemsProcessed(state.iterations() * pool.token_chunks[0].size());
  state.counters["tokens"] = pool.token_chunks[0].size();
  state.counters["output_bytes"] = bytes;
  state.counters["chunk_kb"] = chunk_size / 1024.0;
}
BENCHMARK(BM_ASCII_DecodeBufferSweep)
    ->Unit(benchmark::kMillisecond)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536)
    ->Arg(262144)
    ->Arg(1048576);

//===----------------------------------------------------------------------===//
// Init/deinit overhead
//===----------------------------------------------------------------------===//

void BM_StateInitDeinit(benchmark::State& state) {
  if (!EnsureInitialized(state)) {
    return;
  }
  for (auto _ : state) {
    iree_tokenizer_encode_state_t* encode_state = nullptr;
    IREE_CHECK_OK(iree_tokenizer_encode_state_initialize(
        g_state.tokenizer,
        iree_make_byte_span(g_state.state_storage.data(),
                            g_state.state_storage.size()),
        iree_make_byte_span(g_state.transform_buffer.data(),
                            g_state.transform_buffer.size()),
        iree_tokenizer_offset_run_list_empty(),
        IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &encode_state));
    iree_tokenizer_encode_state_deinitialize(encode_state);
  }
}
BENCHMARK(BM_StateInitDeinit)->Unit(benchmark::kNanosecond);

}  // namespace

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

static int RunMain(int argc, char** argv) {
  // clang-format off
  iree_flags_set_usage("comprehensive-benchmark", R"help(Comprehensive tokenizer encode/decode benchmark.
Tests throughput across ASCII, CJK, and code text for any HuggingFace
tokenizer.json. Uses Google Benchmark for statistical rigor.

QUICK START:
  # Download a tokenizer
  curl -Lo gpt2.json https://huggingface.co/gpt2/resolve/main/tokenizer.json

  # Run with optimization (critical for meaningful numbers)
  iree-bazel-run --copt=-O3 --copt=-march=native --features=thin_lto \
    //runtime/src/iree/tokenizer/tools:comprehensive_benchmark -- \
    --tokenizer_json=gpt2.json --benchmark_min_time=1s

  # Or use the convenience script (downloads all recommended tokenizers)
  runtime/src/iree/tokenizer/tools/run_benchmarks.sh

TEXT INPUT:
  If no text files are provided, embedded default texts (~256KB each,
  ~1KB seed repeated) are used automatically. For more representative
  results, provide natural text files:
    --ascii_text=sherlock.txt     ASCII/English prose
    --cjk_text=japanese.txt       Chinese/Japanese/Korean text
    --code_text=source.c          Source code

POOL ROTATION (--rotate, default: false):
  Allocates 64 full copies of each text corpus at separate heap
  addresses. Each benchmark iteration uses a different copy, so the
  total working set (64x text size) exceeds L3 cache and prevents
  cache warming from inflating throughput by 20-40%%.

  Off by default because the memory footprint is large (~64x corpus
  size per corpus). The run_benchmarks.sh script passes --rotate
  automatically.

BUILD FLAGS (critical for performance):
  --copt=-O3              Full optimization
  --copt=-march=native    CPU-specific instruction set
  --features=thin_lto     Link-time optimization

RECOMMENDED TOKENIZERS:
  Download with:
    curl -Lo <name>.json https://huggingface.co/<repo>/resolve/main/tokenizer.json

  All models below are ungated (no authentication required).

  Algo       Vocab  Name          Repo
  ---------  -----  ----          ----
  BPE         50K   GPT-2         gpt2
  BPE        128K   Llama 3       NousResearch/Meta-Llama-3-8B
  BPE        256K   Gemma 2B      unsloth/gemma-2b
  BPE        151K   Qwen 2.5      Qwen/Qwen2.5-7B
  BPE        250K   BLOOM         bigscience/bloom-560m
  BPE        131K   Mistral NeMo  mistralai/Mistral-Nemo-Instruct-2407
  BPE        129K   DeepSeek V3   deepseek-ai/DeepSeek-V3
  BPE         51K   Whisper       openai/whisper-large-v3
  WordPiece   30K   BERT          google-bert/bert-base-uncased
  Unigram     32K   T5            google-t5/t5-base

  These cover the major tokenizer algorithm families (BPE, WordPiece,
  Unigram) and vocab sizes (30K-256K).

BENCHMARK FLAGS (from Google Benchmark):
  --benchmark_min_time=<t>s     Minimum duration (e.g., 1s, 5s)
  --benchmark_filter=<regex>    Run only matching benchmarks
  --benchmark_format=<fmt>      Output: console, json, csv
  --benchmark_out=<file>        Write results to file
  --benchmark_repetitions=<n>   Repeat for statistics
)help");
  // clang-format on

  // Parse IREE flags first, leaving benchmark flags for google benchmark.
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  ::benchmark::Initialize(&argc, argv);

  if (FLAG_tokenizer_json[0] == '\0') {
    fprintf(stderr,
            "No --tokenizer_json specified; all benchmarks will be skipped.\n"
            "Use --help for download instructions.\n");
  }

  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  // Cleanup.
  if (g_state.tokenizer) {
    iree_tokenizer_free(g_state.tokenizer);
    g_state.tokenizer = nullptr;
  }

  return 0;
}

int main(int argc, char** argv) { return RunMain(argc, argv); }
