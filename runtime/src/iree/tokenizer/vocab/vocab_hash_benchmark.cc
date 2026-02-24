// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/benchmark.h"
#include "iree/tokenizer/vocab/vocab_hash.h"

// Benchmarks for vocab hash table operations at various vocabulary sizes.
// Typical LLM vocabulary sizes:
//   - BERT: ~30K tokens
//   - GPT-2: ~50K tokens
//   - Llama 3: ~128K tokens

//===----------------------------------------------------------------------===//
// Test Data Generation
//===----------------------------------------------------------------------===//

// Generates a deterministic pseudo-random string of given length.
static std::string GenerateString(uint32_t seed, size_t length) {
  std::string result;
  result.reserve(length);
  uint32_t state = seed;
  for (size_t i = 0; i < length; ++i) {
    // Simple LCG for reproducibility.
    state = state * 1103515245 + 12345;
    // Generate printable ASCII characters (32-126).
    char c = 32 + (state >> 16) % 95;
    result.push_back(c);
  }
  return result;
}

// Builds test vocabulary data with deterministic token strings.
struct VocabData {
  std::vector<iree_tokenizer_token_t> tokens;
  std::vector<uint8_t> string_table;
  std::vector<std::string> strings;  // Keep strings alive.

  void Build(size_t token_count, size_t avg_string_length) {
    tokens.clear();
    string_table.clear();
    strings.clear();
    tokens.reserve(token_count);
    strings.reserve(token_count);

    for (size_t i = 0; i < token_count; ++i) {
      // Vary string length around average (50% to 150%).
      size_t string_length =
          avg_string_length / 2 + (i * 7 + 13) % (avg_string_length + 1);
      std::string s = GenerateString(static_cast<uint32_t>(i), string_length);
      strings.push_back(s);

      iree_tokenizer_token_t token = {};
      token.string_offset = static_cast<uint32_t>(string_table.size());
      token.string_length = static_cast<uint16_t>(s.size());
      token.attributes = IREE_TOKENIZER_TOKEN_ATTR_NONE;
      tokens.push_back(token);

      string_table.insert(string_table.end(), s.begin(), s.end());
    }
  }

  iree_const_byte_span_t StringTableSpan() const {
    return iree_const_byte_span_t{string_table.data(), string_table.size()};
  }
};

//===----------------------------------------------------------------------===//
// Build Benchmarks
//===----------------------------------------------------------------------===//

// Measures time to build hash table from vocabulary.

static iree_status_t BenchmarkBuild(iree_benchmark_state_t* benchmark_state,
                                    size_t vocab_size) {
  VocabData data;
  data.Build(vocab_size, 6);  // Average 6 chars per token.

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_tokenizer_vocab_hash_t* hash = nullptr;
    iree_status_t status = iree_tokenizer_vocab_hash_build(
        data.tokens.data(), data.tokens.size(), data.StringTableSpan(),
        IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
        &hash);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    iree_tokenizer_vocab_hash_free(hash);
  }
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_VocabHashBuild_100) {
  return BenchmarkBuild(benchmark_state, 100);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashBuild_100);

IREE_BENCHMARK_FN(BM_VocabHashBuild_1K) {
  return BenchmarkBuild(benchmark_state, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashBuild_1K);

IREE_BENCHMARK_FN(BM_VocabHashBuild_10K) {
  return BenchmarkBuild(benchmark_state, 10000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashBuild_10K);

IREE_BENCHMARK_FN(BM_VocabHashBuild_50K) {
  return BenchmarkBuild(benchmark_state, 50000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashBuild_50K);

IREE_BENCHMARK_FN(BM_VocabHashBuild_128K) {
  return BenchmarkBuild(benchmark_state, 128000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashBuild_128K);

//===----------------------------------------------------------------------===//
// Lookup Hit Benchmarks
//===----------------------------------------------------------------------===//

// Measures time for successful lookups (keys that exist in vocab).

static iree_status_t BenchmarkLookupHit(iree_benchmark_state_t* benchmark_state,
                                        size_t vocab_size,
                                        size_t lookups_per_iteration) {
  VocabData data;
  data.Build(vocab_size, 6);

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_hash_build(
      data.tokens.data(), data.tokens.size(), data.StringTableSpan(),
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));

  // Use prime step to access different parts of vocabulary.
  size_t step = vocab_size > 100 ? 997 : 7;
  size_t index = 0;

  while (iree_benchmark_keep_running(benchmark_state, lookups_per_iteration)) {
    for (size_t i = 0; i < lookups_per_iteration; ++i) {
      const std::string& s = data.strings[index];
      int32_t result = iree_tokenizer_vocab_hash_lookup(
          hash, iree_make_string_view(s.data(), s.size()));
      iree_optimization_barrier(result);
      index = (index + step) % vocab_size;
    }
  }

  iree_tokenizer_vocab_hash_free(hash);
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_VocabHashLookupHit_100) {
  return BenchmarkLookupHit(benchmark_state, 100, 100);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupHit_100);

IREE_BENCHMARK_FN(BM_VocabHashLookupHit_1K) {
  return BenchmarkLookupHit(benchmark_state, 1000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupHit_1K);

IREE_BENCHMARK_FN(BM_VocabHashLookupHit_10K) {
  return BenchmarkLookupHit(benchmark_state, 10000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupHit_10K);

IREE_BENCHMARK_FN(BM_VocabHashLookupHit_50K) {
  return BenchmarkLookupHit(benchmark_state, 50000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupHit_50K);

IREE_BENCHMARK_FN(BM_VocabHashLookupHit_128K) {
  return BenchmarkLookupHit(benchmark_state, 128000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupHit_128K);

//===----------------------------------------------------------------------===//
// Lookup Miss Benchmarks
//===----------------------------------------------------------------------===//

// Measures time for failed lookups (keys not in vocab).
// Miss lookups should be fast due to early termination on empty slot.

static iree_status_t BenchmarkLookupMiss(
    iree_benchmark_state_t* benchmark_state, size_t vocab_size,
    size_t lookups_per_iteration) {
  VocabData data;
  data.Build(vocab_size, 6);

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_hash_build(
      data.tokens.data(), data.tokens.size(), data.StringTableSpan(),
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));

  // Generate strings that won't be in the vocabulary.
  std::vector<std::string> miss_strings;
  miss_strings.reserve(lookups_per_iteration);
  for (size_t i = 0; i < lookups_per_iteration; ++i) {
    // Use offset to ensure different strings than vocab.
    miss_strings.push_back(
        GenerateString(static_cast<uint32_t>(vocab_size + i + 1000000), 6));
  }

  size_t index = 0;
  while (iree_benchmark_keep_running(benchmark_state, lookups_per_iteration)) {
    for (size_t i = 0; i < lookups_per_iteration; ++i) {
      const std::string& s = miss_strings[index];
      int32_t result = iree_tokenizer_vocab_hash_lookup(
          hash, iree_make_string_view(s.data(), s.size()));
      iree_optimization_barrier(result);
      index = (index + 1) % lookups_per_iteration;
    }
  }

  iree_tokenizer_vocab_hash_free(hash);
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_VocabHashLookupMiss_100) {
  return BenchmarkLookupMiss(benchmark_state, 100, 100);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupMiss_100);

IREE_BENCHMARK_FN(BM_VocabHashLookupMiss_1K) {
  return BenchmarkLookupMiss(benchmark_state, 1000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupMiss_1K);

IREE_BENCHMARK_FN(BM_VocabHashLookupMiss_10K) {
  return BenchmarkLookupMiss(benchmark_state, 10000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupMiss_10K);

IREE_BENCHMARK_FN(BM_VocabHashLookupMiss_50K) {
  return BenchmarkLookupMiss(benchmark_state, 50000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupMiss_50K);

IREE_BENCHMARK_FN(BM_VocabHashLookupMiss_128K) {
  return BenchmarkLookupMiss(benchmark_state, 128000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupMiss_128K);

//===----------------------------------------------------------------------===//
// Mixed Lookup Benchmarks
//===----------------------------------------------------------------------===//

// Simulates realistic workload: 90% hits, 10% misses.

static iree_status_t BenchmarkLookupMixed(
    iree_benchmark_state_t* benchmark_state, size_t vocab_size,
    size_t lookups_per_iteration) {
  VocabData data;
  data.Build(vocab_size, 6);

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_hash_build(
      data.tokens.data(), data.tokens.size(), data.StringTableSpan(),
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));

  // Generate some miss strings.
  std::vector<std::string> miss_strings;
  size_t miss_count = lookups_per_iteration / 10;
  for (size_t i = 0; i < miss_count; ++i) {
    miss_strings.push_back(
        GenerateString(static_cast<uint32_t>(vocab_size + i + 1000000), 6));
  }

  size_t hit_index = 0;
  size_t miss_index = 0;
  size_t step = vocab_size > 100 ? 997 : 7;

  while (iree_benchmark_keep_running(benchmark_state, lookups_per_iteration)) {
    for (size_t i = 0; i < lookups_per_iteration; ++i) {
      int32_t result;
      if (i % 10 == 0 && miss_count > 0) {
        // 10% misses.
        const std::string& s = miss_strings[miss_index];
        result = iree_tokenizer_vocab_hash_lookup(
            hash, iree_make_string_view(s.data(), s.size()));
        miss_index = (miss_index + 1) % miss_count;
      } else {
        // 90% hits.
        const std::string& s = data.strings[hit_index];
        result = iree_tokenizer_vocab_hash_lookup(
            hash, iree_make_string_view(s.data(), s.size()));
        hit_index = (hit_index + step) % vocab_size;
      }
      iree_optimization_barrier(result);
    }
  }

  iree_tokenizer_vocab_hash_free(hash);
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_VocabHashLookupMixed_10K) {
  return BenchmarkLookupMixed(benchmark_state, 10000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupMixed_10K);

IREE_BENCHMARK_FN(BM_VocabHashLookupMixed_50K) {
  return BenchmarkLookupMixed(benchmark_state, 50000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupMixed_50K);

IREE_BENCHMARK_FN(BM_VocabHashLookupMixed_128K) {
  return BenchmarkLookupMixed(benchmark_state, 128000, 1000);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookupMixed_128K);

//===----------------------------------------------------------------------===//
// String Length Impact Benchmarks
//===----------------------------------------------------------------------===//

// Measures impact of token string length on lookup performance.

static iree_status_t BenchmarkLookupByLength(
    iree_benchmark_state_t* benchmark_state, size_t avg_length) {
  const size_t vocab_size = 10000;
  VocabData data;
  data.Build(vocab_size, avg_length);

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_hash_build(
      data.tokens.data(), data.tokens.size(), data.StringTableSpan(),
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));

  size_t index = 0;
  const size_t lookups = 1000;
  while (iree_benchmark_keep_running(benchmark_state, lookups)) {
    for (size_t i = 0; i < lookups; ++i) {
      const std::string& s = data.strings[index];
      int32_t result = iree_tokenizer_vocab_hash_lookup(
          hash, iree_make_string_view(s.data(), s.size()));
      iree_optimization_barrier(result);
      index = (index + 997) % vocab_size;
    }
  }

  iree_tokenizer_vocab_hash_free(hash);
  return iree_ok_status();
}

// Short tokens (typical subword tokens).
IREE_BENCHMARK_FN(BM_VocabHashLookup_ShortTokens_3char) {
  return BenchmarkLookupByLength(benchmark_state, 3);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookup_ShortTokens_3char);

// Medium tokens (common words).
IREE_BENCHMARK_FN(BM_VocabHashLookup_MediumTokens_8char) {
  return BenchmarkLookupByLength(benchmark_state, 8);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookup_MediumTokens_8char);

// Long tokens (multi-word or technical terms).
IREE_BENCHMARK_FN(BM_VocabHashLookup_LongTokens_20char) {
  return BenchmarkLookupByLength(benchmark_state, 20);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookup_LongTokens_20char);

// Very long tokens (edge case).
IREE_BENCHMARK_FN(BM_VocabHashLookup_VeryLongTokens_50char) {
  return BenchmarkLookupByLength(benchmark_state, 50);
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookup_VeryLongTokens_50char);

//===----------------------------------------------------------------------===//
// Sequential Access Pattern
//===----------------------------------------------------------------------===//

// Simulates tokenization of text: sequential lookups with locality.

IREE_BENCHMARK_FN(BM_VocabHashLookup_Sequential_50K) {
  const size_t vocab_size = 50000;
  VocabData data;
  data.Build(vocab_size, 6);

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_hash_build(
      data.tokens.data(), data.tokens.size(), data.StringTableSpan(),
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash));

  // Sequential access simulating text tokenization.
  // In real tokenization, consecutive lookups often hit nearby vocabulary
  // entries due to common word patterns.
  const size_t lookups = 1000;
  size_t base_index = 0;

  while (iree_benchmark_keep_running(benchmark_state, lookups)) {
    for (size_t i = 0; i < lookups; ++i) {
      // Small variations around a base index (locality).
      size_t index = (base_index + (i % 100)) % vocab_size;
      const std::string& s = data.strings[index];
      int32_t result = iree_tokenizer_vocab_hash_lookup(
          hash, iree_make_string_view(s.data(), s.size()));
      iree_optimization_barrier(result);
    }
    // Move to different region for next iteration.
    base_index = (base_index + 5000) % vocab_size;
  }

  iree_tokenizer_vocab_hash_free(hash);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_VocabHashLookup_Sequential_50K);
