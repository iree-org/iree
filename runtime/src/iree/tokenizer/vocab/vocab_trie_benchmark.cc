// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks comparing trie-based vs hash-based vocabulary lookup for
// Viterbi-style tokenization access patterns.
//
// The key insight: Viterbi tokenization requires checking all possible
// substrings from each starting position. With hash tables, this is O(N³)
// because each of O(N²) lookups costs O(L) for hashing. With a trie, it's
// O(N²) because a single O(L) traversal finds ALL matching tokens.
//
// Expected results:
// - Trie: time(2N) / time(N) ≈ 4 (quadratic scaling)
// - Hash: time(2N) / time(N) ≈ 8 (cubic scaling)

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/tokenizer/vocab/vocab_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

namespace {

//===----------------------------------------------------------------------===//
// Test Data Setup
//===----------------------------------------------------------------------===//

// Creates a vocabulary of common substrings to ensure matches during benchmark.
// Includes single characters, pairs, and longer tokens.
static void CreateTestVocabulary(std::vector<std::string>* strings,
                                 std::vector<iree_tokenizer_token_t>* tokens,
                                 std::vector<uint8_t>* string_table) {
  strings->clear();
  tokens->clear();
  string_table->clear();

  // Single ASCII characters (common in BPE/Unigram).
  for (char c = 'a'; c <= 'z'; ++c) {
    strings->push_back(std::string(1, c));
  }
  for (char c = 'A'; c <= 'Z'; ++c) {
    strings->push_back(std::string(1, c));
  }
  for (char c = '0'; c <= '9'; ++c) {
    strings->push_back(std::string(1, c));
  }

  // Common pairs.
  const char* pairs[] = {"th", "he", "in", "er", "an", "re", "on", "at",
                         "en", "nd", "ti", "es", "or", "te", "of", "ed"};
  for (const char* p : pairs) {
    strings->push_back(p);
  }

  // Common longer tokens.
  const char* words[] = {"the",  "and",   "ing",   "tion",  "that",  "with",
                         "have", "this",  "will",  "your",  "from",  "they",
                         "been", "which", "their", "would", "there", "about"};
  for (const char* w : words) {
    strings->push_back(w);
  }

  // Build string table and tokens.
  for (const auto& s : *strings) {
    iree_tokenizer_token_t token = {};
    token.string_offset = static_cast<uint32_t>(string_table->size());
    token.string_length = static_cast<uint16_t>(s.size());
    token.attributes = IREE_TOKENIZER_TOKEN_ATTR_NONE;
    tokens->push_back(token);
    string_table->insert(string_table->end(), s.begin(), s.end());
  }
}

// Generates input text of specified length using vocabulary tokens.
static std::string GenerateInput(int64_t length) {
  std::string result;
  result.reserve(length);
  // Use a pattern that creates matches at multiple lengths.
  const char* pattern = "thequickbrownfoxjumpsoverthelazydog";
  size_t pattern_len = strlen(pattern);
  while (result.size() < static_cast<size_t>(length)) {
    size_t remaining = length - result.size();
    size_t to_add = std::min(remaining, pattern_len);
    result.append(pattern, to_add);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Trie Benchmark - Viterbi Traversal
//===----------------------------------------------------------------------===//

// Benchmark trie-based Viterbi-style traversal.
// For each starting position, traverse the trie to find all matching tokens.
// Args: [input_length]
void TrieViterbiTraversal(benchmark::State& state) {
  int64_t input_length = state.range(0);

  std::vector<std::string> strings;
  std::vector<iree_tokenizer_token_t> tokens;
  std::vector<uint8_t> string_table;
  CreateTestVocabulary(&strings, &tokens, &string_table);
  iree_const_byte_span_t table_span = {string_table.data(),
                                       string_table.size()};

  iree_tokenizer_vocab_trie_t* trie = nullptr;
  iree_status_t status = iree_tokenizer_vocab_trie_build(
      tokens.data(), tokens.size(), table_span, iree_allocator_system(), &trie);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("failed to build trie");
    iree_status_ignore(status);
    return;
  }

  std::string input = GenerateInput(input_length);
  int64_t total_matches = 0;

  for (auto _ : state) {
    int64_t matches = 0;

    // Simulate Viterbi forward pass: for each starting position,
    // traverse trie to find all matching tokens.
    for (size_t start = 0; start < input.size(); ++start) {
      iree_tokenizer_trie_cursor_t cursor;
      iree_tokenizer_trie_cursor_reset(&cursor, trie);

      for (size_t i = start; i < input.size(); ++i) {
        if (!iree_tokenizer_trie_cursor_advance(
                &cursor, static_cast<uint8_t>(input[i]))) {
          break;
        }
        if (iree_tokenizer_trie_cursor_token_id(&cursor) >= 0) {
          ++matches;
        }
      }
    }

    benchmark::DoNotOptimize(matches);
    total_matches = matches;
  }

  iree_tokenizer_vocab_trie_free(trie);

  state.SetBytesProcessed(state.iterations() * input.size());
  state.counters["matches"] = total_matches;
}

BENCHMARK(TrieViterbiTraversal)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->ArgName("len");

//===----------------------------------------------------------------------===//
// Hash Benchmark - Viterbi Lookups
//===----------------------------------------------------------------------===//

// Benchmark hash-based Viterbi-style lookup (O(N³) algorithm).
// For each starting position and length, do a hash lookup.
// Limited to smaller sizes - O(N³) is too slow for N>128.
void HashViterbiLookups(benchmark::State& state) {
  int64_t input_length = state.range(0);

  std::vector<std::string> strings;
  std::vector<iree_tokenizer_token_t> tokens;
  std::vector<uint8_t> string_table;
  CreateTestVocabulary(&strings, &tokens, &string_table);
  iree_const_byte_span_t table_span = {string_table.data(),
                                       string_table.size()};

  iree_tokenizer_vocab_hash_t* hash = nullptr;
  iree_status_t status = iree_tokenizer_vocab_hash_build(
      tokens.data(), tokens.size(), table_span,
      IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
      &hash);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("failed to build hash");
    iree_status_ignore(status);
    return;
  }

  std::string input = GenerateInput(input_length);
  int64_t total_matches = 0;

  for (auto _ : state) {
    int64_t matches = 0;

    // Simulate current Viterbi forward pass: for each (start, length) pair,
    // do an independent hash lookup.
    for (size_t start = 0; start < input.size(); ++start) {
      for (size_t length = 1; length <= input.size() - start; ++length) {
        iree_string_view_t substr =
            iree_make_string_view(input.data() + start, length);
        if (iree_tokenizer_vocab_hash_lookup(hash, substr) >= 0) {
          ++matches;
        }
      }
    }

    benchmark::DoNotOptimize(matches);
    total_matches = matches;
  }

  iree_tokenizer_vocab_hash_free(hash);

  state.SetBytesProcessed(state.iterations() * input.size());
  state.counters["matches"] = total_matches;
}

BENCHMARK(HashViterbiLookups)->Arg(64)->Arg(128)->ArgName("len");

//===----------------------------------------------------------------------===//
// Standalone Benchmarks
//===----------------------------------------------------------------------===//

// Benchmark trie construction time.
void TrieBuild(benchmark::State& state) {
  std::vector<std::string> strings;
  std::vector<iree_tokenizer_token_t> tokens;
  std::vector<uint8_t> string_table;
  CreateTestVocabulary(&strings, &tokens, &string_table);
  iree_const_byte_span_t table_span = {string_table.data(),
                                       string_table.size()};

  for (auto _ : state) {
    iree_tokenizer_vocab_trie_t* trie = nullptr;
    iree_status_t status = iree_tokenizer_vocab_trie_build(
        tokens.data(), tokens.size(), table_span, iree_allocator_system(),
        &trie);
    benchmark::DoNotOptimize(trie);
    if (iree_status_is_ok(status)) {
      iree_tokenizer_vocab_trie_free(trie);
    } else {
      state.SkipWithError("build failed");
      iree_status_ignore(status);
      return;
    }
  }

  state.SetItemsProcessed(state.iterations() * tokens.size());
}
BENCHMARK(TrieBuild);

// Benchmark hash construction time for comparison.
void HashBuild(benchmark::State& state) {
  std::vector<std::string> strings;
  std::vector<iree_tokenizer_token_t> tokens;
  std::vector<uint8_t> string_table;
  CreateTestVocabulary(&strings, &tokens, &string_table);
  iree_const_byte_span_t table_span = {string_table.data(),
                                       string_table.size()};

  for (auto _ : state) {
    iree_tokenizer_vocab_hash_t* hash = nullptr;
    iree_status_t status = iree_tokenizer_vocab_hash_build(
        tokens.data(), tokens.size(), table_span,
        IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
        &hash);
    benchmark::DoNotOptimize(hash);
    if (iree_status_is_ok(status)) {
      iree_tokenizer_vocab_hash_free(hash);
    } else {
      state.SkipWithError("build failed");
      iree_status_ignore(status);
      return;
    }
  }

  state.SetItemsProcessed(state.iterations() * tokens.size());
}
BENCHMARK(HashBuild);

//===----------------------------------------------------------------------===//
// Large Vocabulary Build Benchmarks
//===----------------------------------------------------------------------===//

// Creates a large vocabulary simulating a realistic tokenizer vocab.
// Distribution based on real tokenizers (BERT, GPT-2, cl100k, DeepSeek-V3):
//   - 0.1-0.2% single-byte tokens
//   - Peak at 5-7 bytes (30-40% of vocab)
//   - Average length 6-10 bytes
//   - Max length up to 256 bytes
// Total: approximately vocab_size tokens.
static void CreateLargeVocabulary(int64_t vocab_size,
                                  std::vector<iree_tokenizer_token_t>* tokens,
                                  std::vector<uint8_t>* string_table) {
  tokens->clear();
  string_table->clear();
  tokens->reserve(vocab_size);
  // Estimate average token length of 7 bytes (based on real vocab stats).
  string_table->reserve(vocab_size * 7);

  // Length distribution weights based on real tokenizers.
  // Lengths 1-16, derived from cl100k_base/DeepSeek-V3 distributions.
  static const int kLengthWeights[] = {
      1,    // 1 byte:  0.1%
      31,   // 2 bytes: 3.1%
      83,   // 3 bytes: 8.3%
      113,  // 4 bytes: 11.3%
      134,  // 5 bytes: 13.4%
      140,  // 6 bytes: 14.0%
      118,  // 7 bytes: 11.8%
      107,  // 8 bytes: 10.7%
      81,   // 9 bytes: 8.1%
      62,   // 10 bytes: 6.2%
      40,   // 11 bytes: 4.0%
      30,   // 12 bytes: 3.0%
      25,   // 13 bytes: 2.5%
      18,   // 14 bytes: 1.8%
      12,   // 15 bytes: 1.2%
      5,    // 16 bytes: 0.5%
  };
  static const size_t kNumLengths = sizeof(kLengthWeights) / sizeof(int);
  static int total_weight = 0;
  if (total_weight == 0) {
    for (size_t i = 0; i < kNumLengths; ++i) {
      total_weight += kLengthWeights[i];
    }
  }

  // Generate tokens with realistic length distribution.
  // Use a simple LCG for deterministic "random" generation.
  uint64_t seed = 12345;
  auto next_random = [&seed]() -> uint64_t {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed;
  };

  while (tokens->size() < static_cast<size_t>(vocab_size)) {
    // Pick length based on distribution.
    int r = static_cast<int>(next_random() % total_weight);
    size_t length = 1;
    int cumulative = 0;
    for (size_t i = 0; i < kNumLengths; ++i) {
      cumulative += kLengthWeights[i];
      if (r < cumulative) {
        length = i + 1;
        break;
      }
    }

    // Generate token bytes (mix of printable ASCII and some high bytes).
    iree_tokenizer_token_t token = {};
    token.string_offset = static_cast<uint32_t>(string_table->size());
    token.string_length = static_cast<uint16_t>(length);
    token.attributes = IREE_TOKENIZER_TOKEN_ATTR_NONE;
    tokens->push_back(token);

    for (size_t i = 0; i < length; ++i) {
      // 90% printable ASCII (32-126), 10% high bytes (128-255).
      uint64_t rb = next_random();
      uint8_t byte;
      if ((rb % 10) < 9) {
        byte = static_cast<uint8_t>(32 + (rb >> 8) % 95);  // Printable ASCII.
      } else {
        byte = static_cast<uint8_t>(128 + (rb >> 8) % 128);  // High byte.
      }
      string_table->push_back(byte);
    }
  }
}

// Trie build benchmark for large vocabularies.
// Args: [vocab_size]
void TrieBuildLarge(benchmark::State& state) {
  int64_t vocab_size = state.range(0);
  std::vector<iree_tokenizer_token_t> tokens;
  std::vector<uint8_t> string_table;
  CreateLargeVocabulary(vocab_size, &tokens, &string_table);
  iree_const_byte_span_t table_span = {string_table.data(),
                                       string_table.size()};

  for (auto _ : state) {
    iree_tokenizer_vocab_trie_t* trie = nullptr;
    iree_status_t status = iree_tokenizer_vocab_trie_build(
        tokens.data(), tokens.size(), table_span, iree_allocator_system(),
        &trie);
    benchmark::DoNotOptimize(trie);
    if (iree_status_is_ok(status)) {
      iree_tokenizer_vocab_trie_free(trie);
    } else {
      state.SkipWithError("build failed");
      iree_status_ignore(status);
      return;
    }
  }

  state.SetItemsProcessed(state.iterations() * vocab_size);
  state.counters["vocab_size"] = vocab_size;
  state.counters["string_table_bytes"] =
      static_cast<double>(string_table.size());
}
BENCHMARK(TrieBuildLarge)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(50000)
    ->Arg(100000)
    ->ArgName("vocab")
    ->Unit(benchmark::kMillisecond);

// Parameterized hash build benchmark for comparison.
void HashBuildLarge(benchmark::State& state) {
  int64_t vocab_size = state.range(0);
  std::vector<iree_tokenizer_token_t> tokens;
  std::vector<uint8_t> string_table;
  CreateLargeVocabulary(vocab_size, &tokens, &string_table);
  iree_const_byte_span_t table_span = {string_table.data(),
                                       string_table.size()};

  for (auto _ : state) {
    iree_tokenizer_vocab_hash_t* hash = nullptr;
    iree_status_t status = iree_tokenizer_vocab_hash_build(
        tokens.data(), tokens.size(), table_span,
        IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT, iree_allocator_system(),
        &hash);
    benchmark::DoNotOptimize(hash);
    if (iree_status_is_ok(status)) {
      iree_tokenizer_vocab_hash_free(hash);
    } else {
      state.SkipWithError("build failed");
      iree_status_ignore(status);
      return;
    }
  }

  state.SetItemsProcessed(state.iterations() * vocab_size);
  state.counters["vocab_size"] = vocab_size;
  state.counters["string_table_bytes"] =
      static_cast<double>(string_table.size());
}
BENCHMARK(HashBuildLarge)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(50000)
    ->Arg(100000)
    ->ArgName("vocab")
    ->Unit(benchmark::kMillisecond);

}  // namespace
