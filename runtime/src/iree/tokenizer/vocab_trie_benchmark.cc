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
#include "iree/tokenizer/vocab_hash.h"
#include "iree/tokenizer/vocab_trie.h"

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
// Trie Benchmark Fixture
//===----------------------------------------------------------------------===//

class TrieBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    if (!trie_) {
      CreateTestVocabulary(&strings_, &tokens_, &string_table_);
      iree_const_byte_span_t table_span = {string_table_.data(),
                                           string_table_.size()};
      iree_status_t status = iree_tokenizer_vocab_trie_build(
          tokens_.data(), tokens_.size(), table_span, iree_allocator_system(),
          &trie_);
      if (!iree_status_is_ok(status)) {
        state.SkipWithError("failed to build trie");
        iree_status_ignore(status);
        return;
      }
    }
    input_length_ = state.range(0);
    input_ = GenerateInput(input_length_);
  }

  void TearDown(benchmark::State&) override { input_.clear(); }

 protected:
  static std::vector<std::string> strings_;
  static std::vector<iree_tokenizer_token_t> tokens_;
  static std::vector<uint8_t> string_table_;
  static iree_tokenizer_vocab_trie_t* trie_;
  int64_t input_length_ = 0;
  std::string input_;
};

std::vector<std::string> TrieBenchmark::strings_;
std::vector<iree_tokenizer_token_t> TrieBenchmark::tokens_;
std::vector<uint8_t> TrieBenchmark::string_table_;
iree_tokenizer_vocab_trie_t* TrieBenchmark::trie_ = nullptr;

// Benchmark trie-based Viterbi-style traversal.
// For each starting position, traverse the trie to find all matching tokens.
BENCHMARK_DEFINE_F(TrieBenchmark, ViterbiTraversal)(benchmark::State& state) {
  int64_t total_matches = 0;

  for (auto _ : state) {
    int64_t matches = 0;

    // Simulate Viterbi forward pass: for each starting position,
    // traverse trie to find all matching tokens.
    for (size_t start = 0; start < input_.size(); ++start) {
      iree_tokenizer_trie_cursor_t cursor;
      iree_tokenizer_trie_cursor_reset(&cursor, trie_);

      for (size_t i = start; i < input_.size(); ++i) {
        if (!iree_tokenizer_trie_cursor_advance(
                &cursor, static_cast<uint8_t>(input_[i]))) {
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

  state.SetBytesProcessed(state.iterations() * input_.size());
  state.counters["matches"] = total_matches;
}

BENCHMARK_REGISTER_F(TrieBenchmark, ViterbiTraversal)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);

//===----------------------------------------------------------------------===//
// Hash Benchmark Fixture
//===----------------------------------------------------------------------===//

class HashBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    if (!hash_) {
      CreateTestVocabulary(&strings_, &tokens_, &string_table_);
      iree_const_byte_span_t table_span = {string_table_.data(),
                                           string_table_.size()};
      iree_status_t status = iree_tokenizer_vocab_hash_build(
          tokens_.data(), tokens_.size(), table_span,
          IREE_TOKENIZER_VOCAB_HASH_DEFAULT_LOAD_PERCENT,
          iree_allocator_system(), &hash_);
      if (!iree_status_is_ok(status)) {
        state.SkipWithError("failed to build hash");
        iree_status_ignore(status);
        return;
      }
    }
    input_length_ = state.range(0);
    input_ = GenerateInput(input_length_);
  }

  void TearDown(benchmark::State&) override { input_.clear(); }

 protected:
  static std::vector<std::string> strings_;
  static std::vector<iree_tokenizer_token_t> tokens_;
  static std::vector<uint8_t> string_table_;
  static iree_tokenizer_vocab_hash_t* hash_;
  int64_t input_length_ = 0;
  std::string input_;
};

std::vector<std::string> HashBenchmark::strings_;
std::vector<iree_tokenizer_token_t> HashBenchmark::tokens_;
std::vector<uint8_t> HashBenchmark::string_table_;
iree_tokenizer_vocab_hash_t* HashBenchmark::hash_ = nullptr;

// Benchmark hash-based Viterbi-style lookup (current O(N³) algorithm).
// For each starting position and length, do a hash lookup.
BENCHMARK_DEFINE_F(HashBenchmark, ViterbiLookups)(benchmark::State& state) {
  int64_t total_matches = 0;

  for (auto _ : state) {
    int64_t matches = 0;

    // Simulate current Viterbi forward pass: for each (start, length) pair,
    // do an independent hash lookup.
    for (size_t start = 0; start < input_.size(); ++start) {
      for (size_t length = 1; length <= input_.size() - start; ++length) {
        iree_string_view_t substr =
            iree_make_string_view(input_.data() + start, length);
        if (iree_tokenizer_vocab_hash_lookup(hash_, substr) >= 0) {
          ++matches;
        }
      }
    }

    benchmark::DoNotOptimize(matches);
    total_matches = matches;
  }

  state.SetBytesProcessed(state.iterations() * input_.size());
  state.counters["matches"] = total_matches;
}

// Hash benchmark limited to smaller sizes - O(N³) is too slow for N>128.
// At N=128: ~1M hash operations, ~100ms reasonable.
// At N=256: ~8M hash operations, ~1s borderline.
// At N=512+: too slow to benchmark.
BENCHMARK_REGISTER_F(HashBenchmark, ViterbiLookups)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

//===----------------------------------------------------------------------===//
// Standalone Benchmarks
//===----------------------------------------------------------------------===//

// Benchmark trie construction time.
void BM_TrieBuild(benchmark::State& state) {
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
BENCHMARK(BM_TrieBuild);

// Benchmark hash construction time for comparison.
void BM_HashBuild(benchmark::State& state) {
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
BENCHMARK(BM_HashBuild);

}  // namespace
