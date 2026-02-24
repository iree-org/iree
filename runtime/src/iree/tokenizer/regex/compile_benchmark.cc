// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Regex compilation benchmarks.
// Measures compilation time for various pattern types and complexities.
//
// For execution benchmarks (DFA matching speed), see exec_benchmark.cc.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/tokenizer/regex/compile.h"

namespace {

//===----------------------------------------------------------------------===//
// Pattern Compilation Benchmarks
//===----------------------------------------------------------------------===//

// Benchmark helper that compiles a pattern and measures time.
void BM_CompilePattern(benchmark::State& state, const char* pattern) {
  for (auto _ : state) {
    uint8_t* dfa_data = nullptr;
    iree_host_size_t dfa_size = 0;
    iree_tokenizer_regex_compile_error_t error = {0};

    iree_status_t status = iree_tokenizer_regex_compile(
        iree_make_cstring_view(pattern),
        IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
        &dfa_data, &dfa_size, &error);

    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Compilation failed");
      iree_status_ignore(status);
      return;
    }

    benchmark::DoNotOptimize(dfa_data);
    benchmark::DoNotOptimize(dfa_size);

    iree_allocator_free(iree_allocator_system(), dfa_data);
  }
}

//===----------------------------------------------------------------------===//
// Simple Patterns
//===----------------------------------------------------------------------===//

void BM_CompileLiteral(benchmark::State& state) {
  BM_CompilePattern(state, "abc");
}
BENCHMARK(BM_CompileLiteral);

void BM_CompileCharClass(benchmark::State& state) {
  BM_CompilePattern(state, "[a-z]+");
}
BENCHMARK(BM_CompileCharClass);

void BM_CompileShorthand(benchmark::State& state) {
  BM_CompilePattern(state, "\\w+");
}
BENCHMARK(BM_CompileShorthand);

void BM_CompileDot(benchmark::State& state) { BM_CompilePattern(state, ".+"); }
BENCHMARK(BM_CompileDot);

//===----------------------------------------------------------------------===//
// Quantifier Patterns
//===----------------------------------------------------------------------===//

void BM_CompileQuantifierStar(benchmark::State& state) {
  BM_CompilePattern(state, "a*b*c*");
}
BENCHMARK(BM_CompileQuantifierStar);

void BM_CompileQuantifierPlus(benchmark::State& state) {
  BM_CompilePattern(state, "a+b+c+");
}
BENCHMARK(BM_CompileQuantifierPlus);

void BM_CompileQuantifierRange(benchmark::State& state) {
  BM_CompilePattern(state, "a{2,4}b{1,3}c{0,5}");
}
BENCHMARK(BM_CompileQuantifierRange);

//===----------------------------------------------------------------------===//
// Alternation Patterns
//===----------------------------------------------------------------------===//

void BM_CompileAlternation3(benchmark::State& state) {
  BM_CompilePattern(state, "cat|dog|bird");
}
BENCHMARK(BM_CompileAlternation3);

void BM_CompileAlternation10(benchmark::State& state) {
  BM_CompilePattern(state,
                    "a|bb|ccc|dddd|eeeee|ffffff|ggggggg|hhhhhhhh|"
                    "iiiiiiiii|jjjjjjjjjj");
}
BENCHMARK(BM_CompileAlternation10);

void BM_CompileAlternation26(benchmark::State& state) {
  BM_CompilePattern(state,
                    "a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z");
}
BENCHMARK(BM_CompileAlternation26);

//===----------------------------------------------------------------------===//
// Group Patterns
//===----------------------------------------------------------------------===//

void BM_CompileGroup(benchmark::State& state) {
  BM_CompilePattern(state, "(abc)+");
}
BENCHMARK(BM_CompileGroup);

void BM_CompileNestedGroups(benchmark::State& state) {
  BM_CompilePattern(state, "((ab)+c)+");
}
BENCHMARK(BM_CompileNestedGroups);

void BM_CompileCaseInsensitiveGroup(benchmark::State& state) {
  BM_CompilePattern(state, "(?i:abc)");
}
BENCHMARK(BM_CompileCaseInsensitiveGroup);

//===----------------------------------------------------------------------===//
// Unicode Patterns
//===----------------------------------------------------------------------===//

void BM_CompileUnicodeLetter(benchmark::State& state) {
  BM_CompilePattern(state, "\\p{L}+");
}
BENCHMARK(BM_CompileUnicodeLetter);

void BM_CompileUnicodeNumber(benchmark::State& state) {
  BM_CompilePattern(state, "\\p{N}+");
}
BENCHMARK(BM_CompileUnicodeNumber);

void BM_CompileUnicodeComplex(benchmark::State& state) {
  BM_CompilePattern(state, "[^\\p{L}\\p{N}]+");
}
BENCHMARK(BM_CompileUnicodeComplex);

//===----------------------------------------------------------------------===//
// Anchor Patterns
//===----------------------------------------------------------------------===//

void BM_CompileStartAnchor(benchmark::State& state) {
  BM_CompilePattern(state, "^abc");
}
BENCHMARK(BM_CompileStartAnchor);

void BM_CompileEndAnchor(benchmark::State& state) {
  BM_CompilePattern(state, "abc$");
}
BENCHMARK(BM_CompileEndAnchor);

void BM_CompileBothAnchors(benchmark::State& state) {
  BM_CompilePattern(state, "^abc$");
}
BENCHMARK(BM_CompileBothAnchors);

//===----------------------------------------------------------------------===//
// Lookahead Patterns
//===----------------------------------------------------------------------===//

void BM_CompileLookaheadChar(benchmark::State& state) {
  BM_CompilePattern(state, "a(?!b)");
}
BENCHMARK(BM_CompileLookaheadChar);

void BM_CompileLookaheadShorthand(benchmark::State& state) {
  BM_CompilePattern(state, "\\s+(?!\\S)");
}
BENCHMARK(BM_CompileLookaheadShorthand);

void BM_CompileLookaheadWithFallback(benchmark::State& state) {
  BM_CompilePattern(state, "\\s+(?!\\S)|\\s+");
}
BENCHMARK(BM_CompileLookaheadWithFallback);

//===----------------------------------------------------------------------===//
// Real-World Tokenizer Patterns
//===----------------------------------------------------------------------===//

// GPT-2 pattern.
void BM_CompileGPT2(benchmark::State& state) {
  BM_CompilePattern(state,
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
                    "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
}
BENCHMARK(BM_CompileGPT2);

// Llama 3 pattern.
void BM_CompileLlama3(benchmark::State& state) {
  BM_CompilePattern(
      state,
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
}
BENCHMARK(BM_CompileLlama3);

// Qwen2 pattern.
void BM_CompileQwen2(benchmark::State& state) {
  BM_CompilePattern(
      state,
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
}
BENCHMARK(BM_CompileQwen2);

// DeepSeek-V3 patterns (multiple Split pre-tokenizers).
void BM_CompileDeepSeekNumbers(benchmark::State& state) {
  BM_CompilePattern(state, "\\p{N}{1,3}");
}
BENCHMARK(BM_CompileDeepSeekNumbers);

// DeepSeek CJK/Hiragana/Katakana pattern (exact Unicode codepoint ranges).
void BM_CompileDeepSeekCJK(benchmark::State& state) {
  // [一-龥] = CJK Unified Ideographs, [\u3040-\u309F] = Hiragana,
  // [\u30A0-\u30FF] = Katakana
  BM_CompilePattern(state, "[一-龥\u3040-\u309F\u30A0-\u30FF]+");
}
BENCHMARK(BM_CompileDeepSeekCJK);

// DeepSeek main Split pattern (most complex production pattern).
void BM_CompileDeepSeekMain(benchmark::State& state) {
  BM_CompilePattern(state,
                    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|"
                    "[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| "
                    "?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
}
BENCHMARK(BM_CompileDeepSeekMain);

//===----------------------------------------------------------------------===//
// Compile and Load (Combined)
//===----------------------------------------------------------------------===//

// Benchmark compile + load together (typical usage pattern).
void BM_CompileAndLoad(benchmark::State& state, const char* pattern) {
  for (auto _ : state) {
    iree_tokenizer_regex_dfa_t dfa;
    uint8_t* storage = nullptr;
    iree_tokenizer_regex_compile_error_t error = {0};

    iree_status_t status = iree_tokenizer_regex_compile_and_load(
        iree_make_cstring_view(pattern),
        IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
        &dfa, &storage, &error);

    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Compile and load failed");
      iree_status_ignore(status);
      return;
    }

    benchmark::DoNotOptimize(dfa.header);
    benchmark::DoNotOptimize(dfa.transitions);

    iree_allocator_free(iree_allocator_system(), storage);
  }
}

void BM_CompileAndLoadSimple(benchmark::State& state) {
  BM_CompileAndLoad(state, "[a-z]+");
}
BENCHMARK(BM_CompileAndLoadSimple);

void BM_CompileAndLoadGPT2(benchmark::State& state) {
  BM_CompileAndLoad(state,
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
                    "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
}
BENCHMARK(BM_CompileAndLoadGPT2);

//===----------------------------------------------------------------------===//
// Pattern Length Scaling
//===----------------------------------------------------------------------===//

// Generate patterns of increasing complexity.
class PatternScaling : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    int64_t complexity = state.range(0);

    // Build a pattern with N alternation branches.
    pattern_.clear();
    for (int64_t i = 0; i < complexity; ++i) {
      if (i > 0) pattern_ += "|";
      // Each branch is "aaa..." with i+1 'a' characters.
      for (int64_t j = 0; j <= i; ++j) {
        pattern_ += 'a' + (j % 26);
      }
    }
  }

 protected:
  std::string pattern_;
};

BENCHMARK_DEFINE_F(PatternScaling, Alternation)(benchmark::State& state) {
  for (auto _ : state) {
    uint8_t* dfa_data = nullptr;
    iree_host_size_t dfa_size = 0;
    iree_tokenizer_regex_compile_error_t error = {0};

    iree_status_t status = iree_tokenizer_regex_compile(
        iree_make_string_view(pattern_.data(), pattern_.size()),
        IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
        &dfa_data, &dfa_size, &error);

    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Compilation failed");
      iree_status_ignore(status);
      return;
    }

    benchmark::DoNotOptimize(dfa_data);
    state.counters["dfa_size"] = static_cast<double>(dfa_size);

    iree_allocator_free(iree_allocator_system(), dfa_data);
  }
}
BENCHMARK_REGISTER_F(PatternScaling, Alternation)
    ->Arg(2)
    ->Arg(5)
    ->Arg(10)
    ->Arg(20)
    ->Arg(50);

//===----------------------------------------------------------------------===//
// Pathological Patterns (should fail fast, not hang)
//===----------------------------------------------------------------------===//

// These patterns were discovered by fuzzing and caused multi-second compile
// times before iteration limits were tightened. They should now fail with
// RESOURCE_EXHAUSTED within ~100-300ms due to iteration limits.

void BM_PathologicalManyDots(benchmark::State& state) {
  // Pattern with many dots - each dot expands to 256 transitions.
  BM_CompilePattern(state, "..*.......................");
}
BENCHMARK(BM_PathologicalManyDots);

void BM_PathologicalNestedQuantifiers(benchmark::State& state) {
  // Nested quantifiers with wildcards.
  BM_CompilePattern(state, "(.*)+");
}
BENCHMARK(BM_PathologicalNestedQuantifiers);

void BM_PathologicalDotAlternation(benchmark::State& state) {
  // Alternation with wildcards.
  BM_CompilePattern(state, ".*a|.*b|.*c|.*d|.*e");
}
BENCHMARK(BM_PathologicalDotAlternation);

void BM_PathologicalLongDotSequence(benchmark::State& state) {
  // 50 dots in a row - 50 states with 256 transitions each.
  BM_CompilePattern(state,
                    "..................................................");
}
BENCHMARK(BM_PathologicalLongDotSequence);

}  // namespace
