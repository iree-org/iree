// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/regex/exec.h"

// DFA regex executor benchmarks.
// Measures throughput in bytes/s at various input scales.
// Target: >100 MB/s on single core.

namespace {

//===----------------------------------------------------------------------===//
// Test DFA Builder
//===----------------------------------------------------------------------===//

class TestDfaBuilder {
 public:
  void SetNumStates(uint16_t num_states) { num_states_ = num_states; }
  void SetStartState(uint16_t start) { start_state_ = start; }
  void SetFlags(uint16_t flags) { flags_ = flags; }

  void AddTransition(uint16_t state, uint8_t byte, uint16_t next_state) {
    transitions_.push_back({state, byte, next_state});
  }

  void AddTransitionRange(uint16_t state, uint8_t lo, uint8_t hi,
                          uint16_t next_state) {
    for (int b = lo; b <= hi; ++b) {
      AddTransition(state, (uint8_t)b, next_state);
    }
  }

  void SetAccepting(uint16_t state) { accepting_.insert(state); }

  std::vector<uint8_t> Build() {
    size_t header_size = sizeof(iree_tokenizer_regex_dfa_header_t);
    size_t trans_size = (size_t)num_states_ * 256 * sizeof(uint16_t);
    size_t bitmap_words = (num_states_ + 63) / 64;
    size_t bitmap_size = bitmap_words * sizeof(uint64_t);
    size_t total_size = header_size + trans_size + bitmap_size;

    std::vector<uint8_t> data(total_size, 0);

    auto* header = (iree_tokenizer_regex_dfa_header_t*)data.data();
    header->magic = IREE_TOKENIZER_UTIL_REGEX_DFA_MAGIC;
    header->version = IREE_TOKENIZER_UTIL_REGEX_DFA_VERSION;
    header->flags = flags_;
    header->num_states = num_states_;
    header->num_accepting = (uint16_t)accepting_.size();
    header->start_state = start_state_;

    auto* trans = (uint16_t*)(data.data() + header_size);
    for (size_t i = 0; i < (size_t)num_states_ * 256; ++i) {
      trans[i] = IREE_TOKENIZER_UTIL_REGEX_NO_TRANSITION;
    }
    for (const auto& t : transitions_) {
      trans[(size_t)t.state * 256 + t.byte] = t.next_state;
    }

    auto* bitmap = (uint64_t*)(data.data() + header_size + trans_size);
    for (uint16_t state : accepting_) {
      bitmap[state / 64] |= (1ULL << (state % 64));
    }

    return data;
  }

 private:
  struct Transition {
    uint16_t state;
    uint8_t byte;
    uint16_t next_state;
  };
  uint16_t num_states_ = 0;
  uint16_t start_state_ = 0;
  uint16_t flags_ = 0;
  std::vector<Transition> transitions_;
  std::set<uint16_t> accepting_;
};

//===----------------------------------------------------------------------===//
// DFA Patterns
//===----------------------------------------------------------------------===//

static std::vector<uint8_t> BuildLowerAlphaDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransitionRange(0, 'a', 'z', 1);
  builder.AddTransitionRange(1, 'a', 'z', 1);
  builder.SetAccepting(1);
  return builder.Build();
}

static std::vector<uint8_t> BuildDigitsDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  builder.AddTransitionRange(0, '0', '9', 1);
  builder.AddTransitionRange(1, '0', '9', 1);
  builder.SetAccepting(1);
  return builder.Build();
}

static std::vector<uint8_t> BuildNonSpaceDfa() {
  TestDfaBuilder builder;
  builder.SetNumStates(2);
  builder.SetStartState(0);
  for (int b = 0; b < 256; ++b) {
    if (b != ' ') {
      builder.AddTransition(0, (uint8_t)b, 1);
      builder.AddTransition(1, (uint8_t)b, 1);
    }
  }
  builder.SetAccepting(1);
  return builder.Build();
}

//===----------------------------------------------------------------------===//
// Test Data Generation
//===----------------------------------------------------------------------===//

static std::string GenerateWords(size_t total_bytes, size_t avg_word_len) {
  std::string result;
  result.reserve(total_bytes);
  uint32_t seed = 12345;
  while (result.size() < total_bytes) {
    seed = seed * 1103515245 + 12345;
    size_t word_len = avg_word_len / 2 + (seed >> 16) % (avg_word_len + 1);
    if (word_len == 0) word_len = 1;
    for (size_t i = 0; i < word_len && result.size() < total_bytes; ++i) {
      seed = seed * 1103515245 + 12345;
      result.push_back('a' + (seed >> 16) % 26);
    }
    if (result.size() < total_bytes) result.push_back(' ');
  }
  return result;
}

static std::string GenerateDigitsOnly(size_t total_bytes) {
  std::string result;
  result.reserve(total_bytes);
  uint32_t seed = 99999;
  while (result.size() < total_bytes) {
    seed = seed * 1103515245 + 12345;
    result.push_back('0' + (seed >> 16) % 10);
    if (result.size() % 5 == 0 && result.size() < total_bytes) {
      result.push_back(' ');
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Execution Throughput Benchmarks
//===----------------------------------------------------------------------===//

class DfaExecBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    int64_t size = state.range(0);
    text_ = GenerateWords(size, 5);

    if (alpha_dfa_.empty()) {
      alpha_dfa_ = BuildLowerAlphaDfa();
      digits_dfa_ = BuildDigitsDfa();
      nonspace_dfa_ = BuildNonSpaceDfa();
    }
  }

  void TearDown(benchmark::State&) override { text_.clear(); }

 protected:
  static std::vector<uint8_t> alpha_dfa_;
  static std::vector<uint8_t> digits_dfa_;
  static std::vector<uint8_t> nonspace_dfa_;
  std::string text_;
};

std::vector<uint8_t> DfaExecBenchmark::alpha_dfa_;
std::vector<uint8_t> DfaExecBenchmark::digits_dfa_;
std::vector<uint8_t> DfaExecBenchmark::nonspace_dfa_;

// [a-z]+ pattern - typical word matching.
BENCHMARK_DEFINE_F(DfaExecBenchmark, Alpha)(benchmark::State& state) {
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(alpha_dfa_.data(), alpha_dfa_.size()), &dfa);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("dfa load failed");
    iree_status_ignore(status);
    return;
  }

  iree_string_view_t text = iree_make_string_view(text_.data(), text_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    status = iree_tokenizer_regex_count_matches(&dfa, text, &count);
    benchmark::DoNotOptimize(count);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("exec failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * text_.size());
}
BENCHMARK_REGISTER_F(DfaExecBenchmark, Alpha)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(1000000);

// [0-9]+ pattern on text with no digits - measures no-match throughput.
BENCHMARK_DEFINE_F(DfaExecBenchmark, DigitsNoMatch)(benchmark::State& state) {
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(digits_dfa_.data(), digits_dfa_.size()), &dfa);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("dfa load failed");
    iree_status_ignore(status);
    return;
  }

  iree_string_view_t text = iree_make_string_view(text_.data(), text_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    status = iree_tokenizer_regex_count_matches(&dfa, text, &count);
    benchmark::DoNotOptimize(count);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("exec failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * text_.size());
}
BENCHMARK_REGISTER_F(DfaExecBenchmark, DigitsNoMatch)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(1000000);

// [^ ]+ pattern - word boundary splitting.
BENCHMARK_DEFINE_F(DfaExecBenchmark, NonSpace)(benchmark::State& state) {
  iree_tokenizer_regex_dfa_t dfa;
  iree_status_t status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(nonspace_dfa_.data(), nonspace_dfa_.size()),
      &dfa);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("dfa load failed");
    iree_status_ignore(status);
    return;
  }

  iree_string_view_t text = iree_make_string_view(text_.data(), text_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    status = iree_tokenizer_regex_count_matches(&dfa, text, &count);
    benchmark::DoNotOptimize(count);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("exec failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * text_.size());
}
BENCHMARK_REGISTER_F(DfaExecBenchmark, NonSpace)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(1000000);

//===----------------------------------------------------------------------===//
// has_match - Early Exit vs Full Scan
//===----------------------------------------------------------------------===//

class HasMatchBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State&) override {
    if (alpha_dfa_.empty()) {
      alpha_dfa_ = BuildLowerAlphaDfa();
    }
    words_1m_ = GenerateWords(1000000, 5);
    digits_1m_ = GenerateDigitsOnly(1000000);
  }

 protected:
  static std::vector<uint8_t> alpha_dfa_;
  static std::string words_1m_;
  static std::string digits_1m_;
};

std::vector<uint8_t> HasMatchBenchmark::alpha_dfa_;
std::string HasMatchBenchmark::words_1m_;
std::string HasMatchBenchmark::digits_1m_;

// Match at start - should exit immediately.
BENCHMARK_DEFINE_F(HasMatchBenchmark, EarlyMatch)(benchmark::State& state) {
  iree_tokenizer_regex_dfa_t dfa;
  iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(alpha_dfa_.data(), alpha_dfa_.size()), &dfa);

  iree_string_view_t text =
      iree_make_string_view(words_1m_.data(), words_1m_.size());

  for (auto _ : state) {
    bool has_match = false;
    iree_tokenizer_regex_has_match(&dfa, text, &has_match);
    benchmark::DoNotOptimize(has_match);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(HasMatchBenchmark, EarlyMatch);

// No match - must scan entire input.
BENCHMARK_DEFINE_F(HasMatchBenchmark, NoMatch)(benchmark::State& state) {
  iree_tokenizer_regex_dfa_t dfa;
  iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(alpha_dfa_.data(), alpha_dfa_.size()), &dfa);

  iree_string_view_t text =
      iree_make_string_view(digits_1m_.data(), digits_1m_.size());

  for (auto _ : state) {
    bool has_match = false;
    iree_tokenizer_regex_has_match(&dfa, text, &has_match);
    benchmark::DoNotOptimize(has_match);
  }
  state.SetBytesProcessed(state.iterations() * digits_1m_.size());
}
BENCHMARK_REGISTER_F(HasMatchBenchmark, NoMatch);

//===----------------------------------------------------------------------===//
// Streaming Execution - Chunk Size Impact
//===----------------------------------------------------------------------===//

static iree_status_t CountCallback(void* user_data,
                                   iree_tokenizer_regex_match_t match) {
  (void)match;
  (*(iree_host_size_t*)user_data)++;
  return iree_ok_status();
}

class StreamingBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    chunk_size_ = state.range(0);
    if (alpha_dfa_.empty()) {
      alpha_dfa_ = BuildLowerAlphaDfa();
      text_1m_ = GenerateWords(1000000, 5);
    }
  }

 protected:
  static std::vector<uint8_t> alpha_dfa_;
  static std::string text_1m_;
  int64_t chunk_size_ = 0;
};

std::vector<uint8_t> StreamingBenchmark::alpha_dfa_;
std::string StreamingBenchmark::text_1m_;

BENCHMARK_DEFINE_F(StreamingBenchmark, ChunkSize)(benchmark::State& state) {
  iree_tokenizer_regex_dfa_t dfa;
  iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(alpha_dfa_.data(), alpha_dfa_.size()), &dfa);

  for (auto _ : state) {
    iree_tokenizer_regex_exec_state_t exec_state;
    iree_tokenizer_regex_exec_initialize(&exec_state, &dfa);

    iree_host_size_t count = 0;
    for (size_t offset = 0; offset < text_1m_.size();
         offset += (size_t)chunk_size_) {
      size_t length = std::min((size_t)chunk_size_, text_1m_.size() - offset);
      iree_tokenizer_regex_exec_feed(
          &dfa, &exec_state,
          iree_make_string_view(text_1m_.data() + offset, length), offset,
          /*stride=*/NULL, CountCallback, &count);
    }
    iree_tokenizer_regex_exec_finalize(&dfa, &exec_state, text_1m_.size(),
                                       CountCallback, &count);
    benchmark::DoNotOptimize(count);
  }
  state.SetBytesProcessed(state.iterations() * text_1m_.size());
}
BENCHMARK_REGISTER_F(StreamingBenchmark, ChunkSize)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(1000000);

//===----------------------------------------------------------------------===//
// DFA Loading
//===----------------------------------------------------------------------===//

void BM_DfaLoad(benchmark::State& state) {
  static auto dfa_data = BuildLowerAlphaDfa();

  for (auto _ : state) {
    iree_tokenizer_regex_dfa_t dfa;
    iree_tokenizer_regex_dfa_load(
        iree_make_const_byte_span(dfa_data.data(), dfa_data.size()), &dfa);
    benchmark::DoNotOptimize(dfa.header);
  }
}
BENCHMARK(BM_DfaLoad);

//===----------------------------------------------------------------------===//
// Compiled Pattern Benchmarks
//===----------------------------------------------------------------------===//

// RAII wrapper for compiled regex patterns.
// Handles allocation and cleanup of DFA binary data.
class CompiledPattern {
 public:
  explicit CompiledPattern(const char* pattern) {
    iree_tokenizer_regex_compile_error_t error = {0};
    iree_status_t status = iree_tokenizer_regex_compile(
        iree_make_cstring_view(pattern),
        IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
        &data_, &size_, &error);
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, "Failed to compile pattern '%s' at position %zu: %s\n",
              pattern, error.position, error.message);
      iree_status_ignore(status);
      data_ = nullptr;
      size_ = 0;
      return;
    }

    status = iree_tokenizer_regex_dfa_load(
        iree_make_const_byte_span(data_, size_), &dfa_);
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, "Failed to load compiled DFA for '%s'\n", pattern);
      iree_status_ignore(status);
      iree_allocator_free(iree_allocator_system(), data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  ~CompiledPattern() {
    if (data_) {
      iree_allocator_free(iree_allocator_system(), data_);
    }
  }

  // Non-copyable.
  CompiledPattern(const CompiledPattern&) = delete;
  CompiledPattern& operator=(const CompiledPattern&) = delete;

  // Moveable.
  CompiledPattern(CompiledPattern&& other) noexcept
      : data_(other.data_), size_(other.size_), dfa_(other.dfa_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  CompiledPattern& operator=(CompiledPattern&& other) noexcept {
    if (this != &other) {
      if (data_) iree_allocator_free(iree_allocator_system(), data_);
      data_ = other.data_;
      size_ = other.size_;
      dfa_ = other.dfa_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  bool valid() const { return data_ != nullptr; }
  const iree_tokenizer_regex_dfa_t* dfa() const {
    return valid() ? &dfa_ : nullptr;
  }

 private:
  uint8_t* data_ = nullptr;
  iree_host_size_t size_ = 0;
  iree_tokenizer_regex_dfa_t dfa_ = {};
};

// GPT-2 pre-tokenizer pattern (used by GPT-2, RoBERTa, BART, etc).
static const char kGPT2Pattern[] =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
    "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// Llama/Mistral pattern (case-insensitive contractions, multi-digit numbers).
static const char kLlamaPattern[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
    "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// Simple word pattern (just Unicode letters).
static const char kSimpleWordPattern[] = "\\p{L}+";

// Whitespace split pattern.
static const char kWhitespaceSplitPattern[] = "\\S+";

class CompiledPatternBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    // Compile patterns on first use.
    if (!gpt2_pattern_) {
      gpt2_pattern_ = std::make_unique<CompiledPattern>(kGPT2Pattern);
      if (!gpt2_pattern_->valid()) {
        state.SkipWithError("Failed to compile GPT-2 pattern");
        return;
      }
    }
    if (!llama_pattern_) {
      llama_pattern_ = std::make_unique<CompiledPattern>(kLlamaPattern);
    }
    if (!simple_word_pattern_) {
      simple_word_pattern_ =
          std::make_unique<CompiledPattern>(kSimpleWordPattern);
    }
    if (!whitespace_pattern_) {
      whitespace_pattern_ =
          std::make_unique<CompiledPattern>(kWhitespaceSplitPattern);
    }

    if (shakespeare_.empty()) {
      shakespeare_ = GenerateWords(1000000, 5);
    }
  }

 protected:
  static std::unique_ptr<CompiledPattern> gpt2_pattern_;
  static std::unique_ptr<CompiledPattern> llama_pattern_;
  static std::unique_ptr<CompiledPattern> simple_word_pattern_;
  static std::unique_ptr<CompiledPattern> whitespace_pattern_;
  static std::string shakespeare_;
};

std::unique_ptr<CompiledPattern> CompiledPatternBenchmark::gpt2_pattern_;
std::unique_ptr<CompiledPattern> CompiledPatternBenchmark::llama_pattern_;
std::unique_ptr<CompiledPattern> CompiledPatternBenchmark::simple_word_pattern_;
std::unique_ptr<CompiledPattern> CompiledPatternBenchmark::whitespace_pattern_;
std::string CompiledPatternBenchmark::shakespeare_;

// GPT-2 pattern on Shakespeare text - measures real-world tokenizer perf.
BENCHMARK_DEFINE_F(CompiledPatternBenchmark, GPT2Pattern)
(benchmark::State& state) {
  if (!gpt2_pattern_ || !gpt2_pattern_->valid()) {
    state.SkipWithError("GPT-2 pattern not compiled");
    return;
  }

  iree_string_view_t text =
      iree_make_string_view(shakespeare_.data(), shakespeare_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    iree_status_t status =
        iree_tokenizer_regex_count_matches(gpt2_pattern_->dfa(), text, &count);
    benchmark::DoNotOptimize(count);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("exec failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * shakespeare_.size());
}
BENCHMARK_REGISTER_F(CompiledPatternBenchmark, GPT2Pattern)
    ->Unit(benchmark::kMillisecond);

// Llama/Mistral pattern - more complex than GPT-2.
BENCHMARK_DEFINE_F(CompiledPatternBenchmark, LlamaPattern)
(benchmark::State& state) {
  if (!llama_pattern_ || !llama_pattern_->valid()) {
    state.SkipWithError("Llama pattern not compiled");
    return;
  }

  iree_string_view_t text =
      iree_make_string_view(shakespeare_.data(), shakespeare_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    iree_tokenizer_regex_count_matches(llama_pattern_->dfa(), text, &count);
    benchmark::DoNotOptimize(count);
  }
  state.SetBytesProcessed(state.iterations() * shakespeare_.size());
}
BENCHMARK_REGISTER_F(CompiledPatternBenchmark, LlamaPattern)
    ->Unit(benchmark::kMillisecond);

// Simple \p{L}+ pattern - baseline for Unicode category matching.
BENCHMARK_DEFINE_F(CompiledPatternBenchmark, SimpleWordPattern)
(benchmark::State& state) {
  if (!simple_word_pattern_ || !simple_word_pattern_->valid()) {
    state.SkipWithError("Simple word pattern not compiled");
    return;
  }

  iree_string_view_t text =
      iree_make_string_view(shakespeare_.data(), shakespeare_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    iree_tokenizer_regex_count_matches(simple_word_pattern_->dfa(), text,
                                       &count);
    benchmark::DoNotOptimize(count);
  }
  state.SetBytesProcessed(state.iterations() * shakespeare_.size());
}
BENCHMARK_REGISTER_F(CompiledPatternBenchmark, SimpleWordPattern)
    ->Unit(benchmark::kMillisecond);

// \S+ pattern - simple non-whitespace splitting.
BENCHMARK_DEFINE_F(CompiledPatternBenchmark, WhitespaceSplit)
(benchmark::State& state) {
  if (!whitespace_pattern_ || !whitespace_pattern_->valid()) {
    state.SkipWithError("Whitespace pattern not compiled");
    return;
  }

  iree_string_view_t text =
      iree_make_string_view(shakespeare_.data(), shakespeare_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    iree_tokenizer_regex_count_matches(whitespace_pattern_->dfa(), text,
                                       &count);
    benchmark::DoNotOptimize(count);
  }
  state.SetBytesProcessed(state.iterations() * shakespeare_.size());
}
BENCHMARK_REGISTER_F(CompiledPatternBenchmark, WhitespaceSplit)
    ->Unit(benchmark::kMillisecond);

//===----------------------------------------------------------------------===//
// Pathological Pattern Benchmarks
//===----------------------------------------------------------------------===//

// Pattern with many '.' (any byte) metacharacters. Each '.' creates 256
// transitions in the DFA, and sequences of dots can cause state explosion
// during compilation or slow execution due to the large transition tables.
static const char kPathologicalDotPattern[] = "..*.......................";

// Simpler version isolating the core issue: repeated '.' metacharacters.
static const char kRepeatedDotPattern[] = "....................";

// Pattern with alternation and dots. Tests whether alternation combined with
// wildcards causes additional slowdown beyond the dot expansion itself.
static const char kDotAlternationPattern[] = ".*a|.*b|.*c";

class PathologicalBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    if (!dot_pattern_) {
      dot_pattern_ = std::make_unique<CompiledPattern>(kPathologicalDotPattern);
    }
    if (!repeated_dot_pattern_) {
      repeated_dot_pattern_ =
          std::make_unique<CompiledPattern>(kRepeatedDotPattern);
    }
    if (!dot_alternation_pattern_) {
      dot_alternation_pattern_ =
          std::make_unique<CompiledPattern>(kDotAlternationPattern);
    }
    if (binary_input_.empty()) {
      // Generate binary input similar to the fuzzer's slow-unit input.
      binary_input_.resize(1000);
      for (size_t i = 0; i < binary_input_.size(); ++i) {
        binary_input_[i] = static_cast<char>((i * 37 + 0xe5) % 256);
      }
    }
  }

 protected:
  static std::unique_ptr<CompiledPattern> dot_pattern_;
  static std::unique_ptr<CompiledPattern> repeated_dot_pattern_;
  static std::unique_ptr<CompiledPattern> dot_alternation_pattern_;
  static std::string binary_input_;
};

std::unique_ptr<CompiledPattern> PathologicalBenchmark::dot_pattern_;
std::unique_ptr<CompiledPattern> PathologicalBenchmark::repeated_dot_pattern_;
std::unique_ptr<CompiledPattern>
    PathologicalBenchmark::dot_alternation_pattern_;
std::string PathologicalBenchmark::binary_input_;

// Measures performance with many '.' metacharacters.
BENCHMARK_DEFINE_F(PathologicalBenchmark, ManyDots)(benchmark::State& state) {
  if (!dot_pattern_ || !dot_pattern_->valid()) {
    state.SkipWithError("Dot pattern not compiled");
    return;
  }

  iree_string_view_t text =
      iree_make_string_view(binary_input_.data(), binary_input_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    iree_status_t status =
        iree_tokenizer_regex_count_matches(dot_pattern_->dfa(), text, &count);
    benchmark::DoNotOptimize(count);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("exec failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * binary_input_.size());
}
BENCHMARK_REGISTER_F(PathologicalBenchmark, ManyDots)
    ->Unit(benchmark::kMillisecond);

// Repeated dots pattern baseline.
BENCHMARK_DEFINE_F(PathologicalBenchmark, RepeatedDots)
(benchmark::State& state) {
  if (!repeated_dot_pattern_ || !repeated_dot_pattern_->valid()) {
    state.SkipWithError("Repeated dot pattern not compiled");
    return;
  }

  iree_string_view_t text =
      iree_make_string_view(binary_input_.data(), binary_input_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    iree_tokenizer_regex_count_matches(repeated_dot_pattern_->dfa(), text,
                                       &count);
    benchmark::DoNotOptimize(count);
  }
  state.SetBytesProcessed(state.iterations() * binary_input_.size());
}
BENCHMARK_REGISTER_F(PathologicalBenchmark, RepeatedDots)
    ->Unit(benchmark::kMillisecond);

// Dot alternation pattern - tests alternation with wildcards.
BENCHMARK_DEFINE_F(PathologicalBenchmark, DotAlternation)
(benchmark::State& state) {
  if (!dot_alternation_pattern_ || !dot_alternation_pattern_->valid()) {
    state.SkipWithError("Dot alternation pattern not compiled");
    return;
  }

  iree_string_view_t text =
      iree_make_string_view(binary_input_.data(), binary_input_.size());

  for (auto _ : state) {
    iree_host_size_t count = 0;
    iree_tokenizer_regex_count_matches(dot_alternation_pattern_->dfa(), text,
                                       &count);
    benchmark::DoNotOptimize(count);
  }
  state.SetBytesProcessed(state.iterations() * binary_input_.size());
}
BENCHMARK_REGISTER_F(PathologicalBenchmark, DotAlternation)
    ->Unit(benchmark::kMillisecond);

}  // namespace
