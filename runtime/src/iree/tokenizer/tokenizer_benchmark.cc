// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/tokenizer/huggingface/tokenizer_json.h"
#include "iree/tokenizer/testdata/tokenizer_testdata.h"
#include "iree/tokenizer/tokenizer.h"

// End-to-end tokenizer benchmarks testing encode/decode performance
// in both streaming and non-streaming modes.

namespace {

//===----------------------------------------------------------------------===//
// Test Data Access
//===----------------------------------------------------------------------===//

static iree_string_view_t GetTestFile(const char* name) {
  const struct iree_file_toc_t* file_toc = iree_tokenizer_testdata_create();
  for (size_t i = 0; i < iree_tokenizer_testdata_size(); ++i) {
    if (strcmp(file_toc[i].name, name) == 0) {
      return iree_make_string_view((const char*)file_toc[i].data,
                                   file_toc[i].size);
    }
  }
  return iree_string_view_empty();
}

//===----------------------------------------------------------------------===//
// Test Data Generation
//===----------------------------------------------------------------------===//

// Base words that exercise different tokenizer paths.
// All words must be tokenizable by both the BERT and BPE synthetic vocabs.
// Includes punctuation to exercise merge operations in BPE (not just lookups).
static const char* kBaseWords[] = {
    "The",  "quick",        "brown", "fox",   "jumps", "over",  "the",  "lazy",
    "dog.", "Hello",        "world", "This",  "is",    "a",     "test", "with",
    "some", "punctuation,", "and",   "words", "too.",  "Mixed", "in",   "for",
};
static const size_t kBaseWordCount = sizeof(kBaseWords) / sizeof(kBaseWords[0]);

static std::string GenerateTestText(int64_t word_count) {
  if (word_count <= 0) return "";
  std::string result;
  result.reserve(word_count * 8);
  for (int64_t i = 0; i < word_count; ++i) {
    if (i > 0) result += ' ';
    result += kBaseWords[i % kBaseWordCount];
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Streaming Callbacks
//===----------------------------------------------------------------------===//

struct StreamingEncodeContext {
  int64_t total_tokens;
};

static iree_status_t CountTokensCallback(void* user_data,
                                         iree_tokenizer_id_list_t ids) {
  auto* ctx = static_cast<StreamingEncodeContext*>(user_data);
  ctx->total_tokens += ids.count;
  return iree_ok_status();
}

struct StreamingDecodeContext {
  int64_t total_bytes;
};

static iree_status_t CountBytesCallback(void* user_data,
                                        iree_string_view_list_t strings) {
  auto* ctx = static_cast<StreamingDecodeContext*>(user_data);
  for (iree_host_size_t i = 0; i < strings.count; ++i) {
    ctx->total_bytes += strings.values[i].size;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BERT Tokenizer Fixture
//===----------------------------------------------------------------------===//

class BertTokenizerBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    if (!tokenizer_) {
      iree_string_view_t json = GetTestFile("benchmark_bert.json");
      iree_status_t status = iree_tokenizer_from_huggingface_json(
          json, iree_allocator_system(), &tokenizer_);
      if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_ignore(status);
        state.SkipWithError("failed to load BERT tokenizer");
        return;
      }
    }
    word_count_ = state.range(0);
    text_ = GenerateTestText(word_count_);
  }

  void TearDown(benchmark::State&) override { text_.clear(); }

 protected:
  static iree_tokenizer_t* tokenizer_;
  int64_t word_count_ = 0;
  std::string text_;
};

iree_tokenizer_t* BertTokenizerBenchmark::tokenizer_ = nullptr;

BENCHMARK_DEFINE_F(BertTokenizerBenchmark, Encode)(benchmark::State& state) {
  std::vector<int32_t> ids(word_count_ * 2 + 16);
  iree_tokenizer_encode_options_t options = {0, 0};

  for (auto _ : state) {
    iree_host_size_t count = 0;
    iree_status_t status = iree_tokenizer_encode(
        tokenizer_, iree_make_string_view(text_.data(), text_.size()), options,
        ids.data(), ids.size(), &count);
    benchmark::DoNotOptimize(count);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("encode failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * text_.size());
}
BENCHMARK_REGISTER_F(BertTokenizerBenchmark, Encode)
    ->Arg(0)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

BENCHMARK_DEFINE_F(BertTokenizerBenchmark, EncodeStreaming)
(benchmark::State& state) {
  for (auto _ : state) {
    StreamingEncodeContext ctx = {0};
    iree_status_t status = iree_tokenizer_encode_streaming(
        tokenizer_, iree_make_string_view(text_.data(), text_.size()),
        IREE_TOKENIZER_ENCODE_FLAG_DEFAULT, CountTokensCallback, &ctx);
    benchmark::DoNotOptimize(ctx.total_tokens);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("streaming encode failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * text_.size());
}
BENCHMARK_REGISTER_F(BertTokenizerBenchmark, EncodeStreaming)
    ->Arg(0)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

BENCHMARK_DEFINE_F(BertTokenizerBenchmark, Decode)(benchmark::State& state) {
  // Pre-encode to get token IDs.
  std::vector<int32_t> ids(word_count_ * 2 + 16);
  iree_host_size_t id_count = 0;
  iree_tokenizer_encode_options_t options = {0, 0};
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(text_.data(), text_.size()), options,
      ids.data(), ids.size(), &id_count);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("pre-encode failed");
    iree_status_ignore(status);
    return;
  }

  std::vector<char> output(text_.size() * 2 + 16);

  for (auto _ : state) {
    iree_host_size_t output_size = 0;
    status = iree_tokenizer_decode(tokenizer_, ids.data(), id_count,
                                   IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
                                   output.data(), output.size(), &output_size);
    benchmark::DoNotOptimize(output_size);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("decode failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetItemsProcessed(state.iterations() * id_count);
}
BENCHMARK_REGISTER_F(BertTokenizerBenchmark, Decode)
    ->Arg(0)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

BENCHMARK_DEFINE_F(BertTokenizerBenchmark, DecodeStreaming)
(benchmark::State& state) {
  // Pre-encode to get token IDs.
  std::vector<int32_t> ids(word_count_ * 2 + 16);
  iree_host_size_t id_count = 0;
  iree_tokenizer_encode_options_t options = {0, 0};
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(text_.data(), text_.size()), options,
      ids.data(), ids.size(), &id_count);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("pre-encode failed");
    iree_status_ignore(status);
    return;
  }

  for (auto _ : state) {
    StreamingDecodeContext ctx = {0};
    status = iree_tokenizer_decode_streaming(tokenizer_, ids.data(), id_count,
                                             IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
                                             CountBytesCallback, &ctx);
    benchmark::DoNotOptimize(ctx.total_bytes);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("streaming decode failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetItemsProcessed(state.iterations() * id_count);
}
BENCHMARK_REGISTER_F(BertTokenizerBenchmark, DecodeStreaming)
    ->Arg(0)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

//===----------------------------------------------------------------------===//
// BPE Tokenizer Fixture
//===----------------------------------------------------------------------===//

class BpeTokenizerBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    if (!tokenizer_) {
      iree_string_view_t json = GetTestFile("benchmark_bpe.json");
      iree_status_t status = iree_tokenizer_from_huggingface_json(
          json, iree_allocator_system(), &tokenizer_);
      if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_ignore(status);
        state.SkipWithError("failed to load BPE tokenizer");
        return;
      }
    }
    word_count_ = state.range(0);
    text_ = GenerateTestText(word_count_);
  }

  void TearDown(benchmark::State&) override { text_.clear(); }

 protected:
  static iree_tokenizer_t* tokenizer_;
  int64_t word_count_ = 0;
  std::string text_;
};

iree_tokenizer_t* BpeTokenizerBenchmark::tokenizer_ = nullptr;

BENCHMARK_DEFINE_F(BpeTokenizerBenchmark, Encode)(benchmark::State& state) {
  std::vector<int32_t> ids(word_count_ * 4 + 16);
  iree_tokenizer_encode_options_t options = {0, 0};

  for (auto _ : state) {
    iree_host_size_t count = 0;
    iree_status_t status = iree_tokenizer_encode(
        tokenizer_, iree_make_string_view(text_.data(), text_.size()), options,
        ids.data(), ids.size(), &count);
    benchmark::DoNotOptimize(count);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("encode failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * text_.size());
}
BENCHMARK_REGISTER_F(BpeTokenizerBenchmark, Encode)
    ->Arg(0)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

BENCHMARK_DEFINE_F(BpeTokenizerBenchmark, EncodeStreaming)
(benchmark::State& state) {
  for (auto _ : state) {
    StreamingEncodeContext ctx = {0};
    iree_status_t status = iree_tokenizer_encode_streaming(
        tokenizer_, iree_make_string_view(text_.data(), text_.size()),
        IREE_TOKENIZER_ENCODE_FLAG_DEFAULT, CountTokensCallback, &ctx);
    benchmark::DoNotOptimize(ctx.total_tokens);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("streaming encode failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * text_.size());
}
BENCHMARK_REGISTER_F(BpeTokenizerBenchmark, EncodeStreaming)
    ->Arg(0)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

BENCHMARK_DEFINE_F(BpeTokenizerBenchmark, Decode)(benchmark::State& state) {
  // Pre-encode to get token IDs.
  std::vector<int32_t> ids(word_count_ * 4 + 16);
  iree_host_size_t id_count = 0;
  iree_tokenizer_encode_options_t options = {0, 0};
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(text_.data(), text_.size()), options,
      ids.data(), ids.size(), &id_count);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("pre-encode failed");
    iree_status_ignore(status);
    return;
  }

  std::vector<char> output(text_.size() * 2 + 16);

  for (auto _ : state) {
    iree_host_size_t output_size = 0;
    status = iree_tokenizer_decode(tokenizer_, ids.data(), id_count,
                                   IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
                                   output.data(), output.size(), &output_size);
    benchmark::DoNotOptimize(output_size);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("decode failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetItemsProcessed(state.iterations() * id_count);
}
BENCHMARK_REGISTER_F(BpeTokenizerBenchmark, Decode)
    ->Arg(0)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

BENCHMARK_DEFINE_F(BpeTokenizerBenchmark, DecodeStreaming)
(benchmark::State& state) {
  // Pre-encode to get token IDs.
  std::vector<int32_t> ids(word_count_ * 4 + 16);
  iree_host_size_t id_count = 0;
  iree_tokenizer_encode_options_t options = {0, 0};
  iree_status_t status = iree_tokenizer_encode(
      tokenizer_, iree_make_string_view(text_.data(), text_.size()), options,
      ids.data(), ids.size(), &id_count);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("pre-encode failed");
    iree_status_ignore(status);
    return;
  }

  for (auto _ : state) {
    StreamingDecodeContext ctx = {0};
    status = iree_tokenizer_decode_streaming(tokenizer_, ids.data(), id_count,
                                             IREE_TOKENIZER_DECODE_FLAG_DEFAULT,
                                             CountBytesCallback, &ctx);
    benchmark::DoNotOptimize(ctx.total_bytes);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("streaming decode failed");
      iree_status_ignore(status);
      return;
    }
  }
  state.SetItemsProcessed(state.iterations() * id_count);
}
BENCHMARK_REGISTER_F(BpeTokenizerBenchmark, DecodeStreaming)
    ->Arg(0)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000);

//===----------------------------------------------------------------------===//
// Tokenizer Creation Benchmarks
//===----------------------------------------------------------------------===//

void BM_TokenizerCreate_BERT(benchmark::State& state) {
  iree_string_view_t json = GetTestFile("benchmark_bert.json");

  for (auto _ : state) {
    iree_tokenizer_t* tokenizer = nullptr;
    iree_status_t status = iree_tokenizer_from_huggingface_json(
        json, iree_allocator_system(), &tokenizer);
    benchmark::DoNotOptimize(tokenizer);
    if (iree_status_is_ok(status)) {
      iree_tokenizer_free(tokenizer);
    } else {
      state.SkipWithError("create failed");
      iree_status_ignore(status);
      return;
    }
  }
}
BENCHMARK(BM_TokenizerCreate_BERT);

void BM_TokenizerCreate_BPE(benchmark::State& state) {
  iree_string_view_t json = GetTestFile("benchmark_bpe.json");

  for (auto _ : state) {
    iree_tokenizer_t* tokenizer = nullptr;
    iree_status_t status = iree_tokenizer_from_huggingface_json(
        json, iree_allocator_system(), &tokenizer);
    benchmark::DoNotOptimize(tokenizer);
    if (iree_status_is_ok(status)) {
      iree_tokenizer_free(tokenizer);
    } else {
      state.SkipWithError("create failed");
      iree_status_ignore(status);
      return;
    }
  }
}
BENCHMARK(BM_TokenizerCreate_BPE);

}  // namespace
